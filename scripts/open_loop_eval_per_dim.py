#!/usr/bin/env python3
"""Reproducible per-dimension open-loop evaluation for Nero LingBot VLA.

The evaluator feeds the real observation at an anchor frame to the policy,
predicts the complete action chunk, and compares horizon h with the dataset
action at anchor+h.  Episode-tail padding is never scored.

Example (train-replay diagnostic):

  CUDA_VISIBLE_DEVICES=4 python scripts/open_loop_eval_per_dim.py \
    --checkpoint new40=/path/to/global_step_40000/hf_ckpt \
    --data-path /path/to/mobile_transfer_lingbot_v3 \
    --robo-name nero_mobile_xyzquat \
    --episodes 0 1 --anchors-per-episode 4 --repeats 2 \
    --dataset-role train-replay --output-dir /path/to/eval
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
import yaml

from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
from lingbotvla.eval.open_loop_metrics import (
    ACTION_DIM,
    ACTION_DIMENSIONS,
    ACTION_KEYS,
    action_schema_records,
    aggregate_rotation_metrics,
    aggregate_scalar_metrics,
    aggregate_stochastic_metrics,
    build_scalar_sample_frame,
    concatenate_action_bounds,
    deterministic_noise_seed,
    finite_or_none,
    logical_actions,
    logical_normalized_actions,
    select_anchor_offsets,
)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


LOGGER = logging.getLogger("open_loop_eval_per_dim")


def _safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    if not safe:
        raise ValueError(f"Checkpoint label {label!r} has no usable filename characters")
    return safe


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=/path/to/hf_ckpt")
    label, path = value.split("=", 1)
    try:
        safe_label = _safe_label(label)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise argparse.ArgumentTypeError(f"checkpoint directory does not exist: {checkpoint}")
    if not list(checkpoint.glob("*.safetensors")):
        raise argparse.ArgumentTypeError(f"checkpoint has no .safetensors shards: {checkpoint}")
    return safe_label, checkpoint


def training_config_path(checkpoint: Path) -> Path:
    path = checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find checkpoint training config: {path}")
    return path


def load_training_config(checkpoint: Path) -> tuple[Path, dict]:
    path = training_config_path(checkpoint)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or "data" not in config or "train" not in config:
        raise ValueError(f"Invalid training config: {path}")
    return path, config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_file_manifest(checkpoint: Path) -> dict:
    shards = sorted(checkpoint.glob("*.safetensors"))
    index = checkpoint / "model.safetensors.index.json"
    return {
        "safetensor_shards": [
            {"name": shard.name, "size_bytes": shard.stat().st_size}
            for shard in shards
        ],
        "index_sha256": sha256_file(index) if index.is_file() else None,
    }


def resolve_project_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_dataset_bounds(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Dataset raw stats are missing: {stats_path}")
    with stats_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)
    return concatenate_action_bounds(stats)


def load_episode_table(dataset_path: Path) -> pd.DataFrame:
    files = sorted((dataset_path / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquet files under {dataset_path / 'meta/episodes'}")
    table = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "length", "dataset_from_index", "dataset_to_index"]) for path in files],
        ignore_index=True,
    )
    table = table.sort_values("episode_index").reset_index(drop=True)
    if table["episode_index"].duplicated().any():
        raise ValueError("Episode metadata contains duplicate episode_index values")
    actual_length = table["dataset_to_index"] - table["dataset_from_index"]
    if not np.array_equal(actual_length.to_numpy(), table["length"].to_numpy()):
        raise ValueError("Episode metadata length does not match dataset index boundaries")
    return table


def create_dataset(dataset_path: Path, chunk_size: int):
    repo_id = dataset_path.name
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_path)
    delta_timestamps = {
        action_key: [step / metadata.fps for step in range(chunk_size)]
        for action_key in ACTION_KEYS
    }
    dataset = LeRobotDataset(repo_id, root=dataset_path, delta_timestamps=delta_timestamps)
    return dataset, float(metadata.fps)


def to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def image_to_hwc_uint8(value: object) -> np.ndarray:
    image = to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got shape {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image in HWC or CHW layout, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        if finite_max <= 1.5:
            image = image * 255.0
        image = np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def prepare_observation(policy, item: Mapping[str, object]) -> dict:
    observation = dict(item)
    image_keys = tuple(policy.vla.feature_transform.org_features["images"])
    for key in image_keys:
        if key not in observation:
            raise KeyError(f"Dataset sample is missing configured image key: {key}")
        observation[key] = image_to_hwc_uint8(observation[key])
    # Actions are supervision only and should not enter deployment preprocessing.
    for key in list(observation):
        if key.startswith("action."):
            observation.pop(key)
    if "task" not in observation:
        raise KeyError("Dataset sample has no task string")
    return observation


def normalized_ground_truth(policy, item: Mapping[str, object]) -> np.ndarray:
    """Apply the checkpoint's exact training transform to obtain relative/normalized GT."""
    feature_transform = policy.vla.feature_transform
    previous = feature_transform.disabled_image_features
    feature_transform.disabled_image_features = True
    try:
        transformed = feature_transform.apply(dict(item), policy_eval=False)
    finally:
        feature_transform.disabled_image_features = previous
    return logical_normalized_actions(to_numpy(transformed["actions"]))


def validate_dataset_padding(item: Mapping[str, object], valid_length: int, chunk_size: int) -> None:
    """Cross-check metadata-derived validity against every LeRobot action pad mask."""
    expected = np.arange(chunk_size) >= valid_length
    for key in ACTION_KEYS:
        pad_key = f"{key}_is_pad"
        if pad_key not in item:
            raise KeyError(f"Dataset sample is missing {pad_key}")
        actual = to_numpy(item[pad_key]).astype(bool).reshape(-1)
        if actual.shape != (chunk_size,) or not np.array_equal(actual, expected):
            raise ValueError(
                f"Padding mismatch for {pad_key}: expected {valid_length} valid steps, "
                f"got mask shape={actual.shape}, valid={int(np.count_nonzero(~actual))}"
            )


class StreamingParquetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def build_anchor_table(
    episodes: pd.DataFrame,
    selected_episodes: Iterable[int],
    *,
    anchors_per_episode: int | None,
    anchor_stride: int | None,
    include_tail: bool,
    chunk_size: int,
    score_horizon: int,
) -> pd.DataFrame:
    by_episode = episodes.set_index("episode_index")
    rows = []
    for episode in selected_episodes:
        if episode not in by_episode.index:
            raise ValueError(f"Episode {episode} is not present in the dataset")
        row = by_episode.loc[episode]
        offsets = select_anchor_offsets(
            int(row["length"]),
            anchors_per_episode=anchors_per_episode,
            anchor_stride=anchor_stride,
            include_tail=include_tail,
            chunk_size=chunk_size,
        )
        for offset in offsets:
            remaining = int(row["length"]) - int(offset)
            dataset_valid_length = min(chunk_size, remaining)
            valid_length = min(score_horizon, remaining)
            rows.append(
                {
                    "episode": int(episode),
                    "anchor_frame": int(offset),
                    "global_index": int(row["dataset_from_index"]) + int(offset),
                    "valid_length": int(valid_length),
                    "dataset_valid_length": int(dataset_valid_length),
                    "episode_length": int(row["length"]),
                }
            )
    anchors = pd.DataFrame(rows)
    if anchors.empty:
        raise ValueError("Anchor selection is empty")
    return anchors


def load_policy(checkpoint: Path, norm_path: Path | None, robo_name: str, chunk_size: int, use_bf16: bool, use_compile: bool):
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    policy = LingbotVLAv2Server(
        path_to_pi_model=str(checkpoint),
        robot_norm_path=str(norm_path) if norm_path is not None else None,
        use_length=chunk_size,
        chunk_ret=True,
        use_bf16=use_bf16,
        use_fp32=not use_bf16,
        use_compile=use_compile,
    )
    if int(policy.config.chunk_size) != chunk_size or int(policy.config.n_action_steps) != chunk_size:
        raise ValueError(
            f"Checkpoint action chunk is {policy.config.n_action_steps}/{policy.config.chunk_size}, requested {chunk_size}"
        )
    policy.reset(robo_name)
    return policy


def evaluate_checkpoint(
    *,
    label: str,
    checkpoint: Path,
    policy,
    dataset,
    anchors: pd.DataFrame,
    repeats: int,
    base_seed: int,
    chunk_size: int,
    training_min: np.ndarray,
    training_max: np.ndarray,
    output_dir: Path,
    thresholds: tuple[float, float, float],
) -> tuple[Path, Path]:
    sample_path = output_dir / "samples" / f"{label}.parquet"
    rotation_path = output_dir / "rotation_samples" / f"{label}.parquet"
    sample_writer = StreamingParquetWriter(sample_path)
    rotation_writer = StreamingParquetWriter(rotation_path)
    try:
        total = len(anchors) * repeats
        progress = tqdm(total=total, desc=f"{label} inference", unit="chunk")
        for anchor in anchors.itertuples(index=False):
            item = dataset[int(anchor.global_index)]
            validate_dataset_padding(item, int(anchor.dataset_valid_length), chunk_size)
            raw_target = logical_actions({key: to_numpy(item[key]) for key in ACTION_KEYS})
            if raw_target.shape[0] != chunk_size:
                raise ValueError(
                    f"Dataset returned action chunk {raw_target.shape} at index {anchor.global_index}; expected horizon {chunk_size}"
                )
            observation = prepare_observation(policy, item)
            normalized_target = normalized_ground_truth(policy, item)
            for repeat in range(repeats):
                noise_seed = deterministic_noise_seed(base_seed, anchor.episode, anchor.anchor_frame, repeat)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(noise_seed)
                noise = torch.randn(
                    (1, chunk_size, int(policy.config.max_action_dim)),
                    generator=generator,
                    dtype=torch.float32,
                )
                prediction_dict = policy.infer(
                    observation,
                    return_normalized=True,
                    noise=noise,
                )
                normalized_prediction = logical_normalized_actions(prediction_dict.pop("_normalized_actions"))
                raw_prediction = logical_actions(prediction_dict)
                scalar_frame, rotation_frame = build_scalar_sample_frame(
                    checkpoint=label,
                    episode=int(anchor.episode),
                    anchor_frame=int(anchor.anchor_frame),
                    global_index=int(anchor.global_index),
                    repeat=repeat,
                    noise_seed=noise_seed,
                    target=raw_target,
                    predicted=raw_prediction,
                    normalized_target=normalized_target,
                    normalized_prediction=normalized_prediction,
                    valid_length=int(anchor.valid_length),
                    training_min=training_min,
                    training_max=training_max,
                    base_vx_threshold=thresholds[0],
                    base_wz_threshold=thresholds[1],
                    base_height_threshold=thresholds[2],
                )
                sample_writer.write(scalar_frame)
                rotation_writer.write(rotation_frame)
                progress.update(1)
        progress.close()
    finally:
        sample_writer.close()
        rotation_writer.close()
    if not sample_path.is_file() or not rotation_path.is_file():
        raise RuntimeError(f"Checkpoint {label} produced no evaluation samples")
    return sample_path, rotation_path


def aggregate_checkpoint(label: str, sample_path: Path, rotation_path: Path) -> dict[str, pd.DataFrame]:
    per_dim = []
    per_dim_horizon = []
    per_episode_dim = []
    per_repeat_dim = []
    base_activity = []
    stochastic_dim = []
    stochastic_dim_horizon = []
    identity = ["checkpoint", "dimension", "dimension_name", "group", "side", "unit", "model_slot"]
    columns = [
        *identity,
        "episode",
        "anchor_frame",
        "repeat",
        "horizon",
        "activity",
        "target",
        "prediction",
        "normalized_target",
        "normalized_prediction",
        "outside_training_range",
    ]
    for dimension in range(ACTION_DIM):
        frame = pd.read_parquet(sample_path, columns=columns, filters=[("dimension", "=", dimension)])
        if frame.empty:
            raise RuntimeError(f"No samples for {label} dimension {dimension}")
        per_dim.append(aggregate_scalar_metrics(frame, identity))
        per_dim_horizon.append(aggregate_scalar_metrics(frame, [*identity, "horizon"]))
        per_episode_dim.append(aggregate_scalar_metrics(frame, [*identity, "episode"]))
        per_repeat_dim.append(aggregate_scalar_metrics(frame, [*identity, "repeat"]))
        stochastic_dim.append(aggregate_stochastic_metrics(frame, identity))
        stochastic_dim_horizon.append(aggregate_stochastic_metrics(frame, [*identity, "horizon"]))
        if dimension >= 16:
            base_activity.append(aggregate_scalar_metrics(frame, [*identity, "activity"]))

    rotation = pd.read_parquet(rotation_path)
    rotation_overall = aggregate_rotation_metrics(rotation, ["checkpoint", "side"])
    rotation_horizon = aggregate_rotation_metrics(rotation, ["checkpoint", "side", "horizon"])
    rotation_episode = aggregate_rotation_metrics(rotation, ["checkpoint", "side", "episode"])
    return {
        "per_dim": pd.concat(per_dim, ignore_index=True),
        "per_dim_horizon": pd.concat(per_dim_horizon, ignore_index=True),
        "per_episode_dim": pd.concat(per_episode_dim, ignore_index=True),
        "per_repeat_dim": pd.concat(per_repeat_dim, ignore_index=True),
        "base_activity": pd.concat(base_activity, ignore_index=True),
        "stochastic_dim": pd.concat(stochastic_dim, ignore_index=True),
        "stochastic_dim_horizon": pd.concat(stochastic_dim_horizon, ignore_index=True),
        "rotation": rotation_overall,
        "rotation_horizon": rotation_horizon,
        "rotation_episode": rotation_episode,
    }


def save_plots(metrics: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    horizon_metrics = metrics["per_dim_horizon"]
    rotation_horizon = metrics["rotation_horizon"]

    for checkpoint in horizon_metrics["checkpoint"].unique():
        checkpoint_frame = horizon_metrics[horizon_metrics["checkpoint"] == checkpoint]
        matrix = checkpoint_frame.pivot(index="dimension_name", columns="horizon", values="nmae_q01_q99")
        order = [dimension.name for dimension in ACTION_DIMENSIONS]
        matrix = matrix.reindex(order)
        fig, ax = plt.subplots(figsize=(14, 8))
        image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_yticks(np.arange(len(order)), labels=order)
        ax.set_xlabel("Horizon step")
        ax.set_title(f"{checkpoint}: per-dimension NMAE (MAE / GT q01-q99 range)")
        fig.colorbar(image, ax=ax, label="NMAE")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{_safe_label(checkpoint)}_nmae_heatmap.png", dpi=160)
        plt.close(fig)

        for group in ("tcp_position", "tcp_quaternion", "gripper", "base"):
            group_frame = checkpoint_frame[checkpoint_frame["group"] == group]
            fig, ax = plt.subplots(figsize=(10, 5))
            for name, dimension_frame in group_frame.groupby("dimension_name", sort=False):
                ax.plot(dimension_frame["horizon"], dimension_frame["mae"], label=name)
            ax.set_xlabel("Horizon step")
            ax.set_ylabel("MAE (native unit)")
            ax.set_title(f"{checkpoint}: {group} MAE vs horizon")
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{_safe_label(checkpoint)}_{group}_mae_vs_horizon.png", dpi=160)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (checkpoint, side), frame in rotation_horizon.groupby(["checkpoint", "side"], sort=False):
        ax.plot(frame["horizon"], frame["mean_deg"], label=f"{checkpoint}:{side}")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Mean geodesic error (degrees)")
    ax.set_title("TCP orientation error vs horizon")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "rotation_error_vs_horizon.png", dpi=160)
    plt.close(fig)

    per_dim = metrics["per_dim"]
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [dimension.name for dimension in ACTION_DIMENSIONS]
    checkpoints = list(per_dim["checkpoint"].unique())
    width = 0.8 / max(len(checkpoints), 1)
    x = np.arange(len(names))
    for index, checkpoint in enumerate(checkpoints):
        frame = per_dim[per_dim["checkpoint"] == checkpoint].set_index("dimension_name").reindex(names)
        ax.bar(x + (index - (len(checkpoints) - 1) / 2) * width, frame["nmae_q01_q99"], width, label=checkpoint)
    ax.set_xticks(x, names, rotation=60, ha="right")
    ax.set_ylabel("NMAE (MAE / GT q01-q99 range)")
    ax.set_title("Checkpoint comparison by action dimension")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "checkpoint_comparison_nmae.png", dpi=160)
    plt.close(fig)


def build_trajectory_plot_frame(samples: pd.DataFrame) -> pd.DataFrame:
    """Align chunk predictions to episode frames for GT/inference trajectory plots.

    Several anchors (or stochastic repeats) may predict the same episode frame.
    GT must agree at that coordinate; inference is summarized by its mean and
    10th/90th percentiles so overlap is represented instead of silently picking
    one chunk.
    """
    required = {
        "checkpoint",
        "episode",
        "anchor_frame",
        "horizon",
        "dimension",
        "dimension_name",
        "unit",
        "target",
        "prediction",
    }
    missing = sorted(required.difference(samples.columns))
    if missing:
        raise ValueError(f"Trajectory samples are missing columns: {missing}")
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "checkpoint",
                "episode",
                "target_frame",
                "dimension",
                "dimension_name",
                "unit",
                "target",
                "prediction_mean",
                "prediction_p10",
                "prediction_p90",
                "prediction_count",
            ]
        )

    frame = samples.loc[:, sorted(required)].copy()
    frame["target_frame"] = (
        frame["anchor_frame"].to_numpy(dtype=np.int64)
        + frame["horizon"].to_numpy(dtype=np.int64)
    )
    group_columns = [
        "checkpoint",
        "episode",
        "target_frame",
        "dimension",
        "dimension_name",
        "unit",
    ]
    grouped = frame.groupby(group_columns, sort=True, observed=True, dropna=False)
    target_spread = grouped["target"].agg(lambda values: float(values.max() - values.min()))
    max_spread = float(target_spread.max())
    if not np.isfinite(max_spread) or max_spread > 1e-9:
        raise ValueError(
            "GT is inconsistent for predictions aligned to the same episode frame/dimension; "
            f"maximum spread={max_spread:.6g}"
        )
    trajectory = grouped.agg(
        target=("target", "first"),
        prediction_mean=("prediction", "mean"),
        prediction_p10=("prediction", lambda values: float(values.quantile(0.10))),
        prediction_p90=("prediction", lambda values: float(values.quantile(0.90))),
        prediction_count=("prediction", "size"),
    ).reset_index()
    numeric_columns = ["target", "prediction_mean", "prediction_p10", "prediction_p90"]
    if not np.isfinite(trajectory[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Trajectory plot data contain non-finite GT or inference values")
    return trajectory


def save_action_trajectory_plots(
    sample_paths: Mapping[str, Path],
    output_dir: Path,
    fps: float,
    *,
    episodes: Iterable[int] | None = None,
    dpi: int = 160,
) -> dict[str, list[Path]]:
    """Write one 19-dimension GT/inference overview per checkpoint and episode."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and > 0, got {fps}")
    if dpi <= 0:
        raise ValueError(f"dpi must be > 0, got {dpi}")
    plot_dir = output_dir / "plots" / "action_trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)
    requested_episodes = None if episodes is None else [int(episode) for episode in episodes]
    outputs: dict[str, list[Path]] = {}
    columns = [
        "checkpoint",
        "episode",
        "anchor_frame",
        "horizon",
        "dimension",
        "dimension_name",
        "unit",
        "target",
        "prediction",
    ]

    for label, sample_path in sample_paths.items():
        sample_path = Path(sample_path)
        if not sample_path.is_file():
            raise FileNotFoundError(f"Evaluation samples for {label} are missing: {sample_path}")
        if requested_episodes is None:
            episode_values = pd.read_parquet(sample_path, columns=["episode"])["episode"]
            label_episodes = sorted(episode_values.astype(int).unique().tolist())
        else:
            label_episodes = requested_episodes
        outputs[label] = []

        for episode in label_episodes:
            samples = pd.read_parquet(
                sample_path,
                columns=columns,
                filters=[("episode", "=", int(episode))],
            )
            if samples.empty:
                raise ValueError(f"No samples for checkpoint={label}, episode={episode}")
            trajectory = build_trajectory_plot_frame(samples)
            dimensions = list(ACTION_DIMENSIONS)
            fig, axes = plt.subplots(10, 2, figsize=(17, 30), sharex=True)
            flat_axes = axes.reshape(-1)
            has_interval = bool((trajectory["prediction_count"] > 1).any())

            for axis, dimension in zip(flat_axes, dimensions, strict=False):
                dimension_frame = trajectory[
                    trajectory["dimension"] == dimension.index
                ].sort_values("target_frame")
                if dimension_frame.empty:
                    raise ValueError(
                        f"No trajectory data for checkpoint={label}, episode={episode}, "
                        f"dimension={dimension.name}"
                    )
                time_seconds = dimension_frame["target_frame"].to_numpy(dtype=np.float64) / fps
                target = dimension_frame["target"].to_numpy(dtype=np.float64)
                prediction = dimension_frame["prediction_mean"].to_numpy(dtype=np.float64)
                axis.plot(time_seconds, target, color="black", linewidth=1.35, label="GT")
                axis.plot(time_seconds, prediction, color="tab:blue", linewidth=1.0, label="Inference mean")
                if has_interval and (dimension_frame["prediction_count"] > 1).any():
                    axis.fill_between(
                        time_seconds,
                        dimension_frame["prediction_p10"].to_numpy(dtype=np.float64),
                        dimension_frame["prediction_p90"].to_numpy(dtype=np.float64),
                        color="tab:blue",
                        alpha=0.16,
                        linewidth=0,
                        label="Inference P10-P90",
                    )
                mae = float(np.mean(np.abs(prediction - target)))
                axis.set_title(
                    f"[{dimension.index:02d}] {dimension.name} | MAE={mae:.5g} {dimension.unit}",
                    fontsize=10,
                )
                axis.set_ylabel(dimension.unit)
                axis.grid(True, alpha=0.25)

            for axis in flat_axes[len(dimensions):]:
                axis.set_visible(False)
            for axis in axes[-1, :]:
                if axis.get_visible():
                    axis.set_xlabel("Episode time (s)")
            # The last visible dimension is in the first column, so it also needs an x label.
            flat_axes[len(dimensions) - 1].set_xlabel("Episode time (s)")
            handles, legend_labels = flat_axes[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="upper center", ncol=len(legend_labels), fontsize=9)
            fig.suptitle(
                f"{label} · episode {episode} · per-dimension GT vs inference",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.975))
            output_path = plot_dir / f"{_safe_label(label)}_episode_{episode:06d}_gt_vs_inference.png"
            fig.savefig(output_path, dpi=dpi)
            plt.close(fig)
            outputs[label].append(output_path)
    return outputs


def checkpoint_summary(metrics: Mapping[str, pd.DataFrame]) -> dict:
    summaries = {}
    for checkpoint, frame in metrics["per_dim"].groupby("checkpoint", sort=False):
        rotation = metrics["rotation"]
        rotation = rotation[rotation["checkpoint"] == checkpoint]
        base = metrics["base_activity"]
        base = base[(base["checkpoint"] == checkpoint) & (base["activity"] == "active")]
        summaries[checkpoint] = {
            "macro_nmae_q01_q99": float(frame["nmae_q01_q99"].mean(skipna=True)),
            "dimensions_with_undefined_nmae": frame.loc[frame["nmae_q01_q99"].isna(), "dimension_name"].tolist(),
            "normalized_target_abs_gt_3_rate_max": float(frame["normalized_target_abs_gt_3_rate"].max()),
            "normalized_target_abs_gt_5_rate_max": float(frame["normalized_target_abs_gt_5_rate"].max()),
            "normalized_prediction_abs_gt_3_rate_max": float(frame["normalized_prediction_abs_gt_3_rate"].max()),
            "normalized_prediction_abs_gt_5_rate_max": float(frame["normalized_prediction_abs_gt_5_rate"].max()),
            "nonfinite_prediction_count": int(frame["nonfinite_prediction_count"].sum()),
            "rotation": rotation.set_index("side")[["mean_deg", "p95_deg", "p99_deg", "max_deg"]].to_dict("index"),
            "active_base_mae": base.set_index("dimension_name")["mae"].to_dict(),
        }
    return finite_or_none(summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-dimension, per-horizon open-loop evaluation for Nero LingBot VLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True, metavar="LABEL=HF_CKPT")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--robo-name", default="nero_mobile_xyzquat")
    episode_group = parser.add_mutually_exclusive_group(required=True)
    episode_group.add_argument("--episodes", type=int, nargs="+")
    episode_group.add_argument("--all-episodes", action="store_true")
    anchor_group = parser.add_mutually_exclusive_group()
    anchor_group.add_argument("--anchors-per-episode", type=int)
    anchor_group.add_argument("--anchor-stride", type=int)
    parser.add_argument("--include-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument(
        "--score-horizon",
        type=int,
        default=None,
        help="Only score the first N predicted actions; inference still produces the checkpoint's full chunk",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--norm-path", type=Path, default=None, help="Override norm stats for every checkpoint (normally leave unset)")
    parser.add_argument("--dataset-role", choices=("train-replay", "heldout"), required=True)
    parser.add_argument("--base-vx-active-threshold", type=float, default=0.02)
    parser.add_argument("--base-wz-active-threshold", type=float, default=0.02)
    parser.add_argument("--base-height-active-threshold", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.num_denoising_steps <= 0:
        raise ValueError("--num-denoising-steps must be > 0")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path.expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(data_path)
    if args.dataset_role == "train-replay":
        LOGGER.warning(
            "This run is a train-replay diagnostic. It validates the inference/data path but is not evidence of generalization."
        )

    labels = [label for label, _ in args.checkpoint]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Checkpoint labels must be unique: {labels}")
    checkpoint_configs = {}
    configured_chunks = set()
    for label, checkpoint in args.checkpoint:
        config_path, config = load_training_config(checkpoint)
        checkpoint_configs[label] = (checkpoint, config_path, config)
        configured_chunks.add(int(config["train"]["chunk_size"]))
    if len(configured_chunks) != 1:
        raise ValueError(f"Checkpoints use different chunk sizes: {configured_chunks}")
    configured_chunk = configured_chunks.pop()
    chunk_size = args.chunk_size or configured_chunk
    if chunk_size != configured_chunk:
        raise ValueError(f"Requested chunk size {chunk_size} does not match checkpoint chunk size {configured_chunk}")
    score_horizon = args.score_horizon or chunk_size
    if not 1 <= score_horizon <= chunk_size:
        raise ValueError(
            f"--score-horizon must be in [1, {chunk_size}], got {score_horizon}"
        )

    anchors_per_episode = args.anchors_per_episode
    if anchors_per_episode is None and args.anchor_stride is None:
        anchors_per_episode = 20
    episodes = load_episode_table(data_path)
    selected_episodes = episodes["episode_index"].astype(int).tolist() if args.all_episodes else args.episodes
    anchors = build_anchor_table(
        episodes,
        selected_episodes,
        anchors_per_episode=anchors_per_episode,
        anchor_stride=args.anchor_stride,
        include_tail=args.include_tail,
        chunk_size=chunk_size,
        score_horizon=score_horizon,
    )
    anchors.to_csv(output_dir / "anchors.csv", index=False)
    dataset, fps = create_dataset(data_path, chunk_size)
    if len(dataset) != int(episodes["length"].sum()):
        raise ValueError(f"Dataset length {len(dataset)} != episode metadata total {episodes['length'].sum()}")

    all_metrics: dict[str, list[pd.DataFrame]] = {
        "per_dim": [],
        "per_dim_horizon": [],
        "per_episode_dim": [],
        "per_repeat_dim": [],
        "base_activity": [],
        "stochastic_dim": [],
        "stochastic_dim_horizon": [],
        "rotation": [],
        "rotation_horizon": [],
        "rotation_episode": [],
    }
    sample_paths_by_label: dict[str, Path] = {}
    run_checkpoints = []
    for label, checkpoint in args.checkpoint:
        _, config_path, config = checkpoint_configs[label]
        training_data_path = resolve_project_path(config["data"].get("train_path"))
        if training_data_path is None or not training_data_path.is_dir():
            LOGGER.warning(
                "Training dataset path from %s is unavailable; raw range checks will use the evaluation dataset",
                config_path,
            )
            training_data_path = data_path
        training_min, training_max = load_dataset_bounds(training_data_path)
        configured_norm = resolve_project_path(config["data"].get("norm_stats_file"))
        norm_path = args.norm_path.expanduser().resolve() if args.norm_path is not None else configured_norm
        if norm_path is None or not norm_path.is_file():
            raise FileNotFoundError(f"Norm stats for {label} are unavailable: {norm_path}")

        LOGGER.info("Loading checkpoint %s from %s", label, checkpoint)
        policy = load_policy(checkpoint, norm_path, args.robo_name, chunk_size, args.use_bf16, args.use_compile)
        policy.config.num_steps = args.num_denoising_steps
        sample_path, rotation_path = evaluate_checkpoint(
            label=label,
            checkpoint=checkpoint,
            policy=policy,
            dataset=dataset,
            anchors=anchors,
            repeats=args.repeats,
            base_seed=args.seed,
            chunk_size=chunk_size,
            training_min=training_min,
            training_max=training_max,
            output_dir=output_dir,
            thresholds=(args.base_vx_active_threshold, args.base_wz_active_threshold, args.base_height_active_threshold),
        )
        del policy
        gc.collect()
        torch.cuda.empty_cache()

        checkpoint_metrics = aggregate_checkpoint(label, sample_path, rotation_path)
        sample_paths_by_label[label] = sample_path
        for key, frame in checkpoint_metrics.items():
            all_metrics[key].append(frame)
        run_checkpoints.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "training_config": str(config_path),
                "norm_stats": str(norm_path),
                "norm_stats_sha256": sha256_file(norm_path),
                "training_data_for_range": str(training_data_path),
                "checkpoint_files": checkpoint_file_manifest(checkpoint),
                "sample_parquet": str(sample_path),
                "rotation_sample_parquet": str(rotation_path),
            }
        )

    combined = {key: pd.concat(frames, ignore_index=True) for key, frames in all_metrics.items()}
    filenames = {
        "per_dim": "metrics_per_dimension.csv",
        "per_dim_horizon": "metrics_per_dimension_horizon.csv",
        "per_episode_dim": "metrics_per_episode_dimension.csv",
        "per_repeat_dim": "metrics_per_repeat_dimension.csv",
        "base_activity": "metrics_base_activity.csv",
        "stochastic_dim": "metrics_stochastic_per_dimension.csv",
        "stochastic_dim_horizon": "metrics_stochastic_per_dimension_horizon.csv",
        "rotation": "metrics_rotation.csv",
        "rotation_horizon": "metrics_rotation_horizon.csv",
        "rotation_episode": "metrics_rotation_episode.csv",
    }
    for key, filename in filenames.items():
        combined[key].to_csv(output_dir / filename, index=False)
    save_plots(combined, output_dir)
    trajectory_plots = save_action_trajectory_plots(
        sample_paths_by_label,
        output_dir,
        fps,
        episodes=selected_episodes,
    )
    for checkpoint_record in run_checkpoints:
        checkpoint_record["action_trajectory_plots"] = [
            str(path) for path in trajectory_plots[checkpoint_record["label"]]
        ]

    metadata = {
        "schema_version": 1,
        "evaluation_type": "open_loop_action_chunk",
        "dataset_role": args.dataset_role,
        "dataset": str(data_path),
        "dataset_fingerprints": {
            "info_json_sha256": sha256_file(data_path / "meta" / "info.json"),
            "stats_json_sha256": sha256_file(data_path / "meta" / "stats.json"),
        },
        "fps": fps,
        "robot_name": args.robo_name,
        "chunk_size": chunk_size,
        "score_horizon": score_horizon,
        "repeats": args.repeats,
        "base_seed": args.seed,
        "num_denoising_steps": args.num_denoising_steps,
        "use_bf16": args.use_bf16,
        "use_compile": args.use_compile,
        "include_tail": args.include_tail,
        "anchors_per_episode": anchors_per_episode,
        "anchor_stride": args.anchor_stride,
        "episode_count": len(selected_episodes),
        "anchor_count": len(anchors),
        "valid_scalar_sample_count_per_checkpoint": int((anchors["valid_length"] * ACTION_DIM).sum() * args.repeats),
        "thresholds": {
            "base_vx_active": args.base_vx_active_threshold,
            "base_wz_active": args.base_wz_active_threshold,
            "base_height_active": args.base_height_active_threshold,
        },
        "action_schema": action_schema_records(),
        "checkpoints": run_checkpoints,
        "summary": checkpoint_summary(combined),
        "interpretation_warning": (
            "Training-data replay is only a pipeline/fit diagnostic and must not be used as a deployment generalization claim."
            if args.dataset_role == "train-replay"
            else None
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(finite_or_none(metadata), file, ensure_ascii=False, indent=2, allow_nan=False)
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(finite_or_none(metadata), file, allow_unicode=True, sort_keys=False)

    LOGGER.info("Evaluation complete: %s", output_dir)
    LOGGER.info("Per-dimension metrics: %s", output_dir / filenames["per_dim"])
    LOGGER.info("Per-dimension GT/inference plots: %s", output_dir / "plots" / "action_trajectories")
    LOGGER.info("Summary: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
