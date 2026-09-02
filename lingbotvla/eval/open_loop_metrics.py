"""Pure metric and action-mapping utilities for Nero open-loop evaluation.

This module deliberately does not import the VLA model.  Keeping the numerical
logic separate makes it possible to test action-slot mappings, quaternion
handling, episode-tail masking, and metric edge cases without a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ActionDimension:
    index: int
    name: str
    group: str
    side: str
    unit: str
    source_key: str
    source_index: int
    model_slot: int


def _dimensions() -> tuple[ActionDimension, ...]:
    rows: list[ActionDimension] = []
    pose_names = ("x", "y", "z", "qx", "qy", "qz", "qw")
    pose_units = ("m", "m", "m", "quaternion", "quaternion", "quaternion", "quaternion")
    for side_index, side in enumerate(("left", "right")):
        for component, (name, unit) in enumerate(zip(pose_names, pose_units, strict=True)):
            logical_index = side_index * 7 + component
            rows.append(
                ActionDimension(
                    index=logical_index,
                    name=f"{side}_tcp_{name}",
                    group="tcp_position" if component < 3 else "tcp_quaternion",
                    side=side,
                    unit=unit,
                    source_key="action.end.position",
                    source_index=logical_index,
                    model_slot=14 + logical_index,
                )
            )
    rows.extend(
        [
            ActionDimension(14, "left_gripper", "gripper", "left", "dataset_unit", "action.effector.position", 0, 28),
            ActionDimension(15, "right_gripper", "gripper", "right", "dataset_unit", "action.effector.position", 1, 29),
            ActionDimension(16, "base_vx", "base", "base", "m/s", "action.base.position", 0, 36),
            ActionDimension(17, "base_wz", "base", "base", "rad/s", "action.base.position", 1, 37),
            ActionDimension(18, "base_height", "base", "base", "m", "action.base.position", 2, 38),
        ]
    )
    return tuple(rows)


ACTION_DIMENSIONS = _dimensions()
ACTION_KEYS = (
    "action.end.position",
    "action.effector.position",
    "action.base.position",
)
STATE_KEYS = (
    "observation.state.end.position",
    "observation.state.effector.position",
    "observation.state.base.position",
)
MODEL_SLOTS = np.asarray([dimension.model_slot for dimension in ACTION_DIMENSIONS], dtype=np.int64)
ACTION_DIM = len(ACTION_DIMENSIONS)


def action_schema_records() -> list[dict]:
    return [asdict(dimension) for dimension in ACTION_DIMENSIONS]


def logical_actions(values: Mapping[str, object]) -> np.ndarray:
    """Concatenate raw absolute action features in a fixed, documented order."""
    arrays = []
    for key, expected_dim in zip(ACTION_KEYS, (14, 2, 3), strict=True):
        if key not in values:
            raise KeyError(f"Missing required action feature: {key}")
        array = np.asarray(values[key])
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != expected_dim:
            raise ValueError(f"{key} must have shape (H, {expected_dim}), got {array.shape}")
        arrays.append(array.astype(np.float64, copy=False))
    horizons = {array.shape[0] for array in arrays}
    if len(horizons) != 1:
        raise ValueError(f"Action feature horizons do not match: {[a.shape for a in arrays]}")
    return np.concatenate(arrays, axis=-1)


def logical_normalized_actions(padded_actions: object) -> np.ndarray:
    """Select the 19 Nero logical dimensions from LingBot's padded 55 slots."""
    actions = np.asarray(padded_actions)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2:
        raise ValueError(f"Normalized actions must have shape (H, D), got {actions.shape}")
    if actions.shape[1] <= int(MODEL_SLOTS.max()):
        raise ValueError(
            f"Normalized actions only have {actions.shape[1]} slots; need slot {MODEL_SLOTS.max()}"
        )
    return actions[:, MODEL_SLOTS].astype(np.float64, copy=False)


def normalize_quaternion(quaternion: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion / np.maximum(norm, eps)


def align_quaternion_actions(
    predicted: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Normalize quaternions and resolve the q/-q ambiguity against the target.

    Returns copies of both action arrays plus per-arm geodesic errors in degrees.
    Quaternion component errors must be computed from the returned aligned arrays.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected matching (H, {ACTION_DIM}) arrays, got {predicted.shape}, {target.shape}")

    aligned_pred = predicted.copy()
    normalized_target = target.copy()
    geodesic: dict[str, np.ndarray] = {}
    for side, quat_slice in (("left", slice(3, 7)), ("right", slice(10, 14))):
        pred_q = normalize_quaternion(aligned_pred[:, quat_slice])
        target_q = normalize_quaternion(normalized_target[:, quat_slice])
        dot = np.sum(pred_q * target_q, axis=-1)
        pred_q[dot < 0.0] *= -1.0
        dot = np.abs(np.sum(pred_q * target_q, axis=-1))
        angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
        aligned_pred[:, quat_slice] = pred_q
        normalized_target[:, quat_slice] = target_q
        geodesic[side] = np.degrees(angle)
    return aligned_pred, normalized_target, geodesic


def deterministic_noise_seed(
    base_seed: int,
    episode: int,
    anchor_frame: int,
    repeat: int,
) -> int:
    """Return a stable 63-bit seed shared by all evaluated checkpoints."""
    payload = f"{base_seed}:{episode}:{anchor_frame}:{repeat}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"lingbotv").digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def select_anchor_offsets(
    episode_length: int,
    *,
    anchors_per_episode: int | None = None,
    anchor_stride: int | None = None,
    include_tail: bool = True,
    chunk_size: int = 50,
) -> np.ndarray:
    """Select deterministic local frame offsets for one episode."""
    if episode_length <= 0:
        return np.empty((0,), dtype=np.int64)
    if anchors_per_episode is not None and anchor_stride is not None:
        raise ValueError("Specify either anchors_per_episode or anchor_stride, not both")
    last = episode_length - 1 if include_tail else max(episode_length - chunk_size, 0)
    if anchor_stride is not None:
        if anchor_stride <= 0:
            raise ValueError("anchor_stride must be > 0")
        return np.arange(0, last + 1, anchor_stride, dtype=np.int64)
    if anchors_per_episode is None:
        raise ValueError("anchors_per_episode or anchor_stride is required")
    if anchors_per_episode <= 0:
        raise ValueError("anchors_per_episode must be > 0")
    count = min(anchors_per_episode, last + 1)
    return np.unique(np.rint(np.linspace(0, last, num=count)).astype(np.int64))


def _metadata_columns() -> dict[str, np.ndarray]:
    return {
        "dimension": np.asarray([d.index for d in ACTION_DIMENSIONS], dtype=np.int16),
        "dimension_name": np.asarray([d.name for d in ACTION_DIMENSIONS], dtype=object),
        "group": np.asarray([d.group for d in ACTION_DIMENSIONS], dtype=object),
        "side": np.asarray([d.side for d in ACTION_DIMENSIONS], dtype=object),
        "unit": np.asarray([d.unit for d in ACTION_DIMENSIONS], dtype=object),
        "model_slot": np.asarray([d.model_slot for d in ACTION_DIMENSIONS], dtype=np.int16),
    }


def build_scalar_sample_frame(
    *,
    checkpoint: str,
    episode: int,
    anchor_frame: int,
    global_index: int,
    repeat: int,
    noise_seed: int,
    target: np.ndarray,
    predicted: np.ndarray,
    normalized_target: np.ndarray,
    normalized_prediction: np.ndarray,
    valid_length: int,
    training_min: np.ndarray | None = None,
    training_max: np.ndarray | None = None,
    base_vx_threshold: float = 0.02,
    base_wz_threshold: float = 0.02,
    base_height_threshold: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form scalar and quaternion records for one predicted chunk."""
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    normalized_target = np.asarray(normalized_target, dtype=np.float64)
    normalized_prediction = np.asarray(normalized_prediction, dtype=np.float64)
    if not (target.shape == predicted.shape == normalized_target.shape == normalized_prediction.shape):
        raise ValueError(
            "target, predicted, normalized_target, and normalized_prediction must have identical shapes; "
            f"got {target.shape}, {predicted.shape}, {normalized_target.shape}, {normalized_prediction.shape}"
        )
    if target.ndim != 2 or target.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action arrays with shape (H, {ACTION_DIM}), got {target.shape}")
    if not 0 <= valid_length <= target.shape[0]:
        raise ValueError(f"valid_length={valid_length} is outside [0, {target.shape[0]}]")

    target = target[:valid_length]
    predicted = predicted[:valid_length]
    normalized_target = normalized_target[:valid_length]
    normalized_prediction = normalized_prediction[:valid_length]
    predicted, target, geodesic = align_quaternion_actions(predicted, target)

    h = valid_length
    d = ACTION_DIM
    metadata = _metadata_columns()
    horizon = np.repeat(np.arange(h, dtype=np.int16), d)
    dimension = np.tile(metadata["dimension"], h)
    target_flat = target.reshape(-1)
    predicted_flat = predicted.reshape(-1)
    error = predicted_flat - target_flat

    active = np.full((h, d), "all", dtype=object)
    if h:
        active[:, 16] = np.where(np.abs(target[:, 16]) > base_vx_threshold, "active", "idle")
        active[:, 17] = np.where(np.abs(target[:, 17]) > base_wz_threshold, "active", "idle")
        active[:, 18] = np.where(
            np.abs(target[:, 18]) > base_height_threshold,
            "active",
            "idle",
        )

    if training_min is None or training_max is None:
        outside = np.full((h, d), False, dtype=bool)
        train_min_flat = np.full(h * d, np.nan)
        train_max_flat = np.full(h * d, np.nan)
    else:
        training_min = np.asarray(training_min, dtype=np.float64)
        training_max = np.asarray(training_max, dtype=np.float64)
        if training_min.shape != (d,) or training_max.shape != (d,):
            raise ValueError("training_min and training_max must each have shape (19,)")
        outside = (predicted < training_min[None, :]) | (predicted > training_max[None, :])
        train_min_flat = np.tile(training_min, h)
        train_max_flat = np.tile(training_max, h)

    frame = pd.DataFrame(
        {
            "checkpoint": checkpoint,
            "episode": np.full(h * d, episode, dtype=np.int32),
            "anchor_frame": np.full(h * d, anchor_frame, dtype=np.int32),
            "global_index": np.full(h * d, global_index, dtype=np.int64),
            "repeat": np.full(h * d, repeat, dtype=np.int16),
            "noise_seed": np.full(h * d, noise_seed, dtype=np.uint64),
            "horizon": horizon,
            "dimension": dimension,
            "dimension_name": np.tile(metadata["dimension_name"], h),
            "group": np.tile(metadata["group"], h),
            "side": np.tile(metadata["side"], h),
            "unit": np.tile(metadata["unit"], h),
            "model_slot": np.tile(metadata["model_slot"], h),
            "activity": active.reshape(-1),
            "target": target_flat,
            "prediction": predicted_flat,
            "error": error,
            "absolute_error": np.abs(error),
            "squared_error": np.square(error),
            "normalized_target": normalized_target.reshape(-1),
            "normalized_prediction": normalized_prediction.reshape(-1),
            "normalized_error": (normalized_prediction - normalized_target).reshape(-1),
            "outside_training_range": outside.reshape(-1),
            "training_min": train_min_flat,
            "training_max": train_max_flat,
        }
    )

    rotation_rows = []
    for side in ("left", "right"):
        rotation_rows.append(
            pd.DataFrame(
                {
                    "checkpoint": checkpoint,
                    "episode": episode,
                    "anchor_frame": anchor_frame,
                    "global_index": global_index,
                    "repeat": repeat,
                    "noise_seed": np.uint64(noise_seed),
                    "horizon": np.arange(h, dtype=np.int16),
                    "side": side,
                    "rotation_error_deg": geodesic[side],
                }
            )
        )
    rotation_frame = pd.concat(rotation_rows, ignore_index=True) if rotation_rows else pd.DataFrame()
    return frame, rotation_frame


def _summarize_scalar(group: pd.DataFrame, eps: float = 1e-12) -> dict[str, float | int]:
    target = group["target"].to_numpy(dtype=np.float64)
    prediction = group["prediction"].to_numpy(dtype=np.float64)
    normalized_target = group["normalized_target"].to_numpy(dtype=np.float64)
    normalized_prediction = group["normalized_prediction"].to_numpy(dtype=np.float64)
    error = prediction - target
    absolute_error = np.abs(error)
    gt_std = float(np.std(target))
    pred_std = float(np.std(prediction))
    gt_range = float(np.quantile(target, 0.99) - np.quantile(target, 0.01)) if len(target) else np.nan
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if len(target) >= 2 and gt_std > eps and pred_std > eps
        else np.nan
    )
    return {
        "count": int(len(group)),
        "target_mean": float(np.mean(target)),
        "target_std": gt_std,
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": pred_std,
        "bias": float(np.mean(error)),
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "absolute_error_p50": float(np.quantile(absolute_error, 0.50)),
        "absolute_error_p90": float(np.quantile(absolute_error, 0.90)),
        "absolute_error_p95": float(np.quantile(absolute_error, 0.95)),
        "absolute_error_p99": float(np.quantile(absolute_error, 0.99)),
        "absolute_error_max": float(np.max(absolute_error)),
        "correlation": correlation,
        "target_q01_q99_range": gt_range,
        "nmae_q01_q99": float(np.mean(absolute_error) / gt_range) if gt_range > eps else np.nan,
        "normalized_mae": float(np.mean(np.abs(normalized_prediction - normalized_target))),
        "normalized_rmse": float(np.sqrt(np.mean(np.square(normalized_prediction - normalized_target)))),
        "normalized_target_abs_gt_3_rate": float(np.mean(np.abs(normalized_target) > 3.0)),
        "normalized_target_abs_gt_5_rate": float(np.mean(np.abs(normalized_target) > 5.0)),
        "normalized_prediction_abs_gt_3_rate": float(np.mean(np.abs(normalized_prediction) > 3.0)),
        "normalized_prediction_abs_gt_5_rate": float(np.mean(np.abs(normalized_prediction) > 5.0)),
        "outside_training_range_rate": float(np.mean(group["outside_training_range"].astype(bool))),
        "nonfinite_target_count": int(np.count_nonzero(~np.isfinite(target))),
        "nonfinite_prediction_count": int(np.count_nonzero(~np.isfinite(prediction))),
    }


def aggregate_scalar_metrics(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "count"])
    rows: list[dict] = []
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for group_key, group in frame.groupby(grouper, sort=False, dropna=False, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key, strict=True))
        row.update(_summarize_scalar(group))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_rotation_metrics(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "count"])
    rows: list[dict] = []
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for group_key, group in frame.groupby(grouper, sort=False, dropna=False, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group["rotation_error_deg"].to_numpy(dtype=np.float64)
        row = dict(zip(group_columns, group_key, strict=True))
        row.update(
            {
                "count": int(len(values)),
                "mean_deg": float(np.mean(values)),
                "rmse_deg": float(np.sqrt(np.mean(np.square(values)))),
                "p50_deg": float(np.quantile(values, 0.50)),
                "p90_deg": float(np.quantile(values, 0.90)),
                "p95_deg": float(np.quantile(values, 0.95)),
                "p99_deg": float(np.quantile(values, 0.99)),
                "max_deg": float(np.max(values)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_stochastic_metrics(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Summarize prediction spread across repeated flow-noise draws.

    Repeats are first grouped at the exact episode/anchor/horizon/dimension
    coordinate.  The reported spread therefore cannot be contaminated by
    trajectory motion or horizon drift.
    """
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "coordinate_count"])
    coordinate_columns = [*group_columns, "episode", "anchor_frame", "horizon"]
    coordinate_columns = list(dict.fromkeys(coordinate_columns))
    rows = []
    grouper = coordinate_columns[0] if len(coordinate_columns) == 1 else coordinate_columns
    for coordinate_key, coordinate in frame.groupby(grouper, sort=False, dropna=False, observed=True):
        if not isinstance(coordinate_key, tuple):
            coordinate_key = (coordinate_key,)
        values = coordinate["prediction"].to_numpy(dtype=np.float64)
        row = dict(zip(coordinate_columns, coordinate_key, strict=True))
        row.update(
            {
                "repeat_count": int(len(values)),
                "prediction_repeat_std": float(np.std(values)),
                "prediction_repeat_range": float(np.max(values) - np.min(values)),
            }
        )
        rows.append(row)
    coordinates = pd.DataFrame(rows)

    output = []
    outer_grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for group_key, group in coordinates.groupby(outer_grouper, sort=False, dropna=False, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        std = group["prediction_repeat_std"].to_numpy(dtype=np.float64)
        ranges = group["prediction_repeat_range"].to_numpy(dtype=np.float64)
        row = dict(zip(group_columns, group_key, strict=True))
        row.update(
            {
                "coordinate_count": int(len(group)),
                "repeat_count_min": int(group["repeat_count"].min()),
                "repeat_count_max": int(group["repeat_count"].max()),
                "prediction_repeat_std_mean": float(np.mean(std)),
                "prediction_repeat_std_p95": float(np.quantile(std, 0.95)),
                "prediction_repeat_std_max": float(np.max(std)),
                "prediction_repeat_range_mean": float(np.mean(ranges)),
                "prediction_repeat_range_p95": float(np.quantile(ranges, 0.95)),
                "prediction_repeat_range_max": float(np.max(ranges)),
            }
        )
        output.append(row)
    return pd.DataFrame(output)


def concatenate_action_bounds(stats: Mapping[str, Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    """Read raw action min/max arrays from a LeRobot meta/stats.json object."""
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for key, expected_dim in zip(ACTION_KEYS, (14, 2, 3), strict=True):
        if key not in stats:
            raise KeyError(f"Training dataset stats do not contain {key}")
        minimum = np.asarray(stats[key]["min"], dtype=np.float64).reshape(-1)
        maximum = np.asarray(stats[key]["max"], dtype=np.float64).reshape(-1)
        if minimum.size != expected_dim or maximum.size != expected_dim:
            raise ValueError(
                f"{key} stats must have {expected_dim} values, got min={minimum.size}, max={maximum.size}"
            )
        mins.append(minimum)
        maxs.append(maximum)
    return np.concatenate(mins), np.concatenate(maxs)


def finite_or_none(value: object) -> object:
    """Recursively replace non-finite floats for strict JSON output."""
    if isinstance(value, Mapping):
        return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value
