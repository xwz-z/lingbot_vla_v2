#!/usr/bin/env python3
"""Add dual-arm xyz+quaternion task-space fields to a LeRobot v3 dataset.

The source dataset is left untouched.  The converted dataset keeps all source
features and adds:

* ``observation.state.end.position``: current left/right end-effector poses.
* ``action.end.position``: the next feedback pose within the same episode.

Each arm pose is ``[x, y, z, qx, qy, qz, qw]``.  The source FK rotations are
expected to store the first two rotation-matrix columns as Rot6D.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


DEFAULT_LEFT_FK_KEY = "observation.fk.fb.left.gripper_flange"
DEFAULT_RIGHT_FK_KEY = "observation.fk.fb.right.gripper_flange"
DEFAULT_STATE_KEY = "observation.state.end.position"
DEFAULT_ACTION_KEY = "action.end.position"
POSE_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "right_x",
    "right_y",
    "right_z",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
]
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source LeRobot v3 dataset directory")
    parser.add_argument("output", type=Path, help="New converted dataset directory")
    parser.add_argument("--left-fk-key", default=DEFAULT_LEFT_FK_KEY)
    parser.add_argument("--right-fk-key", default=DEFAULT_RIGHT_FK_KEY)
    parser.add_argument("--state-key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--action-key", default=DEFAULT_ACTION_KEY)
    parser.add_argument(
        "--media-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to duplicate files below videos/images/depth (default: copy)",
    )
    return parser.parse_args()


def load_info(source: Path) -> dict:
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected a LeRobot v3.0 dataset, got {info.get('codebase_version')!r}"
        )
    return info


def validate_paths(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source directory or a child of it")
    return source, output


def copy_dataset(source: Path, output: Path, media_mode: str) -> None:
    media_roots = {"videos", "images", "depth"}

    def copy_file(src: str, dst: str) -> str:
        relative = Path(src).relative_to(source)
        is_media = bool(relative.parts) and relative.parts[0] in media_roots
        if not is_media or media_mode == "copy":
            return shutil.copy2(src, dst)
        if media_mode == "hardlink":
            os.link(src, dst)
        else:
            os.symlink(Path(src).resolve(), dst)
        return dst

    shutil.copytree(source, output, copy_function=copy_file)


def table_matrix(table: pa.Table, key: str, width: int) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Missing required parquet feature: {key}")
    values = np.asarray(table[key].to_pylist(), dtype=np.float32)
    if values.shape != (len(table), width):
        raise ValueError(f"{key} must have shape [N, {width}], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} contains NaN or infinite values")
    return values


def rot6d_to_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Convert two matrix columns to normalized xyzw quaternions."""
    first = rot6d[:, :3].astype(np.float64)
    second = rot6d[:, 3:6].astype(np.float64)

    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Rot6D first column contains a near-zero vector")
    first /= first_norm

    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Rot6D columns are degenerate")
    second /= second_norm

    third = np.cross(first, second)
    matrices = np.stack((first, second, third), axis=2)
    quaternions = Rotation.from_matrix(matrices).as_quat()
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions


def make_pose(fk: np.ndarray) -> np.ndarray:
    pose = np.concatenate((fk[:, :3], rot6d_to_xyzw(fk[:, 3:9])), axis=1)
    return pose.astype(np.float32)


def enforce_quaternion_continuity(poses: np.ndarray, ordered_indices: np.ndarray) -> None:
    """Choose equivalent quaternion signs continuously inside one episode."""
    for quat_start in (3, 10):
        first_index = ordered_indices[0]
        if poses[first_index, quat_start + 3] < 0:
            poses[first_index, quat_start : quat_start + 4] *= -1
        previous = poses[first_index, quat_start : quat_start + 4].copy()
        for index in ordered_indices[1:]:
            current = poses[index, quat_start : quat_start + 4]
            if np.dot(previous, current) < 0:
                poses[index, quat_start : quat_start + 4] *= -1
            previous = poses[index, quat_start : quat_start + 4].copy()


def load_and_convert(
    parquet_files: list[Path], total_frames: int, left_key: str, right_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_fk = np.empty((total_frames, 9), dtype=np.float32)
    right_fk = np.empty((total_frames, 9), dtype=np.float32)
    episode_indices = np.empty(total_frames, dtype=np.int64)
    frame_indices = np.empty(total_frames, dtype=np.int64)
    seen = np.zeros(total_frames, dtype=bool)

    for path in parquet_files:
        table = pq.read_table(path, columns=[left_key, right_key, "episode_index", "frame_index", "index"])
        indices = np.asarray(table["index"].to_pylist(), dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= total_frames):
            raise ValueError(f"Out-of-range global index in {path}")
        if np.any(seen[indices]) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"Duplicate global index in {path}")
        seen[indices] = True
        left_fk[indices] = table_matrix(table, left_key, 9)
        right_fk[indices] = table_matrix(table, right_key, 9)
        episode_indices[indices] = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)
        frame_indices[indices] = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)

    if not seen.all():
        missing = np.flatnonzero(~seen)
        raise ValueError(f"Dataset is missing {len(missing)} global frame indices; first: {missing[:10].tolist()}")

    state_pose = np.concatenate((make_pose(left_fk), make_pose(right_fk)), axis=1)
    action_pose = np.empty_like(state_pose)
    for episode_index in np.unique(episode_indices):
        members = np.flatnonzero(episode_indices == episode_index)
        order = members[np.argsort(frame_indices[members], kind="stable")]
        expected = np.arange(len(order), dtype=np.int64)
        if not np.array_equal(frame_indices[order], expected):
            raise ValueError(f"Episode {episode_index} frame_index is not contiguous from zero")
        enforce_quaternion_continuity(state_pose, order)
        action_pose[order[:-1]] = state_pose[order[1:]]
        action_pose[order[-1]] = state_pose[order[-1]]

    return state_pose, action_pose, episode_indices, frame_indices


def fixed_list_array(values: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def update_huggingface_metadata(table: pa.Table, state_key: str, action_key: str) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        huggingface = json.loads(metadata[b"huggingface"])
        features = huggingface.setdefault("info", {}).setdefault("features", {})
        feature_schema = {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": 14,
            "_type": "List",
        }
        features[state_key] = feature_schema
        features[action_key] = feature_schema.copy()
        metadata[b"huggingface"] = json.dumps(huggingface, separators=(",", ":")).encode()
    return table.replace_schema_metadata(metadata)


def rewrite_data_parquet(
    source_root: Path,
    output_root: Path,
    state_pose: np.ndarray,
    action_pose: np.ndarray,
    state_key: str,
    action_key: str,
) -> None:
    for source_path in sorted((source_root / "data").glob("*/*.parquet")):
        output_path = output_root / source_path.relative_to(source_root)
        table = pq.read_table(output_path)
        if state_key in table.column_names or action_key in table.column_names:
            raise ValueError(f"Output feature already exists in {source_path}")
        indices = np.asarray(table["index"].to_pylist(), dtype=np.int64).reshape(-1)
        table = table.append_column(state_key, fixed_list_array(state_pose[indices]))
        table = table.append_column(action_key, fixed_list_array(action_pose[indices]))
        table = update_huggingface_metadata(table, state_key, action_key)
        pq.write_table(table, output_path, compression="snappy", use_dictionary=True)


def compute_stats(values: np.ndarray) -> dict[str, list]:
    values64 = values.astype(np.float64)
    return {
        "min": values64.min(axis=0).tolist(),
        "max": values64.max(axis=0).tolist(),
        "mean": values64.mean(axis=0).tolist(),
        "std": values64.std(axis=0).tolist(),
        "count": [len(values64)],
        "q01": np.quantile(values64, 0.01, axis=0).tolist(),
        "q10": np.quantile(values64, 0.10, axis=0).tolist(),
        "q50": np.quantile(values64, 0.50, axis=0).tolist(),
        "q90": np.quantile(values64, 0.90, axis=0).tolist(),
        "q99": np.quantile(values64, 0.99, axis=0).tolist(),
    }


def update_global_stats(output: Path, values_by_key: dict[str, np.ndarray]) -> None:
    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    for key, values in values_by_key.items():
        stats[key] = compute_stats(values)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def update_episode_stats(
    output: Path, values_by_key: dict[str, np.ndarray], episode_indices: np.ndarray
) -> None:
    for path in sorted((output / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        row_episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64).reshape(-1)
        for feature_key, values in values_by_key.items():
            episode_stats = {
                int(ep): compute_stats(values[episode_indices == ep]) for ep in row_episodes
            }
            for stat_key in STAT_KEYS:
                name = f"stats/{feature_key}/{stat_key}"
                if name in table.column_names:
                    raise ValueError(f"Episode metadata already contains {name}")
                rows = [episode_stats[int(ep)][stat_key] for ep in row_episodes]
                value_type = pa.int64() if stat_key == "count" else pa.float64()
                table = table.append_column(name, pa.array(rows, type=pa.list_(value_type)))
        pq.write_table(table, path, compression="snappy", use_dictionary=True)


def update_info(
    output: Path,
    info: dict,
    state_key: str,
    action_key: str,
    left_key: str,
    right_key: str,
) -> None:
    feature_info = {"dtype": "float32", "shape": [14], "names": POSE_NAMES}
    info["features"][state_key] = feature_info
    info["features"][action_key] = feature_info.copy()
    info["task_space_conversion"] = {
        "representation": "dual_arm_xyz_quaternion_xyzw",
        "state_source": {"left": left_key, "right": right_key},
        "action_semantics": "next_feedback_pose_within_episode_last_self",
        "rotation_source": "rot6d_matrix_columns_c0_c1",
        "quaternion_order": "xyzw",
    }
    (output / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_result(
    output: Path,
    state_key: str,
    action_key: str,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> None:
    total = 0
    state_all: list[np.ndarray] = []
    action_all: list[np.ndarray] = []
    indices_all: list[np.ndarray] = []
    for path in sorted((output / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=[state_key, action_key, "index"])
        state_all.append(table_matrix(table, state_key, 14))
        action_all.append(table_matrix(table, action_key, 14))
        indices_all.append(np.asarray(table["index"].to_pylist(), dtype=np.int64).reshape(-1))
        total += len(table)

    states = np.empty((total, 14), dtype=np.float32)
    actions = np.empty_like(states)
    indices = np.concatenate(indices_all)
    states[indices] = np.concatenate(state_all)
    actions[indices] = np.concatenate(action_all)

    for start in (3, 10):
        norms = np.linalg.norm(states[:, start : start + 4], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise AssertionError("Converted quaternions are not unit length")
    for episode in np.unique(episode_indices):
        members = np.flatnonzero(episode_indices == episode)
        order = members[np.argsort(frame_indices[members], kind="stable")]
        for start in (3, 10):
            adjacent_dots = np.sum(
                states[order[:-1], start : start + 4]
                * states[order[1:], start : start + 4],
                axis=1,
            )
            if np.any(adjacent_dots < 0):
                raise AssertionError(
                    f"Episode {episode} contains discontinuous quaternion signs"
                )
        if not np.array_equal(actions[order[:-1]], states[order[1:]]):
            raise AssertionError(f"Episode {episode} action is not next-frame state")
        if not np.array_equal(actions[order[-1]], states[order[-1]]):
            raise AssertionError(f"Episode {episode} last action is not self")


def main() -> None:
    args = parse_args()
    source, output = validate_paths(args.source, args.output)
    info = load_info(source)
    for key in (args.left_fk_key, args.right_fk_key):
        feature = info.get("features", {}).get(key)
        if feature is None:
            raise KeyError(f"Missing required feature in info.json: {key}")
        if feature.get("shape") != [9]:
            raise ValueError(f"{key} must be xyz+Rot6D with shape [9], got {feature.get('shape')}")
    for key in (args.state_key, args.action_key):
        if key in info["features"]:
            raise ValueError(f"Feature already exists: {key}")

    parquet_files = sorted((source / "data").glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data files below {source / 'data'}")

    total_frames = int(info["total_frames"])
    state_pose, action_pose, episode_indices, frame_indices = load_and_convert(
        parquet_files,
        total_frames,
        args.left_fk_key,
        args.right_fk_key,
    )

    temporary = output.with_name(f".{output.name}.converting-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary path already exists: {temporary}")
    try:
        copy_dataset(source, temporary, args.media_mode)
        rewrite_data_parquet(
            source,
            temporary,
            state_pose,
            action_pose,
            args.state_key,
            args.action_key,
        )
        values_by_key = {args.state_key: state_pose, args.action_key: action_pose}
        update_global_stats(temporary, values_by_key)
        update_episode_stats(temporary, values_by_key, episode_indices)
        update_info(
            temporary,
            info,
            args.state_key,
            args.action_key,
            args.left_fk_key,
            args.right_fk_key,
        )
        validate_result(
            temporary,
            args.state_key,
            args.action_key,
            episode_indices,
            frame_indices,
        )
        temporary.rename(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"Converted {total_frames} frames across {len(np.unique(episode_indices))} episodes")
    print(f"State feature : {args.state_key} [14] (left xyz+xyzw, right xyz+xyzw)")
    print(f"Action feature: {args.action_key} [14] (next feedback pose; last frame self)")
    print(f"Output        : {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
