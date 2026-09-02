#!/usr/bin/env python3
"""Convert the Nero mobile-transfer LeRobot v2.1 export to LingBot v3.

This is intentionally dataset-specific.  The source stores dual-arm absolute
TCP poses as xyz+Rot6D, gripper commands, and mobile-base commands inside flat
state/action vectors.  The output exposes the six canonical LingBot features:

* observation.state.end.position / action.end.position: 14-D dual-arm
  xyz+quaternion (xyzw)
* observation.state.effector.position / action.effector.position: 2-D
  left/right gripper values
* observation.state.base.position / action.base.position: 3-D
  [vx, wz, height] and [vx_cmd, wz_cmd, height_cmd]

Actions remain the contemporaneous absolute controller commands from the
source.  They are not replaced by next-frame feedback and are not time-shifted.
The three MJPEG camera files are linked/copied without re-encoding.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


STATE_KEY = "observation.state"
ACTION_KEY = "action"
END_STATE_KEY = "observation.state.end.position"
END_ACTION_KEY = "action.end.position"
EFFECTOR_STATE_KEY = "observation.state.effector.position"
EFFECTOR_ACTION_KEY = "action.effector.position"
BASE_STATE_KEY = "observation.state.base.position"
BASE_ACTION_KEY = "action.base.position"

CAMERA_KEYS = (
    "observation.images.cam0",
    "observation.images.cam1",
    "observation.images.cam2",
)
CAMERA_ROLES = {
    "observation.images.cam0": "head_fpv",
    "observation.images.cam1": "left_wrist",
    "observation.images.cam2": "right_wrist",
}

END_NAMES = [
    "left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw",
    "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw",
]
EFFECTOR_NAMES = ["left_gripper", "right_gripper"]
BASE_STATE_NAMES = ["base_vx", "base_wz", "base_height"]
BASE_ACTION_NAMES = ["base_vx_cmd", "base_wz_cmd", "base_height_cmd"]
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source mobile_transfer_lerobot_clean directory")
    parser.add_argument("output", type=Path, help="New LeRobot v3 output directory")
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize videos (default: hardlink; falls back to copy across filesystems)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N episodes; useful for a smoke test",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


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


def validate_source(source: Path, info: dict) -> list[dict]:
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected source codebase_version v2.1, got {info.get('codebase_version')!r}")
    if info.get("robot_type") != "nero_dual_arm":
        raise ValueError(f"Expected robot_type nero_dual_arm, got {info.get('robot_type')!r}")
    if float(info.get("fps", 0)) != 30.0:
        raise ValueError(f"Expected 30 Hz source data, got {info.get('fps')!r}")
    features = info.get("features", {})
    if features.get(STATE_KEY, {}).get("length") != 26:
        raise ValueError("observation.state must be the known 26-D mobile-transfer layout")
    if features.get(ACTION_KEY, {}).get("length") != 23:
        raise ValueError("action must be the known 23-D mobile-transfer layout")
    for key in CAMERA_KEYS:
        if key not in features:
            raise KeyError(f"Missing source camera feature: {key}")

    episodes = read_jsonl(source / "meta" / "episodes.jsonl")
    expected = list(range(len(episodes)))
    actual = [int(row["episode_index"]) for row in episodes]
    if actual != expected:
        raise ValueError("episodes.jsonl must contain consecutive episode indices starting at zero")
    return episodes


def matrix(table: pa.Table, key: str, width: int) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"Missing parquet column: {key}")
    values = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if values.shape != (len(table), width):
        raise ValueError(f"{key} must have shape [{len(table)}, {width}], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} contains NaN or infinity")
    return values


def rot6d_to_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Convert two rotation-matrix columns to normalized xyzw quaternions."""
    first = rot6d[:, :3].copy()
    second = rot6d[:, 3:6].copy()
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
    rotation_matrices = np.stack((first, second, third), axis=2)
    quaternions = Rotation.from_matrix(rotation_matrices).as_quat()
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions


def make_dual_arm_pose(flat: np.ndarray) -> np.ndarray:
    left = np.concatenate((flat[:, 0:3], rot6d_to_xyzw(flat[:, 3:9])), axis=1)
    right = np.concatenate((flat[:, 10:13], rot6d_to_xyzw(flat[:, 13:19])), axis=1)
    return np.concatenate((left, right), axis=1)


def make_quaternions_continuous(poses: np.ndarray, first_reference: np.ndarray | None = None) -> None:
    """Pick equivalent quaternion signs continuously for each arm in one episode."""
    for start in (3, 10):
        first = poses[0, start : start + 4]
        if first_reference is None:
            should_flip = first[3] < 0
        else:
            should_flip = np.dot(first, first_reference[start : start + 4]) < 0
        if should_flip:
            poses[0, start : start + 4] *= -1
        for index in range(1, len(poses)):
            previous = poses[index - 1, start : start + 4]
            current = poses[index, start : start + 4]
            if np.dot(previous, current) < 0:
                poses[index, start : start + 4] *= -1


def fixed_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def hf_metadata(widths: dict[str, int]) -> dict[bytes, bytes]:
    features = {}
    for key, width in widths.items():
        features[key] = {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": width,
            "_type": "List",
        }
    for key, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[key] = {"dtype": dtype, "_type": "Value"}
    payload = {"info": {"features": features}}
    return {b"huggingface": json.dumps(payload, separators=(",", ":")).encode()}


def converted_table(
    source_table: pa.Table,
    episode_index: int,
    global_from_index: int,
    fps: float,
) -> tuple[pa.Table, dict[str, np.ndarray]]:
    source_state = matrix(source_table, STATE_KEY, 26)
    source_action = matrix(source_table, ACTION_KEY, 23)
    frame_index = np.asarray(source_table["frame_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(frame_index, np.arange(len(source_table), dtype=np.int64)):
        raise ValueError(f"Episode {episode_index}: frame_index is not contiguous from zero")
    source_episode = np.asarray(source_table["episode_index"].to_pylist(), dtype=np.int64)
    if not np.all(source_episode == episode_index):
        raise ValueError(f"Episode {episode_index}: parquet episode_index disagrees with filename")

    end_state = make_dual_arm_pose(source_state)
    end_action = make_dual_arm_pose(source_action)
    make_quaternions_continuous(end_state)
    make_quaternions_continuous(end_action, first_reference=end_state[0])

    values = {
        END_STATE_KEY: end_state.astype(np.float32),
        END_ACTION_KEY: end_action.astype(np.float32),
        EFFECTOR_STATE_KEY: source_state[:, [9, 19]].astype(np.float32),
        EFFECTOR_ACTION_KEY: source_action[:, [9, 19]].astype(np.float32),
        BASE_STATE_KEY: source_state[:, 23:26].astype(np.float32),
        BASE_ACTION_KEY: source_action[:, 20:23].astype(np.float32),
    }
    length = len(source_table)
    arrays = {key: fixed_list(value) for key, value in values.items()}
    arrays.update(
        {
            "timestamp": pa.array(frame_index.astype(np.float32) / np.float32(fps), type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(global_from_index, global_from_index + length, dtype=np.int64)),
            "task_index": pa.array(np.zeros(length, dtype=np.int64)),
        }
    )
    table = pa.table(arrays)
    widths = {key: value.shape[1] for key, value in values.items()}
    return table.replace_schema_metadata(hf_metadata(widths)), values


def compute_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def link_video(source: Path, target: Path, mode: str) -> str:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty source video: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
        return "copy"
    if mode == "symlink":
        target.symlink_to(source.resolve())
        return "symlink"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)
        return "copy"


def feature_info() -> dict:
    numeric = {
        END_STATE_KEY: (14, END_NAMES),
        END_ACTION_KEY: (14, END_NAMES),
        EFFECTOR_STATE_KEY: (2, EFFECTOR_NAMES),
        EFFECTOR_ACTION_KEY: (2, EFFECTOR_NAMES),
        BASE_STATE_KEY: (3, BASE_STATE_NAMES),
        BASE_ACTION_KEY: (3, BASE_ACTION_NAMES),
    }
    features = {
        key: {"dtype": "float32", "shape": [width], "names": names}
        for key, (width, names) in numeric.items()
    }
    for key, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[key] = {"dtype": dtype, "shape": [1], "names": None}
    for key in CAMERA_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "mjpeg",
                "video.pix_fmt": "yuvj420p",
                "video.is_depth_map": False,
                "video.fps": 30.0,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    return features


def write_metadata(
    output: Path,
    total_episodes: int,
    total_frames: int,
    instruction: str,
    episode_rows: list[dict],
    global_values: dict[str, list[np.ndarray]],
    video_modes_used: set[str],
    source: Path,
) -> None:
    meta = output / "meta"
    episodes_dir = meta / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "nero_mobile_dual_arm",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": 30.0,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": feature_info(),
        "camera_roles": CAMERA_ROLES,
        "task_space_conversion": {
            "source": str(source),
            "source_version": "v2.1-custom",
            "representation": "dual_arm_xyz_quaternion_xyzw",
            "pose_frame": "base_link",
            "tcp_frame": "dex1_tcp",
            "rotation_source": "rot6d_matrix_columns_c0_c1",
            "action_semantics": "same_timestamp_absolute_controller_command_no_shift",
            "base_state": "vx_wz_height",
            "base_action": "vx_cmd_wz_cmd_height_cmd",
            "omitted_source_state": "slam_map_x_y_yaw",
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stats = {key: compute_stats(np.concatenate(parts, axis=0)) for key, parts in global_values.items()}
    (meta / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"task_index": [0]}, index=[instruction]).to_parquet(meta / "tasks.parquet")
    pq.write_table(pa.Table.from_pylist(episode_rows), episodes_dir / "file-000.parquet", compression="snappy")

    report = {
        "source": str(source),
        "episodes": total_episodes,
        "frames": total_frames,
        "instruction": instruction,
        "video_modes_used": sorted(video_modes_used),
        "actions_shifted": False,
        "source_video_reencoded": False,
        "output_features": list(feature_info()),
    }
    (meta / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def add_episode_stats(row: dict, values: dict[str, np.ndarray]) -> None:
    for key, feature_values in values.items():
        stats = compute_stats(feature_values)
        for name in STAT_NAMES:
            row[f"stats/{key}/{name}"] = stats[name]


def validate_output(output: Path, total_episodes: int, total_frames: int) -> None:
    info = read_json(output / "meta" / "info.json")
    if info["total_episodes"] != total_episodes or info["total_frames"] != total_frames:
        raise AssertionError("Output info.json totals are inconsistent")
    seen = 0
    for episode_index in range(total_episodes):
        path = output / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
        table = pq.read_table(path)
        state = matrix(table, END_STATE_KEY, 14)
        action = matrix(table, END_ACTION_KEY, 14)
        for poses in (state, action):
            for start in (3, 10):
                norms = np.linalg.norm(poses[:, start : start + 4], axis=1)
                if not np.allclose(norms, 1.0, atol=1e-5):
                    raise AssertionError(f"Non-unit quaternion in {path}")
                dots = np.sum(poses[:-1, start : start + 4] * poses[1:, start : start + 4], axis=1)
                if np.any(dots < -1e-6):
                    raise AssertionError(f"Quaternion sign discontinuity in {path}")
        expected_index = np.arange(seen, seen + len(table), dtype=np.int64)
        actual_index = np.asarray(table["index"].to_pylist(), dtype=np.int64)
        if not np.array_equal(actual_index, expected_index):
            raise AssertionError(f"Non-contiguous global index in {path}")
        seen += len(table)
    if seen != total_frames:
        raise AssertionError(f"Validated {seen} frames, expected {total_frames}")


def main() -> None:
    args = parse_args()
    source, output = validate_paths(args.source, args.output)
    source_info = read_json(source / "meta" / "info.json")
    source_episodes = validate_source(source, source_info)
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be at least 1")
        source_episodes = source_episodes[: args.max_episodes]
    task_rows = read_jsonl(source / "meta" / "tasks.jsonl")
    if len(task_rows) != 1 or int(task_rows[0]["task_index"]) != 0:
        raise ValueError("This converter expects exactly one task with task_index 0")
    instruction = str(task_rows[0]["task"]).strip()
    if not instruction:
        raise ValueError("Source task instruction is empty")

    temporary = output.with_name(f".{output.name}.converting-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    global_values: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            END_STATE_KEY, END_ACTION_KEY, EFFECTOR_STATE_KEY, EFFECTOR_ACTION_KEY,
            BASE_STATE_KEY, BASE_ACTION_KEY,
        )
    }
    episode_rows: list[dict] = []
    video_modes_used: set[str] = set()
    global_index = 0
    try:
        (temporary / "data" / "chunk-000").mkdir(parents=True)
        for output_episode_index, source_episode in enumerate(source_episodes):
            source_episode_index = int(source_episode["episode_index"])
            source_data = source / "data" / "chunk-000" / f"episode_{source_episode_index:06d}.parquet"
            if not source_data.is_file():
                raise FileNotFoundError(source_data)
            source_table = pq.read_table(source_data)
            expected_length = int(source_episode["length"])
            if len(source_table) != expected_length:
                raise ValueError(
                    f"Episode {source_episode_index}: metadata length {expected_length} != parquet length {len(source_table)}"
                )
            table, values = converted_table(source_table, output_episode_index, global_index, 30.0)
            output_data = temporary / "data" / "chunk-000" / f"file-{output_episode_index:03d}.parquet"
            pq.write_table(table, output_data, compression="snappy", use_dictionary=True)
            for key, feature_values in values.items():
                global_values[key].append(feature_values)

            row = {
                "episode_index": output_episode_index,
                "tasks": [instruction],
                "length": expected_length,
                "data/chunk_index": 0,
                "data/file_index": output_episode_index,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + expected_length,
                "source_episode_index": source_episode_index,
            }
            for camera_key in CAMERA_KEYS:
                source_video = (
                    source / "videos" / "chunk-000" / camera_key / f"episode_{source_episode_index:06d}.mp4"
                )
                output_video = (
                    temporary / "videos" / camera_key / "chunk-000" / f"file-{output_episode_index:03d}.mp4"
                )
                video_modes_used.add(link_video(source_video, output_video, args.video_mode))
                row[f"videos/{camera_key}/chunk_index"] = 0
                row[f"videos/{camera_key}/file_index"] = output_episode_index
                row[f"videos/{camera_key}/from_timestamp"] = 0.0
                row[f"videos/{camera_key}/to_timestamp"] = expected_length / 30.0
            add_episode_stats(row, values)
            episode_rows.append(row)
            global_index += expected_length
            print(
                f"[{output_episode_index + 1}/{len(source_episodes)}] "
                f"episode {source_episode_index}: {expected_length} frames",
                flush=True,
            )

        write_metadata(
            temporary,
            len(source_episodes),
            global_index,
            instruction,
            episode_rows,
            global_values,
            video_modes_used,
            source,
        )
        validate_output(temporary, len(source_episodes), global_index)
        temporary.rename(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"Converted {global_index} frames across {len(source_episodes)} episodes")
    print("Pose          : dual-arm xyz+quaternion(xyzw), absolute in base_link")
    print("Action timing : unchanged same-timestamp controller commands (no shift)")
    print("Base          : state[vx,wz,height], action[vx_cmd,wz_cmd,height_cmd]")
    print(f"Videos        : {', '.join(sorted(video_modes_used))}; no re-encoding")
    print(f"Output        : {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
