#!/usr/bin/env python3
"""Validate Nero mobile normalization over every valid 50-step action chunk.

This intentionally reads only parquet control columns.  It reproduces the
dataset transform used by training: each future TCP action is expressed
relative to the state at the chunk origin, while gripper and base actions stay
absolute.  Episode-tail padding is excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


STATE_KEYS = (
    "observation.state.end.position",
    "observation.state.effector.position",
    "observation.state.base.position",
)
ACTION_KEYS = (
    "action.end.position",
    "action.effector.position",
    "action.base.position",
)
ALL_KEYS = ACTION_KEYS + STATE_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("old_stats", type=Path)
    parser.add_argument("new_stats", type=Path)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )


def quat_inverse(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    result = q.copy()
    result[..., :3] *= -1
    return result


def quat_canonicalize(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.where(q[..., 3:4] < 0, -q, q)


def relative_end(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    if action.shape != state.shape or action.shape[-1] != 14:
        raise ValueError(f"Expected matching (..., 14) arrays, got {action.shape}, {state.shape}")
    parts = []
    for start in (0, 7):
        a_xyz = action[..., start : start + 3]
        s_xyz = state[..., start : start + 3]
        a_q = quat_normalize(action[..., start + 3 : start + 7])
        s_q = quat_normalize(state[..., start + 3 : start + 7])
        rel_q = quat_canonicalize(quat_multiply(quat_inverse(s_q), a_q))
        parts.append(np.concatenate((a_xyz - s_xyz, rel_q), axis=-1))
    return np.concatenate(parts, axis=-1)


def absolute_end(relative: np.ndarray, state: np.ndarray) -> np.ndarray:
    parts = []
    for start in (0, 7):
        r_xyz = relative[..., start : start + 3]
        s_xyz = state[..., start : start + 3]
        r_q = quat_normalize(relative[..., start + 3 : start + 7])
        s_q = quat_normalize(state[..., start + 3 : start + 7])
        absolute_q = quat_canonicalize(quat_multiply(s_q, r_q))
        parts.append(np.concatenate((s_xyz + r_xyz, absolute_q), axis=-1))
    return np.concatenate(parts, axis=-1)


class RunningMoments:
    def __init__(self, width: int):
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.total_sq = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.total.size)
        self.count += values.shape[0]
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def result(self) -> dict[str, np.ndarray | int]:
        mean = self.total / self.count
        std = np.sqrt(np.maximum(0.0, self.total_sq / self.count - np.square(mean)))
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "min": self.minimum,
            "max": self.maximum,
        }


def load_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads(path.read_text(encoding="utf-8"))["norm_stats"]
    return {
        key: {name: np.asarray(values, dtype=np.float64) for name, values in item.items() if isinstance(values, list)}
        for key, item in raw.items()
    }


def normalize(values: np.ndarray, item: dict[str, np.ndarray]) -> np.ndarray:
    return (values - item["mean"]) / (item["std"] + 1e-6)


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
    result = np.quantile(values, levels)
    return {f"p{level * 100:g}": float(value) for level, value in zip(levels, result, strict=True)}


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    files = sorted(args.dataset.glob("data/chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files below {args.dataset}")
    old = load_stats(args.old_stats)
    new = load_stats(args.new_stats)
    for key in ALL_KEYS:
        if key not in old or key not in new:
            raise KeyError(f"Missing normalization feature: {key}")

    raw_moments = {key: RunningMoments(len(old[key]["mean"])) for key in ALL_KEYS}
    normalized_max = {
        label: {key: np.zeros_like(stats[key]["mean"]) for key in ALL_KEYS}
        for label, stats in (("old", old), ("new", new))
    }
    chunk_max = {"old": [], "new": []}
    chunk_rms = {"old": [], "new": []}
    nonfinite = {"old": 0, "new": 0}
    roundtrip_error = {"old": 0.0, "new": 0.0}
    pose_xyz_error = {"old": 0.0, "new": 0.0}
    pose_angle_error_deg = {"old": 0.0, "new": 0.0}
    frame_count = 0
    valid_action_count = 0

    columns = list(ALL_KEYS) + ["episode_index", "frame_index"]
    for file_index, path in enumerate(files):
        table = pq.read_table(path, columns=columns)
        episode_ids = np.asarray(table["episode_index"].to_numpy())
        frame_ids = np.asarray(table["frame_index"].to_numpy())
        if len(np.unique(episode_ids)) != 1 or not np.array_equal(frame_ids, np.arange(len(frame_ids))):
            raise ValueError(f"Expected one ordered episode per file: {path}")
        episode = {
            key: np.asarray(table[key].to_pylist(), dtype=np.float64)
            for key in ALL_KEYS
        }
        length = len(frame_ids)
        frame_count += length
        for key in STATE_KEYS:
            values = episode[key]
            raw_moments[key].update(values)
            for label, stats in (("old", old), ("new", new)):
                z = normalize(values, stats[key])
                normalized_max[label][key] = np.maximum(normalized_max[label][key], np.max(np.abs(z), axis=0))
                nonfinite[label] += int((~np.isfinite(z)).sum())
                recovered = z * (stats[key]["std"] + 1e-6) + stats[key]["mean"]
                roundtrip_error[label] = max(roundtrip_error[label], float(np.max(np.abs(recovered - values))))

        episode_chunk_max = {
            "old": np.zeros(length, dtype=np.float64),
            "new": np.zeros(length, dtype=np.float64),
        }
        episode_chunk_sum_sq = {
            "old": np.zeros(length, dtype=np.float64),
            "new": np.zeros(length, dtype=np.float64),
        }
        episode_chunk_elements = np.zeros(length, dtype=np.int64)

        state_end = episode["observation.state.end.position"]
        for horizon in range(min(args.chunk_size, length)):
            origins = length - horizon
            transformed = {
                "action.end.position": relative_end(
                    episode["action.end.position"][horizon:], state_end[:origins]
                ),
                "action.effector.position": episode["action.effector.position"][horizon:],
                "action.base.position": episode["action.base.position"][horizon:],
            }
            valid_action_count += origins
            for key, values in transformed.items():
                raw_moments[key].update(values)
                episode_chunk_elements[:origins] += values.shape[-1]
                for label, stats in (("old", old), ("new", new)):
                    z = normalize(values, stats[key])
                    normalized_max[label][key] = np.maximum(
                        normalized_max[label][key], np.max(np.abs(z), axis=0)
                    )
                    episode_chunk_max[label][:origins] = np.maximum(
                        episode_chunk_max[label][:origins], np.max(np.abs(z), axis=1)
                    )
                    episode_chunk_sum_sq[label][:origins] += np.square(z).sum(axis=1)
                    nonfinite[label] += int((~np.isfinite(z)).sum())
                    recovered = z * (stats[key]["std"] + 1e-6) + stats[key]["mean"]
                    roundtrip_error[label] = max(
                        roundtrip_error[label], float(np.max(np.abs(recovered - values)))
                    )

                    if key == "action.end.position":
                        absolute = absolute_end(recovered, state_end[:origins])
                        target = episode[key][horizon:]
                        for start in (0, 7):
                            pose_xyz_error[label] = max(
                                pose_xyz_error[label],
                                float(np.max(np.abs(absolute[:, start : start + 3] - target[:, start : start + 3]))),
                            )
                            dot = np.abs(
                                np.sum(
                                    quat_normalize(absolute[:, start + 3 : start + 7])
                                    * quat_normalize(target[:, start + 3 : start + 7]),
                                    axis=-1,
                                )
                            )
                            angle = np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))
                            pose_angle_error_deg[label] = max(
                                pose_angle_error_deg[label], float(np.max(angle))
                            )

        for label in ("old", "new"):
            chunk_max[label].append(episode_chunk_max[label])
            chunk_rms[label].append(
                np.sqrt(episode_chunk_sum_sq[label] / episode_chunk_elements)
            )
        if (file_index + 1) % 25 == 0 or file_index + 1 == len(files):
            print(f"scanned {file_index + 1}/{len(files)} episodes", flush=True)

    report: dict[str, object] = {
        "dataset": str(args.dataset),
        "episodes": len(files),
        "frames": frame_count,
        "chunk_size": args.chunk_size,
        "valid_action_rows_across_chunks": valid_action_count,
        "stats_reproduction": {},
        "normalization": {},
    }
    reproduction = report["stats_reproduction"]
    assert isinstance(reproduction, dict)
    for key, moments in raw_moments.items():
        actual = moments.result()
        reproduction[key] = {
            "count": actual["count"],
            "max_abs_mean_error": float(np.max(np.abs(actual["mean"] - old[key]["mean"]))),
            "max_abs_std_error": float(np.max(np.abs(actual["std"] - old[key]["std"]))),
            "max_abs_min_error": float(np.max(np.abs(actual["min"] - old[key]["min"]))),
            "max_abs_max_error": float(np.max(np.abs(actual["max"] - old[key]["max"]))),
        }

    normalization = report["normalization"]
    assert isinstance(normalization, dict)
    for label, stats in (("old", old), ("new", new)):
        all_chunk_max = np.concatenate(chunk_max[label])
        all_chunk_rms = np.concatenate(chunk_rms[label])
        typical_ranges = {}
        for key in ALL_KEYS:
            q01 = normalize(stats[key]["q01"], stats[key])
            q99 = normalize(stats[key]["q99"], stats[key])
            typical_ranges[key] = {
                "normalized_q01_min": float(np.min(q01)),
                "normalized_q99_max": float(np.max(q99)),
                "largest_recorded_abs_per_dim": normalized_max[label][key].tolist(),
            }
        normalization[label] = {
            "nonfinite_values": nonfinite[label],
            "affine_roundtrip_max_abs_error": roundtrip_error[label],
            "pose_roundtrip_max_xyz_error": pose_xyz_error[label],
            "pose_roundtrip_max_angle_error_deg": pose_angle_error_deg[label],
            "chunk_max_abs_quantiles": quantiles(all_chunk_max),
            "chunk_rms_quantiles": quantiles(all_chunk_rms),
            "chunk_max_abs_threshold_counts": {
                f"gt_{threshold}": int(np.count_nonzero(all_chunk_max > threshold))
                for threshold in (3, 5, 10, 20, 50)
            },
            "chunk_max_abs_threshold_fractions": {
                f"gt_{threshold}": float(np.mean(all_chunk_max > threshold))
                for threshold in (3, 5, 10, 20, 50)
            },
            "typical_and_recorded_ranges": typical_ranges,
        }

    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
