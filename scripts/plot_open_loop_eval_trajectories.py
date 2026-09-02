#!/usr/bin/env python3
"""Generate per-dimension GT/inference plots from a completed evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.open_loop_eval_per_dim import save_action_trajectory_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot all 19 action dimensions as GT versus inference from an existing evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Only plot selected episodes; by default use every episode present in each sample parquet",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    summary_path = evaluation_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Evaluation summary is missing: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    fps = float(summary["fps"])
    sample_paths = {
        sample_path.stem: sample_path
        for sample_path in sorted((evaluation_dir / "samples").glob("*.parquet"))
    }
    if not sample_paths:
        raise FileNotFoundError(f"No sample parquet files under {evaluation_dir / 'samples'}")
    outputs = save_action_trajectory_plots(
        sample_paths,
        evaluation_dir,
        fps,
        episodes=args.episodes,
        dpi=args.dpi,
    )
    count = sum(len(paths) for paths in outputs.values())
    print(f"Generated {count} plot(s) under {evaluation_dir / 'plots' / 'action_trajectories'}")


if __name__ == "__main__":
    main()
