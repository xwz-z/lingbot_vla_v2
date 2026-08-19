#!/usr/bin/env python3
"""Build range-capped mean/std statistics from existing LingBot stats.

The native ``meanstd`` normalizer is affine and does not clip values.  For a
sparse control dimension, a valid command at the physical limit can therefore
be tens of standard deviations away from the mean.  This utility preserves the
empirical mean while flooring every scale so that all values covered by the
recorded training min/max map inside ``[-max_abs, max_abs]``.

The result remains fully compatible with LingBot's existing ``meanstd``
normalizer and inverse normalizer.  No model or data-loader code change is
required, but checkpoints trained with the original scales are not compatible
with the generated statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing LingBot norm-stats JSON")
    parser.add_argument("output", type=Path, help="Output JSON; must not overwrite source")
    parser.add_argument(
        "--max-abs",
        type=float,
        default=3.0,
        help="Maximum absolute normalized value over recorded min/max (default: 3.0)",
    )
    return parser.parse_args()


def _finite_vector(stats: dict, key: str, name: str) -> list[float]:
    value = stats.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key}.{name} must be a non-empty list")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{key}.{name} contains a non-finite value")
    return result


def build_stats(source: Path, max_abs: float, source_label: str | None = None) -> dict:
    if not math.isfinite(max_abs) or max_abs <= 0:
        raise ValueError(f"--max-abs must be finite and positive, got {max_abs!r}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    norm_stats = payload.get("norm_stats")
    if not isinstance(norm_stats, dict) or not norm_stats:
        raise ValueError(f"{source} does not contain a non-empty 'norm_stats' object")

    adjusted_dimensions = 0
    total_dimensions = 0
    max_scale_inflation = 1.0
    feature_report: dict[str, dict[str, float | int]] = {}

    for key, stats in norm_stats.items():
        if not isinstance(stats, dict):
            raise ValueError(f"{key} stats must be an object")
        mean = _finite_vector(stats, key, "mean")
        empirical_std = _finite_vector(stats, key, "std")
        minimum = _finite_vector(stats, key, "min")
        maximum = _finite_vector(stats, key, "max")
        lengths = {len(mean), len(empirical_std), len(minimum), len(maximum)}
        if len(lengths) != 1:
            raise ValueError(f"{key} mean/std/min/max dimensions do not match")

        robust_std = []
        feature_adjusted = 0
        feature_max_inflation = 1.0
        for center, scale, low, high in zip(mean, empirical_std, minimum, maximum):
            if scale < 0 or low > high:
                raise ValueError(f"{key} has invalid std or min/max ordering")
            max_deviation = max(abs(low - center), abs(high - center))
            scale_floor = max_deviation / max_abs
            new_scale = max(scale, scale_floor)
            robust_std.append(new_scale)
            total_dimensions += 1
            if new_scale > scale * (1.0 + 1e-12):
                adjusted_dimensions += 1
                feature_adjusted += 1
            if scale > 0:
                inflation = new_scale / scale
            elif new_scale > 0:
                inflation = math.inf
            else:
                inflation = 1.0
            feature_max_inflation = max(feature_max_inflation, inflation)
            max_scale_inflation = max(max_scale_inflation, inflation)

        stats["empirical_std"] = empirical_std
        stats["std"] = robust_std
        feature_report[key] = {
            "dimensions": len(mean),
            "adjusted_dimensions": feature_adjusted,
            "max_scale_inflation": feature_max_inflation,
        }

    payload["normalization_profile"] = {
        "type": "range_capped_meanstd",
        "source_stats": source_label or source.name,
        "formula": "(value - mean) / (std + 1e-6)",
        "scale_rule": "std=max(empirical_std,max(abs(min-mean),abs(max-mean))/max_abs)",
        "max_abs_over_recorded_training_range": max_abs,
        "clips_values": False,
        "preserves_empirical_mean": True,
        "adjusted_dimensions": adjusted_dimensions,
        "total_dimensions": total_dimensions,
        "max_scale_inflation": max_scale_inflation,
        "features": feature_report,
    }
    return payload


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the source stats file")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    payload = build_stats(source, args.max_abs, source_label=str(args.source))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    profile = payload["normalization_profile"]
    print(f"Wrote {output}")
    print(
        f"Adjusted {profile['adjusted_dimensions']}/{profile['total_dimensions']} dimensions; "
        f"recorded training range is bounded by ±{args.max_abs:g}."
    )


if __name__ == "__main__":
    main()
