import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from lingbotvla.eval.open_loop_metrics import ACTION_DIM, build_scalar_sample_frame, finite_or_none
from scripts.open_loop_eval_per_dim import (
    StreamingParquetWriter,
    aggregate_checkpoint,
    build_trajectory_plot_frame,
    build_anchor_table,
    checkpoint_summary,
    save_action_trajectory_plots,
    save_plots,
)


class OpenLoopReportIntegrationTest(unittest.TestCase):
    def test_score_horizon_is_separate_from_dataset_chunk_validity(self):
        episodes = pd.DataFrame(
            {
                "episode_index": [0],
                "length": [12],
                "dataset_from_index": [100],
                "dataset_to_index": [112],
            }
        )
        anchors = build_anchor_table(
            episodes,
            [0],
            anchors_per_episode=2,
            anchor_stride=None,
            include_tail=True,
            chunk_size=50,
            score_horizon=10,
        )
        self.assertEqual(anchors["anchor_frame"].tolist(), [0, 11])
        self.assertEqual(anchors["valid_length"].tolist(), [10, 1])
        self.assertEqual(anchors["dataset_valid_length"].tolist(), [12, 1])

    def test_parquet_aggregation_plot_and_strict_json(self):
        horizon = 4
        target = np.zeros((horizon, ACTION_DIM), dtype=np.float64)
        target[:, 6] = 1.0
        target[:, 13] = 1.0
        target[:, 16:19] = 0.1
        prediction = target.copy()
        prediction[:, 0] += np.arange(horizon) * 0.01
        normalized_target = target.copy()
        normalized_prediction = normalized_target + 0.05
        scalar, rotation = build_scalar_sample_frame(
            checkpoint="synthetic",
            episode=0,
            anchor_frame=0,
            global_index=0,
            repeat=0,
            noise_seed=1,
            target=target,
            predicted=prediction,
            normalized_target=normalized_target,
            normalized_prediction=normalized_prediction,
            valid_length=horizon,
            training_min=np.full(ACTION_DIM, -1.0),
            training_max=np.full(ACTION_DIM, 1.0),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "samples.parquet"
            rotation_path = root / "rotation.parquet"
            sample_writer = StreamingParquetWriter(sample_path)
            rotation_writer = StreamingParquetWriter(rotation_path)
            sample_writer.write(scalar)
            rotation_writer.write(rotation)
            sample_writer.close()
            rotation_writer.close()

            metrics = aggregate_checkpoint("synthetic", sample_path, rotation_path)
            self.assertEqual(len(metrics["per_dim"]), ACTION_DIM)
            self.assertEqual(len(metrics["per_dim_horizon"]), ACTION_DIM * horizon)
            self.assertEqual(len(metrics["rotation_horizon"]), 2 * horizon)
            self.assertIn("normalized_mae", metrics["per_dim"].columns)
            self.assertIn("active", set(metrics["base_activity"]["activity"]))

            save_plots(metrics, root)
            self.assertTrue((root / "plots" / "synthetic_nmae_heatmap.png").is_file())
            trajectory = build_trajectory_plot_frame(scalar)
            self.assertEqual(len(trajectory), ACTION_DIM * horizon)
            self.assertEqual(sorted(trajectory["target_frame"].unique().tolist()), [0, 1, 2, 3])
            trajectory_plots = save_action_trajectory_plots(
                {"synthetic": sample_path},
                root,
                fps=30.0,
                episodes=[0],
                dpi=40,
            )
            self.assertEqual(len(trajectory_plots["synthetic"]), 1)
            self.assertTrue(trajectory_plots["synthetic"][0].is_file())
            summary = finite_or_none(checkpoint_summary(metrics))
            encoded = json.dumps(summary, allow_nan=False)
            self.assertIn("synthetic", encoded)


if __name__ == "__main__":
    unittest.main()
