import unittest

import numpy as np
import pandas as pd

from lingbotvla.eval.open_loop_metrics import (
    ACTION_DIM,
    MODEL_SLOTS,
    aggregate_rotation_metrics,
    aggregate_scalar_metrics,
    aggregate_stochastic_metrics,
    align_quaternion_actions,
    build_scalar_sample_frame,
    concatenate_action_bounds,
    deterministic_noise_seed,
    logical_actions,
    logical_normalized_actions,
    select_anchor_offsets,
)


def identity_action_chunk(horizon: int) -> np.ndarray:
    actions = np.zeros((horizon, ACTION_DIM), dtype=np.float64)
    actions[:, 6] = 1.0
    actions[:, 13] = 1.0
    return actions


class ActionMappingTest(unittest.TestCase):
    def test_raw_and_model_slot_mapping(self):
        raw = logical_actions(
            {
                "action.end.position": np.arange(2 * 14).reshape(2, 14),
                "action.effector.position": np.arange(2 * 2).reshape(2, 2) + 100,
                "action.base.position": np.arange(2 * 3).reshape(2, 3) + 200,
            }
        )
        self.assertEqual(raw.shape, (2, 19))
        np.testing.assert_array_equal(raw[0], np.r_[np.arange(14), [100, 101], [200, 201, 202]])

        padded = np.tile(np.arange(55), (2, 1))
        normalized = logical_normalized_actions(padded)
        np.testing.assert_array_equal(normalized[0], MODEL_SLOTS)

    def test_mapping_rejects_wrong_feature_width(self):
        with self.assertRaisesRegex(ValueError, "action.end.position"):
            logical_actions(
                {
                    "action.end.position": np.zeros((2, 13)),
                    "action.effector.position": np.zeros((2, 2)),
                    "action.base.position": np.zeros((2, 3)),
                }
            )


class QuaternionMetricTest(unittest.TestCase):
    def test_sign_equivalent_quaternions_have_zero_error(self):
        target = identity_action_chunk(2)
        predicted = target.copy()
        predicted[:, 3:7] *= -1.0
        predicted[:, 10:14] *= -1.0
        aligned, normalized_target, geodesic = align_quaternion_actions(predicted, target)
        np.testing.assert_allclose(aligned, normalized_target, atol=1e-12)
        np.testing.assert_allclose(geodesic["left"], 0.0, atol=1e-12)
        np.testing.assert_allclose(geodesic["right"], 0.0, atol=1e-12)

    def test_known_ninety_degree_rotation(self):
        target = identity_action_chunk(1)
        predicted = target.copy()
        predicted[0, 5] = np.sin(np.pi / 4.0)
        predicted[0, 6] = np.cos(np.pi / 4.0)
        _, _, geodesic = align_quaternion_actions(predicted, target)
        self.assertAlmostEqual(geodesic["left"][0], 90.0, places=10)


class SampleAndAggregationTest(unittest.TestCase):
    def test_tail_mask_and_base_activity(self):
        target = identity_action_chunk(5)
        target[:, 16] = [0.0, 0.03, 0.0, 0.0, 0.0]
        target[:, 17] = [0.0, 0.0, -0.03, 0.0, 0.0]
        target[:, 18] = [0.0, 0.0, 0.006, 0.0, 0.0]
        prediction = target + 0.1
        # Restore unit quaternions so their component error remains meaningful.
        prediction[:, 3:7] = target[:, 3:7]
        prediction[:, 10:14] = target[:, 10:14]
        frame, rotation = build_scalar_sample_frame(
            checkpoint="test",
            episode=4,
            anchor_frame=98,
            global_index=1234,
            repeat=0,
            noise_seed=7,
            target=target,
            predicted=prediction,
            normalized_target=np.zeros_like(target),
            normalized_prediction=np.zeros_like(target),
            valid_length=3,
            training_min=np.full(ACTION_DIM, -1.0),
            training_max=np.full(ACTION_DIM, 1.0),
        )
        self.assertEqual(len(frame), 3 * ACTION_DIM)
        self.assertEqual(len(rotation), 3 * 2)
        self.assertEqual(frame["horizon"].max(), 2)
        vx = frame[frame["dimension"] == 16].sort_values("horizon")
        self.assertEqual(vx["activity"].tolist(), ["idle", "active", "idle"])
        wz = frame[frame["dimension"] == 17].sort_values("horizon")
        self.assertEqual(wz["activity"].tolist(), ["idle", "idle", "active"])
        height = frame[frame["dimension"] == 18].sort_values("horizon")
        self.assertEqual(height["activity"].tolist(), ["idle", "idle", "active"])

    def test_constant_target_has_undefined_correlation_and_nmae(self):
        frame = pd.DataFrame(
            {
                "dimension_name": ["constant"] * 3,
                "target": [5.4, 5.4, 5.4],
                "prediction": [5.3, 5.4, 5.5],
                "normalized_target": [0.0, 0.0, 0.0],
                "normalized_prediction": [0.0, 0.0, 0.0],
                "outside_training_range": [False, False, False],
            }
        )
        result = aggregate_scalar_metrics(frame, ["dimension_name"]).iloc[0]
        self.assertTrue(np.isnan(result["correlation"]))
        self.assertTrue(np.isnan(result["nmae_q01_q99"]))
        self.assertAlmostEqual(result["mae"], 0.2 / 3.0)

    def test_rotation_aggregation(self):
        frame = pd.DataFrame(
            {
                "checkpoint": ["x"] * 3,
                "side": ["left"] * 3,
                "rotation_error_deg": [0.0, 30.0, 60.0],
            }
        )
        result = aggregate_rotation_metrics(frame, ["checkpoint", "side"]).iloc[0]
        self.assertAlmostEqual(result["mean_deg"], 30.0)
        self.assertAlmostEqual(result["rmse_deg"], np.sqrt(1500.0))

    def test_stochastic_spread_only_compares_matching_coordinates(self):
        frame = pd.DataFrame(
            {
                "checkpoint": ["x"] * 4,
                "dimension": [0] * 4,
                "episode": [0] * 4,
                "anchor_frame": [10] * 4,
                "horizon": [0, 0, 1, 1],
                "prediction": [1.0, 3.0, 100.0, 104.0],
            }
        )
        result = aggregate_stochastic_metrics(frame, ["checkpoint", "dimension"]).iloc[0]
        # Population std is 1 at horizon 0 and 2 at horizon 1; trajectory drift
        # between predictions near 2 and 102 must not enter the spread.
        self.assertAlmostEqual(result["prediction_repeat_std_mean"], 1.5)
        self.assertEqual(result["repeat_count_min"], 2)
        self.assertEqual(result["repeat_count_max"], 2)


class SamplingAndStatsTest(unittest.TestCase):
    def test_anchor_selection_and_tail_policy(self):
        np.testing.assert_array_equal(
            select_anchor_offsets(10, anchors_per_episode=3, include_tail=True, chunk_size=5),
            [0, 4, 9],
        )
        np.testing.assert_array_equal(
            select_anchor_offsets(10, anchors_per_episode=3, include_tail=False, chunk_size=5),
            [0, 2, 5],
        )
        np.testing.assert_array_equal(
            select_anchor_offsets(10, anchor_stride=4, include_tail=True, chunk_size=5),
            [0, 4, 8],
        )

    def test_noise_seed_is_reproducible_and_index_sensitive(self):
        seed = deterministic_noise_seed(42, 3, 10, 1)
        self.assertEqual(seed, deterministic_noise_seed(42, 3, 10, 1))
        self.assertNotEqual(seed, deterministic_noise_seed(42, 3, 11, 1))
        self.assertLess(seed, 1 << 63)

    def test_concatenate_bounds(self):
        stats = {
            "action.end.position": {"min": list(range(14)), "max": list(range(14, 28))},
            "action.effector.position": {"min": [30, 31], "max": [40, 41]},
            "action.base.position": {"min": [50, 51, 52], "max": [60, 61, 62]},
        }
        minimum, maximum = concatenate_action_bounds(stats)
        np.testing.assert_array_equal(minimum, np.r_[np.arange(14), [30, 31], [50, 51, 52]])
        np.testing.assert_array_equal(maximum, np.r_[np.arange(14, 28), [40, 41], [60, 61, 62]])


if __name__ == "__main__":
    unittest.main()
