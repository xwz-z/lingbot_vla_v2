import unittest
from pathlib import Path

import torch.nn as nn

from lingbotvla.utils.model_utils import audit_expert_only_parameters, validate_expert_only_parameters


class _ExpertContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwenvl = nn.Sequential(nn.Linear(4, 4), nn.Dropout())
        self.qwen_expert = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))


class _FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwenvl_with_expert = _ExpertContainer()
        self.state_proj = nn.Linear(2, 4)
        self.action_in_proj = nn.Linear(2, 4)
        self.action_out_proj = nn.Linear(4, 2)
        self.action_time_mlp_in = nn.Linear(8, 4)
        self.action_time_mlp_out = nn.Linear(4, 4)
        self.depth_align_head = nn.Linear(4, 3)


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FlowModel()


def _expert_only_policy():
    policy = _Policy()
    policy.model.qwenvl_with_expert.qwenvl.requires_grad_(False)
    return policy


class ExpertOnlyParameterAuditTest(unittest.TestCase):
    def test_accepts_native_expert_only_boundary(self):
        policy = _expert_only_policy()

        stats = validate_expert_only_parameters(policy)

        self.assertEqual(stats["vlm_trainable"], 0)
        self.assertGreater(stats["action_expert_trainable"], 0)
        self.assertGreater(stats["projection_trainable"], 0)
        self.assertGreater(stats["auxiliary_trainable"], 0)

    def test_rejects_trainable_qwenvl_parameter(self):
        policy = _expert_only_policy()
        next(policy.model.qwenvl_with_expert.qwenvl.parameters()).requires_grad_(True)

        with self.assertRaisesRegex(RuntimeError, "Qwen-VL backbone"):
            validate_expert_only_parameters(policy)

    def test_rejects_frozen_action_expert(self):
        policy = _expert_only_policy()
        policy.model.qwenvl_with_expert.qwen_expert.requires_grad_(False)

        with self.assertRaisesRegex(RuntimeError, "action expert"):
            validate_expert_only_parameters(policy)

    def test_audit_accounts_for_every_trainable_parameter(self):
        stats = audit_expert_only_parameters(_expert_only_policy())

        self.assertEqual(
            stats["trainable"],
            stats["action_expert_trainable"]
            + stats["projection_trainable"]
            + stats["auxiliary_trainable"],
        )


class ExpertOnlyConfigTest(unittest.TestCase):
    def test_unitree_config_preserves_approved_training_settings(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/vla/real_robot/unitree_mobile_xyzquat_expert_only.yaml"
        )
        self._assert_approved_settings(config_path)

    def test_real_robot_template_preserves_approved_training_settings(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/vla/real_robot/real_robot_expert_only.yaml"
        )
        self._assert_approved_settings(config_path)

    def _assert_approved_settings(self, config_path: Path):
        config_text = config_path.read_text(encoding="utf-8")

        for approved_setting in (
            "  use_future_image: true",
            "  freeze_vision_encoder: true",
            "  train_expert_only: true",
            "  train_state_proj: true",
            "  use_moe_expert_lr: true",
            "  lr: 5.0e-5",
        ):
            self.assertIn(approved_setting, config_text)


if __name__ == "__main__":
    unittest.main()
