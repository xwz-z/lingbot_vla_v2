import unittest

import torch

from lingbotvla.models.vla.action_loss_utils import build_action_loss_mask


class ActionPaddingMaskTest(unittest.TestCase):
    def test_excludes_padded_timesteps_and_inactive_joints(self):
        losses = torch.tensor([[[1.0, 100.0], [3.0, 100.0], [1000.0, 1000.0]]])
        joint_mask = torch.tensor([[[True, False], [True, False], [True, False]]])
        action_is_pad = torch.tensor([[False, False, True]])

        mask = build_action_loss_mask(losses, joint_mask, action_is_pad)
        masked_losses = losses * mask
        loss = masked_losses.sum() / mask.sum()

        self.assertAlmostEqual(loss.item(), 2.0)

    def test_repeat_loss_repeats_both_masks(self):
        losses = torch.tensor([
            [[1.0], [1000.0]],
            [[3.0], [1000.0]],
        ])
        joint_mask = torch.ones((1, 2, 1), dtype=torch.bool)
        action_is_pad = torch.tensor([[False, True]])

        mask = build_action_loss_mask(
            losses,
            joint_mask,
            action_is_pad,
            repeat_batch=True,
        )
        loss = (losses * mask).sum() / mask.sum()

        self.assertAlmostEqual(loss.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
