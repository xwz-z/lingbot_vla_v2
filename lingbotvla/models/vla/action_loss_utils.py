import torch
from torch import Tensor


def build_action_loss_mask(
    losses: Tensor,
    joint_mask: Tensor | None,
    action_is_pad: Tensor | None,
    *,
    repeat_batch: bool = False,
) -> Tensor | None:
    """Combine semantic joint validity and episode-boundary validity."""
    loss_mask = None

    if joint_mask is not None:
        if repeat_batch:
            joint_mask = joint_mask.repeat(2, 1, 1)
        if joint_mask.ndim != 3 or joint_mask.shape != losses.shape:
            raise ValueError(
                f"joint_mask must match losses shape {tuple(losses.shape)}, "
                f"got {tuple(joint_mask.shape)}"
            )
        loss_mask = joint_mask.to(dtype=torch.bool)

    if action_is_pad is not None:
        if repeat_batch:
            action_is_pad = action_is_pad.repeat(2, 1)
        if action_is_pad.ndim != 2 or action_is_pad.shape != losses.shape[:2]:
            raise ValueError(
                f"action_is_pad must match losses batch/chunk shape {tuple(losses.shape[:2])}, "
                f"got {tuple(action_is_pad.shape)}"
            )
        timestep_mask = (~action_is_pad.to(dtype=torch.bool)).unsqueeze(-1).expand_as(losses)
        loss_mask = timestep_mask if loss_mask is None else loss_mask & timestep_mask

    return loss_mask
