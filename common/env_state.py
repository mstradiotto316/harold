"""Small helpers for resetting per-environment policy state."""

from __future__ import annotations

import torch


def reset_policy_state_buffers(
    env_ids: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    prev_target_delta: torch.Tensor,
    actions_smooth: torch.Tensor | None = None,
    action_delay_buffer: torch.Tensor | None = None,
) -> None:
    """Clear per-environment action history on episode reset."""
    actions[env_ids] = 0.0
    previous_actions[env_ids] = 0.0
    prev_target_delta[env_ids] = 0.0
    if actions_smooth is not None:
        actions_smooth[env_ids] = 0.0
    if action_delay_buffer is not None:
        action_delay_buffer[env_ids] = 0.0
