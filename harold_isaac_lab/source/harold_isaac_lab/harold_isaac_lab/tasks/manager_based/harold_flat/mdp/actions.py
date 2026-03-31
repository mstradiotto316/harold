"""EMA-filtered joint position action for sim-to-real transfer.

Training without action filtering creates a mismatch with deployment, where
EMA smoothing (beta=0.18) is applied before sending commands to servos.
This action class applies the same EMA filter during training so the policy
learns to produce movements that are compatible with the deployment pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass


class EMAJointPositionAction(JointPositionAction):
    """Joint position action with exponential moving average filtering.

    Applies EMA smoothing to raw policy outputs before the affine transformation
    (scale + offset). This matches the deployment pipeline where actions are
    smoothed before being sent to hardware servos.

    The EMA formula: smooth_t = (1 - beta) * smooth_{t-1} + beta * raw_t
    Lower beta = stronger smoothing (slower response).
    """

    cfg: "EMAJointPositionActionCfg"

    def __init__(self, cfg: "EMAJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._ema_beta = cfg.ema_beta
        # Buffer for smoothed raw actions (before scale+offset)
        self._smooth_actions = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        # Store raw actions from the policy network
        self._raw_actions[:] = actions
        # Apply EMA filter on raw actions (before scale + offset)
        self._smooth_actions = (
            (1.0 - self._ema_beta) * self._smooth_actions
            + self._ema_beta * self._raw_actions
        )
        # Apply affine transformation to smoothed actions
        self._processed_actions = self._smooth_actions * self._scale + self._offset
        # Clip if configured
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        # Zero the EMA buffer on reset — first post-reset step will be
        # attenuated (beta * raw), which provides a gentle startup.
        self._smooth_actions[env_ids] = 0.0


@configclass
class EMAJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for EMA-filtered joint position action."""

    class_type: type = EMAJointPositionAction

    ema_beta: float = 0.2
    """EMA filter coefficient. Lower = stronger smoothing.
    Default 0.2 matches deployment pipeline (action_converter.py).
    Range: 0.1 (very smooth) to 1.0 (no filtering)."""
