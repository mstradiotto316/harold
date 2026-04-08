"""Harold flat locomotion environment with displacement tracking and servo drift DR.

Subclasses ManagerBasedRLEnv to add:
1. Per-episode displacement metrics logged to TensorBoard
2. Servo midpoint drift domain randomization (persistent per-episode joint offsets)
"""

from __future__ import annotations

import torch
from typing import Sequence

from isaaclab.envs import ManagerBasedRLEnv


# Servo drift range in radians. ST3215 servos lose their zero point on impact.
# Hardware observation: ~5° (0.087 rad) drift after a fall on one knee.
SERVO_DRIFT_RANGE = 0.10  # ±0.10 rad (~5.7°), validated free up to ±0.15


class HaroldFlatEnv(ManagerBasedRLEnv):
    """Manager-based Harold env with displacement metrics and servo drift DR.

    Displacement: Tracks x_displacement, total_displacement, and avg_velocity
    per episode and logs them via extras["log"] for TensorBoard.

    Servo drift: At each episode reset, applies a random per-joint offset to the
    action manager's default position. This simulates ST3215 servo midpoint drift
    where the zero position shifts after impacts. The offset persists for the
    entire episode, so the policy must cope with asymmetric joint positions.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Displacement tracking
        self._episode_start_pos = self.scene["robot"].data.root_pos_w[:, :2].clone()
        self._step_count = torch.zeros(self.num_envs, device=self.device)

        # Store the original default joint positions (before any drift)
        # The action manager's _offset is initialized from default_joint_pos
        action_term = self.action_manager._terms["joint_pos"]
        self._original_joint_offset = action_term._offset.clone()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._step_count += 1
        return obs, reward, terminated, truncated, info

    def _reset_idx(self, env_ids: Sequence[int]):
        # Compute displacement BEFORE parent resets positions
        pending_metrics = None
        if len(env_ids) > 0:
            env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)
            valid = self._step_count[env_ids_t] > 0
            if valid.any():
                valid_ids = env_ids_t[valid]
                current_pos = self.scene["robot"].data.root_pos_w[valid_ids, :2]
                start_pos = self._episode_start_pos[valid_ids]
                disp = current_pos - start_pos
                total_disp = torch.linalg.norm(disp, dim=1)
                episode_time = self._step_count[valid_ids] * self.step_dt
                avg_vel = total_disp / episode_time.clamp(min=0.1)

                pending_metrics = {
                    "Episode_Metric/x_displacement": disp[:, 0].mean().item(),
                    "Episode_Metric/y_displacement": disp[:, 1].mean().item(),
                    "Episode_Metric/total_displacement": total_disp.mean().item(),
                    "Episode_Metric/avg_velocity": avg_vel.mean().item(),
                }

        # Parent resets the environments and initializes extras["log"]
        super()._reset_idx(env_ids)

        # Merge displacement metrics into the log dict
        if pending_metrics is not None:
            self.extras["log"].update(pending_metrics)

        # Apply servo drift: random per-joint offset to action default positions
        # This persists for the entire episode, simulating drifted servo zero points
        if len(env_ids) > 0:
            env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=self.device)

            action_term = self.action_manager._terms["joint_pos"]
            drift = torch.empty_like(self._original_joint_offset[env_ids_t]).uniform_(
                -SERVO_DRIFT_RANGE, SERVO_DRIFT_RANGE
            )
            action_term._offset[env_ids_t] = self._original_joint_offset[env_ids_t] + drift

            # Reset displacement trackers
            self._episode_start_pos[env_ids_t] = self.scene["robot"].data.root_pos_w[env_ids_t, :2].clone()
            self._step_count[env_ids_t] = 0
