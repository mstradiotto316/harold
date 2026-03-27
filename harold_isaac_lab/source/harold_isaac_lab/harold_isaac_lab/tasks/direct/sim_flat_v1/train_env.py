# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mutable research surface for sim_flat_v1.

All 14 Spot flat locomotion reward terms, plus observations and action processing.
Faithfully ported from the manager-based Spot reference in IsaacLab.
"""

from __future__ import annotations

import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sim_flat_v1_env import SimFlatV1Env


# ---------------------------------------------------------------------------
# Action processing
# ---------------------------------------------------------------------------

def process_actions(env: SimFlatV1Env, actions: torch.Tensor) -> None:
    """Convert normalized actions to joint position targets.

    Spot reference: JointPositionActionCfg(scale=0.2, use_default_offset=True)
    target = default_joint_pos + action_scale * actions
    """
    env._processed_actions = env._robot.data.default_joint_pos + env.cfg.action_scale * actions


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def compute_observations(env: SimFlatV1Env) -> dict:
    """Build 48D observation vector (no noise — Spot has enable_corruption=False).

    Components (order matches Spot reference):
      [0:3]   base_lin_vel (body frame)
      [3:6]   base_ang_vel (body frame)
      [6:9]   projected_gravity (body frame)
      [9:12]  velocity commands (vx, vy, yaw)
      [12:24] joint_pos - default_joint_pos
      [24:36] joint_vel (default is 0, so rel == abs)
      [36:48] last_action
    """
    obs = torch.cat([
        env._robot.data.root_lin_vel_b,                                    # (N, 3)
        env._robot.data.root_ang_vel_b,                                    # (N, 3)
        env._robot.data.projected_gravity_b,                               # (N, 3)
        env._commands,                                                     # (N, 3)
        env._robot.data.joint_pos - env._robot.data.default_joint_pos,     # (N, 12)
        env._robot.data.joint_vel,                                         # (N, 12)
        env._actions,                                                      # (N, 12)
    ], dim=-1)

    return {"policy": obs}


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def compute_rewards(env: SimFlatV1Env) -> torch.Tensor:
    """Compute all 14 Spot reward terms. Returns per-env total reward."""
    cfg = env.cfg.rewards
    dt = env.cfg.sim.dt * env.cfg.decimation  # step_dt

    # ----- Shared quantities -----
    root_lin_vel_b = env._robot.data.root_lin_vel_b
    root_ang_vel_b = env._robot.data.root_ang_vel_b
    projected_gravity = env._robot.data.projected_gravity_b
    joint_pos = env._robot.data.joint_pos
    joint_vel = env._robot.data.joint_vel
    default_joint_pos = env._robot.data.default_joint_pos
    applied_torque = env._robot.data.applied_torque
    commands = env._commands

    cmd_norm = torch.norm(commands, dim=1)
    body_vel_xy = torch.linalg.norm(root_lin_vel_b[:, :2], dim=1)

    contact_sensor = env._contact_sensor
    foot_ids = env._foot_body_ids

    # ===== TASK REWARDS =====

    # 1. Air time reward (weight=5.0)
    current_air_time = contact_sensor.data.current_air_time[:, foot_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, foot_ids]
    mode_time = cfg.air_time_mode_time
    vel_threshold = cfg.air_time_velocity_threshold

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd_expanded = cmd_norm.unsqueeze(1).expand(-1, len(foot_ids))
    body_vel_expanded = body_vel_xy.unsqueeze(1).expand(-1, len(foot_ids))
    air_time_reward = torch.where(
        torch.logical_or(cmd_expanded > 0.0, body_vel_expanded > vel_threshold),
        torch.where(t_max < mode_time, t_min, torch.zeros_like(t_min)),
        stance_cmd_reward,
    )
    air_time_reward = torch.sum(air_time_reward, dim=1)

    # 2. Base linear velocity reward (weight=5.0)
    target_xy = commands[:, :2]
    lin_vel_error = torch.linalg.norm(target_xy - root_lin_vel_b[:, :2], dim=1)
    vel_cmd_mag = torch.linalg.norm(target_xy, dim=1)
    velocity_scaling = torch.clamp(
        1.0 + cfg.base_linear_velocity_ramp_rate * (vel_cmd_mag - cfg.base_linear_velocity_ramp_at_vel),
        min=1.0,
    )
    base_lin_vel_reward = torch.exp(-lin_vel_error / cfg.base_linear_velocity_std) * velocity_scaling

    # 3. Base angular velocity reward (weight=5.0)
    target_yaw = commands[:, 2]
    ang_vel_error = torch.abs(target_yaw - root_ang_vel_b[:, 2])
    base_ang_vel_reward = torch.exp(-ang_vel_error / cfg.base_angular_velocity_std)

    # 4. Foot clearance reward (weight=0.5)
    foot_pos_z = env._robot.data.body_pos_w[:, env._foot_body_ids_robot, 2]
    foot_z_error = torch.square(foot_pos_z - cfg.foot_clearance_target_height)
    foot_vel_xy = torch.linalg.norm(
        env._robot.data.body_lin_vel_w[:, env._foot_body_ids_robot, :2], dim=2
    )
    foot_vel_tanh = torch.tanh(cfg.foot_clearance_tanh_mult * foot_vel_xy)
    foot_clearance_reward = torch.exp(-torch.sum(foot_z_error * foot_vel_tanh, dim=1) / cfg.foot_clearance_std)

    # 5. Gait reward — trot enforcement (weight=10.0)
    gait_reward = _compute_gait_reward(env, cfg)

    # ===== PENALTIES =====

    # 6. Action smoothness penalty (weight=-1.0)
    action_smoothness_penalty = torch.linalg.norm(env._actions - env._previous_actions, dim=1)

    # 7. Air time variance penalty (weight=-1.0)
    last_air_time = contact_sensor.data.last_air_time[:, foot_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, foot_ids]
    air_time_variance_penalty = (
        torch.var(torch.clip(last_air_time, max=0.5), dim=1)
        + torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    )

    # 8. Base motion penalty (weight=-2.0)
    base_motion_penalty = (
        0.8 * torch.square(root_lin_vel_b[:, 2])
        + 0.2 * torch.sum(torch.abs(root_ang_vel_b[:, :2]), dim=1)
    )

    # 9. Base orientation penalty (weight=-3.0)
    base_orientation_penalty = torch.linalg.norm(projected_gravity[:, :2], dim=1)

    # 10. Foot slip penalty (weight=-0.5)
    net_forces_hist = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(
        torch.norm(net_forces_hist[:, :, foot_ids], dim=-1), dim=1
    )[0] > cfg.foot_slip_threshold
    foot_planar_vel = torch.linalg.norm(
        env._robot.data.body_lin_vel_w[:, env._foot_body_ids_robot, :2], dim=2
    )
    foot_slip_penalty = torch.sum(is_contact * foot_planar_vel, dim=1)

    # 11. Joint acceleration penalty (weight=-1e-4, hip joints only)
    joint_acc = env._robot.data.joint_acc
    joint_acc_penalty = torch.linalg.norm(joint_acc[:, env._hip_joint_ids], dim=1)

    # 12. Joint position penalty (weight=-0.7)
    joint_pos_error = torch.linalg.norm(joint_pos - default_joint_pos, dim=1)
    # 5x scale when standing (no command AND body velocity below threshold)
    is_standing = torch.logical_and(
        cmd_norm <= 0.0,
        body_vel_xy <= cfg.joint_position_velocity_threshold,
    )
    joint_pos_penalty = torch.where(
        is_standing,
        cfg.joint_position_stand_still_scale * joint_pos_error,
        joint_pos_error,
    )

    # 13. Joint torques penalty (weight=-5e-4)
    joint_torques_penalty = torch.linalg.norm(applied_torque, dim=1)

    # 14. Joint velocity penalty (weight=-1e-2, hip joints only)
    joint_vel_penalty = torch.linalg.norm(joint_vel[:, env._hip_joint_ids], dim=1)

    # ===== WEIGHTED SUM =====
    rewards = {
        "air_time": cfg.air_time_weight * air_time_reward,
        "base_linear_velocity": cfg.base_linear_velocity_weight * base_lin_vel_reward,
        "base_angular_velocity": cfg.base_angular_velocity_weight * base_ang_vel_reward,
        "foot_clearance": cfg.foot_clearance_weight * foot_clearance_reward,
        "gait": cfg.gait_weight * gait_reward,
        "action_smoothness": cfg.action_smoothness_weight * action_smoothness_penalty,
        "air_time_variance": cfg.air_time_variance_weight * air_time_variance_penalty,
        "base_motion": cfg.base_motion_weight * base_motion_penalty,
        "base_orientation": cfg.base_orientation_weight * base_orientation_penalty,
        "foot_slip": cfg.foot_slip_weight * foot_slip_penalty,
        "joint_acceleration": cfg.joint_acceleration_weight * joint_acc_penalty,
        "joint_position": cfg.joint_position_weight * joint_pos_penalty,
        "joint_torques": cfg.joint_torques_weight * joint_torques_penalty,
        "joint_velocity": cfg.joint_velocity_weight * joint_vel_penalty,
    }

    total = sum(rewards.values()) * dt

    # Track per-term episode sums
    for key, value in rewards.items():
        env._episode_sums[key] += value * dt

    # Track metrics
    env._episode_sums["vx_w_mean"] += root_lin_vel_b[:, 0]
    env._episode_sums["vy_w_mean"] += root_lin_vel_b[:, 1]
    env._episode_sums["cmd_vx_mean"] += commands[:, 0]

    return total


# ---------------------------------------------------------------------------
# Gait reward helper
# ---------------------------------------------------------------------------

def _compute_gait_reward(env: SimFlatV1Env, cfg) -> torch.Tensor:
    """Trot gait enforcement: sync FL/HR + FR/HL, antisync cross-pairs.

    6-term product: 2 sync rewards * 4 async rewards.
    Only active when cmd > 0 or body velocity > threshold.
    """
    contact_sensor = env._contact_sensor
    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time
    std = cfg.gait_std
    max_err_sq = cfg.gait_max_err ** 2

    p0_a, p0_b = env._sync_pair_0  # FL, HR
    p1_a, p1_b = env._sync_pair_1  # FR, HL

    def sync_reward(f0: int, f1: int) -> torch.Tensor:
        se_air = torch.clip(torch.square(air_time[:, f0] - air_time[:, f1]), max=max_err_sq)
        se_contact = torch.clip(torch.square(contact_time[:, f0] - contact_time[:, f1]), max=max_err_sq)
        return torch.exp(-(se_air + se_contact) / std)

    def async_reward(f0: int, f1: int) -> torch.Tensor:
        se_0 = torch.clip(torch.square(air_time[:, f0] - contact_time[:, f1]), max=max_err_sq)
        se_1 = torch.clip(torch.square(contact_time[:, f0] - air_time[:, f1]), max=max_err_sq)
        return torch.exp(-(se_0 + se_1) / std)

    sync_r = sync_reward(p0_a, p0_b) * sync_reward(p1_a, p1_b)
    async_r = (
        async_reward(p0_a, p1_a)
        * async_reward(p0_b, p1_b)
        * async_reward(p0_a, p1_b)
        * async_reward(p1_a, p0_b)
    )

    cmd_norm = torch.norm(env._commands, dim=1)
    body_vel = torch.linalg.norm(env._robot.data.root_lin_vel_b[:, :2], dim=1)

    return torch.where(
        torch.logical_or(cmd_norm > 0.0, body_vel > cfg.gait_velocity_threshold),
        sync_r * async_r,
        torch.zeros_like(sync_r),
    )
