"""Mutable research surface for autoresearch.

This file contains the reward computation, observation construction, and action
processing for Harold's RL training. The autoresearch agent may freely edit this
file to explore new reward functions, observation representations, and action
processing pipelines.

The frozen infrastructure (physics setup, sensors, resets, termination, Isaac Lab
interface) stays in harold_isaac_lab_env.py and must not be duplicated here.

RULES:
- Do NOT change observation_space (must remain 48D) or action_space (12D)
- Do NOT write to actuators directly (env._robot.write_*)
- Do NOT modify reset/termination logic
- Do NOT access env.cfg.sim or physics parameters
- reward tensor must be shape [num_envs]
- observation dict must have key 'policy' with shape [num_envs, 48]
"""

import json
import torch


def compute_body_frame_command_errors(root_lin_vel_b: torch.Tensor, commands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return absolute X/Y command errors in the body frame."""
    return (
        torch.abs(root_lin_vel_b[:, 0] - commands[:, 0]),
        torch.abs(root_lin_vel_b[:, 1] - commands[:, 1]),
    )


def extract_sim_time_scalar(time_buffer: torch.Tensor, env_index: int = 0) -> float:
    """Serialize a single environment timestamp for JSON logging."""
    if time_buffer.ndim == 0:
        return float(time_buffer.item())
    return float(time_buffer[env_index].item())


def compute_rewards(env) -> torch.Tensor:
    """Compute per-step rewards for all environments.

    Session 54: Spot-aligned reward structure. No existence rewards (upright, stance_height).
    Only penalize bad states, reward locomotion. Standing earns ~4.2/step, walking ~13.0/step.

    Args:
        env: HaroldIsaacLabEnv instance (access state via env._robot, env.cfg, etc.)

    Returns:
        Total reward tensor of shape [num_envs]
    """
    cfg = env.cfg.rewards

    # === Extract quantities ===
    root_lin_vel_w = env._robot.data.root_lin_vel_w
    root_lin_vel_b = env._robot.data.root_lin_vel_b
    root_ang_vel_b = env._robot.data.root_ang_vel_b
    projected_gravity = env._robot.data.projected_gravity_b
    joint_acc = env._robot.data.joint_acc
    applied_torque = env._robot.data.applied_torque

    vx_b = root_lin_vel_b[:, 0]
    vy_b = root_lin_vel_b[:, 1]
    wz = root_ang_vel_b[:, 2]

    cmd_vx = env._commands[:, 0]
    cmd_vy = env._commands[:, 1]
    cmd_yaw = env._commands[:, 2]

    # === TASK REWARDS (exponential kernel + linear velocity bootstrap) ===
    lin_vel_error = torch.sum(
        torch.square(torch.stack([vx_b - cmd_vx, vy_b - cmd_vy], dim=1)), dim=1
    )
    track_lin_vel_xy = torch.exp(-lin_vel_error / (cfg.track_lin_vel_xy_std ** 2))

    # Linear velocity reward: proportional to forward velocity, capped at commanded.
    # Provides smooth gradient from 0 to cmd_vx — every tiny forward movement gets
    # rewarded, unlike the exponential which saturates at zero for large errors.
    # Weight 10.0 — aggressive forward signal. 16384 envs has upright=0.966, can afford aggression.
    linear_vel_reward = 10.0 * torch.clamp(vx_b / cmd_vx.clamp(min=0.05), 0.0, 1.0) * (cmd_vx > 0.05).float()

    ang_vel_error = torch.square(wz - cmd_yaw)
    track_ang_vel_z = torch.exp(-ang_vel_error / (cfg.track_ang_vel_z_std ** 2))

    # === BASE ORIENTATION PENALTY (Spot-aligned, replaces upright reward) ===
    # Spot: -3.0 * norm(projected_gravity_xy). Zero when level, negative when tilted.
    base_orientation_penalty = -cfg.base_orientation_weight * torch.norm(
        projected_gravity[:, :2], dim=1
    )

    # === BASE MOTION PENALTY (Spot-aligned, replaces lin_vel_z + ang_vel_xy) ===
    # Spot: -2.0 * (0.8 * vz² + 0.2 * |omega_xy|)
    vz_w = root_lin_vel_w[:, 2]
    omega_xy_norm = torch.norm(root_ang_vel_b[:, :2], dim=1)
    base_motion_penalty = -cfg.base_motion_weight * (
        0.8 * torch.square(vz_w) + 0.2 * omega_xy_norm
    )

    # === ACTION SMOOTHNESS (Spot-aligned, replaces action_rate) ===
    # Spot: -1.0 * norm(action_diff). L2 norm, not sum-of-squares.
    action_diff = env._actions - env._previous_actions
    action_smoothness = -cfg.action_smoothness_weight * torch.norm(action_diff, dim=1)

    # === SMOOTHNESS PENALTIES ===
    dof_torques = cfg.dof_torques_weight * torch.sum(torch.square(applied_torque), dim=1)
    dof_acc = cfg.dof_acc_weight * torch.sum(torch.square(joint_acc), dim=1)

    # === SHOULDER JOINT VELOCITY PENALTY (Spot's hip joint vel) ===
    # Harold's shoulders (indices 0-3) = Spot's hips.
    shoulder_joint_vel = -cfg.shoulder_joint_vel_weight * torch.sum(
        torch.square(env._robot.data.joint_vel[:, :4]), dim=1
    )

    # === GAIT: FEET AIR TIME ===
    first_contact = env._contact_sensor.compute_first_contact(env.step_dt)[:, env._feet_ids]
    last_air_time = env._contact_sensor.data.last_air_time[:, env._feet_ids]
    air_time_reward = torch.sum(
        (last_air_time - cfg.feet_air_time_threshold) * first_contact.float(), dim=1
    )
    cmd_magnitude = torch.norm(env._commands[:, :2], dim=1)
    air_time_reward = air_time_reward * (cmd_magnitude > 0.05).float()

    # === CONTINUOUS GAIT REWARD (from Spot) ===
    current_air_time = env._contact_sensor.data.current_air_time[:, env._feet_ids]
    current_contact_time = env._contact_sensor.data.current_contact_time[:, env._feet_ids]

    gait_std = 0.2   # 2x looser than Spot's 0.1 (Harold is smaller/lighter)
    gait_max_err = 0.3  # Spot uses 0.2; wider for Harold's morphology

    def _sync_reward(foot_a, foot_b):
        se_air = torch.clip(torch.square(current_air_time[:, foot_a] - current_air_time[:, foot_b]), max=gait_max_err**2)
        se_contact = torch.clip(torch.square(current_contact_time[:, foot_a] - current_contact_time[:, foot_b]), max=gait_max_err**2)
        return torch.exp(-(se_air + se_contact) / gait_std)

    def _async_reward(foot_a, foot_b):
        se_0 = torch.clip(torch.square(current_air_time[:, foot_a] - current_contact_time[:, foot_b]), max=gait_max_err**2)
        se_1 = torch.clip(torch.square(current_contact_time[:, foot_a] - current_air_time[:, foot_b]), max=gait_max_err**2)
        return torch.exp(-(se_0 + se_1) / gait_std)

    sync_reward = _sync_reward(0, 3) * _sync_reward(1, 2)
    async_reward = (_async_reward(0, 1) * _async_reward(3, 2)
                    * _async_reward(0, 2) * _async_reward(3, 1))
    gait_velocity_threshold = 0.1
    body_vel = torch.linalg.norm(root_lin_vel_b[:, :2], dim=1)
    gait_active = torch.logical_or(cmd_magnitude > 0.05, body_vel > gait_velocity_threshold).float()
    continuous_gait_reward = sync_reward * async_reward * gait_active

    # === UNDESIRED CONTACTS ===
    net_contact_forces = env._contact_sensor.data.net_forces_w_history[:, 0]
    undesired_forces = torch.norm(
        net_contact_forces[:, env._undesired_contact_body_ids], dim=-1
    )
    undesired_contacts = torch.sum(
        (undesired_forces > cfg.undesired_contacts_threshold).float(), dim=1
    )

    # === PER-FOOT CONTACT + SLIP METRICS ===
    foot_forces = torch.norm(net_contact_forces[:, env._feet_ids], dim=-1)
    foot_contact = foot_forces > env._foot_contact_force_threshold
    env._foot_contact_count += foot_contact.float()
    env._foot_contact_force_peak = torch.maximum(env._foot_contact_force_peak, foot_forces)

    air_time_sample = torch.where(first_contact, last_air_time, torch.zeros_like(last_air_time))
    env._foot_air_time_sum += air_time_sample
    env._foot_air_time_sumsq += air_time_sample * air_time_sample
    env._foot_air_time_count += first_contact.float()

    foot_lin_vel_xy = env._robot.data.body_lin_vel_w[:, env._feet_body_ids, :2]
    foot_slip_speed = torch.linalg.vector_norm(foot_lin_vel_xy, dim=-1)
    slip_sample = foot_slip_speed * foot_contact.float()
    env._foot_slip_speed_sum += slip_sample
    env._foot_slip_speed_count += foot_contact.float()

    # === FOOT SLIP PENALTY (Spot-aligned) ===
    foot_slip_penalty = -cfg.foot_slip_weight * torch.sum(slip_sample, dim=1)

    # === FORWARD MOTION BONUS (Harold-specific bootstrap, Spot has none) ===
    # Gate by tilt quality (orientation penalty replaces upright reward).
    base_tilt = torch.norm(projected_gravity[:, :2], dim=1)
    posture_quality = (1.0 - base_tilt).clamp(0.0, 1.0)
    forward_motion = cfg.forward_motion_weight * vx_b * posture_quality * (cmd_vx > 0.05).float()

    # === JOINT POSITION REGULARIZATION (Spot original direction) ===
    # Spot: 5x when standing with NO command (keeps tidy when idle).
    # Safe now: existence rewards removed, so standing at default pose earns ~4/step not ~10/step.
    joint_pos_error = torch.linalg.norm(
        env._robot.data.joint_pos - env._robot.data.default_joint_pos, dim=1
    )
    has_move_cmd = cmd_magnitude > 0.05
    is_standing = body_vel < cfg.joint_pos_velocity_threshold
    standing_no_command = (~has_move_cmd) & is_standing
    joint_pos_penalty = -cfg.joint_pos_weight * torch.where(
        standing_no_command,
        cfg.joint_pos_stand_still_scale * joint_pos_error,  # 5x when standing with NO command
        joint_pos_error,                                      # 1x otherwise
    )

    # === AIR TIME VARIANCE PENALTY (Spot-aligned) ===
    last_contact_time = env._contact_sensor.data.last_contact_time[:, env._feet_ids]
    air_time_var = torch.var(torch.clip(last_air_time, max=0.5), dim=1)
    contact_time_var = torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    air_time_variance_penalty = -cfg.air_time_variance_weight * (air_time_var + contact_time_var)

    # === FOOT CLEARANCE REWARD (from Spot) ===
    foot_pos_z = env._robot.data.body_pos_w[:, env._feet_body_ids, 2]
    foot_clearance_target = 0.05  # 5cm (Spot uses 10cm, Harold is smaller)
    foot_z_error = torch.square(foot_pos_z - foot_clearance_target)
    foot_xy_vel = torch.linalg.norm(
        env._robot.data.body_lin_vel_w[:, env._feet_body_ids, :2], dim=2
    )
    foot_velocity_gate = torch.tanh(2.0 * foot_xy_vel)
    foot_clearance = foot_z_error * foot_velocity_gate
    foot_clearance_reward = 0.5 * torch.exp(-torch.sum(foot_clearance, dim=1) / 0.05)

    # === COMPUTE TOTAL ===
    # EXP-750: Best minimal config (EXP-746 baseline) — seed sweep.
    # Pure minimal + air_time=2.5 produces best vx (0.025 at EXP-746).
    # No gait/smoothness — those killed early exploration.
    rewards = {
        "track_lin_vel_xy": cfg.track_lin_vel_xy_weight * track_lin_vel_xy,
        "track_ang_vel_z": 0.0 * track_ang_vel_z,  # disabled
        "base_orientation_penalty": -0.5 * torch.norm(projected_gravity[:, :2], dim=1),  # light safety
        "base_motion_penalty": 0.0 * base_motion_penalty,  # disabled
        "action_smoothness": 0.0 * action_smoothness,  # disabled
        "dof_torques": dof_torques,  # keep tiny torque penalty
        "dof_acc": dof_acc,  # keep tiny acc penalty
        "shoulder_joint_vel": 0.0 * shoulder_joint_vel,  # disabled
        "feet_air_time": 5.0 * air_time_reward,  # Full Spot (aggressive at 16384)
        "undesired_contacts": cfg.undesired_contacts_weight * undesired_contacts,  # keep safety
        "forward_motion": forward_motion,  # keep forward incentive
        "foot_slip_penalty": 0.0 * foot_slip_penalty,  # disabled
        "continuous_gait_reward": 0.0 * continuous_gait_reward,  # disabled
        "joint_pos_penalty": 0.0 * joint_pos_penalty,  # disabled
        "air_time_variance_penalty": 0.0 * air_time_variance_penalty,  # disabled
        "foot_clearance_reward": 0.0 * foot_clearance_reward,  # disabled
        "linear_vel_reward": linear_vel_reward,  # strong linear vel incentive
    }

    total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

    for key, value in rewards.items():
        if key not in env._episode_sums:
            env._episode_sums[key] = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        env._episode_sums[key] += value

    # === TELEMETRY (not in reward, just metrics) ===
    upright = -projected_gravity[:, 2]
    env._episode_sums["vx_w_mean"] += env._robot.data.root_lin_vel_w[:, 0]
    env._episode_sums["vy_w_mean"] += torch.abs(env._robot.data.root_lin_vel_w[:, 1])
    env._episode_sums["upright_mean"] += upright.clamp(0.0, 1.0)

    # Height metric (telemetry only, no longer a reward)
    pos_z = env._height_scanner.data.pos_w[:, 2].unsqueeze(1)
    ray_z = env._height_scanner.data.ray_hits_w[..., 2]
    ray_z = torch.where(torch.isfinite(ray_z), ray_z, pos_z)
    height_data = pos_z - ray_z
    current_height = torch.mean(height_data, dim=1)
    target_height = env.cfg.gait.target_height
    height_error = torch.abs(current_height - target_height)
    height_reward = torch.tanh(3.0 * torch.exp(-5.0 * height_error))
    env._episode_sums["height_reward"] += height_reward

    body_contact_penalty = -undesired_contacts
    env._episode_sums["body_contact_penalty"] += body_contact_penalty
    cmd_vx_error, cmd_vy_error = compute_body_frame_command_errors(root_lin_vel_b, env._commands)
    env._episode_sums["cmd_vx_error"] += cmd_vx_error
    env._episode_sums["cmd_vy_error"] += cmd_vy_error
    env._episode_sums["cmd_yaw_error"] += torch.abs(wz - cmd_yaw)

    return total_reward


def compute_observations(env) -> dict:
    """Construct the 48D observation vector for policy input.

    Args:
        env: HaroldIsaacLabEnv instance

    Returns:
        Dict with 'policy' key containing observation tensor [num_envs, 48]
    """
    # Update temporal state
    env._time += env.step_dt

    # Base observation components (48D)
    base_obs = [
        env._robot.data.root_lin_vel_b,                                      # (3D)
        env._robot.data.root_ang_vel_b,                                      # (3D)
        env._robot.data.projected_gravity_b,                                 # (3D)
        env._robot.data.joint_pos - env._robot.data.default_joint_pos,       # (12D)
        env._robot.data.joint_vel,                                           # (12D)
        env._commands,                                                       # (3D)
        env._prev_target_delta,                                              # (12D)
    ]

    obs = torch.cat(base_obs, dim=-1)  # [batch_size, 48]

    # Apply observation noise if domain randomization is enabled
    if env.cfg.domain_randomization.enable_randomization:
        obs = env._add_observation_noise(obs)

    observations = {"policy": obs}

    # Update previous actions
    env._previous_actions.copy_(env._actions)

    # Simulation playback logging
    if env._policy_log_dir is not None and env.num_envs > 0:
        entry = {
            "step": int(env._policy_log_step),
            "sim_time": extract_sim_time_scalar(env._time),
            "observation": obs[0].detach().cpu().tolist(),
            "command": env._commands[0].detach().cpu().tolist(),
            "raw_action": env._actions[0].detach().cpu().tolist(),
            "processed_action": env._processed_actions[0].detach().cpu().tolist(),
        }
        smoothed_actions = getattr(env, "_actions_smooth", None)
        if smoothed_actions is not None:
            entry["smoothed_action"] = smoothed_actions[0].detach().cpu().tolist()

        with open(env._policy_log_file, "a", encoding="utf-8") as f:
            json.dump(entry, f)
            f.write("\n")
        env._policy_log_step += 1

    return observations


def process_actions(env, actions: torch.Tensor) -> None:
    """Process raw policy actions into joint position targets.

    Takes normalized [-1, 1] policy outputs and converts to joint angle targets
    around the default pose, with EMA filtering and optional domain randomization.

    Args:
        env: HaroldIsaacLabEnv instance
        actions: Raw policy output [num_envs, 12] in range [-1, 1]
    """
    # Action copy
    env._actions.copy_(actions)

    # Low-pass filter (EMA) for stability and sim2real.
    # Seed freshly-reset envs with the current action to avoid cold-start attenuation.
    if not hasattr(env, "_actions_smooth"):
        env._actions_smooth = env._actions.clone()
    else:
        # Detect reset envs: _actions_smooth is zeroed by reset_policy_state_buffers.
        # Seed those envs so first action isn't attenuated to 20%.
        reset_mask = (env._actions_smooth.abs().sum(dim=1) == 0)
        if reset_mask.any():
            env._actions_smooth[reset_mask] = env._actions[reset_mask]
    beta = getattr(env.cfg, "action_filter_beta", 0.2)
    env._actions_smooth = (1.0 - beta) * env._actions_smooth + beta * env._actions

    # Apply action noise and delays if domain randomization is enabled
    if env.cfg.domain_randomization.enable_randomization:
        actions_to_use = env._add_action_noise(env._actions_smooth)
    else:
        actions_to_use = env._actions_smooth

    # Scale around default pose with per-joint ranges
    target = env._robot.data.default_joint_pos + env.cfg.action_scale * env._joint_range * actions_to_use
    env._processed_actions = torch.clamp(
        target,
        env._JOINT_ANGLE_MIN,
        env._JOINT_ANGLE_MAX,
    )

    # Store target delta for next observation
    env._prev_target_delta = env._processed_actions - env._robot.data.default_joint_pos
