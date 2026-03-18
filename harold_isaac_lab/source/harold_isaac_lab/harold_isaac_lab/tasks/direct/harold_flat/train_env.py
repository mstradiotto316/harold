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

    # Body-frame velocities for reward computation (commands are body-frame).
    # BUG-1 fix: was using root_lin_vel_w which diverges from commands after yaw.
    vx_b = root_lin_vel_b[:, 0]
    vy_b = root_lin_vel_b[:, 1]
    wz = root_ang_vel_b[:, 2]

    cmd_vx = env._commands[:, 0]
    cmd_vy = env._commands[:, 1]
    cmd_yaw = env._commands[:, 2]

    # === TASK REWARDS (exponential kernel) ===
    lin_vel_error = torch.sum(
        torch.square(torch.stack([vx_b - cmd_vx, vy_b - cmd_vy], dim=1)), dim=1
    )
    track_lin_vel_xy = torch.exp(-lin_vel_error / (cfg.track_lin_vel_xy_std ** 2))

    ang_vel_error = torch.square(wz - cmd_yaw)
    track_ang_vel_z = torch.exp(-ang_vel_error / (cfg.track_ang_vel_z_std ** 2))

    # === MOTION QUALITY PENALTIES ===
    vz_w = root_lin_vel_w[:, 2]
    lin_vel_z = torch.square(vz_w)  # world-frame vertical velocity (not body-frame Z)
    ang_vel_xy = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)

    # === SMOOTHNESS PENALTIES ===
    dof_torques = torch.sum(torch.square(applied_torque), dim=1)
    dof_acc = torch.sum(torch.square(joint_acc), dim=1)
    action_rate = torch.sum(
        torch.square(env._actions - env._previous_actions), dim=1
    )

    # === GAIT: FEET AIR TIME ===
    first_contact = env._contact_sensor.compute_first_contact(env.step_dt)[:, env._feet_ids]
    last_air_time = env._contact_sensor.data.last_air_time[:, env._feet_ids]
    air_time_reward = torch.sum(
        (last_air_time - cfg.feet_air_time_threshold) * first_contact.float(), dim=1
    )
    # Only reward air time when commanded to move
    cmd_magnitude = torch.norm(env._commands[:, :2], dim=1)
    air_time_reward = air_time_reward * (cmd_magnitude > 0.05).float()

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

    # === STABILITY: UPRIGHT ===
    upright = -projected_gravity[:, 2]

    # === PITCH PENALTY ===
    # projected_gravity[:, 0] is the forward component of gravity in body frame.
    # Positive = nose-down pitch. Penalize quadratically to discourage nose-diving.
    pitch_penalty = -3.0 * torch.square(projected_gravity[:, 0])

    # === HEIGHT METRIC (terrain-relative) ===
    pos_z = env._height_scanner.data.pos_w[:, 2].unsqueeze(1)
    ray_z = env._height_scanner.data.ray_hits_w[..., 2]
    ray_z = torch.where(torch.isfinite(ray_z), ray_z, pos_z)
    height_data = pos_z - ray_z
    current_height = torch.mean(height_data, dim=1)
    target_height = env.cfg.gait.target_height
    height_error = torch.abs(current_height - target_height)
    height_reward = torch.tanh(3.0 * torch.exp(-5.0 * height_error))

    # === BODY CONTACT METRIC ===
    body_contact_penalty = -undesired_contacts

    # === FORWARD MOTION BONUS ===
    # Direct reward for body-frame forward velocity, gated by posture quality.
    # Bug fixes applied: uses vx_b (body-frame), upright.clamp(0.0, 1.0) (proper gate).
    forward_motion = cfg.forward_motion_weight * vx_b * upright.clamp(0.0, 1.0) * (cmd_vx > 0.05).float()

    # === STANCE HEIGHT REWARD ===
    stance_height = 4.0 * height_reward

    # === FOOT SLIP PENALTY ===
    foot_slip_penalty = -0.1 * torch.sum(slip_sample, dim=1)

    # === JOINT ACTIVITY REWARD ===
    # Incentivize joint movement when commanded to move. Provides gradient from
    # standing (zero joint vel = 0) toward motion. Smooth periodic motion (gait)
    # is favored over jittering by action_rate and dof_acc penalties.
    joint_vel_norm = torch.sum(torch.abs(env._robot.data.joint_vel), dim=1)
    joint_activity = torch.tanh(joint_vel_norm / 10.0)  # saturates at high vel
    joint_activity_reward = 0.3 * joint_activity * (cmd_magnitude > 0.05).float()

    # === FOOT LIFT REWARD ===
    # Reward upward (Z+) velocity of feet with 2x bonus when foot is airborne.
    # Ground gradient: provides initial nudge to start lifting.
    # Airborne bonus: makes actual lifting 2x more rewarding than ground vibration.
    foot_vel_z = env._robot.data.body_lin_vel_w[:, env._feet_body_ids, 2]
    foot_lift_speed = torch.clamp(foot_vel_z, min=0.0)  # only upward
    base_lift = torch.tanh(foot_lift_speed / 0.5)
    airborne_mult = 1.0 + (~foot_contact).float()  # 1.0 on ground, 2.0 airborne
    foot_lift_reward = 0.3 * torch.sum(base_lift * airborne_mult, dim=1) * (cmd_magnitude > 0.05).float()

    # === COMPUTE TOTAL ===
    rewards = {
        "track_lin_vel_xy": cfg.track_lin_vel_xy_weight * track_lin_vel_xy,
        "track_ang_vel_z": cfg.track_ang_vel_z_weight * track_ang_vel_z,
        "lin_vel_z": cfg.lin_vel_z_weight * lin_vel_z,
        "ang_vel_xy": cfg.ang_vel_xy_weight * ang_vel_xy,
        "dof_torques": cfg.dof_torques_weight * dof_torques,
        "dof_acc": cfg.dof_acc_weight * dof_acc,
        "action_rate": cfg.action_rate_weight * action_rate,
        "feet_air_time": cfg.feet_air_time_weight * air_time_reward,
        "undesired_contacts": cfg.undesired_contacts_weight * undesired_contacts,
        "upright": cfg.upright_weight * upright,
        "forward_motion": forward_motion,
        "stance_height": stance_height,
        "foot_slip_penalty": foot_slip_penalty,
        "joint_activity_reward": joint_activity_reward,
        "foot_lift_reward": foot_lift_reward,
        "pitch_penalty": pitch_penalty,
    }

    total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

    for key, value in rewards.items():
        env._episode_sums[key] += value

    # Telemetry: keep world-frame speed diagnostic metrics separate from body-frame command tracking.
    env._episode_sums["vx_w_mean"] += env._robot.data.root_lin_vel_w[:, 0]
    env._episode_sums["vy_w_mean"] += torch.abs(env._robot.data.root_lin_vel_w[:, 1])
    env._episode_sums["upright_mean"] += upright.clamp(0.0, 1.0)
    env._episode_sums["height_reward"] += height_reward
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
