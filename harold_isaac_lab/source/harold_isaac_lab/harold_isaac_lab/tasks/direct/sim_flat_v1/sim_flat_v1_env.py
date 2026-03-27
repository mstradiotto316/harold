# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRLEnv for sim_flat_v1 — exact Spot flat locomotion reproduction.

Clean implementation: no CPG, no backlash, no scripted gait, no env var overrides.
"""

from __future__ import annotations

from typing import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis, sample_uniform
import isaaclab.sim as sim_utils

from .sim_flat_v1_env_cfg import SimFlatV1EnvCfg
from . import train_env


class SimFlatV1Env(DirectRLEnv):
    """Spot flat locomotion environment (direct RL).

    Faithful reproduction of the Isaac Lab Spot velocity-tracking task,
    translated from manager-based to direct-env architecture.
    """

    cfg: SimFlatV1EnvCfg

    def __init__(self, cfg: SimFlatV1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- Action buffers ---
        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self._processed_actions = self._robot.data.default_joint_pos.clone()

        # --- Velocity commands (vx, vy, yaw_rate) ---
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_timer = torch.zeros(self.num_envs, device=self.device)

        # --- Time tracking ---
        self._time = torch.zeros(self.num_envs, device=self.device)

        # --- Body/joint index lookups ---
        # Feet: fl_foot, fr_foot, hl_foot, hr_foot
        self._foot_body_ids = self._contact_sensor.find_bodies(".*_foot")[0]
        print(f"Foot body IDs: {self._foot_body_ids} "
              f"({[self._contact_sensor.body_names[i] for i in self._foot_body_ids]})")

        # Synced foot pairs for trot gait: (FL, HR) and (FR, HL)
        self._sync_pair_0 = (
            self._contact_sensor.find_bodies("fl_foot")[0][0],
            self._contact_sensor.find_bodies("hr_foot")[0][0],
        )
        self._sync_pair_1 = (
            self._contact_sensor.find_bodies("fr_foot")[0][0],
            self._contact_sensor.find_bodies("hl_foot")[0][0],
        )

        # Undesired contact bodies: body + legs (for termination)
        self._undesired_contact_body_ids = self._contact_sensor.find_bodies(
            ["body", ".*leg"]
        )[0]
        print(f"Undesired contact body IDs: {self._undesired_contact_body_ids} "
              f"({[self._contact_sensor.body_names[i] for i in self._undesired_contact_body_ids]})")

        # Hip joint indices (for selective joint_acc and joint_vel penalties)
        self._hip_joint_ids = self._robot.find_joints(".*_h[xy]")[0]
        print(f"Hip joint IDs: {self._hip_joint_ids} "
              f"({[self._robot.data.joint_names[i] for i in self._hip_joint_ids]})")

        # Foot body indices on the robot (for foot_clearance reward)
        self._foot_body_ids_robot = self._robot.find_bodies(".*_foot")[0]

        # --- Episode tracking for TensorBoard ---
        self._reward_keys = [
            "air_time", "base_linear_velocity", "base_angular_velocity",
            "foot_clearance", "gait",
            "action_smoothness", "air_time_variance", "base_motion",
            "base_orientation", "foot_slip", "joint_acceleration",
            "joint_position", "joint_torques", "joint_velocity",
        ]
        self._metric_keys = [
            "vx_w_mean", "vy_w_mean", "cmd_vx_mean",
        ]
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [*self._reward_keys, *self._metric_keys]
        }

        # --- Domain randomization: interval push buffers ---
        dr = cfg.domain_randomization
        self._push_timer = torch.zeros(self.num_envs, device=self.device)
        self._push_interval = torch.empty(self.num_envs, device=self.device).uniform_(
            *dr.push_interval_range
        )

        # --- Startup randomization flag ---
        self._startup_randomized = False

        # --- Episode start positions (for displacement tracking) ---
        self._episode_start_pos = self._robot.data.root_pos_w.clone()

        # --- Termination tracking ---
        self._termination_masks = {
            "body_contact": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "terrain_oob": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

        print(f"\n{'='*60}")
        print(f"sim_flat_v1: Spot flat locomotion (direct env)")
        print(f"  Envs: {self.num_envs}")
        print(f"  Obs: {cfg.observation_space}, Act: {cfg.action_space}")
        print(f"  Physics dt: {cfg.sim.dt}s, Decimation: {cfg.decimation}, Control: {1.0/(cfg.sim.dt * cfg.decimation):.0f} Hz")
        print(f"  Episode: {cfg.episode_length_s}s ({self.max_episode_length} steps)")
        print(f"  Action scale: {cfg.action_scale}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        """Create robot, contact sensor, terrain, and lighting."""
        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Environment cloning
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=750.0, color=(0.8, 0.85, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Startup randomization (friction, mass) — applied once
    # ------------------------------------------------------------------
    def _apply_startup_randomization(self) -> None:
        """Randomize physics material and base mass (Spot reference: startup events).

        Uses CPU tensors as required by the PhysX API.
        """
        if self._startup_randomized:
            return
        self._startup_randomized = True

        dr = self.cfg.domain_randomization
        num_envs = self.num_envs
        env_ids_cpu = torch.arange(num_envs, device="cpu")

        # --- Friction randomization ---
        # PhysX material properties are per-shape: (num_envs, max_shapes, 3)
        # We modify the existing material buffer rather than creating from scratch
        materials = self._robot.root_physx_view.get_material_properties()  # CPU tensor
        max_shapes = materials.shape[1]

        # Sample per-env friction values on CPU
        static_friction = torch.empty(num_envs, 1, device="cpu").uniform_(*dr.static_friction_range)
        dynamic_friction = torch.empty(num_envs, 1, device="cpu").uniform_(*dr.dynamic_friction_range)
        restitution = torch.zeros(num_envs, 1, device="cpu")

        # Assign to all shapes
        materials[:, :, 0] = static_friction.expand(-1, max_shapes)
        materials[:, :, 1] = dynamic_friction.expand(-1, max_shapes)
        materials[:, :, 2] = restitution.expand(-1, max_shapes)

        self._robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

        # --- Mass randomization: add [-2.5, 2.5] kg to body ---
        body_idx = self._robot.find_bodies("body")[0]
        if len(body_idx) > 0:
            body_id = body_idx[0]
            masses = self._robot.root_physx_view.get_masses()  # CPU tensor
            mass_delta = torch.empty(num_envs, device="cpu").uniform_(*dr.base_mass_range)
            masses[:, body_id] += mass_delta
            self._robot.root_physx_view.set_masses(masses, env_ids_cpu)

        print(f"  Startup randomization applied: friction=[{dr.static_friction_range}], mass_delta=[{dr.base_mass_range}]")

    # ------------------------------------------------------------------
    # Pre-physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions and resample commands."""
        self._previous_actions[:] = self._actions
        self._actions[:] = actions.clamp(-1.0, 1.0)

        # Process actions via train_env
        train_env.process_actions(self, self._actions)

        # Time tracking
        self._time += self.cfg.sim.dt * self.cfg.decimation

        # Command resampling (every 10s)
        resample_mask = self._command_timer >= self.cfg.commands.resample_time
        if torch.any(resample_mask):
            resample_ids = torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)
            self._sample_commands(resample_ids)
            self._command_timer[resample_ids] = 0.0
        self._command_timer += self.cfg.sim.dt * self.cfg.decimation

    # ------------------------------------------------------------------
    # Apply action
    # ------------------------------------------------------------------
    def _apply_action(self) -> None:
        """Write joint position targets to sim and apply interval pushes."""
        # Apply startup randomization on first step
        self._apply_startup_randomization()

        self._robot.set_joint_position_target(self._processed_actions)

        # Interval push: every 10-15 seconds, push robot with random XY velocity
        dt = self.cfg.sim.dt * self.cfg.decimation
        self._push_timer += dt
        push_mask = self._push_timer >= self._push_interval
        if torch.any(push_mask):
            push_ids = torch.nonzero(push_mask, as_tuple=False).squeeze(-1)
            dr = self.cfg.domain_randomization
            vel = self._robot.data.root_lin_vel_w[push_ids].clone()
            vel[:, 0] += torch.empty(len(push_ids), device=self.device).uniform_(*dr.push_vel_range)
            vel[:, 1] += torch.empty(len(push_ids), device=self.device).uniform_(*dr.push_vel_range)
            root_vel = torch.cat([vel, self._robot.data.root_ang_vel_w[push_ids]], dim=-1)
            self._robot.write_root_velocity_to_sim(root_vel, push_ids)
            # Reset push timer
            self._push_timer[push_ids] = 0.0
            self._push_interval[push_ids] = torch.empty(
                len(push_ids), device=self.device
            ).uniform_(*dr.push_interval_range)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """Delegate to train_env.compute_observations()."""
        return train_env.compute_observations(self)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Delegate to train_env.compute_rewards()."""
        return train_env.compute_rewards(self)

    # ------------------------------------------------------------------
    # Terminations
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Timeout + illegal body/leg contact + terrain out of bounds."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Body/leg contact termination
        threshold = self.cfg.termination.body_contact_threshold
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # Max force over history for undesired bodies
        is_contact = torch.max(
            torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1),
            dim=1
        )[0] > threshold
        body_contact_terminated = is_contact.any(dim=-1) if is_contact.dim() > 1 else is_contact

        # Terrain out of bounds
        oob_distance = self.cfg.termination.terrain_out_of_bounds_distance
        if hasattr(self._terrain, 'env_origins'):
            dist_from_origin = torch.norm(
                self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2],
                dim=1
            )
            terrain_oob = dist_from_origin > oob_distance
        else:
            terrain_oob = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._termination_masks["body_contact"] = body_contact_terminated
        self._termination_masks["terrain_oob"] = terrain_oob

        terminated = body_contact_terminated | terrain_oob
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset environments with domain randomization."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        prev_episode_steps = self.episode_length_buf[env_ids].clone()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Stagger episode lengths on full reset
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._robot.data.default_joint_pos[env_ids]

        # Get default state
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # Apply terrain origins
        if hasattr(self._terrain, 'env_origins'):
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        else:
            default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # --- Reset domain randomization ---
        dr = self.cfg.domain_randomization
        num = len(env_ids)

        # Pose randomization
        default_root_state[:, 0] += torch.empty(num, device=self.device).uniform_(*dr.pose_x_range)
        default_root_state[:, 1] += torch.empty(num, device=self.device).uniform_(*dr.pose_y_range)
        # Yaw randomization via quaternion
        yaw_angles = torch.empty(num, device=self.device).uniform_(*dr.pose_yaw_range)
        yaw_quats = quat_from_angle_axis(yaw_angles, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(num, -1))
        # Compose with default quaternion (w, x, y, z)
        default_root_state[:, 3:7] = self._quat_multiply(yaw_quats, default_root_state[:, 3:7])

        # Velocity randomization
        default_root_state[:, 7] += torch.empty(num, device=self.device).uniform_(*dr.vel_x_range)
        default_root_state[:, 8] += torch.empty(num, device=self.device).uniform_(*dr.vel_y_range)
        default_root_state[:, 9] += torch.empty(num, device=self.device).uniform_(*dr.vel_z_range)
        default_root_state[:, 10] += torch.empty(num, device=self.device).uniform_(*dr.vel_roll_range)
        default_root_state[:, 11] += torch.empty(num, device=self.device).uniform_(*dr.vel_pitch_range)
        default_root_state[:, 12] += torch.empty(num, device=self.device).uniform_(*dr.vel_yaw_range)

        # Joint randomization (clamp to soft limits)
        joint_pos += torch.empty_like(joint_pos).uniform_(*dr.joint_pos_range)
        joint_vel += torch.empty_like(joint_vel).uniform_(*dr.joint_vel_range)
        # Clamp to soft joint limits
        soft_limits = self._robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, min=soft_limits[..., 0], max=soft_limits[..., 1])
        soft_vel_limits = self._robot.data.soft_joint_vel_limits[env_ids]
        joint_vel = torch.clamp(joint_vel, min=-soft_vel_limits, max=soft_vel_limits)

        # Write to sim
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample commands
        self._sample_commands(env_ids)
        self._command_timer[env_ids] = 0.0
        self._time[env_ids] = 0.0

        # Reset push timers
        self._push_timer[env_ids] = 0.0
        self._push_interval[env_ids] = torch.empty(
            num, device=self.device
        ).uniform_(*dr.push_interval_range)

        # --- TensorBoard logging ---
        log = {}
        if len(env_ids) > 0:
            valid = prev_episode_steps > 0
            if torch.any(valid):
                step_counts = prev_episode_steps[valid].float().clamp(min=1.0)
                for key in self._reward_keys:
                    values = self._episode_sums[key][env_ids][valid] / step_counts
                    log[f'Episode_Reward/{key}'] = torch.mean(values)
                for key in self._metric_keys:
                    values = self._episode_sums[key][env_ids][valid] / step_counts
                    log[f'Episode_Metric/{key}'] = torch.mean(values)

                # Displacement tracking
                current_pos = self._robot.data.root_pos_w[env_ids]
                valid_env_ids = env_ids[valid]
                x_disp = current_pos[valid, 0] - self._episode_start_pos[valid_env_ids, 0]
                log['Episode_Metric/x_displacement'] = torch.mean(x_disp)

            # Termination logging
            body_contact_count = torch.count_nonzero(self._termination_masks["body_contact"][env_ids]).float()
            terrain_oob_count = torch.count_nonzero(self._termination_masks["terrain_oob"][env_ids]).float()
            timeout_count = torch.count_nonzero(self.reset_time_outs[env_ids]).float()
            log['Episode_Termination/body_contact'] = body_contact_count
            log['Episode_Termination/terrain_oob'] = terrain_oob_count
            log['Episode_Termination/time_out'] = timeout_count

            # Reset episode sums
            for key in [*self._reward_keys, *self._metric_keys]:
                self._episode_sums[key][env_ids] = 0.0

        # Store new start positions
        self._episode_start_pos[env_ids] = self._robot.data.root_pos_w[env_ids].clone()

        self.extras['log'] = log

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        """Sample velocity commands for the given environments."""
        cmd = self.cfg.commands
        num = len(env_ids)
        commands = torch.zeros(num, 3, device=self.device)
        commands[:, 0] = torch.empty(num, device=self.device).uniform_(cmd.vx_min, cmd.vx_max)
        commands[:, 1] = torch.empty(num, device=self.device).uniform_(cmd.vy_min, cmd.vy_max)
        commands[:, 2] = torch.empty(num, device=self.device).uniform_(cmd.yaw_min, cmd.yaw_max)

        # Standing environments (zero velocity)
        standing_mask = torch.rand(num, device=self.device) < cmd.standing_probability
        commands[standing_mask] = 0.0

        self._commands[env_ids] = commands

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (w, x, y, z format)."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)
