from typing import Sequence
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
import math
from .harold_isaac_lab_env_cfg import HaroldIsaacLabEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis


class HaroldIsaacLabEnv(DirectRLEnv):
    """Environment class for Harold robot, integrated with RL training frameworks."""

    cfg: HaroldIsaacLabEnvCfg

    def __init__(self, cfg: HaroldIsaacLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- Action Buffers & Initial Joint Targets ---
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._processed_actions = ( self.cfg.action_scale * self._actions ) + self._robot.data.default_joint_pos
        
        # --- Command Tensor for Velocity Tracking (X, Y, Yaw) ---
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # --- Joint Angle Limits Configuration ---
        self._JOINT_ANGLE_MAX = torch.tensor([0.3491, 0.3491, 0.3491, 0.3491, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853], device=self.device)
        self._JOINT_ANGLE_MIN = torch.tensor([-0.3491, -0.3491, -0.3491, -0.3491, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853], device=self.device)

        # --- Debug: Print Robot Body & Joint Info ---
        print("--------------------------------")
        print("Body Names: ", self._robot.data.body_names)
        print("Body Masses: ", self._robot.data.default_mass[0])
        print()

        print("--------------------------------")
        print("Joint Names: ", self._robot.data.joint_names)
        print("Joint Positions: ", torch.round(self._robot.data.joint_pos[0] * 100) / 100)
        print()

        # --- Contact Sensor Body ID Extraction ---
        self._contact_ids, contact_names = self._contact_sensor.find_bodies(".*", preserve_order=True)
        self._body_contact_id, body_names = self._contact_sensor.find_bodies(".*body", preserve_order=True)
        self._shoulder_contact_ids, shoulder_names = self._contact_sensor.find_bodies(".*shoulder", preserve_order=True)
        self._thigh_contact_ids, thigh_names = self._contact_sensor.find_bodies(".*thigh", preserve_order=True)
        self._calf_contact_ids, calf_names = self._contact_sensor.find_bodies(".*calf", preserve_order=True)
        self._undesired_contact_body_ids, undesired_names = self._contact_sensor.find_bodies(".*(body|thigh|shoulder).*", preserve_order=True)

        # --- Observation & Curriculum State Trackers ---
        self._time = torch.zeros(self.num_envs, device=self.device)
        self._prev_lin_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._decimation_counter = 0
        self._policy_step = 0
        
        # --- Episode Reward Logging Buffers ---
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_xy_lin_commands",
                "track_yaw_commands",
                "height_reward",
                "velocity_jitter",
            ]
        }

    def _setup_scene(self) -> None:
        """Creates and configures the robot, sensors, and terrain."""

        # --- Robot Articulation Setup ---
        # Instantiate robot articulation from config and add to scene
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # --- Contact Sensor Setup ---
        # Create and register a contact sensor for all robot bodies
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # --- Terrain Setup ---
        # Synchronize terrain config with number of envs and spacing, then spawn terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs  # TODO: fix inefficiency: repeated assignment per scene init
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # --- Height Scanner Setup ---
        # Spawn a ray-caster sensor for terrain height measurements
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # --- Environment Cloning & Collision Filtering ---
        # Duplicate envs, optionally without copying prim data, and disable collisions with the ground
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # --- Lighting Setup ---
        # Add a dome light to illuminate the scene
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # --- Visualization Markers Setup ---
        # Instantiate and configure arrow markers for command vs actual velocity
        command_arrow_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/command_arrows",
            markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 0.25),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                )
            }
        )
        
        # Configure a smaller arrow prototype for actual velocity
        actual_arrow_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/actual_arrows",
            markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 0.25),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                )
            }
        )

        self._cmd_marker = VisualizationMarkers(command_arrow_cfg)
        self._act_marker = VisualizationMarkers(actual_arrow_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Prepare, scale, and clamp raw policy actions before each physics step."""
        # --- Action cloning ---
        self._actions = actions.clone()

        # --- Action scaling & clamping to joint limits ---
        self._processed_actions = torch.clamp(
            self.cfg.action_scale * self._actions,
            self._JOINT_ANGLE_MIN,
            self._JOINT_ANGLE_MAX,
        )

    def _apply_action(self) -> None:
        """Apply processed actions to joints, handle optional logging, and update visual markers."""
        
        # --- Logging (commented for later use) ---
        """
        log_dir = "/home/matteo/Desktop/Harold_V5/policy_playback_test/"
        if self._decimation_counter % 18 == 0: # Log every 18 steps
            with open(log_dir + "actions.log", "a") as f:
                f.write(f"{self._actions.tolist()}\n") # Write tensor data only

        # Log scaled actions to a file
        log_dir = "/home/matteo/Desktop/Harold_V5/policy_playback_test/"
        if self._decimation_counter % 18 == 0: # Log every 18 steps
            with open(log_dir + "processed_actions.log", "a") as f:
                f.write(f"{self._processed_actions.tolist()}\n") # Write tensor data only
        """
        
        # --- Send joint targets to robot ---
        self._robot.set_joint_position_target(self._processed_actions)

        # --- Decimation counter for logging/diagnostics ---
        self._decimation_counter += 1  # Increment counter
        # ----- UPDATE ARROW MARKERS -----
        base_pos = self._height_scanner.data.pos_w  # [num_envs, 3]
        marker_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # Commanded arrow: orientation from commanded X/Y velocity
        cmd_vel = self._commands[:, :2]
        cmd_angle = torch.atan2(cmd_vel[:, 1], cmd_vel[:, 0])
        marker_ori_cmd = quat_from_angle_axis(cmd_angle, torch.tensor((0.0, 0.0, 1.0), device=self.device))
        marker_pos_cmd = base_pos + torch.tensor((0.0, 0.0, 0.25), device=self.device)
        self._cmd_marker.visualize(marker_pos_cmd, marker_ori_cmd, marker_indices=marker_idx)
        # Actual arrow: orientation from actual X/Y root velocity
        act_vel = self._robot.data.root_lin_vel_b[:, :2]
        act_angle = torch.atan2(act_vel[:, 1], act_vel[:, 0])
        marker_ori_act = quat_from_angle_axis(act_angle, torch.tensor((0.0, 0.0, 1.0), device=self.device))
        marker_pos_act = base_pos + torch.tensor((0.0, 0.0, 0.40), device=self.device)
        self._act_marker.visualize(marker_pos_act, marker_ori_act, marker_indices=marker_idx)

    def _get_observations(self) -> dict:
        """Gather all the relevant states for the policy's observation."""

        # Increment policy life-step counter (one per RL step)
        self._policy_step += 1
        # Occasionally log curriculum alpha
        if self._policy_step % 10000 == 0:
            alpha = min(self._policy_step / self.cfg.curriculum.phase_transition_steps, 1.0)
            print(f"[Curriculum] Policy Step {self._policy_step}, alpha = {alpha:.4f}")

        # Calculate sinusoidal values
        self._time += self.step_dt

        # Create foot cycle signals with amplitude based on commanded velocity
        gait_freq = self.cfg.gait.frequency  # Use config value
        gait_amplitude = torch.clip(torch.abs(self._commands[:, 0]) * 4.0, 0.0, 1.0)
        
        # Trotting gait - diagonal pairs
        foot_cycle_1 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + 0)  # Front Left
        foot_cycle_2 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + math.pi)  # Front Right
        foot_cycle_3 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + math.pi)  # Back Left  
        foot_cycle_4 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + 0)  # Back Right

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._commands,
                    self._actions - self._robot.data.default_joint_pos
                    #foot_cycle_1.unsqueeze(-1),
                    #foot_cycle_2.unsqueeze(-1),
                    #foot_cycle_3.unsqueeze(-1),
                    #foot_cycle_4.unsqueeze(-1),
                )
                if tensor is not None
            ],
            dim=-1,
        )

        observations = {"policy": obs}

        # Update previous actions
        self._previous_actions = self._actions.clone()


        # ============================ LOGGING FOR SIMULATION PLAYBACK =============================
        # Log observations to a file
        
        """
        log_dir = "/home/matteo/Desktop/Harold_V5/policy_playback_test/"
        if self._decimation_counter % 18 == 0: # Log every 18 steps
            with open(log_dir + "observations.log", "a") as f:
                f.write(f"{obs[0].tolist()}\n")
        """
        

        return observations

    def _get_rewards(self) -> torch.Tensor:

        """
        # Linear velocity tracking =====================================================================================
        """
        lin_vel_error = self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]
        lin_vel_error_abs = torch.sum(torch.abs(lin_vel_error), dim=1)
        # Use exponential reward for smoother learning, but with a more forgiving decay
        lin_vel_reward = torch.exp(-2.0 * lin_vel_error_abs)  # Changed from -2.0 to -1.0

        # Add small reward for any forward movement (exploration bonus)
        forward_vel = self._robot.data.root_lin_vel_b[:, 0]
        # Give bigger reward for forward movement, even if not at commanded speed
        forward_progress_reward = torch.clip(forward_vel, -0.5, 1.0)  # Increased upper limit from 0.5 to 1.0

        """
        # Yaw Rate Tracking ============================================================================================
        """
        yaw_error_tracking_abs = torch.abs(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])

        """
        # z velocity tracking ==========================================================================================
        """
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        """
        # angular velocity x/y =========================================================================================
        """
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        
        """
        # Get joint torques ============================================================================================
        """
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        """
        # joint acceleration ===========================================================================================
        """
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        """
        # Get action rate ==============================================================================================
        """
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        """
        # Get feet air time ============================================================================================
        """

        # Track how long each foot has been in the air for
        foot_1_air_time = self._contact_sensor.data.current_air_time[:, self._calf_contact_ids[0]]
        foot_2_air_time = self._contact_sensor.data.current_air_time[:, self._calf_contact_ids[1]]
        foot_3_air_time = self._contact_sensor.data.current_air_time[:, self._calf_contact_ids[2]]
        foot_4_air_time = self._contact_sensor.data.current_air_time[:, self._calf_contact_ids[3]]
        
        # Create foot cycle signals with amplitude based on commanded velocity
        gait_freq = self.cfg.gait.frequency  # Use config value
        gait_amplitude = torch.clip(torch.abs(self._commands[:, 0]) * 4.0, 0.0, 1.0)
        
        # Trotting gait - diagonal pairs
        foot_cycle_1 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + 0)  # Front Left
        foot_cycle_2 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + math.pi)  # Front Right
        foot_cycle_3 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + math.pi)  # Back Left  
        foot_cycle_4 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + 0)  # Back Right

        # If cycle is <0 then the foot should be on the ground so target air time is 0.0
        # If cycle is >0 then the foot should be in the air so target air time is the cycle value
        foot_1_target_air_time = torch.where(foot_cycle_1 < 0.0, torch.zeros_like(foot_cycle_1), foot_cycle_1) # Front left 1st
        foot_2_target_air_time = torch.where(foot_cycle_2 < 0.0, torch.zeros_like(foot_cycle_2), foot_cycle_2) # Front right 2nd
        foot_3_target_air_time = torch.where(foot_cycle_3 < 0.0, torch.zeros_like(foot_cycle_3), foot_cycle_3) # Back left 3rd
        foot_4_target_air_time = torch.where(foot_cycle_4 < 0.0, torch.zeros_like(foot_cycle_4), foot_cycle_4) # Back right 4th

        foot_1_error = torch.abs(foot_1_target_air_time - foot_1_air_time)
        foot_2_error = torch.abs(foot_2_target_air_time - foot_2_air_time)
        foot_3_error = torch.abs(foot_3_target_air_time - foot_3_air_time)
        foot_4_error = torch.abs(foot_4_target_air_time - foot_4_air_time)

        # Sum of absolute errors
        foot_error = foot_1_error + foot_2_error + foot_3_error + foot_4_error

        
        """
        # Undersired contacts ==========================================================================================
        """
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > self.cfg.termination.contact_force_threshold
        )
        contacts = torch.sum(is_contact, dim=1)

        """
        # Height Reward ================================================================================================
        """
        # Get height data from scanner
        height_data = self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2]

        # Calculate mean height for each environment
        current_height = torch.mean(height_data, dim=1)

        # Define target height and calculate error
        target_height = self.cfg.gait.target_height  # Use config value
        height_error = torch.abs(current_height - target_height)

        # Convert to reward using exponential form that saturates
        # Use tanh to cap the reward so standing isn't over-rewarded
        height_reward = torch.tanh(3.0 * torch.exp(-5.0 * height_error))

        


        """
        # XY acceleration penalty ======================================================================================
        """
        # Get root body xy acceleration (first 2 components of linear acceleration)
        xy_acceleration = self._robot.data.body_acc_w[:, 0, :2]  # Shape: [num_envs, 2]
        xy_acceleration_error = torch.sum(torch.square(xy_acceleration), dim=1)

        """
        # Body orientation (roll and pitch) ============================================================================
        """
        orientation_error = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        """
        # Velocity Jitter ==============================================================================================
        """
        # Compute velocity jitter penalty: angle between last and current root_xy velocity (skip if too small)
        curr_vel = self._robot.data.root_lin_vel_b[:, :2]
        prev_vel = self._prev_lin_vel
        dot = torch.sum(prev_vel * curr_vel, dim=1)
        norm_prev = torch.norm(prev_vel, dim=1)
        norm_curr = torch.norm(curr_vel, dim=1)
        # Minimum speed threshold for applying jitter penalty (m/s)
        eps = 1e-3
        valid = (norm_prev > eps) & (norm_curr > eps)
        cos_raw = torch.where(valid, dot / (norm_prev * norm_curr + 1e-8), torch.zeros_like(dot))
        cos_val = torch.clamp(cos_raw, -1.0, 1.0)
        jitter_angle = torch.where(valid, torch.acos(cos_val), torch.zeros_like(dot))
        # Update previous velocity for next step
        self._prev_lin_vel = curr_vel.clone()

        # Use configuration values for reward weights
        rewards = {
            "track_xy_lin_commands": lin_vel_reward * self.step_dt * self.cfg.rewards.track_xy_lin_commands,
            #"track_yaw_commands": yaw_error_tracking_abs * self.step_dt * self.cfg.rewards.track_yaw_commands,
            #"forward_progress": forward_progress_reward * self.step_dt * self.cfg.rewards.forward_progress,
            #"lin_vel_z_l2": z_vel_error * self.step_dt * self.cfg.rewards.lin_vel_z_l2,
            #"ang_vel_xy_l2": ang_vel_error * self.step_dt * self.cfg.rewards.ang_vel_xy_l2,
            #"dof_torques_l2": joint_torques * self.step_dt * self.cfg.rewards.dof_torques_l2,
            #"dof_acc_l2": joint_accel * self.step_dt * self.cfg.rewards.dof_acc_l2,
            #"action_rate_l2": action_rate * self.step_dt * self.cfg.rewards.action_rate_l2,
            #"feet_air_time": foot_error * self.step_dt * self.cfg.rewards.feet_air_time,
            #"undesired_contacts": contacts * self.step_dt * self.cfg.rewards.undesired_contacts,
            "height_reward": height_reward * self.step_dt * self.cfg.rewards.height_reward,
            #"xy_acceleration_l2": xy_acceleration_error * self.step_dt * self.cfg.rewards.xy_acceleration_l2,
            #"orientation_l2": orientation_error * self.step_dt * self.cfg.rewards.orientation_l2,
            "velocity_jitter": jitter_angle * self.step_dt * self.cfg.rewards.velocity_jitter,
            #"alive_bonus": torch.ones_like(forward_vel) * self.step_dt * self.cfg.rewards.alive_bonus,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        # Terminate if episode length exceeds max episode length
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminate if undesired contacts are detected
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        body_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._body_contact_id], dim=-1), dim=1)[0] > self.cfg.termination.contact_force_threshold, dim=1)
        shoulder_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._shoulder_contact_ids], dim=-1), dim=1)[0] > self.cfg.termination.contact_force_threshold, dim=1)
        thigh_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._thigh_contact_ids], dim=-1), dim=1)[0] > self.cfg.termination.contact_force_threshold, dim=1)
        contact_terminated = body_contact | shoulder_contact | thigh_contact

        # Terminate if robot has fallen (orientation too far from upright)
        # projected_gravity_b[:, 2] is the z component in body frame, should be close to -1 when upright
        orientation_terminated = self._robot.data.projected_gravity_b[:, 2] > self.cfg.termination.orientation_threshold

        

        # Combine all termination conditions (fell-over only); keep timeout separate
        terminated = contact_terminated | orientation_terminated

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset specified environment(s) to default state."""

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=self.max_episode_length)
        
        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        #self._commands[env_ids, 0] = 0.5  # X velocity
        #self._commands[env_ids, 1] = 0.0  # Y velocity
        #self._commands[env_ids, 2] = 0.0  # Yaw rate

        # Curriculum-scheduled commands: sample and scale by curriculum factor
        alpha = min(self._policy_step / self.cfg.curriculum.phase_transition_steps, 1.0)
        sampled_commands = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        self._commands[env_ids] = sampled_commands * alpha

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset previous velocity for jitter penalty
        self._prev_lin_vel[env_ids] = self._robot.data.root_lin_vel_b[env_ids, :2].clone()

        # Reset time for sinusoidal observation
        self._time[env_ids] = 0

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            # Normalize by max episode length (seconds)
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        # Log termination reasons
        extras = dict()
        # Check termination types for environments that actually terminated (not timed out)
        if torch.any(self.reset_terminated[env_ids]):
            # Get contact forces and orientation for terminated envs
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            body_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._body_contact_id], dim=-1), dim=1)[0] > 0.2, dim=1)
            orientation_fallen = self._robot.data.projected_gravity_b[:, 2] > -0.5
            
            # Count different termination types
            contact_term_count = torch.sum(body_contact[env_ids] & self.reset_terminated[env_ids]).item()
            orientation_term_count = torch.sum(orientation_fallen[env_ids] & self.reset_terminated[env_ids]).item()
            
            extras["Episode_Termination/contact"] = contact_term_count
            extras["Episode_Termination/orientation"] = orientation_term_count
        else:
            extras["Episode_Termination/contact"] = 0
            extras["Episode_Termination/orientation"] = 0
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def __del__(self):
        pass








