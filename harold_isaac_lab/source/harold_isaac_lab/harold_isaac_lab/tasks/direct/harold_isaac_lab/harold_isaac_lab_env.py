from typing import Sequence
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
import math
from .harold_isaac_lab_env_cfg import HaroldIsaacLabEnvCfg


class HaroldIsaacLabEnv(DirectRLEnv):
    """Environment class for Harold robot, integrated with ROS 2 and RL-Games."""

    cfg: HaroldIsaacLabEnvCfg

    def __init__(self, cfg: HaroldIsaacLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions tensor holds the raw position commands outputted by the policy
        #self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        #self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        #self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Commands tensor has shape [num_envs, 3], the three dimensions are: X lin vel, Y lin vel, Yaw rate
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Angle limits for each joint
        self._JOINT_ANGLE_MAX = torch.tensor([0.3491, 0.3491, 0.3491, 0.3491, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853, 0.7853], device=self.device)
        self._JOINT_ANGLE_MIN = torch.tensor([-0.3491, -0.3491, -0.3491, -0.3491, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853, -0.7853], device=self.device)

        # Print body names and masses
        print("--------------------------------")
        print("Body Names: ", self._robot.data.body_names)
        print("Body Masses: ", self._robot.data.default_mass[0])
        print()

        # Print joint names and positions
        print("--------------------------------")
        print("Joint Names: ", self._robot.data.joint_names)
        print("Joint Positions: ", torch.round(self._robot.data.joint_pos[0] * 100) / 100)
        print()

        # Contact sensor IDs (for debugging or specialized foot contact checks)
        self._contact_ids, contact_names = self._contact_sensor.find_bodies(".*", preserve_order=True)
        self._body_contact_id, body_names = self._contact_sensor.find_bodies(".*body", preserve_order=True)
        self._shoulder_contact_ids, shoulder_names = self._contact_sensor.find_bodies(".*shoulder", preserve_order=True)
        self._thigh_contact_ids, thigh_names = self._contact_sensor.find_bodies(".*thigh", preserve_order=True)
        self._calf_contact_ids, calf_names = self._contact_sensor.find_bodies(".*calf", preserve_order=True)
        self._undesired_contact_body_ids, undesired_names = self._contact_sensor.find_bodies(".*(body|thigh|shoulder).*", preserve_order=True)

        # Print contact sensor IDs and names (NOTE: For some reason these are added in a depth first manner by regex)
        print("--------------------------------")
        print("CONTACT IDS: ", self._contact_ids, "\nALL CONTACT NAMES: ", contact_names)
        print("BODY CONTACT ID: ", self._body_contact_id, "\nBODY NAMES: ", body_names)
        print("SHOULDER CONTACT IDS: ", self._shoulder_contact_ids, "\nSHOULDER NAMES: ", shoulder_names)
        print("THIGH CONTACT IDS: ", self._thigh_contact_ids, "\nTHIGH NAMES: ", thigh_names)
        print("calf CONTACT IDS: ", self._calf_contact_ids, "\ncalf NAMES: ", calf_names)
        print("UNDESIRED CONTACT BODY IDS: ", self._undesired_contact_body_ids, "\nALL UNDESIRED CONTACT NAMES: ", undesired_names)
        print()

        # Add time tracking for temporal (sinusoidal) observations
        self._time = torch.zeros(self.num_envs, device=self.device)

        """
        # Randomize robot friction 
        # TODO: IS THIS ACTUALLY DOING ANYTHING? WE HAVE SEVERAL MATERIALS DEFINED IN THE USD FILE, ARE WE MODIFYING ALL OF THEM?
        env_ids = self._robot._ALL_INDICES
        mat_props = self._robot.root_physx_view.get_material_properties()
        mat_props[:, :, :2].uniform_(1.0, 2.0)
        self._robot.root_physx_view.set_material_properties(mat_props, env_ids.cpu())
        """

        """
        # Initialize ROS 2
        rclpy.init()
        self.ros2_node = rclpy.create_node('joint_state_publisher')
        self.joint_state_publisher = self.ros2_node.create_publisher(JointState, 'joint_states', 10)

        # Start ROS 2 spinning in a separate thread
        self.ros2_thread = threading.Thread(target=rclpy.spin, args=(self.ros2_node,), daemon=True)
        self.ros2_thread.start()
        self.ros2_counter = 0
        """

        # Decimation counter (For action and observation logging)
        self._decimation_counter = 0
        
        # Logging - episode sums only
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_xy_lin_commands",
                "track_yaw_commands",
                "forward_progress",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "height_reward",
                "xy_acceleration_l2",
                "orientation_l2",
                "alive_bonus",
            ]
        }

    def _setup_scene(self) -> None:
        """Creates and configures the robot, sensors, and terrain."""

        # Create robot articulation
        self._robot = Articulation(self.cfg.robot)

        # Add robot articulation to scene
        self.scene.articulations["robot"] = self._robot

        # Create body contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Create terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Create height scanner (add this after terrain setup)
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called before physics steps. Used to process and scale actions."""
        self._actions = actions.clone()
        self._processed_actions = torch.clamp(
            (self.cfg.action_scale * self._actions),
            self._JOINT_ANGLE_MIN,
            self._JOINT_ANGLE_MAX,
        )

    def _apply_action(self) -> None:
        """Actually apply the actions to the joints."""
        
        # Log raw actions to a file
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
        
        
        self._robot.set_joint_position_target(self._processed_actions)
        self._decimation_counter += 1 # Increment counter

    def publish_ROS2_joint_states(self):
        """Publishes the current joint states to a ROS 2 topic."""

        # Get joint positions and velocities
        joint_positions = self._robot.data.joint_pos[0].tolist()
        joint_velocities = self._robot.data.joint_vel[0].tolist()
        joint_names = self._robot.data.joint_names

        # Create JointState message
        msg = JointState()
        msg.header.stamp = self.ros2_node.get_clock().now().to_msg()
        msg.name = joint_names
        msg.position = joint_positions

        # Publish the message
        self.joint_state_publisher.publish(msg)
        print('Publishing joint states!', msg)

    def _get_observations(self) -> dict:
        """Gather all the relevant states for the policy's observation."""

        # Calculate sinusoidal values
        self._time += self.step_dt

        # Create foot cycle signals with amplitude based on commanded velocity
        gait_freq = 1.5  # Reduced from 2.0 Hz for easier learning
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
                    self._robot.data.joint_pos,
                    self._robot.data.joint_vel,
                    self._commands,
                    self._actions,
                    foot_cycle_1.unsqueeze(-1),
                    foot_cycle_2.unsqueeze(-1),
                    foot_cycle_3.unsqueeze(-1),
                    foot_cycle_4.unsqueeze(-1),
                )
                if tensor is not None
            ],
            dim=-1,
        )

        observations = {"policy": obs}

        # Update previous actions
        self._previous_actions = self._processed_actions.clone()

        # ============================== ROS2 JOINT STATE STREAMING ================================
        #self.publish_ROS2_joint_states()

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
        lin_vel_reward = torch.exp(-1.0 * lin_vel_error_abs)  # Changed from -2.0 to -1.0

        # Add small reward for any forward movement (exploration bonus)
        forward_vel = self._robot.data.root_lin_vel_b[:, 0]
        # Give bigger reward for forward movement, even if not at commanded speed
        forward_progress_reward = torch.clip(forward_vel, -0.5, 1.0)  # Increased upper limit from 0.5 to 1.0

        """
        # Yaw Rate Tracking ============================================================================================
        """
        yaw_error_tracking_abs = torch.abs(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])

        #print("Env 0 Yaw Commands: ", self._commands[:, 2][0].tolist())
        #print("Env 0 Yaw Actual: ", self._robot.data.root_ang_vel_b[:, 2][0].tolist())
        #print("yaw_error_tracking_abs: ", yaw_error_tracking_abs[0])
        #print()

        """
        # z velocity tracking ==========================================================================================
        """
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        #print("z_vel_error: ", z_vel_error[0])
        #print()

        """
        # angular velocity x/y =========================================================================================
        """
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

        #print("ang_vel_error: ", ang_vel_error[0])
        #print()
        
        """
        # Get joint torques ============================================================================================
        """
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        #print("joint_torques: ", joint_torques[0])
        #print()

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

        #print("Feet air times: ", foot_1_air_time[0], foot_2_air_time[0], foot_3_air_time[0], foot_4_air_time[0])
        
        # Create foot cycle signals with amplitude based on commanded velocity
        gait_freq = 1.5  # Reduced from 2.0 Hz for easier learning
        gait_amplitude = torch.clip(torch.abs(self._commands[:, 0]) * 4.0, 0.0, 1.0)
        
        # Trotting gait - diagonal pairs
        foot_cycle_1 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + 0)  # Front Left
        foot_cycle_2 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + math.pi)  # Front Right
        foot_cycle_3 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + math.pi)  # Back Left  
        foot_cycle_4 = gait_amplitude * torch.sin(2 * math.pi * gait_freq * self._time + 0)  # Back Right

        #print("Feet cycles: ", foot_cycle_1[0], foot_cycle_2[0], foot_cycle_3[0], foot_cycle_4[0])

        # If cycle is <0 then the foot should be on the ground so target air time is 0.0
        # If cycle is >0 then the foot should be in the air so target air time is the cycle value
        foot_1_target_air_time = torch.where(foot_cycle_1 < 0.0, torch.zeros_like(foot_cycle_1), foot_cycle_1) # Front left 1st
        foot_2_target_air_time = torch.where(foot_cycle_2 < 0.0, torch.zeros_like(foot_cycle_2), foot_cycle_2) # Front right 2nd
        foot_3_target_air_time = torch.where(foot_cycle_3 < 0.0, torch.zeros_like(foot_cycle_3), foot_cycle_3) # Back left 3rd
        foot_4_target_air_time = torch.where(foot_cycle_4 < 0.0, torch.zeros_like(foot_cycle_4), foot_cycle_4) # Back right 4th

        #print("Feet air time targets: ", foot_1_target_air_time[0], foot_2_target_air_time[0], foot_3_target_air_time[0], foot_4_target_air_time[0])

        foot_1_error = torch.abs(foot_1_target_air_time - foot_1_air_time)
        foot_2_error = torch.abs(foot_2_target_air_time - foot_2_air_time)
        foot_3_error = torch.abs(foot_3_target_air_time - foot_3_air_time)
        foot_4_error = torch.abs(foot_4_target_air_time - foot_4_air_time)

        #print("Foot 1 error: ", foot_1_error[0])
        #print("Foot 2 error: ", foot_2_error[0])
        #print("Foot 3 error: ", foot_3_error[0])
        #print("Foot 4 error: ", foot_4_error[0])

        # Sum of absolute errors
        foot_error = foot_1_error + foot_2_error + foot_3_error + foot_4_error

        #print("Foot error: ", foot_error[0])

        
        """
        # Undersired contacts ==========================================================================================
        """
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 0.1
        )
        contacts = torch.sum(is_contact, dim=1)

        """
        # Height Reward ================================================================================================
        """
        # Get height data from scanner
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2]
        ).clip(-1.0, 1.0)

        # Calculate mean height for each environment
        current_height = torch.mean(height_data, dim=1)

        # Define target height and calculate error
        target_height = 0.16  # Lowered from 0.18 to encourage more dynamic walking
        height_error = torch.abs(current_height - target_height)

        # Convert to reward using exponential form that saturates
        # Use tanh to cap the reward so standing isn't over-rewarded
        height_reward = torch.tanh(3.0 * torch.exp(-5.0 * height_error))

        #print("Current height from scanner: ", current_height[0], " Height reward: ", height_reward[0])
        #print("Target height: ", target_height)
        #print('Height reward: ', height_reward)

        


        """
        # XY acceleration penalty ======================================================================================
        """
        # Get root body xy acceleration (first 2 components of linear acceleration)
        xy_acceleration = self._robot.data.body_acc_w[:, 0, :2]  # Shape: [num_envs, 2]
        xy_acceleration_error = torch.sum(torch.square(xy_acceleration), dim=1)

        #print("xy_acceleration: ", xy_acceleration[0].tolist())
        #print("xy_acceleration_error: ", xy_acceleration_error[0])

        """
        # Body orientation (roll and pitch) ============================================================================
        """
        orientation_error = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_xy_lin_commands": lin_vel_reward * self.step_dt * 8.0,  # Reduced from 10.0
            "track_yaw_commands": yaw_error_tracking_abs * self.step_dt * -1.0,
            "forward_progress": forward_progress_reward * self.step_dt * 10.0,  # Reduced from 15.0
            "lin_vel_z_l2": z_vel_error * self.step_dt * -2.0,  # Increased from -1.0
            "ang_vel_xy_l2": ang_vel_error * self.step_dt * -0.1,  # Increased from -0.05
            "dof_torques_l2": joint_torques * self.step_dt * -0.001,
            "dof_acc_l2": joint_accel * self.step_dt * 0.0,
            "action_rate_l2": action_rate * self.step_dt * -0.001,
            "feet_air_time": foot_error * self.step_dt * -0.5,
            "undesired_contacts": contacts * self.step_dt * -1.0,
            "height_reward": height_reward * self.step_dt * 2.0,  # Increased from 1.0
            "xy_acceleration_l2": xy_acceleration_error * self.step_dt * 0.0,
            "orientation_l2": orientation_error * self.step_dt * -2.0,  # Increased from -0.5
            "alive_bonus": torch.ones_like(forward_vel) * self.step_dt * 1.0,  # Reduced from 2.0
        }

        #print("Commands: ", self._commands[0].tolist())
        #print()
        #print("Avg lin vel: ", avg_lin_vel[0].tolist())
        #print("track_xy_lin_commands: ", rewards["track_xy_lin_commands"][0])
        #print()
        #print("Avg yaw vel: ", avg_yaw_vel[0].tolist())
        #print("track_yaw_commands: ", rewards["track_yaw_commands"][0])
        #print()
        #print("lin_vel_z_l2: ", rewards["lin_vel_z_l2"][0])
        #print("ang_vel_xy_l2: ", rewards["ang_vel_xy_l2"][0])
        #print("dof_torques_l2: ", rewards["dof_torques_l2"][0])
        #print("feet_air_time: ", rewards["feet_air_time"][0])
        #print("height_reward: ", rewards["height_reward"][0])
        #print("------------------------------------------------------------------")
        #print()
        
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
        body_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._body_contact_id], dim=-1), dim=1)[0] > 0.2, dim=1)
        shoulder_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._shoulder_contact_ids], dim=-1), dim=1)[0] > 0.2, dim=1)
        thigh_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._thigh_contact_ids], dim=-1), dim=1)[0] > 0.2, dim=1)
        contact_terminated = body_contact | shoulder_contact | thigh_contact

        # Terminate if robot has fallen (orientation too far from upright)
        # projected_gravity_b[:, 2] is the z component in body frame, should be close to -1 when upright
        orientation_terminated = self._robot.data.projected_gravity_b[:, 2] > -0.5  # Robot tilted more than 60 degrees

        # Combine all termination conditions
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
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new commands with simple curriculum
        # Start with small velocities and increase over time
        progress = min(1.0, self._decimation_counter / 50000.0)  # Full speed after ~50k steps
        max_vel = 0.25 * progress
        min_vel = 0.05 * progress if progress > 0.2 else 0.0  # Add minimum velocity after 20% progress
        
        self._commands[env_ids, 0] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(min_vel, max(max_vel, 0.05))
        self._commands[env_ids, 1] = 0.0  # Y velocity
        self._commands[env_ids, 2] = 0.0  # Yaw rate

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset time for sinusoidal observation
        self._time[env_ids] = 0

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
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
        # Shutdown ROS 2 when environment is destroyed
        #print("SHUTTING DOWN ROS 2!!!!!!!!!!!!!!!!!")
        #self.ros2_node.destroy_node()
        #rclpy.shutdown()
        #self.ros2_thread.join()
        pass








