from typing import Sequence
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
import math
from .harold_isaac_lab_env_cfg import HaroldIsaacLabEnvCfg

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque


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
        
        # Logging
        # Track episode sums for currently running episodes
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_xy_lin_commands",
                "track_yaw_commands",
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
            ]
        }

        # Track completed episodes history (for tensorboard-style logging)
        self.reward_history = {
            key: deque(maxlen=100)  # Store last 100 episodes like tensorboard
            for key in list(self._episode_sums.keys()) + ["total_reward"]
        }

        # Track completed episodes and intervals
        self._completed_episodes = 0
        self._print_interval = 20  # Print every 20 episodes
        self._plot_interval = 100  # Plot every 100 episodes

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
        
        # sine_wave_1: period matches target air time (0.2s)
        sine_wave_1 = torch.sin(2 * math.pi * 5 * self._time)  # (0.2s) 5 Hz frequency
        sine_wave_2 = torch.sin(2 * math.pi * 2.5 * self._time)  # (0.4s) 2.5 Hz frequency

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
                    #self._previous_actions,
                    sine_wave_1.unsqueeze(-1),
                    sine_wave_2.unsqueeze(-1),
                )
                if tensor is not None
            ],
            dim=-1,
        )

        observations = {"policy": obs}

        # Update previous actions
        self._previous_actions = self._processed_actions.clone()

        # ================================= PRINT REWARD STATISTICS ================================
        # Call info function to print reward statistics
        self._get_info()

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
        lin_vel_error_abs = torch.sum(torch.abs(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)

        #print("Env 0 XY Commands: ", self._commands[:, :2][0].tolist())
        #print("Env 0 XY Actual: ", self._robot.data.root_lin_vel_b[:, :2][0].tolist())
        #print("lin_vel_error_abs: ", lin_vel_error_abs[0])
        #print()

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
        
        # Create foot cycle signals --> 1s period, offset by pi/2 each, signal is btw -1 (grounded) and 1 (in air)
        #Order: Front left, back right, front right, back left
        foot_cycle_1 = torch.sin(2 * math.pi * 0.125 * self._time + math.pi/2) # Front Left
        foot_cycle_2 = torch.sin(2 * math.pi * 0.125 * self._time + math.pi) # Front Right
        foot_cycle_3 = torch.sin(2 * math.pi * 0.125 * self._time + 3*math.pi/2) # Back Left
        foot_cycle_4 = torch.sin(2 * math.pi * 0.125 * self._time + 2*math.pi) # Back Right # was 0.125 before

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
        target_height = 0.18 #0.20  # meters from ground
        height_error = torch.abs(current_height - target_height)

        # Convert to reward using same exponential form as velocity rewards
        height_reward = torch.exp(-5.0 * height_error)

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
            "track_xy_lin_commands": lin_vel_error_abs * self.step_dt * -3.5, #-5.5, #-2.0, #(CONFIRMED -4.0)
            "track_yaw_commands": yaw_error_tracking_abs * self.step_dt * -2.0, #(CONFIRMED -4.0)
            "lin_vel_z_l2": z_vel_error * self.step_dt * 0.0, #-10.0,
            "ang_vel_xy_l2": ang_vel_error * self.step_dt * 0.0, #-5, #-0.05,
            "dof_torques_l2": joint_torques * self.step_dt * -0.01, #(CONFIRMED -0.01)
            "dof_acc_l2": joint_accel * self.step_dt * 0.0, #-0.5e-6, #-1.0e-6, #-2.5e-7,
            "action_rate_l2": action_rate * self.step_dt * -0.01, #(CONFIRMED -0.01)
            #"feet_air_time": air_time_reward * self.step_dt * 0.3, #(CONFIRMED 0.3)
            "feet_air_time": foot_error * self.step_dt * -1.0,
            "undesired_contacts": contacts * self.step_dt * -5.0,
            "height_reward": height_reward * self.step_dt * 10.0,
            "xy_acceleration_l2": xy_acceleration_error * self.step_dt * 0.0, #-0.5 #-0.15,
            "orientation_l2": orientation_error * self.step_dt * -3.0,
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

        # Combine all termination conditions
        terminated = contact_terminated

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

        # Randomize commands only for environments that are resetting
        # Get commands for resetting envs
        temp_commands = self._commands[env_ids]

        # Overall commands tests
        # TEST 1 (COMPLETE): Commands all 0, with height reward results in the robot standing upright and still
        # TEST 2 (PENDING): Commands all 0, with height reward and feet air time reward gives us in place trotting.

        # X velocity 
        # TEST 1 (COMPLETE): Works with set command at 0.25
        # TEST 2 (FAILED): Works with set command at -0.25
        # TEST 3 (COMPLETE): Works with range of 0 to 0.25
        temp_commands[:, 0].uniform_(0.25, 0.25)
        
        # Y velocity
        # TEST 1 (FAILED): Works with set command at 0.25
        # TEST 2 (PENDING): Works with set command at -0.25
        # TEST 3 (PENDING): Works with range of 0 to 0.25
        temp_commands[:, 1].uniform_(0.0, 0.0)

        # Yaw rate (Note: The policy takes longer to train with yaw rate commands)
        # TEST 1 (COMPLETE): Works with set command at 0.25
        # TEST 2 (PENDING): Works with set command at -0.25
        # TEST 3 (PENDING): Works with range of 0 to 0.25
        temp_commands[:, 2].uniform_(0.0, 0.0)  

        # Write back the randomized commands
        self._commands[env_ids] = temp_commands  

        # Reset to default root pose/vel and joint state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset time for sinusoidal observation
        self._time[env_ids] = 0

    def _get_info(self) -> dict:
        info = {}
        terminated = self._get_dones()[0]
        
        if torch.any(terminated):
            self._completed_episodes += 1
            
            # Calculate total reward for terminated environments
            total_reward = sum(value[terminated].cpu().numpy()[0] for value in self._episode_sums.values())
            
            # Store individual components and total reward
            for key, value in self._episode_sums.items():
                self.reward_history[key].append(value[terminated].cpu().numpy()[0])
            self.reward_history["total_reward"].append(total_reward)
            
            # Print statistics every N episodes
            if self._completed_episodes % self._print_interval == 0:
                print(f"\n=== Reward Statistics (last 100 episodes) ===")
                print("                          Mean    Min      Max")
                print("--------------------------------------------------------")
                
                for key, values in self.reward_history.items():
                    if values:  # Check if we have data
                        values_array = np.array(list(values))
                        mean_val = np.mean(values_array)
                        min_val = np.min(values_array)
                        max_val = np.max(values_array)
                        print(f"{key:25}: {mean_val:8.3f} {min_val:8.3f} {max_val:8.3f}")
                
                print("--------------------------------------------------------\n")
            
            
            # Reset episode sums for terminated environments
            for key in self._episode_sums:
                self._episode_sums[key][terminated] = 0.0
                
            # Update plots every plot interval
            if self._completed_episodes % self._plot_interval == 0:
                self._update_plots()




            # CUSTOM CODE: Add reward component tracking to tensorboard
            if hasattr(self, "agent") and self._completed_episodes % self._print_interval == 0:
                for key, buf in self.reward_history.items():
                    if buf:                                    # avoid empty deque
                        self.agent.track_data(f"Episode/{key}",
                                            float(np.mean(buf)))
                
        return info

    def _update_plots(self):
        """Create and update reward component plots showing rolling averages."""
        plt.ioff()  # Turn off interactive mode
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplot grid
        num_components = len(self.reward_history)
        cols = 3
        rows = (num_components + cols - 1) // cols
        
        for idx, (key, values) in enumerate(self.reward_history.items(), 1):
            plt.subplot(rows, cols, idx)
            if values:  # Check if we have data
                values_array = np.array(list(values))
                # Calculate number of episodes (each value represents the average of last 100 episodes)
                episodes = np.arange(len(values_array))
                
                # Plot rolling average
                plt.plot(episodes, values_array, label='100-episode average')
                plt.title(key)
                plt.xlabel('Episodes (x100)')
                plt.ylabel('Average Value')
                plt.grid(True)
        
        plt.tight_layout()
        
        # Save and close
        plt.savefig('/home/matteo/Desktop/Harold_V5/reward_plots/reward_components.png')
        plt.close(fig)

    def __del__(self):
        # Shutdown ROS 2 when environment is destroyed
        print("SHUTTING DOWN ROS 2!!!!!!!!!!!!!!!!!")
        self.ros2_node.destroy_node()
        rclpy.shutdown()
        self.ros2_thread.join()









