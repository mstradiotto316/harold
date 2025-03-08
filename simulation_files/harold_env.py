from typing import Sequence
import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import ContactSensor
import math
from .harold_env_cfg import HaroldEnvCfg
from omni.isaac.lab.utils.math import quat_conjugate, quat_mul

# Ros Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque


class HaroldEnv(DirectRLEnv):
    """Environment class for Harold robot, integrated with ROS 2 and RL-Games."""

    cfg: HaroldEnvCfg

    def __init__(self, cfg: HaroldEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions tensor holds the raw torque commands outputted by the policy
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

        # Commands tensor has shape [num_envs, 3], the three dimensions are:
        # 1. X linear velocity
        # 2. Y linear velocity
        # 3. Yaw angular velocity
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Print body names
        print("Body Names: ", self._robot.data.body_names)

        # Print joint names
        print("Joint Names: ", self._robot.data.joint_names)

        # Contact sensor IDs (for debugging or specialized foot contact checks)
        self._contact_ids, _ = self._contact_sensor.find_bodies(".*")
        print("CONTACT IDS: ", self._contact_ids)
        self._body_contact_id, _ = self._contact_sensor.find_bodies(".*body")
        print("BODY CONTACT ID: ", self._body_contact_id)
        self._shoulder_contact_ids, _ = self._contact_sensor.find_bodies(".*shoulder")
        print("SHOULDER CONTACT IDS: ", self._shoulder_contact_ids)
        self._thigh_contact_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        print("THIGH CONTACT IDS: ", self._thigh_contact_ids)
        self._knee_contact_ids, _ = self._contact_sensor.find_bodies(".*knee")
        print("KNEE CONTACT IDS: ", self._knee_contact_ids)
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        print("UNDESIRED CONTACT BODY IDS: ", self._undesired_contact_body_ids)

        # Print body masses
        print("Body Masses: ", self._robot.data.default_mass[0])

        # Print joint positions
        print("Joint Positions: ", torch.round(self._robot.data.joint_pos[0] * 100) / 100)

        # Prepare processed actions tensor
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

        # Add time tracking for sinusoidal observation
        self._time = torch.zeros(self.num_envs, device=self.device)

        # Randomize robot friction 
        # TODO: IS THIS ACTUALLY DOING ANYTHING? WE HAVE SEVERAL MATERIALS DEFINED IN THE USD FILE, ARE WE MODIFYING ALL OF THEM?
        env_ids = self._robot._ALL_INDICES
        mat_props = self._robot.root_physx_view.get_material_properties()
        mat_props[:, :, :2].uniform_(1.0, 2.0)
        self._robot.root_physx_view.set_material_properties(mat_props, env_ids.cpu())

        # Initialize ROS 2
        rclpy.init()
        self.ros2_node = rclpy.create_node('joint_state_publisher')
        self.joint_state_publisher = self.ros2_node.create_publisher(JointState, 'joint_states', 10)

        # Start ROS 2 spinning in a separate thread
        self.ros2_thread = threading.Thread(target=rclpy.spin, args=(self.ros2_node,), daemon=True)
        self.ros2_thread.start()
        self.ros2_counter = 0

        # Decimation counter (For action and observation logging)
        self._decimation_counter = 0

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "stay_in_place",
                "no_rotation",
                "height_position",
                "velocity_control",
                "flat_orientation",
                "feet_on_ground",
                "dof_torques_l2",
            ]
        }

        # Track completed episodes
        self._completed_episodes = 0
        self._print_interval = 20  # Print every 20 episodes
        self._plot_interval = 100  # Plot every 100 episodes
        self._episode_stats = {key: [] for key in self._episode_sums.keys()}  # Changed to lists

        # Add plotting-related attributes
        self.reward_history = {
            key: deque(maxlen=1000)  # Store last 1000 episodes' means
            for key in [
                "stay_in_place",
                "no_rotation",
                "height_position",
                "velocity_control", 
                "flat_orientation",
                "feet_on_ground",
                "dof_torques_l2",
                "total_reward"
            ]
        }

        """
        
                    # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "direction_reward",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "diagonal_feet_coordination",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }

        # Track completed episodes
        self._completed_episodes = 0
        self._print_interval = 20  # Print every 20 episodes
        self._plot_interval = 100  # Plot every 100 episodes
        self._episode_stats = {key: [] for key in self._episode_sums.keys()}  # Changed to lists

        # Add plotting-related attributes
        self.reward_history = {
            key: deque(maxlen=1000)  # Store last 1000 episodes' means
            for key in [
                "track_lin_vel_xy_exp",
                "direction_reward",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "diagonal_feet_coordination",
                "undesired_contacts",
                "flat_orientation_l2",
                "total_reward"
            ]
        }
    
        
        """

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

        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called before physics steps. Used to process and scale actions."""
        self._actions = actions.clone()
        # self._actions = self._actions * self.cfg.action_scale
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

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

        # Update previous actions
        self._previous_actions = self._actions.clone()

        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w, dim=-1)
        knee_contact_forces = contact_forces[:, self._knee_contact_ids]
        knee_contact_binary = (knee_contact_forces > 1.0).float()

        # Calculate sinusoidal values
        self._time += self.step_dt
        
        # sine_wave_1: period matches target air time (0.2s)
        # frequency = 1/period = 1/0.2 = 5 Hz
        sine_wave_1 = torch.sin(2 * math.pi * 5 * self._time)  # 5 Hz frequency
        
        # sine_wave_2: period is double (0.4s)
        # frequency = 1/period = 1/0.4 = 2.5 Hz
        sine_wave_2 = torch.sin(2 * math.pi * 2.5 * self._time)  # 2.5 Hz frequency
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos,
                    self._robot.data.joint_vel,
                    sine_wave_1.unsqueeze(-1),
                    sine_wave_2.unsqueeze(-1),
                )
                if tensor is not None
            ],
            dim=-1,
        )


        # Log observations to a file
        """
        log_dir = "/home/matteo/Desktop/Harold_V5/policy_playback_test/"
        if self._decimation_counter % 18 == 0: # Log every 18 steps
            with open(log_dir + "observations.log", "a") as f:
                f.write(f"{obs[0].tolist()}\n")
        """

        observations = {"policy": obs}

        # OPTION 1: Publish every other step joint states to ROS2 Bridge
        """
        if self.ros2_counter % 2 == 0:
            print("Publishing joint states!")
            self.publish_ROS2_joint_states()
        self.ros2_counter += 1
        print("ROS2 Counter: ", self.ros2_counter)
        """
        
        # OPTION 2: Publish joint states to ROS2 Bridge
        #self.publish_ROS2_joint_states()

        # Call info function to print reward statistics
        self._get_info()

        return observations

    def _get_rewards(self) -> torch.Tensor:



        
        """
        ================== walking reward ==================

            def _get_rewards(self) -> torch.Tensor:

        """
        # Linear velocity tracking
        """

        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        """
        # Yaw rate tracking
        """

        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        """
        # z velocity tracking
        """
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        """
        # angular velocity x/y
        """
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        
        """
        # Get joint torques
        """
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        """
        # joint acceleration
        """
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        """
        # Get action rate
        """
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        """
        # Get feet air time
        """

        # Get the contact forces on all body parts
        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w, dim=-1)
        # Shape: [num_envs, num_bodies]

        # Get the contact forces on the feet
        knee_contact_forces = contact_forces[:, self._knee_contact_ids]
        # Shape: [num_envs, 4] where 4 = number of feet
        
        # Get air time for diagonal pairs explicitly
        paired_air_time = torch.stack([
            self._contact_sensor.data.last_air_time[:, [0, 3]],  # FL + BR
            self._contact_sensor.data.last_air_time[:, [1, 2]]   # FR + BL
        ], dim=1)
        # Shape: [num_envs, 2, 2]

        target_air_time = 0.2 #0.5
        # Calculate the error between the average air time of the diagonal pairs and the target air time
        pair_air_time_error = torch.abs(paired_air_time.mean(dim=-1) - target_air_time)
        # Calculate the air time reward as the negative of the error
        air_time = -torch.sum(pair_air_time_error, dim=1)


        """
        # Diagonal coordination
        """
        
        # Get diagonal pairs of feet (FL+BR, FR+BL)
        diagonal_pairs = torch.stack([
            knee_contact_forces[:, [0, 3]],  # FL + BR
            knee_contact_forces[:, [1, 2]]   # FR + BL
        ], dim=1)
        # Shape: [num_envs, 2, 2] where 2,2 represents two diagonal pairs
        
        # Convert to binary air status (1 for air, 0 for ground)
        diagonal_pairs_in_air = (diagonal_pairs < 0.2).float()
        # Shape: [num_envs, 2, 2]

        # Reward when diagonal pairs are coordinated (both air or both ground)
        diagonal_coordination = torch.sum(
            (diagonal_pairs_in_air.prod(dim=-1) + (1 - diagonal_pairs_in_air).prod(dim=-1)),
            dim=1
        )
        # Shape: [num_envs]


        """
        # Undersired contacts
        """
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        
        """
        # Flat orientation
        """
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # Direction reward: Reward moving in the commanded direction
        direction_reward = torch.where(
            self._commands[:, 0] == 0.0,  # First check if command is zero
            torch.where(
                torch.abs(self._robot.data.root_lin_vel_b[:, 0]) < 0.1,  # If command is zero, check if robot is nearly stationary
                1.0,  # Reward for being still when commanded to be still
                -20.0  # Penalty for moving when commanded to be still
            ),
            torch.where(
                self._commands[:, 0] * self._robot.data.root_lin_vel_b[:, 0] > 0.0,  # If command is non-zero, check direction
                1.0,  # Reward for matching direction
                -20.0  # Penalty for wrong direction
            )
        )

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.step_dt * 5.0,
            "lin_vel_z_l2": z_vel_error * self.step_dt * -2.0,
            "direction_reward": direction_reward * self.step_dt * 0.5,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.step_dt * 0.5,
            "ang_vel_xy_l2": ang_vel_error * self.step_dt * -0.05,
            "dof_torques_l2": joint_torques * self.step_dt * -1e-4,
            #"dof_acc_l2": joint_accel * self.step_dt * -2.5e-7, #-2.5e-7,
            #"action_rate_l2": action_rate * self.step_dt * -0.01, #-0.01,
            "feet_air_time": air_time * self.step_dt * 3.0,
            "diagonal_feet_coordination": diagonal_coordination * self.step_dt * 0.5,
            #"undesired_contacts": contacts * self.step_dt * -1.0, #-1.0,
            #"flat_orientation_l2": flat_orientation * self.step_dt * -20.0 #-20.0,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


        """




        # Stay in place - penalize any horizontal movement
        stay_in_place = torch.sum(torch.square(self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        stay_in_place_reward = torch.exp(-stay_in_place / 0.05)  # Sharper falloff for movement
        
        # Don't rotate - penalize any angular velocity
        no_rotation = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        no_rotation_reward = torch.exp(-no_rotation / 0.05)  # Sharper falloff for rotation
        
        # Body height control using time-based target
        # Get current body height
        body_height = self._robot.data.root_pos_w[:, 2]
        
        # Define push-up height parameters
        min_height = 0.05  # Lowest position in push-up
        max_height = 0.20  # Highest position in push-up
        height_range = max_height - min_height
        
        # Define time-based target using sinusoidal pattern
        # Complete one push-up cycle every 5 seconds
        cycle_period = 5.0
        cycle_freq = 1.0 / cycle_period
        
        # Calculate target height based on time
        # sin oscillates between -1 and 1, we rescale to oscillate between min_height and max_height
        target_height = min_height + 0.5 * height_range * (1 + torch.sin(2 * math.pi * cycle_freq * self._time - math.pi/2))
        
        # Reward being close to target height
        height_position_error = torch.abs(body_height - target_height)
        height_position_reward = torch.exp(-height_position_error / 0.05)
        
        # Maintain proper orientation (body flat/parallel to ground)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        flat_orientation_reward = torch.exp(-flat_orientation / 0.25)
        
        # All feet should stay on ground for pushups
        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w, dim=-1)
        knee_contact_forces = contact_forces[:, self._knee_contact_ids]
        feet_on_ground = torch.mean((knee_contact_forces > 1.0).float(), dim=1)  # Percentage of feet on ground
        
        # Limit joint torques for smooth motion
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        
        rewards = {
            "stay_in_place": stay_in_place_reward * self.step_dt * 5.0,
            "no_rotation": no_rotation_reward * self.step_dt * 2.0,
            "height_position": height_position_reward * self.step_dt * 4.0,
            "velocity_control": torch.zeros_like(body_height),  # Zero placeholder for consistency
            "flat_orientation": flat_orientation_reward * self.step_dt * 3.0,
            "feet_on_ground": feet_on_ground * self.step_dt * 2.0,
            "dof_torques_l2": joint_torques * self.step_dt * -1e-4,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value
            else:
                # Initialize new reward components if they don't exist
                self._episode_sums[key] = value
        
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

        """
        # TODO: Add random command sampling
        #self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        #self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(0.0, 1.0)
        self._commands[env_ids, 0] = 0.2 #0.5 # FORWARD VELOCITY
        self._commands[env_ids, 1] = 0.0 #0.5 # LEFT/RIGHT VELOCITY
        self._commands[env_ids, 2] = 0.0 #0.5 # YAW RATE
        """

        # Reset commands to zero for push-up task (no movement commanded)
        self._commands[env_ids, 0] = 0.0  # FORWARD VELOCITY
        self._commands[env_ids, 1] = 0.0  # LEFT/RIGHT VELOCITY
        self._commands[env_ids, 2] = 0.0  # YAW RATE

        # Reset to default root pose/vel and joint state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset time for sinusoidal observation
        if env_ids is not None:
            self._time[env_ids] = 0

    def _get_info(self) -> dict:
        info = {}
        terminated = self._get_dones()[0]
        
        if torch.any(terminated):
            self._completed_episodes += 1
            
            # Update episode statistics
            for key, value in self._episode_sums.items():
                # Store individual values for each terminated environment
                terminated_values = value[terminated].cpu().numpy()
                self._episode_stats[key].extend(terminated_values.tolist())
            
            # Print and store statistics every N episodes
            if self._completed_episodes % self._print_interval == 0:
                print(f"\n=== Reward Statistics (over {len(self._episode_stats[list(self._episode_stats.keys())[0]])} episodes) ===")
                print("                          Env Mean    Min      Max")
                print("--------------------------------------------------------")
                
                # Calculate and store means for each component
                total_rewards = np.zeros(len(self._episode_stats[list(self._episode_stats.keys())[0]]))
                for key, values in self._episode_stats.items():
                    values_array = np.array(values)
                    mean_val = np.mean(values_array)
                    min_val = np.min(values_array)
                    max_val = np.max(values_array)
                    print(f"{key:25}: {mean_val:8.3f} {min_val:8.3f} {max_val:8.3f}")
                    total_rewards += values_array
                    
                    # Store the mean value for plotting
                    self.reward_history[key].append(mean_val)
                
                # Calculate and store total reward mean
                total_reward_mean = np.mean(total_rewards)
                self.reward_history["total_reward"].append(total_reward_mean)
                
                # Print total reward statistics
                print("--------------------------------------------------------")
                print(f"{'Total Reward':25}: {total_reward_mean:8.3f} {np.min(total_rewards):8.3f} {np.max(total_rewards):8.3f}")
                print("--------------------------------------------------------\n")

                # Reset episode statistics
                self._episode_stats = {key: [] for key in self._episode_sums.keys()}
                
                # Reset episode sums for terminated environments
                for key in self._episode_sums:
                    self._episode_sums[key][terminated] = 0.0
                
            # Update plots every plot interval
            if self._completed_episodes % self._plot_interval == 0:
                self._update_plots()
                
        return info

    def _update_plots(self):
        """Create and update reward component plots."""
        # Create figure without opening a window
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
                episodes = np.arange(len(values_array)) * self._print_interval
                
                plt.plot(episodes, values_array)
                plt.title(key)
                plt.xlabel('Episodes')
                plt.ylabel('Mean Value')
                plt.grid(True)
        
        plt.tight_layout()
        
        # Save and close
        fig.savefig('/home/matteo/Desktop/Harold_V5/reward_plots/reward_components.png')
        plt.close(fig)

    def __del__(self):
        # Shutdown ROS 2 when environment is destroyed
        print("SHUTTING DOWN ROS 2!!!!!!!!!!!!!!!!!")
        self.ros2_node.destroy_node()
        rclpy.shutdown()
        self.ros2_thread.join()

