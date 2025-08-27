from typing import Sequence
import sys
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
import math
from .harold_isaac_lab_env_cfg import HaroldIsaacLabEnvCfg
# Visualization arrows removed for push-up task
from isaaclab.utils.math import quat_from_angle_axis, sample_uniform
from isaaclab.utils.noise import gaussian_noise, uniform_noise


class HaroldIsaacLabEnv(DirectRLEnv):
    """Reinforcement Learning Environment for Harold Quadruped Robot.
    
    This class implements a sophisticated RL training environment for a 12-DOF quadruped robot
    with multi-terrain support and comprehensive reward shaping. The
    environment features progressive terrain difficulty scaling, velocity command tracking,
    and energy-efficient locomotion rewards.
    
    Key Features:
        - 12-DOF quadruped with 3 joints per leg (shoulder, thigh, calf)
        - Multi-terrain training with various difficulty levels
        - Multi-component reward system (velocity tracking, height maintenance, energy efficiency)
        - Contact-based termination and gait analysis
        - Real-time visualization with velocity command/actual arrows
        - 45-dimensional observation space (no commands)
        - Physics simulation at 360Hz with 20Hz policy updates (18:1 decimation)
    
    State Spaces:
        - Observation: 48D vector [root_vel(6) + gravity(3) + joint_pos(12) + joint_vel(12) + commands(3) + actions(12)]
        - Action: 12D joint position targets (clamped to safety limits)
        - Terrain: 200 unique patches (10 difficulty levels × 20 variations)
    
    Args:
        cfg: Configuration object containing robot, terrain, reward, and simulation parameters
        render_mode: Visualization mode ('human' for GUI, None for headless)
        **kwargs: Additional arguments passed to parent DirectRLEnv
    """

    cfg: HaroldIsaacLabEnvCfg

    def __init__(self, cfg: HaroldIsaacLabEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the Harold RL environment with configuration and state buffers.
        
        Sets up action/observation buffers, joint limits, contact sensors, terrain,
        and visualization markers. Initializes reward tracking and debugging outputs.
        
        Args:
            cfg: Environment configuration containing robot, terrain, and training parameters
            render_mode: Rendering mode - 'human' enables GUI visualization, None for headless
            **kwargs: Additional keyword arguments passed to DirectRLEnv parent class
        """

        super().__init__(cfg, render_mode, **kwargs)

        # --- Action Buffers & Initial Joint Targets ---
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._processed_actions = ( self.cfg.action_scale * self._actions ) + self._robot.data.default_joint_pos
        
        # --- Per-joint range scaling for natural action space ---
        # Shoulders get smaller range (0.25) for stability, thighs and calves get larger (0.5)
        self._joint_range = torch.tensor(
            [0.35, 0.35, 0.35, 0.35,  # Shoulders (FL, FR, BL, BR) - increased to aid lateral control
             0.5, 0.5, 0.5, 0.5,       # Thighs (FL, FR, BL, BR)
             0.5, 0.5, 0.5, 0.5],      # Calves (FL, FR, BL, BR)
            device=self.device
        )
        
        # --- Track previous target deltas for observation space ---
        self._prev_target_delta = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
        # Commands removed for push-up playback (no velocity tracking)
        self._commands = None

        # --- Joint Angle Limits Configuration ---
        # Safety limits for Harold's 12 joints to prevent mechanical damage
        # 
        # Joint Classification and Limits:
        # - Shoulders (0-3): ±20° (±0.3491 rad) - limited swing to prevent leg collision
        # - Thighs (4-7): ±45° (±0.7853 rad) - primary lift/extension joints
        # - Calves (8-11): ±45° (±0.7853 rad) - foot positioning and ground contact
        #
        # Joint Naming Convention (matches hardware and USD file):
        # Index | Name           | Type     | Limit   | Function
        # ------|----------------|----------|---------|---------------------------
        #   0   | FL_shoulder    | Shoulder | ±20°    | Front-Left leg swing
        #   1   | FR_shoulder    | Shoulder | ±20°    | Front-Right leg swing  
        #   2   | BL_shoulder    | Shoulder | ±20°    | Back-Left leg swing
        #   3   | BR_shoulder    | Shoulder | ±20°    | Back-Right leg swing
        #   4   | FL_thigh       | Thigh    | ±45°    | Front-Left leg lift
        #   5   | FR_thigh       | Thigh    | ±45°    | Front-Right leg lift
        #   6   | BL_thigh       | Thigh    | ±45°    | Back-Left leg lift
        #   7   | BR_thigh       | Thigh    | ±45°    | Back-Right leg lift
        #   8   | FL_calf        | Calf     | ±45°    | Front-Left foot position
        #   9   | FR_calf        | Calf     | ±45°    | Front-Right foot position
        #  10   | BL_calf        | Calf     | ±45°    | Back-Left foot position
        #  11   | BR_calf        | Calf     | ±45°    | Back-Right foot position
        #
        # Hardware Safety Notes:
        # - Limits prevent servo stall and mechanical binding
        # - Conservative ranges ensure sim-to-real transfer safety
        # - FeeTech STS3215 servos have built-in position limits
        # - These software limits provide additional protection layer
        # Push-up task: allow wider leg motion (shoulders ±30°, thighs/calves ±90°)
        self._JOINT_ANGLE_MAX = torch.tensor([
            0.5236, 0.5236, 0.5236, 0.5236,  # shoulders
            1.5708, 1.5708, 1.5708, 1.5708,  # thighs
            1.5708, 1.5708, 1.5708, 1.5708   # calves
        ], device=self.device)
        self._JOINT_ANGLE_MIN = -self._JOINT_ANGLE_MAX.clone()

        # --- Push-up playback parameters (match firmware timing at 20 Hz) ---
        self._pushup_settle_steps = 16       # ~0.8s at 20 Hz (0.8 * 20)
        self._pushup_steps_phase = 24        # 1.2s down or up phase at 20 Hz (1.2 * 20)
        self._pushup_pause_steps = 6         # ~0.3s pause at top/bottom at 20 Hz (0.3 * 20)
        self._pushup_reps = 5
        self._pushup_total_steps = self._pushup_settle_steps + self._pushup_reps * (
            self._pushup_steps_phase + self._pushup_pause_steps + self._pushup_steps_phase + self._pushup_pause_steps
        )
        self._pushup_step_counter = 0

        # --- Debug: Print Robot Body & Joint Info ---
        print("--------------------------------")
        print("Body Names: ", self._robot.data.body_names)
        print("Body Masses: ", self._robot.data.default_mass[0])
        print()

        print("--------------------------------")
        print("Joint Names: ", self._robot.data.joint_names)
        print("Joint Positions: ", torch.round(self._robot.data.joint_pos[0] * 100) / 100)
        print()

        # --- Contact Sensor Body ID Extraction (Isaac Lab Standard Pattern) ---
        # Get base body for termination detection
        self._base_id, _ = self._contact_sensor.find_bodies(".*body")
        
        # Get feet for gait analysis and air time rewards (Harold's feet are calf ends)
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*calf")
        
        # Get undesired contact bodies (body, thighs, shoulders should not touch ground)
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*(body|thigh|shoulder)")

        # --- Observation & State Trackers ---
        self._time = torch.zeros(self.num_envs, device=self.device)
        self._decimation_counter = 0
        self._last_terrain_level = 0  # Track current terrain difficulty level

        # --- Episode Reward Logging Buffers ---
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_xy_lin_commands",
                "track_yaw_commands",
                "height_reward",
                "torque_penalty",
                "feet_air_time_reward"
            ]
        }
        # --- Additional alignment diagnostics (episodic sums) ---
        self._metric_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "alignment_cos",      # cosine between cmd and actual velocity
                "v_par_cmd_ratio",    # along-track speed / commanded speed
                "lateral_speed"        # perpendicular speed magnitude
            ]
        }

    def _setup_scene(self) -> None:
        """Initialize and configure the complete simulation scene.
        
        Sets up the robot articulation, contact sensors, terrain generation, height scanning,
        lighting, and visualization markers. Configures environment cloning and collision
        filtering for multi-environment parallel training.
        
        Scene Components:
            - Robot: 12-DOF quadruped articulation with physics properties
            - Contact Sensor: Multi-body contact detection for feet, body, and limbs
            - Terrain: Procedurally generated or imported terrain meshes
            - Height Scanner: Ray-casting sensor for terrain height measurements
            - Lighting: Dome light for scene illumination
            - Markers: Velocity visualization arrows (GUI only)
        
        Note:
            This method is called once during environment initialization and handles
            the creation and configuration of all simulation assets.
        """

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

        # Disable visualization markers in push-up task
        self._markers_enabled = False
        self._cmd_marker = None
        self._act_marker = None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process and validate policy actions before physics simulation.
        
        Takes raw policy outputs and scales them around the default pose using per-joint
        ranges. This allows natural motion without fighting gravity. Actions are processed
        at policy frequency (20Hz) but applied at physics frequency (360Hz).
        
        Args:
            actions: Raw policy actions tensor [num_envs, 12] in normalized range [-1, 1]
            
        Processing Pipeline:
            1. Copy actions to internal buffer
            2. Scale by action_scale * per_joint_range around default pose
            3. Clamp to joint angle limits for hardware safety
            4. Store target deltas for next observation
            5. Store processed actions for physics application
            
        Safety Features:
            - Joint limits prevent mechanical damage: shoulders ±20°, others ±45°
            - Action scaling allows tuning of policy output sensitivity
            - Clamping ensures actions never exceed safe operating ranges
        """
        # Ignore incoming policy actions and drive a push-up trajectory instead
        # Playback runs at policy rate (decimation set to 9 => 20 Hz), matching firmware timing
        self._actions.zero_()

        # Determine current phase of the push-up routine
        step = int(self._pushup_step_counter)
        thigh_deg_start, calf_deg_start = -10.0, 20.0
        thigh_deg_end, calf_deg_end = -45.0, 90.0

        # Default to athletic stance
        thigh_deg, calf_deg = thigh_deg_start, calf_deg_start

        if step >= self._pushup_settle_steps:
            # Compute step within the repetition cycle
            local = (step - self._pushup_settle_steps) % (
                self._pushup_steps_phase + self._pushup_pause_steps + self._pushup_steps_phase + self._pushup_pause_steps
            )
            # Down phase [0, steps_phase)
            if local < self._pushup_steps_phase:
                t = float(local) / float(self._pushup_steps_phase)
                thigh_deg = thigh_deg_start + (thigh_deg_end - thigh_deg_start) * t
                calf_deg = calf_deg_start + (calf_deg_end - calf_deg_start) * t
            # Bottom pause [steps_phase, steps_phase+pause)
            elif local < self._pushup_steps_phase + self._pushup_pause_steps:
                thigh_deg = thigh_deg_end
                calf_deg = calf_deg_end
            # Up phase [steps_phase+pause, 2*steps_phase+pause)
            elif local < self._pushup_steps_phase + self._pushup_pause_steps + self._pushup_steps_phase:
                up_local = local - (self._pushup_steps_phase + self._pushup_pause_steps)
                t = float(up_local) / float(self._pushup_steps_phase)
                thigh_deg = thigh_deg_end + (thigh_deg_start - thigh_deg_end) * t
                calf_deg = calf_deg_end + (calf_deg_start - calf_deg_end) * t
            # Top pause
            else:
                thigh_deg = thigh_deg_start
                calf_deg = calf_deg_start

        # Advance counter until routine completion, then hold athletic stance
        if self._pushup_step_counter < self._pushup_total_steps:
            self._pushup_step_counter += 1

        # Convert degrees to radians
        deg2rad = math.pi / 180.0
        thigh = thigh_deg * deg2rad
        calf = calf_deg * deg2rad

        # Build target joint vector: shoulders unchanged (default), thighs/calves per playback
        target = self._robot.data.default_joint_pos.clone()
        # Apply per-leg direction multipliers to mimic firmware DIR[] pattern: [FL:+, FR:-, BL:+, BR:-]
        # Signs per leg order [FL, FR, BL, BR]
        # Flip LEFT thigh joints; keep RIGHT thighs unchanged
        thigh_signs = torch.tensor([-1.0, -1.0, -1.0, -1.0], device=self.device)
        # Flip RIGHT calf joints; keep LEFT calves unchanged
        calf_signs = torch.tensor([-1.0, -1.0, -1.0, -1.0], device=self.device)
        target[:, 4:8] = target[:, 4:8] + thigh * thigh_signs
        target[:, 8:12] = target[:, 8:12] + calf * calf_signs

        # Clamp to push-up joint limits
        self._processed_actions = torch.clamp(target, self._JOINT_ANGLE_MIN, self._JOINT_ANGLE_MAX)
        # Store delta for observations
        self._prev_target_delta = self._processed_actions - self._robot.data.default_joint_pos

    def _apply_action(self) -> None:
        
        # --- Send joint targets to robot ---
        self._robot.set_joint_position_target(self._processed_actions)
        
        # --- Decimation counter for logging/diagnostics ---
        self._decimation_counter += 1  # Increment counter

    def _get_observations(self) -> dict:

        # Update temporal state for time-based observations
        self._time += self.step_dt

        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,                                      # (3D) Body linear velocity  
                    self._robot.data.root_ang_vel_b,                                      # (3D) Body angular velocity
                    self._robot.data.projected_gravity_b,                                 # (3D) Gravity in body frame
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,     # (12D) Joint angles (relative)
                    self._robot.data.joint_vel,                                          # (12D) Joint velocities
                    # commands removed
                    self._prev_target_delta                                              # (12D) Previous target deltas
                )
                if tensor is not None
            ],
            dim=-1,  # Concatenate along feature dimension -> [batch_size, 45]
        )
        
        observations = {"policy": obs}

        # Update previous actions
        self._previous_actions.copy_(self._actions)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Push-up playback task returns zero rewards (no RL objective)."""
        zeros = torch.zeros(self.num_envs, device=self.device)
        # Update episode sums as zeros for consistent logging keys
        for key in self._episode_sums.keys():
            self._episode_sums[key] += zeros
        return zeros

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        # Only time-out termination retained for push-up playback
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset specified environments to initial state with terrain selection.
        
        Resets robot pose, joint states, command generation, and terrain assignment based
        Implements sophisticated terrain system where
        environments are assigned progressively harder terrain patches during training.
        
        Args:
            env_ids: Sequence of environment indices to reset. If None, resets all environments.
            
        Reset Operations:
        
        1. Robot State Reset:
           - Joint positions: Return to neutral pose
           - Joint velocities: Zero initial velocity
           - Root pose: Default position + terrain offset
           - Root velocity: Zero initial velocity
           
        2. Command Generation:
           - X/Y velocity: Uniform random [-0.3, 0.3] m/s
           - Yaw rate: Uniform random [-0.2, 0.2] rad/s
           
        3. Terrain Assignment:
           - Select terrain level randomly from available levels
           - Random terrain within allowed difficulty range
           - Update environment-specific terrain tracking
           - Set spawn position to selected terrain patch
           
        4. State Buffer Reset:
           - Action buffers: Zero previous actions
           - Velocity tracking: Reset for jitter calculation
           - Time buffers: Reset temporal observations
           - Reward tracking: Clear episode accumulations
           
        5. Logging and Analytics:
           - Episode reward summaries (normalized by episode length)
           - Termination reason categorization
           - Training progress tracking
           
        Terrain Grid Structure:
           - Level 0: Flat terrain, Level 9: Maximum difficulty
           - Random selection from all terrain levels
           
        Performance Optimizations:
           - Batch processing of multiple environment resets
           - Staggered episode lengths prevent synchronized resets
           - Efficient tensor operations for state updates
        """

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

        # Zero commands for stationary push-up playback
        # No commands in push-up task

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        
        # Fixed spawn position: do not offset by terrain origins; place at scene origin
        default_root_state[:, :3] = torch.tensor((0.0, 0.0, default_root_state[0, 2]), device=self.device)
            
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Domain randomization disabled for push-up playback

        # Reset time and playback counter
        self._time[env_ids] = 0
        self._pushup_step_counter = 0

        # Initialize to athletic stance target immediately
        deg2rad = math.pi / 180.0
        thigh = -10.0 * deg2rad
        calf = 20.0 * deg2rad
        target = self._robot.data.default_joint_pos.clone()
        # Apply per-leg signs at reset as well
        thigh_signs = torch.tensor([-1.0, -1.0, -1.0, -1.0], device=self.device)
        calf_signs = torch.tensor([-1.0, -1.0, -1.0, -1.0], device=self.device)
        target[:, 4:8] = target[:, 4:8] + thigh * thigh_signs
        target[:, 8:12] = target[:, 8:12] + calf * calf_signs
        self._processed_actions = torch.clamp(target, self._JOINT_ANGLE_MIN, self._JOINT_ANGLE_MAX)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            # Normalize by max episode length (seconds)
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        # Add alignment diagnostics (normalized similarly for consistency)
        for key in self._metric_sums.keys():
            episodic_sum_avg = torch.mean(self._metric_sums[key][env_ids])
            extras["Episode_Metric/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._metric_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        # Log termination reasons
        extras = dict()
        # Check termination types for environments that actually terminated (not timed out)
        if torch.any(self.reset_terminated[env_ids]):
            # Get contact forces and orientation for terminated envs
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            body_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 0.2, dim=1)
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
