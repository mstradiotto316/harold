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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
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
        - 48-dimensional observation space including robot state, commands, and terrain info
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
            [0.25, 0.25, 0.25, 0.25,  # Shoulders (FL, FR, BL, BR)
             0.5, 0.5, 0.5, 0.5,       # Thighs (FL, FR, BL, BR)
             0.5, 0.5, 0.5, 0.5],      # Calves (FL, FR, BL, BR)
            device=self.device
        )
        
        # --- Track previous target deltas for observation space ---
        self._prev_target_delta = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
        # --- Command Tensor for Velocity Tracking (X, Y, Yaw) ---
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

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
        
        # Track which terrain level each environment is on
        self._env_terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # ------------------------------------------------------------------
        # Pre-allocate constant tensors to avoid per-step allocations and
        # potential memory leaks.
        # ------------------------------------------------------------------
        # Index 0 selects the single arrow prototype for every environment.
        self._marker_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # Constant width of the arrow (y and z scale components).
        self._arrow_width = torch.full((self.num_envs,), 1.0, device=self.device)
        
        # Pre-allocate tensors for marker computations to avoid repeated allocations (only if GUI enabled)
        if self.sim.has_gui():
            self._cmd_scale_buffer = torch.zeros((self.num_envs, 3), device=self.device)
            self._act_scale_buffer = torch.zeros((self.num_envs, 3), device=self.device)
        else:
            self._cmd_scale_buffer = None
            self._act_scale_buffer = None

        # ------------------------------------------------------------------
        # Constant vectors used each physics step. Creating them here avoids
        # implicit CPU → GPU copies that occur when ``torch.tensor`` is
        # called inside the hot loop.
        # ------------------------------------------------------------------
        self._up_vec = torch.tensor((0.0, 0.0, 1.0), device=self.device)
        self._arrow_offset_cmd = torch.tensor((0.0, 0.0, 0.25), device=self.device)
        self._arrow_offset_act = torch.tensor((0.0, 0.0, 0.40), device=self.device)

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
        
        # --- Domain Randomization Buffers ---
        # Store randomized parameters per environment for consistency within episodes
        # Initialize with default values from configuration
        default_friction = self.cfg.sim.physics_material.static_friction
        default_stiffness = self.cfg.robot.actuators["all_joints"].stiffness
        default_damping = self.cfg.robot.actuators["all_joints"].damping
        
        self._randomized_friction = torch.ones(self.num_envs, device=self.device) * default_friction
        self._randomized_mass_scale = torch.ones(self.num_envs, device=self.device)
        self._randomized_stiffness = torch.ones(self.num_envs, 12, device=self.device) * default_stiffness
        self._randomized_damping = torch.ones(self.num_envs, 12, device=self.device) * default_damping
        
        # Action delay buffer for simulating control latency
        if cfg.domain_randomization.add_action_delay:
            max_delay = cfg.domain_randomization.action_delay_steps[1]
            self._action_delay_buffer = torch.zeros(
                self.num_envs, max_delay + 1, self.cfg.action_space, device=self.device
            )
            self._action_delays = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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

        # --- Visualization Markers Setup ---
        # Only create markers when GUI is enabled to save memory in headless training
        self._markers_enabled = self.sim.has_gui()
        if self._markers_enabled:
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

            # Create arrow marker instances
            self._cmd_marker = VisualizationMarkers(command_arrow_cfg)
            self._act_marker = VisualizationMarkers(actual_arrow_cfg)
        else:
            # Set markers to None for headless training
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
        # --- Action copy ---
        self._actions.copy_(actions)
        
        # --- Apply action noise and delays if domain randomization is enabled ---
        if self.cfg.domain_randomization.enable_randomization:
            actions_to_use = self._add_action_noise(self._actions)
        else:
            actions_to_use = self._actions

        # --- Action scaling around default pose with per-joint ranges ---
        # Scale each joint by a safe fraction of its mechanical range
        # This allows the policy to work around the default pose instead of fighting gravity
        target = self._robot.data.default_joint_pos + self.cfg.action_scale * self._joint_range * actions_to_use
        self._processed_actions = torch.clamp(
            target,
            self._JOINT_ANGLE_MIN,
            self._JOINT_ANGLE_MAX,
        )
        
        # --- Store target delta for next observation ---
        self._prev_target_delta = self._processed_actions - self._robot.data.default_joint_pos

    def _apply_action(self) -> None:
        """Apply processed joint targets to robot and update visualization.
        
        Sends the processed joint position targets to the robot's actuators and
        handles optional data logging and visualization marker updates.
        
        Operations:
            1. Send joint position targets to robot actuators
            2. Increment decimation counter for timing control
            3. Update visualization markers (every 18 physics steps)
            4. Optional action logging for replay/analysis (commented)
            
        Performance Optimizations:
            - Markers only update at policy frequency (18x less often)
            - Headless training skips all visualization computations
            - Pre-allocated buffers prevent memory allocations in hot loop
        """
        
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
        
        # --- Apply external forces if domain randomization is enabled ---
        if self.cfg.domain_randomization.enable_randomization:
            self._apply_external_forces()

        # --- Decimation counter for logging/diagnostics ---
        self._decimation_counter += 1  # Increment counter

        # ----- UPDATE ARROW MARKERS (OPTIMIZED) -----
        # Only update markers at policy frequency (every 18 physics steps) and when markers are enabled
        # This reduces computational overhead by 18x compared to updating every physics step
        if self._decimation_counter % 18 == 0 and self._markers_enabled:
            self._update_visualization_markers()

    def _update_visualization_markers(self) -> None:
        """Update 3D arrow markers showing commanded vs actual velocities.
        
        Creates visual arrows above each robot showing:
        - Red arrows: Commanded velocity direction and magnitude
        - Green arrows: Actual velocity direction and magnitude
        
        This provides real-time visual feedback on tracking performance during training.
        Only called when GUI is enabled and at policy frequency (20Hz) for performance.
        
        Marker Properties:
            - Position: Above robot base by fixed offsets
            - Orientation: Aligned with velocity vector direction
            - Scale: Proportional to velocity magnitude with minimum visibility
            - Colors: Red for commands, green for actual velocities
            
        Performance Features:
            - Uses pre-allocated tensor buffers to avoid allocations
            - Early return for headless training
            - Batch processing for all environments simultaneously
        """
        # Early return if markers are disabled (headless training)
        if not self._markers_enabled:
            return
            
        base_pos = self._height_scanner.data.pos_w  # [num_envs, 3]
        marker_idx = self._marker_idx

        # Commanded arrow: orientation from commanded X/Y velocity
        cmd_vel = self._commands[:, :2]
        cmd_angle = torch.atan2(cmd_vel[:, 1], cmd_vel[:, 0])
        cmd_magnitude = torch.norm(cmd_vel, dim=1)
        marker_ori_cmd = quat_from_angle_axis(cmd_angle, self._up_vec)
        marker_pos_cmd = base_pos + self._arrow_offset_cmd
        
        # Use pre-allocated buffer with minimum scale to ensure visibility when commands exist
        # If no command exists (magnitude near 0), arrow will be tiny but visible for debugging
        self._cmd_scale_buffer[:, 0] = torch.clamp(cmd_magnitude * 2.0, min=0.02)  # Minimum length 0.02
        self._cmd_scale_buffer[:, 1] = self._arrow_width
        self._cmd_scale_buffer[:, 2] = self._arrow_width
        self._cmd_marker.visualize(marker_pos_cmd, marker_ori_cmd, marker_indices=marker_idx, scales=self._cmd_scale_buffer)
        
        # Actual arrow: orientation from actual X/Y root velocity
        act_vel = self._robot.data.root_lin_vel_b[:, :2]
        act_angle = torch.atan2(act_vel[:, 1], act_vel[:, 0])
        act_magnitude = torch.norm(act_vel, dim=1)
        marker_ori_act = quat_from_angle_axis(act_angle, self._up_vec)
        marker_pos_act = base_pos + self._arrow_offset_act
        
        # Use pre-allocated buffer to avoid tensor allocations
        self._act_scale_buffer[:, 0] = act_magnitude * 5.0
        self._act_scale_buffer[:, 1] = self._arrow_width
        self._act_scale_buffer[:, 2] = self._arrow_width
        self._act_marker.visualize(marker_pos_act, marker_ori_act, marker_indices=marker_idx, scales=self._act_scale_buffer)

    def _get_contact_sensor_data(self) -> dict:
        """Extract and process contact sensor data for rewards and termination logic.
        
        Processes contact forces from all robot bodies and computes derived metrics
        for gait analysis, termination detection, and reward calculation. Uses force
        history to ensure reliable contact detection despite simulation noise.
        
        Returns:
            Dict containing processed contact data:
                - base_contact_forces: Forces on main body (for termination)
                - undesired_contact_forces: Forces on body/thighs/shoulders (penalties)
                - feet_contact_forces: Forces on feet (for gait analysis)
                - first_contact: Boolean mask of feet making first contact this step
                - last_air_time: Duration each foot was airborne before contact
                - net_forces_w_history: Raw force history for advanced analysis
                
        Body Groups:
            - Base: Main robot body (contact triggers termination)
            - Feet: Calf endpoints (desired contact points for locomotion)
            - Undesired: Body, thighs, shoulders (contact penalized)
            
        Force Processing:
            - Uses maximum over force history window for noise robustness
            - Computes L2 norm of 3D contact forces for scalar metrics
            - Tracks air time for stepping pattern analysis
        """
        # Get contact forces with history
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        
        # Base contact detection (for termination)
        base_contact_forces = torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0]
        
        # Undesired contact detection (for rewards)
        undesired_contact_forces = torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0]
        
        # Feet contact and air time data (for gait rewards)
        feet_contact_forces = torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0]
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        
        return {
            "base_contact_forces": base_contact_forces,
            "undesired_contact_forces": undesired_contact_forces,
            "feet_contact_forces": feet_contact_forces,
            "first_contact": first_contact,
            "last_air_time": last_air_time,
            "net_forces_w_history": net_contact_forces
        }

    def _get_observations(self) -> dict:
        """Construct the 48-dimensional observation vector for policy input.
        
        Assembles robot state, sensor data, and command information into a structured
        observation that enables the policy to perform locomotion control. Updates
        terrain difficulty for each environment.
        
        Returns:
            Dict with 'policy' key containing 48D observation tensor:
                - root_lin_vel_b (3): Linear velocity in body frame [m/s]
                - root_ang_vel_b (3): Angular velocity in body frame [rad/s] 
                - projected_gravity_b (3): Gravity vector in body frame (for orientation)
                - joint_pos - default (12): Joint angles relative to neutral pose [rad]
                - joint_vel (12): Joint angular velocities [rad/s]
                - commands (3): Velocity commands [vx, vy, yaw_rate]
                - prev_target_delta (12): Previous joint targets relative to neutral pose [rad]
                
        State Tracking:
            - Increments policy step counter for tracking
            - Updates time buffer for temporal observations
            - Uses previous joint target deltas for temporal consistency
            
        Note:
            Observation design ensures policy receives sufficient information for
            stable quadruped locomotion while maintaining manageable dimensionality.
            Previous target deltas correctly represent what the policy commanded,
            not raw actions, providing proper temporal feedback.
        """

        
        # Update temporal state for time-based observations
        self._time += self.step_dt

        # ================== OBSERVATION SPACE CONSTRUCTION (48-D) ==================
        # Carefully designed observation vector providing policy with sufficient
        # information for stable locomotion while maintaining manageable dimensionality
        # 
        # Observation Components (Total: 48 dimensions):
        # 
        # 1. Root Linear Velocity (3D) - [vx, vy, vz] in body frame [m/s]
        #    - Primary feedback for velocity tracking objectives
        #    - Body frame ensures rotation-invariant representation
        #    - Critical for track_xy_lin_commands reward component
        # 
        # 2. Root Angular Velocity (3D) - [wx, wy, wz] in body frame [rad/s] 
        #    - Essential for yaw rate tracking and stability control
        #    - Roll/pitch rates indicate balance state
        #    - Used in track_yaw_commands reward component
        #
        # 3. Projected Gravity (3D) - gravity vector in body frame
        #    - Provides orientation information without explicit angles
        #    - More robust than Euler angles (no singularities)
        #    - z-component used for orientation termination detection
        #    - Enables upright balance and fall detection
        #
        # 4. Joint Positions Relative to Default (12D) - [q_i - q_default] [rad]
        #    - Joint angles relative to neutral standing pose
        #    - Normalized representation reduces learning complexity
        #    - Provides leg configuration information for gait control
        #    - Order: [shoulders(4), thighs(4), calves(4)]
        #
        # 5. Joint Velocities (12D) - [dq_i/dt] [rad/s]
        #    - Joint angular velocities for dynamic control
        #    - Essential for smooth motion and momentum management
        #    - Used in torque penalty calculations
        #    - Enables predictive control and oscillation damping
        #
        # 6. Velocity Commands (3D) - [vx_cmd, vy_cmd, yaw_rate_cmd]
        #    - Target velocities for tracking objectives
        #    - Provides policy with explicit task specification
        #    - Directly used in reward function calculations
        #
        # 7. Previous Target Deltas (12D) - [target_prev - q_default] [rad]
        #    - Memory of previous joint position targets relative to default
        #    - Correctly represents what the policy commanded (not raw actions)
        #    - Enables temporal consistency and smooth control
        #    - Helps prevent action oscillations and instability
        #    - Provides recurrence without explicit memory networks
        #
        # Design Principles:
        # - All observations in consistent units and coordinate frames
        # - Relative representations (joint_pos - default) improve learning
        # - Body frame velocities ensure rotation invariance
        # - No proprioceptive terrain information (generalizes to unknown terrain)
        # - Sufficient information for closed-loop control without redundancy
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,                                      # (3D) Body linear velocity  
                    self._robot.data.root_ang_vel_b,                                      # (3D) Body angular velocity
                    self._robot.data.projected_gravity_b,                                 # (3D) Gravity in body frame
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,     # (12D) Joint angles (relative)
                    self._robot.data.joint_vel,                                          # (12D) Joint velocities
                    self._commands,                                                      # (3D) Velocity commands 
                    self._prev_target_delta                                              # (12D) Previous target deltas
                )
                if tensor is not None
            ],
            dim=-1,  # Concatenate along feature dimension -> [batch_size, 48]
        )

        # Apply observation noise if domain randomization is enabled
        if self.cfg.domain_randomization.enable_randomization:
            obs = self._add_observation_noise(obs)
        
        observations = {"policy": obs}

        # Update previous actions
        self._previous_actions.copy_(self._actions)


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
        """Compute multi-component reward signal for reinforcement learning.
        
        Implements a sophisticated reward function combining velocity tracking,
        height maintenance, energy efficiency, and gait quality metrics. Each
        component is carefully tuned for stable quadruped locomotion training.
        
        Returns:
            Total reward tensor [num_envs] combining all reward components
            
        Reward Components:
        
        1. Linear Velocity Tracking (Directional):
           - Decomposes velocity into parallel and perpendicular components
           - Elliptical Gaussian: exp(-(e_par/0.25)² + (e_perp/0.08)²)
           - Lateral drift penalized 3x more strictly than along-track error
           - Heavily penalizes sideways movement and backwards motion
           
        2. Yaw Velocity Tracking:
           - Exponential reward for matching commanded yaw rate
           - Gaussian penalty: exp(-(error/0.4)²)
           - Prevents stationary spinning behaviors
           
        3. Height Maintenance:
           - Maintains target height above terrain using ray-casting
           - Tanh-based reward: tanh(3*exp(-5*|height_error|))
           - NaN-safe terrain height computation
           - Critical for stable locomotion
           
        4. Torque Penalty:
           - Quadratic penalty on joint torques: sum(torque²)
           - Encourages energy-efficient movements
           - Prevents aggressive actuator usage
           
        5. Feet Air Time Reward:
           - Rewards proper stepping patterns (0.4s optimal air time)
           - Uses exponential reward curve to encourage stepping
           - Only active when robot speed > 0.05 m/s
           - Reduces sliding and shuffling behaviors
           
        Mathematical Formulation:
           Total = Σ(component_value * weight)
           Note: Rewards are not multiplied by step_dt to avoid double normalization
           with the episodic logging that divides by max_episode_length_s
           
        """
        
        # ==================== LINEAR VELOCITY TRACKING ====================
        # DIRECTIONAL: Decompose velocity into parallel and perpendicular components
        # relative to command direction. Penalize sideways drift much more than along-track error.
        
        # Get velocity and command vectors (N,2) in body frame
        v = self._robot.data.root_lin_vel_b[:, :2]
        cmd = self._commands[:, :2]
        
        # Compute unit command direction
        eps = 1e-6
        cmd_mag = torch.linalg.vector_norm(cmd, dim=1)                    # (N,)
        u = torch.where(cmd_mag[:, None] > eps, cmd / cmd_mag[:, None], torch.zeros_like(cmd))  # unit cmd dir
        
        # Decompose velocity into parallel and perpendicular components
        v_par = (v * u).sum(dim=1)                                       # scalar component along u
        v_perp = v - v_par[:, None] * u                                  # vector perpendicular to u
        e_par = v_par - cmd_mag                                          # signed along-track error
        e_perp = torch.linalg.vector_norm(v_perp, dim=1)                 # sideways speed magnitude
        
        # Elliptical Gaussian: much tighter on lateral drift
        c_par, c_perp = 0.25, 0.08  # lateral 3x stricter
        Q = (e_par / c_par)**2 + (e_perp / c_perp)**2
        lin_vel_reward = torch.exp(-Q)
        
        # Gate on actual robot speed to prevent reward exploitation when standing still
        actual_speed = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        lin_vel_reward *= (actual_speed > 0.05)

        # ==================== YAW VELOCITY TRACKING ====================
        # AGGRESSIVE: Much more punishment for sitting still (0.05 vs 0.25 normalization)
        #yaw_vel_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        #yaw_vel_reward = torch.exp(-yaw_vel_error / 0.05)
        # yaw
        err_yaw = torch.abs(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_vel_reward = torch.exp(-(err_yaw / 0.4)**2)   # sigma ≈ 0.4 rad/s

        # ==================== HEIGHT MAINTENANCE ====================
        # Get height data from scanner and compute mean height (NaN-safe)
        pos_z = self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
        ray_z = self._height_scanner.data.ray_hits_w[..., 2]
        # Replace missing hits (NaNs or Infs) with base position to yield zero diff.
        ray_z = torch.where(torch.isfinite(ray_z), ray_z, pos_z)
        height_data = pos_z - ray_z
        current_height = torch.mean(height_data, dim=1)
        
        # Calculate height error and convert to reward
        target_height = self.cfg.gait.target_height
        height_error = torch.abs(current_height - target_height)
        height_reward = torch.tanh(3.0 * torch.exp(-5.0 * height_error))

        # ==================== TORQUE PENALTY ====================
        # Penalize large actuator efforts to encourage energy-efficient motions.
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        # ==================== FEET AIR TIME REWARD ====================
        # Reward proper stepping patterns to reduce sliding and shuffling (Fixed implementation)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # Reward feet that achieve optimal air time (0.15 seconds for 40cm robot) when they land
        # Uses exponential reward curve to encourage proper stepping patterns instead of penalizing them
        optimal_air_time = 0.4 #0.25 #0.5 #0.15  # Appropriate for Harold's 40cm scale (was 0.3s - too long)
        air_time_error = torch.abs(last_air_time - optimal_air_time)
        # Gate the reward on actual robot speed (already calculated above for velocity tracking)
        air_time_reward = torch.sum(torch.exp(-air_time_error * 10.0) * first_contact, dim=1) * ( #3.0 #10.0
            actual_speed > 0.05  # was: torch.norm(self._commands[:, :2], dim=1) > 0.03
        )

        # ==================== REWARD ASSEMBLY ====================
        # Note: Rewards are NOT multiplied by step_dt here to avoid double normalization.
        # The episodic sum is already normalized by max_episode_length_s when logged.
        rewards = {
            "track_xy_lin_commands": lin_vel_reward * self.cfg.rewards.track_xy_lin_commands,
            "track_yaw_commands": yaw_vel_reward * self.cfg.rewards.track_yaw_commands,
            "height_reward": height_reward * self.cfg.rewards.height_reward,
            "torque_penalty": joint_torques * self.cfg.rewards.torque_penalty,
            "feet_air_time_reward": air_time_reward * self.cfg.rewards.feet_air_time
        }
        
        # Sum all rewards
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Update episode sums for logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination and timeout conditions.
        
        Evaluates various failure modes and episode length limits to decide when
        to reset environments. Uses contact forces and robot orientation to detect
        falls, collisions, and other undesirable states.
        
        Returns:
            Tuple of (terminated, timed_out) boolean tensors:
                - terminated: Environments that failed and need immediate reset
                - timed_out: Environments that reached maximum episode length
                
        Termination Conditions:
        
        1. Undesired Contact (Immediate Reset):
           - Body, shoulders, or thighs touching ground
           - Contact force threshold: 0.05N (scaled for 2kg robot)
           - Prevents damage and encourages proper gait
           - Uses force history for reliable detection
           
        2. Orientation Failure (Currently Disabled):
           - Robot tipped over (gravity_z > threshold)
           - Threshold: -0.5 (allows some tilt tolerance)
           - Commented out pending threshold calibration
           
        3. Episode Timeout:
           - Maximum episode length reached (time-based cutoff)
           - Prevents infinite episodes during exploration
           - Separate from failure-based termination
           
        Contact Force Analysis:
           - Uses net_forces_w_history for noise robustness
           - Computes L2 norm of 3D contact forces
           - Per-body contact detection with body-specific IDs
           - Thresholds tuned for 2kg Harold robot mass
           
        Safety Features:
           - Conservative contact thresholds prevent hardware damage
           - Force history smoothing reduces false positives
           - Immediate reset on any undesired body contact
        """

        # Terminate if episode length exceeds max episode length
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # ==================== CONTACT CHECK (Harold 2kg Robot - Scaled Thresholds) ====================
        # Get contact forces with history for reliable detection
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        
        # Undesired contact termination (body, shoulders, thighs touching ground) - IMMEDIATE RESET
        # Very sensitive threshold for 2kg robot - any significant contact = reset
        undesired_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 0.05, 
            dim=1
        )
        
        contact_terminated = undesired_contact  # Any undesired contact = immediate reset

        # ==================== ORIENTATION CHECK ====================
        # Check if robot has fallen (z component of gravity in body frame should be close to -1 when upright)
        orientation_terminated = self._robot.data.projected_gravity_b[:, 2] > self.cfg.termination.orientation_threshold

        # Combine all failure conditions
        terminated = contact_terminated #| orientation_terminated

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

        # Sample velocity commands for reset environments
        sampled_commands = torch.zeros_like(self._commands[env_ids])
        # X and Y velocity commands: -0.3 to 0.3 m/s (appropriate for 40cm robot = 0.75 body lengths/sec)
        sampled_commands[:, :2].uniform_(-0.3, 0.3)
        # Yaw rate commands: -0.2 to 0.2 rad/s (increased for better turning agility on small robot)
        # TEMPORARILY SET TO 0 FOR TESTING
        sampled_commands[:, 2].uniform_(0.0, 0.0)
        # Use full command range
        self._commands[env_ids] = sampled_commands

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        
        # Use terrain origins if available, otherwise use scene origins
        if hasattr(self._terrain, 'env_origins'):
            # Implement terrain selection
            if self.cfg.terrain.terrain_type == "generator" and hasattr(self._terrain, 'terrain_origins'):
                # Get the terrain grid dimensions
                num_rows = self.cfg.terrain.terrain_generator.num_rows  # 10 difficulty levels
                num_cols = self.cfg.terrain.terrain_generator.num_cols  # 20 variations per level
                
                # Debug: Print terrain information once
                #if env_ids[0] == 0:  # Only print for first environment
                #
                #    print(f"[Terrain Debug] terrain_origins shape: {self._terrain.terrain_origins.shape}")
                #    print(f"[Terrain Debug] env_origins shape: {self._terrain.env_origins.shape}")
                #    print(f"[Terrain Debug] Total terrains: {len(self._terrain.terrain_origins)}")
                
                # For each resetting environment, assign a terrain level randomly
                for i, env_id in enumerate(env_ids):
                    # Convert to integer for indexing
                    env_idx = int(env_id)
                    
                    # Randomly select a terrain level from all available levels
                    #num_rows = self.cfg.terrain.terrain_generator.num_rows  # 10 difficulty levels
                    terrain_level = torch.randint(0, self.cfg.terrain.max_init_terrain_level, (1,), device=self.device).item()
                    # Randomly select a column within that terrain level
                    terrain_col = torch.randint(0, num_cols, (1,), device=self.device).item()
                    
                    # Update environment's terrain level tracking
                    self._env_terrain_levels[env_idx] = terrain_level
                    
                    # Set the environment origin to the selected terrain patch
                    if hasattr(self._terrain, 'terrain_origins'):
                        # terrain_origins has shape [num_rows, num_cols, 3]
                        # Index it properly with [row, col] to get the 3D position
                        self._terrain.env_origins[env_idx, :] = self._terrain.terrain_origins[terrain_level, terrain_col, :]
            
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        else:
            default_root_state[:, :3] += self.scene.env_origins[env_ids]
            
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Apply domain randomization on reset
        if self.cfg.domain_randomization.enable_randomization and self.cfg.domain_randomization.randomize_on_reset:
            self._randomize_robot_properties(env_ids)
            self._randomize_physics_materials(env_ids)
            
            # Randomize action delays if enabled
            if self.cfg.domain_randomization.add_action_delay:
                delay_min, delay_max = self.cfg.domain_randomization.action_delay_steps
                self._action_delays[env_ids] = torch.randint(
                    delay_min, delay_max + 1, (len(env_ids),), device=self.device
                )

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

    # ==========================================
    # Domain Randomization Methods
    # ==========================================
    
    def _randomize_robot_properties(self, env_ids: torch.Tensor) -> None:
        """Randomize robot physical properties for specified environments.
        
        Applies randomization to mass, inertia, joint properties, and actuator
        characteristics. Called during environment reset for sim-to-real transfer.
        
        Args:
            env_ids: Indices of environments to randomize
        """
        if not self.cfg.domain_randomization.enable_randomization:
            return
            
        num_envs_to_randomize = len(env_ids)
        
        # Randomize joint stiffness
        if self.cfg.domain_randomization.randomize_joint_stiffness:
            stiffness_min, stiffness_max = self.cfg.domain_randomization.stiffness_range
            self._randomized_stiffness[env_ids] = sample_uniform(
                stiffness_min, stiffness_max, (num_envs_to_randomize, 12), self.device
            )
            # Note: In Direct workflow, actuator properties are typically set at initialization
            # Dynamic modification would require accessing the underlying PhysX articulation
            # For now, store the values for potential use in custom PD control
        
        # Randomize joint damping
        if self.cfg.domain_randomization.randomize_joint_damping:
            damping_min, damping_max = self.cfg.domain_randomization.damping_range
            self._randomized_damping[env_ids] = sample_uniform(
                damping_min, damping_max, (num_envs_to_randomize, 12), self.device
            )
            # Note: In Direct workflow, actuator properties are typically set at initialization
            # Dynamic modification would require accessing the underlying PhysX articulation
            # For now, store the values for potential use in custom PD control
        
        # Randomize mass (scale all link masses proportionally)
        if self.cfg.domain_randomization.randomize_mass:
            mass_min, mass_max = self.cfg.domain_randomization.mass_range
            self._randomized_mass_scale[env_ids] = sample_uniform(
                mass_min, mass_max, (num_envs_to_randomize,), self.device
            )
            # Note: Mass randomization requires modifying body properties
            # This is more complex in Direct workflow and may require USD modifications
    
    def _randomize_physics_materials(self, env_ids: torch.Tensor) -> None:
        """Randomize physics material properties for specified environments.
        
        Modifies friction and restitution coefficients for ground contact.
        
        Args:
            env_ids: Indices of environments to randomize
        """
        if not self.cfg.domain_randomization.enable_randomization:
            return
            
        num_envs_to_randomize = len(env_ids)
        
        # Randomize friction
        if self.cfg.domain_randomization.randomize_friction:
            friction_min, friction_max = self.cfg.domain_randomization.friction_range
            self._randomized_friction[env_ids] = sample_uniform(
                friction_min, friction_max, (num_envs_to_randomize,), self.device
            )
            # Note: In Direct workflow, material properties are typically set at scene creation
            # Dynamic modification requires accessing PhysX APIs directly
    
    def _add_observation_noise(self, observations: torch.Tensor) -> torch.Tensor:
        """Add noise to observations to simulate sensor imperfections.
        
        Applies Gaussian noise to different observation components based on
        configuration settings. Noise models realistic sensor characteristics.
        
        Args:
            observations: Clean observation tensor [num_envs, obs_dim]
            
        Returns:
            Noisy observation tensor with same shape
        """
        if not self.cfg.domain_randomization.enable_randomization:
            return observations
        if not self.cfg.domain_randomization.randomize_per_step:
            return observations
            
        noisy_obs = observations.clone()
        
        # Add IMU noise (angular velocity: indices 3-6, gravity: indices 6-9)
        if self.cfg.domain_randomization.add_imu_noise:
            # Angular velocity noise
            noisy_obs[:, 3:6] = gaussian_noise(
                observations[:, 3:6],
                self.cfg.domain_randomization.imu_angular_velocity_noise
            )
            # Gravity projection noise
            noisy_obs[:, 6:9] = gaussian_noise(
                observations[:, 6:9],
                self.cfg.domain_randomization.imu_gravity_noise
            )
        
        # Add joint noise (positions: indices 9-21, velocities: indices 21-33)
        if self.cfg.domain_randomization.add_joint_noise:
            # Joint position noise
            noisy_obs[:, 9:21] = gaussian_noise(
                observations[:, 9:21],
                self.cfg.domain_randomization.joint_position_noise
            )
            # Joint velocity noise
            noisy_obs[:, 21:33] = gaussian_noise(
                observations[:, 21:33],
                self.cfg.domain_randomization.joint_velocity_noise
            )
        
        return noisy_obs
    
    def _add_action_noise(self, actions: torch.Tensor) -> torch.Tensor:
        """Add noise and delays to actions to simulate control imperfections.
        
        Applies Gaussian noise and optional time delays to action commands.
        
        Args:
            actions: Clean action tensor [num_envs, action_dim]
            
        Returns:
            Noisy/delayed action tensor with same shape
        """
        if not self.cfg.domain_randomization.enable_randomization:
            return actions
            
        noisy_actions = actions.clone()
        
        # Add action noise
        if self.cfg.domain_randomization.add_action_noise:
            noisy_actions = gaussian_noise(
                noisy_actions,
                self.cfg.domain_randomization.action_noise
            )
        
        # Apply action delays (if enabled)
        if self.cfg.domain_randomization.add_action_delay and hasattr(self, '_action_delay_buffer'):
            # Shift buffer and insert new actions
            self._action_delay_buffer[:, 1:] = self._action_delay_buffer[:, :-1].clone()
            self._action_delay_buffer[:, 0] = noisy_actions
            
            # Select delayed actions based on per-env delays
            delayed_actions = torch.zeros_like(noisy_actions)
            for i in range(self.num_envs):
                delay = self._action_delays[i]
                delayed_actions[i] = self._action_delay_buffer[i, delay]
            
            return delayed_actions
        
        return noisy_actions
    
    def _apply_external_forces(self) -> None:
        """Apply random external forces/torques to robot bodies.
        
        Simulates environmental disturbances like wind or collisions.
        Called during physics step with configured probability.
        """
        if not self.cfg.domain_randomization.enable_randomization:
            return
        if not self.cfg.domain_randomization.apply_external_forces:
            return
            
        # Sample which environments get forces this step
        force_probs = torch.rand(self.num_envs, device=self.device)
        apply_force = force_probs < self.cfg.domain_randomization.external_force_probability
        
        if apply_force.any():
            # Sample random forces and torques
            force_min, force_max = self.cfg.domain_randomization.external_force_range
            torque_min, torque_max = self.cfg.domain_randomization.external_torque_range
            
            forces = torch.zeros(self.num_envs, 3, device=self.device)
            torques = torch.zeros(self.num_envs, 3, device=self.device)
            
            # Generate random forces for selected environments
            num_forced = apply_force.sum()
            forces[apply_force, :2] = sample_uniform(
                -force_max, force_max, (num_forced, 2), self.device
            )
            forces[apply_force, 2] = sample_uniform(
                force_min, force_max, (num_forced,), self.device
            )
            
            torques[apply_force] = sample_uniform(
                -torque_max, torque_max, (num_forced, 3), self.device
            )
            
            # Apply forces to robot base
            self._robot.set_external_force_and_torque(
                forces, torques, body_ids=[self._base_id[0]]
            )

    def __del__(self):
        pass