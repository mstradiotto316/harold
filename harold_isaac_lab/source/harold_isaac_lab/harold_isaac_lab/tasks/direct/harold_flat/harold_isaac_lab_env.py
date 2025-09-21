from typing import Sequence
import sys
import os
import json
from pathlib import Path
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
        
        # --- Per-joint range scaling for natural action space (from config) ---
        self._joint_range = torch.tensor(self.cfg.joint_range, device=self.device)
        
        # --- Track previous target deltas for observation space ---
        self._prev_target_delta = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        
        # --- Command Tensor for Velocity Tracking (X, Y, Yaw) ---
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # --- Joint Angle Limits ---
        self._JOINT_ANGLE_MAX = torch.tensor(self.cfg.joint_angle_max, device=self.device)
        self._JOINT_ANGLE_MIN = torch.tensor(self.cfg.joint_angle_min, device=self.device)

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
        # Warmup per-env counters (hold default pose for a few policy steps after reset)
        self._warmup_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # No undesired-contact timers in simplified flat task

        # --- Optional policy logging for sim-to-real dataset capture ---
        self._policy_log_dir: Path | None = None
        self._policy_log_step = 0
        log_dir_env = os.getenv("HAROLD_POLICY_LOG_DIR")
        if log_dir_env:
            log_dir = Path(log_dir_env).expanduser().resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            self._policy_log_dir = log_dir
            self._policy_log_file = self._policy_log_dir / "policy_steps.jsonl"
            if self._policy_log_file.exists():
                self._policy_log_file.unlink()

            actuator_cfg = self.cfg.robot.actuators.get("all_joints")
            control_dt = getattr(self.cfg.sim, "dt", None)
            if control_dt is None:
                control_dt = self.step_dt
            metadata = {
                "decimation": int(self.cfg.decimation),
                "control_dt": float(control_dt),
                "action_scale": float(self.cfg.action_scale),
                "joint_range": [float(x) for x in self.cfg.joint_range],
                "joint_angle_min": [float(x) for x in self.cfg.joint_angle_min],
                "joint_angle_max": [float(x) for x in self.cfg.joint_angle_max],
                "action_filter_beta": float(getattr(self.cfg, "action_filter_beta", 0.0)),
            }
            if actuator_cfg is not None:
                metadata["actuator"] = {
                    "type": actuator_cfg.__class__.__name__,
                    "stiffness": float(actuator_cfg.stiffness),
                    "damping": float(actuator_cfg.damping),
                    "effort_limit_sim": float(actuator_cfg.effort_limit_sim),
                }

            with open(self._policy_log_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
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
                "track_forward_speed",
                "height_reward",
                "torque_penalty",
                "feet_air_time_reward",
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

        # --- Low-pass filter (EMA) on actions for stability and sim2real ---
        # a_t_smooth = (1 - beta) * a_{t-1}_smooth + beta * a_t_raw
        # beta in (0,1]; lower beta = stronger smoothing
        if not hasattr(self, "_actions_smooth"):
            self._actions_smooth = torch.zeros_like(self._actions)
        beta = getattr(self.cfg, "action_filter_beta", 0.2)
        self._actions_smooth = (1.0 - beta) * self._actions_smooth + beta * self._actions
        
        # --- Apply action noise and delays if domain randomization is enabled ---
        if self.cfg.domain_randomization.enable_randomization:
            actions_to_use = self._add_action_noise(self._actions_smooth)
        else:
            actions_to_use = self._actions_smooth

        # --- Warmup gating: zero actions while warmup steps remain (keeps default pose)
        warmup_mask = self._warmup_steps_remaining > 0
        if torch.any(warmup_mask):
            actions_to_use = actions_to_use.clone()
            actions_to_use[warmup_mask] = 0.0
            # Decrement per-env warmup counters (only at policy rate)
            self._warmup_steps_remaining[warmup_mask] -= 1

        # --- Map actions to joint targets according to control mode ---
        mode = getattr(self.cfg, "control_mode", "residual_cpg")

        if mode == "raw":
            # Direct mapping: joint deltas around default
            target = self._robot.data.default_joint_pos + self.cfg.action_scale * self._joint_range * actions_to_use
        else:
            # Generate scripted trot gait as base
            base = self._compute_cpg_targets()
            if mode == "cpg":
                target = base
            else:  # residual_cpg
                # Add small residual on top of base to allow learning fine corrections
                residual = self.cfg.residual_scale * actions_to_use
                residual = self.cfg.action_scale * self._joint_range * residual
                target = base + residual

        self._processed_actions = torch.clamp(target, self._JOINT_ANGLE_MIN, self._JOINT_ANGLE_MAX)
        
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

    def _compute_cpg_targets(self) -> torch.Tensor:
        """Compute simple symmetric trot joint targets (shoulders neutral).

        - Diagonal pairing: (FL, BR) in phase; (FR, BL) opposite phase.
        - Thigh: sinusoid with amplitude `cpg_thigh_amp`.
        - Calf: cosine with amplitude `cpg_calf_amp` plus small knee bias delta.
        - Shoulders: 0.0 (neutral) to avoid lateral roll.
        """
        f = float(self.cfg.cpg_frequency_hz)
        phi = 2.0 * math.pi * f * self._time  # [N]
        # Phases per leg: order [FL, FR, BL, BR]
        # FL, BR: phi; FR, BL: phi+pi
        phi_fl = phi
        phi_fr = phi + math.pi
        phi_bl = phi + math.pi
        phi_br = phi

        # Stack per-leg phases (N,4)
        phases = torch.stack([phi_fl, phi_fr, phi_bl, phi_br], dim=1)

        # Amplitudes
        Ath = float(self.cfg.cpg_thigh_amp)
        Acf = float(self.cfg.cpg_calf_amp)
        knee_bias = float(self.cfg.cpg_knee_bias_delta)

        # Default pose
        q0 = self._robot.data.default_joint_pos  # (N,12)

        # Build target deltas per joint group
        # Shoulders (4): zero deltas
        d_sh = torch.zeros(self.num_envs, 4, device=self.device)

        # Thighs (4): Ath * sin(phase)
        d_th = Ath * torch.sin(phases)

        # Calves (4): Acf * cos(phase) + knee_bias
        d_cf = Acf * torch.cos(phases) + knee_bias

        # Concatenate in RL joint order
        d_all = torch.cat([d_sh, d_th, d_cf], dim=1)
        return torch.clamp(q0 + d_all, self._JOINT_ANGLE_MIN, self._JOINT_ANGLE_MAX)

    def _get_contact_sensor_data(self) -> dict:
        """Extract and process contact sensor data for termination and gait rewards.

        Returns a dict with selected contact metrics used by the flat task:
            - undesired_contact_forces: max force over history on non-foot bodies
            - first_contact: boolean mask of first foot contacts this step
            - last_air_time: air time duration for each foot prior to contact
        """
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # Undesired contact detection (body/thighs/shoulders touching ground)
        undesired_contact_forces = torch.max(
            torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1
        )[0]

        # Feet air-time support data
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]

        return {
            "undesired_contact_forces": undesired_contact_forces,
            "first_contact": first_contact,
            "last_air_time": last_air_time,
            "net_forces_w_history": net_contact_forces,
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
        #    - Primary feedback for velocity tracking
        #    - Body frame ensures rotation-invariant representation
        # 
        # 2. Root Angular Velocity (3D) - [wx, wy, wz] in body frame [rad/s] 
        #    - Stability and balance cues
        #    - Roll/pitch rates indicate balance state
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
        if self._policy_log_dir is not None and self.num_envs > 0:
            entry = {
                "step": int(self._policy_log_step),
                "sim_time": float(self._time),
                "observation": obs[0].detach().cpu().tolist(),
                "command": self._commands[0].detach().cpu().tolist(),
                "raw_action": self._actions[0].detach().cpu().tolist(),
                "processed_action": self._processed_actions[0].detach().cpu().tolist(),
            }
            smoothed_actions = getattr(self, "_actions_smooth", None)
            if smoothed_actions is not None:
                entry["smoothed_action"] = smoothed_actions[0].detach().cpu().tolist()

            with open(self._policy_log_file, "a", encoding="utf-8") as f:
                json.dump(entry, f)
                f.write("\n")
            self._policy_log_step += 1

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute simplified reward: forward speed tracking + height + torque.

        Forward-speed reward is gated on minimum body height to prevent belly-dragging.
        """
        
        # ==================== LINEAR VELOCITY TRACKING ====================
        # DIRECTIONAL: Decompose velocity into parallel and perpendicular components
        # relative to command direction. Penalize sideways drift much more than along-track error.
        
        # Simplified forward speed tracking (vx only) with stronger incentive to move
        vx = self._robot.data.root_lin_vel_b[:, 0]
        vx_cmd = self._commands[:, 0]
        err_vx = vx - vx_cmd
        sigma_vx = 0.18  # tighter tolerance so idle policies score poorly
        # Baseline subtraction so standing still at non-zero command yields ~0
        lin_raw = torch.exp(- (err_vx / sigma_vx) ** 2)
        lin_base = torch.exp(- (vx_cmd / sigma_vx) ** 2)
        lin_vel_reward = lin_raw - lin_base
        # Small underspeed penalty to discourage stalling when command is positive
        k_under = 0.5
        lin_vel_reward = lin_vel_reward - k_under * torch.clamp(vx_cmd - vx, min=0.0, max=1.0)

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

        # Gate forward-speed reward on minimum body height
        min_h = float(getattr(self.cfg.gait, "forward_reward_min_height", 0.12))
        height_gate = (current_height > min_h).float()
        lin_vel_reward = lin_vel_reward * height_gate

        # ==================== TORQUE PENALTY ====================
        # Penalize large actuator efforts to encourage energy-efficient motions.
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        # ==================== FEET AIR TIME REWARD ====================
        # Encourage stepping behavior when the robot is moving forward
        cs = self._get_contact_sensor_data()
        first_contact = cs["first_contact"]
        last_air_time = cs["last_air_time"]
        speed_xy = torch.linalg.vector_norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        optimal_air_time = 0.4
        air_time_error = torch.abs(last_air_time - optimal_air_time)
        air_time_gate = (speed_xy > 0.05).float()
        air_time_reward = torch.sum(torch.exp(-air_time_error * 10.0) * first_contact.float(), dim=1) * air_time_gate

        # ==================== REWARD ASSEMBLY ====================
        # Note: Rewards are NOT multiplied by step_dt here to avoid double normalization.
        # The episodic sum is already normalized by max_episode_length_s when logged.
        rewards = {
            "track_forward_speed": lin_vel_reward * self.cfg.rewards.track_forward_speed,
            "height_reward": height_reward * self.cfg.rewards.height_reward,
            "torque_penalty": joint_torques * self.cfg.rewards.torque_penalty,
            "feet_air_time_reward": air_time_reward * self.cfg.rewards.feet_air_time,
        }
        
        # Sum all rewards
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Update episode sums for logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination and timeout using contact-based logic.

        - Immediate reset on undesired contacts (body/thigh/shoulder contacting ground)
        - Orientation termination disabled to allow recovery from moderate tilts
        - Time-based episode limit respected
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Contact-based termination
        cs = self._get_contact_sensor_data()
        undesired_contact_forces = cs["undesired_contact_forces"]
        contact_terminated = torch.any(
            undesired_contact_forces > self.cfg.termination.undesired_contact_force_threshold,
            dim=1,
        )

        # Orientation check (disabled in termination composition)
        _orientation_terminated = self._robot.data.projected_gravity_b[:, 2] > self.cfg.termination.orientation_threshold

        terminated = contact_terminated
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset specified environments and set fixed forward command.

        - Resets robot to neutral pose and zero velocities
        - Assigns terrain origins
        - Sets commands to [vx=fixed_forward_speed, vy=0, yaw=0]
        - Clears episode sums and timing buffers
        """

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Desynchronize episodes when all envs reset (e.g., at start of training)
        # to avoid large synchronous reset spikes in reward/throughput.
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset action buffers
        self._actions[env_ids].zero_()
        self._previous_actions[env_ids].zero_()
        # Reset warmup counters
        self._warmup_steps_remaining[env_ids] = int(getattr(self.cfg, "warmup_policy_steps", 0))
        # No undesired-contact timers in simplified task

        # Fixed deployment-focused velocity command (forward-only)
        num = len(env_ids)
        forward_cmd = torch.zeros(num, 3, device=self.device)
        # Use configurable fixed forward speed for straight-walk training
        forward_cmd[:, 0] = float(getattr(self.cfg, "fixed_forward_speed", 0.25))
        self._commands[env_ids] = forward_cmd

        if self._policy_log_dir is not None:
            # During logging runs, hold a fixed forward command to simplify replay analysis
            self._commands[env_ids] = 0.0
            self._commands[env_ids, 0] = float(getattr(self.cfg, "fixed_forward_speed", 0.25))

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        # Use terrain origins if available, otherwise use scene origins
        if hasattr(self._terrain, "env_origins"):
            # Implement terrain selection
            if self.cfg.terrain.terrain_type == "generator" and hasattr(self._terrain, "terrain_origins"):
                num_cols = self.cfg.terrain.terrain_generator.num_cols  # 20 variations per level
                # For each resetting environment, assign a terrain level randomly
                for env_id in env_ids:
                    env_idx = int(env_id)
                    # Randomly select a terrain level from allowed initial range
                    terrain_level = torch.randint(0, self.cfg.terrain.max_init_terrain_level, (1,), device=self.device).item()
                    # Randomly select a column within that terrain level
                    terrain_col = torch.randint(0, num_cols, (1,), device=self.device).item()
                    # Update environment's terrain level tracking
                    self._env_terrain_levels[env_idx] = terrain_level
                    # Set the environment origin to the selected terrain patch
                    self._terrain.env_origins[env_idx, :] = self._terrain.terrain_origins[terrain_level, terrain_col, :]

            origins = self._terrain.env_origins
        else:
            origins = self.scene.env_origins

        # Spread spawn locations for parallel environments while keeping env 0 at reference origin
        if len(env_ids) > 0:
            spacing = getattr(self.cfg.scene, "env_spacing", 2.0)
            cols = max(int(math.sqrt(max(self.num_envs - 1, 1))) + 1, 1)
            for env_id in env_ids:
                env_idx = int(env_id)
                if env_idx == 0:
                    continue
                row = (env_idx - 1) // cols
                col = (env_idx - 1) % cols
                origins[env_idx, 0] = row * spacing
                origins[env_idx, 1] = col * spacing

        default_root_state[:, :3] += origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Apply domain randomization on reset
        dr = self.cfg.domain_randomization
        if dr.enable_randomization and dr.randomize_on_reset:
            self._randomize_robot_properties(env_ids)
            self._randomize_physics_materials(env_ids)
            # Randomize action delays if enabled
            if dr.add_action_delay:
                delay_min, delay_max = dr.action_delay_steps
                self._action_delays[env_ids] = torch.randint(
                    delay_min, delay_max + 1, (len(env_ids),), device=self.device
                )

        # Reset time for sinusoidal observation
        self._time[env_ids] = 0

        # Logging and analytics
        log = {}
        # Episode reward summaries (normalized by episode length)
        for key, buf in self._episode_sums.items():
            log[f"Episode_Reward/{key}"] = torch.mean(buf[env_ids]) / self.max_episode_length_s
            buf[env_ids] = 0.0
        self.extras["log"] = {}
        self.extras["log"].update(log)
        # Termination reason logging removed for simplified flat task

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
