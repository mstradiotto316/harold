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
from harold_isaac_lab.common.stance import load_rl_default_pose


class HaroldIsaacLabEnv(DirectRLEnv):
    """Reinforcement Learning Environment for Harold Quadruped Robot.
    
    This class implements a sophisticated RL training environment for a 12-DOF quadruped robot
    mirroring the rough-terrain task logic while using the flat-terrain generator.
    The environment features comprehensive reward shaping, velocity command tracking,
    and energy-efficient locomotion rewards.
    
    Key Features:
        - 12-DOF quadruped with 3 joints per leg (shoulder, thigh, calf)
        - Flat-terrain training (terrain curriculum disabled in this variant)
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

        Environment Variables:
            HAROLD_SCRIPTED_GAIT: Set to "1" to enable scripted walking gait mode
            HAROLD_SCRIPTED_GAIT_FREQ: Override gait frequency (default: cfg.scripted_gait.frequency)

        Args:
            cfg: Environment configuration containing robot, terrain, and training parameters
            render_mode: Rendering mode - 'human' enables GUI visualization, None for headless
            **kwargs: Additional keyword arguments passed to DirectRLEnv parent class
        """
        # --- Environment variable overrides ---
        # Scripted gait mode - validated in Session 21-22
        if os.getenv("HAROLD_SCRIPTED_GAIT", "0") == "1":
            cfg.scripted_gait.enabled = True
            print("=" * 60)
            print("SCRIPTED GAIT MODE ENABLED")
            print("Session 21: vx=+0.141 m/s achieved (141% of target)")
            print("Session 22: Real robot walks forward with this gait")
            print("=" * 60)
            freq_override = os.getenv("HAROLD_SCRIPTED_GAIT_FREQ")
            if freq_override:
                try:
                    cfg.scripted_gait.frequency = float(freq_override)
                    print(f"Scripted gait frequency overridden to {cfg.scripted_gait.frequency} Hz")
                except ValueError:
                    print(f"WARNING: Invalid HAROLD_SCRIPTED_GAIT_FREQ='{freq_override}', using default.")

        # CPG mode (Phase 2 - structured action space)
        if os.getenv("HAROLD_CPG", "0") == "1":
            cfg.cpg.enabled = True
            print("=" * 60)
            print("CPG MODE ENABLED (Phase 2)")
            print("Policy outputs corrections to CPG base trajectory.")
            print(f"Base frequency: {cfg.cpg.base_frequency} Hz")
            print(f"Residual scale: {cfg.cpg.residual_scale}")
            print("=" * 60)
        cfg.observation_space = 50 if cfg.cpg.enabled else 48

        # Optional gait amplitude scaling (for sim↔hardware mismatch diagnostics)
        amp_scale_env = os.getenv("HAROLD_GAIT_AMP_SCALE")
        if amp_scale_env:
            try:
                amp_scale = float(amp_scale_env)
                if amp_scale <= 0:
                    raise ValueError("scale must be > 0")
                def _scale_gait(gait_cfg):
                    thigh_mid = (gait_cfg.stance_thigh + gait_cfg.swing_thigh) / 2.0
                    thigh_amp = (gait_cfg.swing_thigh - gait_cfg.stance_thigh) / 2.0
                    calf_mid = (gait_cfg.stance_calf + gait_cfg.swing_calf) / 2.0
                    calf_amp = (gait_cfg.swing_calf - gait_cfg.stance_calf) / 2.0
                    gait_cfg.swing_thigh = thigh_mid + thigh_amp * amp_scale
                    gait_cfg.stance_thigh = thigh_mid - thigh_amp * amp_scale
                    gait_cfg.swing_calf = calf_mid + calf_amp * amp_scale
                    gait_cfg.stance_calf = calf_mid - calf_amp * amp_scale
                    gait_cfg.shoulder_amplitude = gait_cfg.shoulder_amplitude * amp_scale
                _scale_gait(cfg.scripted_gait)
                _scale_gait(cfg.cpg)
                print("=" * 60)
                print(f"GAIT AMPLITUDE SCALE ENABLED: {amp_scale}x")
                print("=" * 60)
            except ValueError:
                print(f"WARNING: Invalid HAROLD_GAIT_AMP_SCALE='{amp_scale_env}', ignoring.")

        # Command tracking mode (for controllability)
        if os.getenv("HAROLD_CMD_TRACK", "0") == "1":
            print("=" * 60)
            print("COMMAND TRACKING MODE ENABLED")
            print("Policy will learn to follow commanded velocity.")
            print(f"Linear vel weight: {cfg.rewards.track_lin_vel_xy_weight}")
            print(f"Angular vel weight: {cfg.rewards.track_ang_vel_z_weight}")
            print("=" * 60)

        # Variable command mode (for speed control)
        if os.getenv("HAROLD_VAR_CMD", "0") == "1":
            cfg.commands.variable_commands = True
            print("=" * 60)
            print("VARIABLE COMMAND MODE ENABLED")
            print(f"vx range: [{cfg.commands.vx_min}, {cfg.commands.vx_max}] m/s")
            print(f"vy range: [{cfg.commands.vy_min}, {cfg.commands.vy_max}] m/s")
            print(f"yaw range: [{cfg.commands.yaw_min}, {cfg.commands.yaw_max}] rad/s")
            print(f"Zero velocity probability: {cfg.commands.zero_velocity_prob}")
            print("=" * 60)

        # Dynamic command mode (commands change mid-episode)
        if os.getenv("HAROLD_DYN_CMD", "0") == "1":
            cfg.commands.dynamic_commands = True
            # Dynamic implies variable commands
            cfg.commands.variable_commands = True
            print("=" * 60)
            print("DYNAMIC COMMAND MODE ENABLED")
            print(f"Command change interval: {cfg.commands.command_change_interval}s")
            print(f"Command change probability: {cfg.commands.command_change_prob}")
            print("=" * 60)

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

        # --- Dynamic Command Timer (Phase 3) ---
        # Tracks time since last command change for each environment
        self._command_timer = torch.zeros(self.num_envs, device=self.device)

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

        default_mass_tensor = self._robot.data.default_mass
        if default_mass_tensor.ndim == 2:
            total_mass = default_mass_tensor.sum(dim=1)
        else:
            scalar_mass = default_mass_tensor.flatten().sum()
            total_mass = torch.full(
                (self.num_envs,),
                float(scalar_mass.item() if isinstance(scalar_mass, torch.Tensor) else scalar_mass),
                device=self.device,
            )
        if total_mass.numel() == 1 and self.num_envs > 1:
            total_mass = total_mass.repeat(self.num_envs)
        self._robot_total_mass = total_mass.detach().clone().to(self.device)

        # --- Contact Sensor Body ID Extraction (Isaac Lab Standard Pattern) ---
        # Get base body for termination detection
        self._base_id, _ = self._contact_sensor.find_bodies(".*body")
        
        # Get feet for gait analysis and air time rewards (Harold's feet are calf ends)
        self._feet_ids, foot_names = self._contact_sensor.find_bodies(".*calf")
        foot_names_lower = [str(name).lower() for name in foot_names]
        desired_order = ("fl_calf", "fr_calf", "bl_calf", "br_calf")
        reorder_slots: list[int] = []
        for expected in desired_order:
            matches = [i for i, name in enumerate(foot_names_lower) if expected in name]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one '{expected}' calf in contact sensor results, found {matches} from {foot_names}."
                )
            reorder_slots.append(matches[0])
        if isinstance(self._feet_ids, torch.Tensor):
            index_tensor = torch.tensor(reorder_slots, device=self._feet_ids.device, dtype=torch.long)
            self._feet_ids = self._feet_ids.index_select(0, index_tensor)
        else:
            self._feet_ids = [self._feet_ids[i] for i in reorder_slots]
        if not isinstance(self._feet_ids, torch.Tensor):
            self._feet_ids = torch.tensor(self._feet_ids, device=self.device, dtype=torch.long)
        self._front_foot_slots = torch.tensor((0, 1), device=self.device, dtype=torch.long)
        self._rear_foot_slots = torch.tensor((2, 3), device=self.device, dtype=torch.long)
        
        # Get undesired contact bodies (body, thighs, shoulders should not touch ground)
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*(body|thigh|shoulder)")

        # --- Observation & State Trackers ---
        self._time = torch.zeros(self.num_envs, device=self.device)

        # --- CPG State (Phase 2) ---
        # CPG phase tracks position in the gait cycle (0 to 1, repeating)
        self._cpg_phase = torch.zeros(self.num_envs, device=self.device)

        # Ready stance (canonical default pose) loaded from config/stance.yaml
        self._ready_pose = torch.tensor(load_rl_default_pose(), device=self.device)
        self._decimation_counter = 0

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
        # Session 36: Simplified reward structure (~11 terms)
        self._reward_keys = [
            "track_lin_vel_xy",
            "track_ang_vel_z",
            "lin_vel_z",
            "ang_vel_xy",
            "dof_torques",
            "dof_acc",
            "action_rate",
            "feet_air_time",
            "undesired_contacts",
            "upright",
            "forward_motion",  # Session 36e: bootstrap walking
        ]

        self._metric_keys = [
            "vx_w_mean",
            "vy_w_mean",
            "upright_mean",
            "height_reward",
            "body_contact_penalty",
            "cmd_vx_error",
            "cmd_vy_error",
            "cmd_yaw_error",
        ]
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [*self._reward_keys, *self._metric_keys]
        }

        # --- Gait Observability Buffers ---
        # Track per-foot contact, air time, slip speed, and peak forces.
        self._foot_contact_force_threshold = 1.0
        self._foot_contact_count = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_contact_force_peak = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_air_time_sum = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_air_time_sumsq = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_air_time_count = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_slip_speed_sum = torch.zeros((self.num_envs, 4), device=self.device)
        self._foot_slip_speed_count = torch.zeros((self.num_envs, 4), device=self.device)
        self._episode_start_pos = self._robot.data.root_pos_w.clone()
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

        # Linear velocity bias buffer (per-episode calibration error)
        # Session 29: Hardware IMU has per-session calibration drift
        self._lin_vel_bias = torch.zeros(self.num_envs, 3, device=self.device)

        # Action delay buffer for simulating control latency
        if cfg.domain_randomization.add_action_delay:
            max_delay = cfg.domain_randomization.action_delay_steps[1]
            self._action_delay_buffer = torch.zeros(
                self.num_envs, max_delay + 1, self.cfg.action_space, device=self.device
            )
            self._action_delays = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # --- Backlash Hysteresis State (Session 37) ---
        # Track "engaged position" where gears are meshed. Commands within the
        # dead zone don't move output; only when exceeding half_backlash does
        # output follow (with offset). This models real servo backlash.
        if getattr(cfg, 'backlash', None) is not None and cfg.backlash.enable_backlash:
            self._backlash_enabled = True
            self._engaged_position = None  # Initialized on first _apply_action
            # Per-joint backlash values
            if cfg.backlash.per_joint_backlash is not None:
                self._backlash_rad = torch.tensor(
                    cfg.backlash.per_joint_backlash, device=self.device
                )
            else:
                self._backlash_rad = torch.full(
                    (12,), cfg.backlash.backlash_rad, device=self.device
                )
            print("=" * 60)
            print("BACKLASH HYSTERESIS ENABLED (Session 37)")
            print(f"Dead zone: {cfg.backlash.backlash_rad:.2f} rad ({math.degrees(cfg.backlash.backlash_rad):.1f}°)")
            print("Policy must learn to overdrive joints to compensate.")
            print("=" * 60)
        else:
            self._backlash_enabled = False

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
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.8, 0.85, 0.9))  # Softer light for better contrast
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

        If scripted_gait is enabled, policy actions are ignored and a predefined
        walking trajectory is used instead (for Phase 1 verification).

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
        # --- Dynamic command update (Phase 3) ---
        self._update_dynamic_commands()

        # --- Check for scripted gait mode (Phase 1 - FAILED, debugging only) ---
        if self.cfg.scripted_gait.enabled:
            self._apply_scripted_gait()
            return

        # --- Check for CPG mode (Phase 2 - structured action space) ---
        if self.cfg.cpg.enabled:
            self._apply_cpg_action(actions)
            return

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

    def _apply_cpg_action(self, actions: torch.Tensor) -> None:
        """Apply CPG-based action processing (Phase 2).

        The CPG generates a base walking trajectory. The policy's 12D output
        is interpreted as CORRECTIONS to this trajectory, providing:
        1. Walking structure from the CPG
        2. Reactive balance control from the policy

        Architecture:
            target = CPG_base + policy_output * residual_scale * joint_range

        Args:
            actions: Policy output tensor [num_envs, 12] in range [-1, 1]
        """
        cfg = self.cfg.cpg

        # --- Update CPG phase ---
        # Phase advances based on time and frequency
        self._cpg_phase = (self._time * cfg.base_frequency) % 1.0

        # --- Compute CPG base trajectory ---
        cpg_targets = self._compute_cpg_base_trajectory()

        # Store for potential reward computation (CPG tracking reward)
        self._cpg_base_targets = cpg_targets.clone()

        # --- Apply policy corrections ---
        # Store actions for logging
        self._actions.copy_(actions)

        # Smooth the actions
        if not hasattr(self, "_actions_smooth"):
            self._actions_smooth = torch.zeros_like(self._actions)
        beta = getattr(self.cfg, "action_filter_beta", 0.2)
        self._actions_smooth = (1.0 - beta) * self._actions_smooth + beta * self._actions

        # Compute corrections: policy output scaled by residual_scale and joint_range
        corrections = cfg.residual_scale * self._joint_range * self._actions_smooth

        # Combine CPG base + policy corrections
        target = cpg_targets + corrections

        # Clamp to joint limits
        self._processed_actions = torch.clamp(
            target,
            self._JOINT_ANGLE_MIN,
            self._JOINT_ANGLE_MAX,
        )

        # Store target delta for observation
        self._prev_target_delta = self._processed_actions - self._robot.data.default_joint_pos

    def _compute_cpg_base_trajectory(self) -> torch.Tensor:
        """Compute the CPG base trajectory for all joints.

        Session 24: NOW USES the same trajectory computation as ScriptedGaitCfg,
        which has been validated to work in simulation (vx=+0.141 m/s) and on
        real hardware (walking forward).

        Generates a diagonal trot gait pattern:
        - FL + BR are in phase (pair 0)
        - FR + BL are 180° out of phase (pair 1)

        Returns:
            Joint targets tensor [num_envs, 12]
        """
        cfg = self.cfg.cpg

        # Phase for each diagonal pair - same as scripted gait
        phase_fl_br = self._cpg_phase  # [num_envs]
        phase_fr_bl = (self._cpg_phase + 0.5) % 1.0  # 180° offset

        # Use the PROVEN trajectory method from ScriptedGaitCfg
        # This is the exact same computation that achieves vx=+0.141 m/s
        thigh_fl, calf_fl = self._compute_leg_trajectory(phase_fl_br, cfg)
        thigh_br, calf_br = self._compute_leg_trajectory(phase_fl_br, cfg)
        thigh_fr, calf_fr = self._compute_leg_trajectory(phase_fr_bl, cfg)
        thigh_bl, calf_bl = self._compute_leg_trajectory(phase_fr_bl, cfg)

        # Shoulder oscillation for balance (hardware-aligned sin)
        shoulder_fl = cfg.shoulder_amplitude * torch.sin(2 * math.pi * phase_fl_br)
        shoulder_br = cfg.shoulder_amplitude * torch.sin(2 * math.pi * phase_fl_br)
        shoulder_fr = cfg.shoulder_amplitude * torch.sin(2 * math.pi * phase_fr_bl)
        shoulder_bl = cfg.shoulder_amplitude * torch.sin(2 * math.pi * phase_fr_bl)

        # Assemble joint targets [shoulders(4), thighs(4), calves(4)]
        targets = torch.zeros(self.num_envs, 12, device=self.device)

        # Shoulders (indices 0-3): FL, FR, BL, BR
        targets[:, 0] = shoulder_fl
        targets[:, 1] = shoulder_fr
        targets[:, 2] = shoulder_bl
        targets[:, 3] = shoulder_br

        # Thighs (indices 4-7): FL, FR, BL, BR
        targets[:, 4] = thigh_fl
        targets[:, 5] = thigh_fr
        targets[:, 6] = thigh_bl
        targets[:, 7] = thigh_br

        # Calves (indices 8-11): FL, FR, BL, BR
        targets[:, 8] = calf_fl
        targets[:, 9] = calf_fr
        targets[:, 10] = calf_bl
        targets[:, 11] = calf_br

        return targets

    def _apply_scripted_gait(self) -> None:
        """Apply scripted walking gait trajectory (Phase 1 verification).

        Generates a predefined diagonal trot pattern to verify that the robot's
        physics simulation can support walking. This bypasses the policy entirely.

        Gait Pattern:
            - Diagonal trot: FL+BR alternate with FR+BL
            - Duty-cycle swing/stance split for clearance (hardware-aligned)
            - Configurable frequency and amplitude

        Joint Order: [shoulders(4), thighs(4), calves(4)]
            - Shoulders: FL, FR, BL, BR (indices 0-3)
            - Thighs: FL, FR, BL, BR (indices 4-7)
            - Calves: FL, FR, BL, BR (indices 8-11)
        """
        cfg = self.cfg.scripted_gait

        # Diagnostic output every 50 steps (2.5 seconds at 20Hz)
        if not hasattr(self, "_scripted_gait_step_count"):
            self._scripted_gait_step_count = 0
            self._scripted_gait_vx_sum = 0.0
            self._scripted_gait_height_sum = 0.0
            self._scripted_gait_upright_sum = 0.0

        self._scripted_gait_step_count += 1
        vx = self._robot.data.root_lin_vel_w[0, 0].item()
        height = self._robot.data.root_pos_w[0, 2].item()
        upright = -self._robot.data.projected_gravity_b[0, 2].item()
        self._scripted_gait_vx_sum += vx
        self._scripted_gait_height_sum += height
        self._scripted_gait_upright_sum += upright

        if self._scripted_gait_step_count % 50 == 0:
            avg_vx = self._scripted_gait_vx_sum / 50
            avg_height = self._scripted_gait_height_sum / 50
            avg_upright = self._scripted_gait_upright_sum / 50
            in_warmup = self._time[0].item() < 2.0
            mode = "WARMUP" if in_warmup else "GAIT"

            # Show current joint positions for debugging
            joint_pos = self._robot.data.joint_pos[0]
            thighs = joint_pos[4:8].cpu().numpy()
            calves = joint_pos[8:12].cpu().numpy()

            print(f"[SCRIPTED GAIT {mode}] Step {self._scripted_gait_step_count}: "
                  f"vx={avg_vx:.4f} m/s, height={avg_height:.4f} m, upright={avg_upright:.3f}", flush=True)
            print(f"  Thighs: {thighs}, Calves: {calves}", flush=True)

            self._scripted_gait_vx_sum = 0.0
            self._scripted_gait_height_sum = 0.0
            self._scripted_gait_upright_sum = 0.0

        # Compute gait phase from time (0 to 1, repeating)
        phase = (self._time * cfg.frequency) % 1.0  # [num_envs]

        # Check for static pose test mode (hold pose indefinitely, no gait)
        static_test = os.getenv("HAROLD_STATIC_TEST", "0") == "1"

        # Warmup period: hold standing pose for first 3 seconds to stabilize
        # In static test mode, warmup is forever (no gait phase)
        warmup_time = 9999.0 if static_test else 3.0  # seconds
        in_warmup = (self._time < warmup_time).all()

        # DEBUG: Show what mode we're in
        if self._scripted_gait_step_count == 1:
            print(f"[DEBUG] Initial time: {self._time[0].item():.3f}s, warmup_time: {warmup_time}", flush=True)

        if in_warmup:
            # During warmup, hold pose that matches gait mid-stance for smooth transition
            # Use average of stance/swing thigh and calf (matches firmware stance)
            mid_thigh = (cfg.stance_thigh + cfg.swing_thigh) / 2.0
            mid_calf = (cfg.stance_calf + cfg.swing_calf) / 2.0
            targets = torch.tensor(
                [
                    0.0, 0.0, 0.0, 0.0,                             # Shoulders: neutral
                    mid_thigh, mid_thigh, mid_thigh, mid_thigh,     # Thighs: mid-stance
                    mid_calf, mid_calf, mid_calf, mid_calf          # Calves: mid-stance
                ],
                device=self.device,
                dtype=torch.float32
            ).unsqueeze(0).expand(self.num_envs, -1)
            if self._scripted_gait_step_count % 20 == 0:
                print(f"[DEBUG WARMUP] t={self._time[0].item():.2f}s, holding gait-matched pose (thigh={mid_thigh:.2f}, calf={mid_calf:.2f})", flush=True)
        else:
            # After warmup, apply the gait pattern

            # Diagonal pair phases:
            # Pair 0 (FL + BR): phase offset 0.0
            # Pair 1 (FR + BL): phase offset 0.5
            phase_fl_br = phase
            phase_fr_bl = (phase + 0.5) % 1.0

            # Compute leg trajectories for each diagonal pair
            thigh_fl, calf_fl = self._compute_leg_trajectory(phase_fl_br, cfg)
            thigh_br, calf_br = self._compute_leg_trajectory(phase_fl_br, cfg)
            thigh_fr, calf_fr = self._compute_leg_trajectory(phase_fr_bl, cfg)
            thigh_bl, calf_bl = self._compute_leg_trajectory(phase_fr_bl, cfg)

            # Shoulder oscillation for balance (hardware-aligned sin)
            shoulder_fl = cfg.shoulder_amplitude * torch.sin(2 * math.pi * phase_fl_br)
            shoulder_br = cfg.shoulder_amplitude * torch.sin(2 * math.pi * phase_fl_br)
            shoulder_fr = cfg.shoulder_amplitude * torch.sin(2 * math.pi * phase_fr_bl)
            shoulder_bl = cfg.shoulder_amplitude * torch.sin(2 * math.pi * phase_fr_bl)

            # Assemble joint targets
            # Order: [shoulders(4), thighs(4), calves(4)]
            targets = torch.zeros(self.num_envs, 12, device=self.device)

            # Shoulders (indices 0-3): FL, FR, BL, BR
            targets[:, 0] = shoulder_fl
            targets[:, 1] = shoulder_fr
            targets[:, 2] = shoulder_bl
            targets[:, 3] = shoulder_br

            # Thighs (indices 4-7): FL, FR, BL, BR
            targets[:, 4] = thigh_fl
            targets[:, 5] = thigh_fr
            targets[:, 6] = thigh_bl
            targets[:, 7] = thigh_br

            # Calves (indices 8-11): FL, FR, BL, BR
            targets[:, 8] = calf_fl
            targets[:, 9] = calf_fr
            targets[:, 10] = calf_bl
            targets[:, 11] = calf_br

        # Clamp to joint limits
        self._processed_actions = torch.clamp(
            targets,
            self._JOINT_ANGLE_MIN,
            self._JOINT_ANGLE_MAX,
        )

        # Store for observation (even though we're not using policy)
        self._prev_target_delta = self._processed_actions - self._robot.data.default_joint_pos

    def _compute_leg_trajectory(self, phase: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute smooth leg trajectory based on gait phase.

        Args:
            phase: Gait phase for this leg [0, 1) tensor of shape [num_envs]
            cfg: ScriptedGaitCfg with trajectory parameters

        Returns:
            Tuple of (thigh_angle, calf_angle) tensors

        Trajectory Design (hardware-aligned):
            - Duty-cycle splits stance vs swing for leg clearance.
            - Stance: thigh sweeps back, calf holds extension.
            - Swing: thigh moves forward, calf flexes quickly (distal joint moves more).
            - Avoids common pitfall: calf must move more than thigh since it sits at
              the end of the thigh link.
        """
        # Guard against invalid duty cycles (avoid div-by-zero).
        duty = min(max(float(cfg.duty_cycle), 0.05), 0.95)
        duty_t = torch.tensor(duty, device=phase.device, dtype=phase.dtype)

        def smoothstep(x: torch.Tensor) -> torch.Tensor:
            return x * x * (3.0 - 2.0 * x)

        stance_mask = phase < duty_t
        stance_phase = torch.where(stance_mask, phase / duty_t, torch.zeros_like(phase))
        swing_phase = torch.where(
            stance_mask, torch.zeros_like(phase), (phase - duty_t) / (1.0 - duty_t)
        )

        stance_blend = smoothstep(stance_phase)
        swing_blend = smoothstep(swing_phase)

        thigh_stance = cfg.swing_thigh + (cfg.stance_thigh - cfg.swing_thigh) * stance_blend
        thigh_swing = cfg.stance_thigh + (cfg.swing_thigh - cfg.stance_thigh) * swing_blend
        thigh = torch.where(stance_mask, thigh_stance, thigh_swing)

        calf_stance = torch.full_like(phase, cfg.stance_calf)
        lift = 0.5 - 0.5 * torch.cos(2.0 * math.pi * swing_phase)
        calf_swing = cfg.stance_calf + (cfg.swing_calf - cfg.stance_calf) * lift
        calf = torch.where(stance_mask, calf_stance, calf_swing)

        return thigh, calf

    def _apply_action(self) -> None:
        """Apply processed joint targets to robot and update visualization.

        Sends the processed joint position targets to the robot's actuators and
        handles optional data logging and visualization marker updates.

        Operations:
            1. Apply backlash hysteresis (Session 37) if enabled
            2. Send joint position targets to robot actuators
            3. Increment decimation counter for timing control
            4. Update visualization markers (every 18 physics steps)
            5. Optional action logging for replay/analysis (commented)

        Performance Optimizations:
            - Markers only update at policy frequency (18x less often)
            - Headless training skips all visualization computations
            - Pre-allocated buffers prevent memory allocations in hot loop
        """
        # --- Apply backlash hysteresis (Session 37) ---
        # Modifies the target to simulate gear backlash dead zone
        effective_target = self._apply_backlash_hysteresis(self._processed_actions)

        # --- Send joint targets to robot ---
        self._robot.set_joint_position_target(effective_target)
        
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

    def _get_observations(self) -> dict:
        """Construct the observation vector for policy input.

        Session 36: Pure RL = 48D (no gait phase)
        Session 37: CPG mode = 50D (adds 2D gait phase: sin/cos)

        Returns:
            Dict with 'policy' key containing observation tensor:
                - root_lin_vel_b (3): Linear velocity in body frame [m/s]
                - root_ang_vel_b (3): Angular velocity in body frame [rad/s]
                - projected_gravity_b (3): Gravity vector in body frame (for orientation)
                - joint_pos - default (12): Joint angles relative to neutral pose [rad]
                - joint_vel (12): Joint angular velocities [rad/s]
                - commands (3): Velocity commands [vx, vy, yaw_rate]
                - prev_target_delta (12): Previous joint targets relative to neutral pose [rad]
                - [CPG only] gait_phase (2): sin/cos of CPG phase for timing
        """

        # Update temporal state for time-based observations
        self._time += self.step_dt

        # Base observation components (48D)
        base_obs = [
            self._robot.data.root_lin_vel_b,                                      # (3D) Body linear velocity
            self._robot.data.root_ang_vel_b,                                      # (3D) Body angular velocity
            self._robot.data.projected_gravity_b,                                 # (3D) Gravity in body frame
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,     # (12D) Joint angles (relative)
            self._robot.data.joint_vel,                                          # (12D) Joint velocities
            self._commands,                                                      # (3D) Velocity commands
            self._prev_target_delta,                                             # (12D) Previous target deltas
        ]

        # Add gait phase for CPG mode (2D: sin/cos for continuous phase representation)
        if self.cfg.cpg.enabled:
            phase = self._cpg_phase.unsqueeze(-1)  # [num_envs, 1]
            gait_phase = torch.cat([
                torch.sin(2 * math.pi * phase),  # [num_envs, 1]
                torch.cos(2 * math.pi * phase),  # [num_envs, 1]
            ], dim=-1)  # [num_envs, 2]
            base_obs.append(gait_phase)

        obs = torch.cat(base_obs, dim=-1)  # [batch_size, 48 or 50]

        # Apply observation noise if domain randomization is enabled
        if self.cfg.domain_randomization.enable_randomization:
            obs = self._add_observation_noise(obs)

        # Apply observation clipping if enabled (matches deployment)
        # Session 29: Hardware deployment clips normalized obs to ±5.0
        # We clip raw obs at ±50 to approximate effect (before normalization)
        if getattr(self.cfg, 'clip_observations', False):
            clip_val = getattr(self.cfg, 'clip_observations_value', 5.0)
            # Use 10x clip_val for raw obs (pre-normalization approximation)
            obs = torch.clamp(obs, -clip_val * 10, clip_val * 10)

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
        """Simplified reward structure following Isaac Lab reference pattern.

        Session 36: Pure RL with ~10 core terms for clean gradient signals.
        Reference: isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
        """
        cfg = self.cfg.rewards

        # === Extract quantities ===
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_acc = self._robot.data.joint_acc
        applied_torque = self._robot.data.applied_torque

        vx = root_lin_vel_w[:, 0]
        vy = root_lin_vel_w[:, 1]
        vz_b = root_lin_vel_b[:, 2]
        wz = root_ang_vel_b[:, 2]

        cmd_vx = self._commands[:, 0]
        cmd_vy = self._commands[:, 1]
        cmd_yaw = self._commands[:, 2]

        # === TASK REWARDS (exponential kernel) ===
        lin_vel_error = torch.sum(
            torch.square(torch.stack([vx - cmd_vx, vy - cmd_vy], dim=1)), dim=1
        )
        track_lin_vel_xy = torch.exp(-lin_vel_error / (cfg.track_lin_vel_xy_std ** 2))

        ang_vel_error = torch.square(wz - cmd_yaw)
        track_ang_vel_z = torch.exp(-ang_vel_error / (cfg.track_ang_vel_z_std ** 2))

        # === MOTION QUALITY PENALTIES ===
        lin_vel_z = torch.square(vz_b)
        ang_vel_xy = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)

        # === SMOOTHNESS PENALTIES ===
        dof_torques = torch.sum(torch.square(applied_torque), dim=1)
        dof_acc = torch.sum(torch.square(joint_acc), dim=1)
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )

        # === GAIT: FEET AIR TIME ===
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time_reward = torch.sum(
            (last_air_time - cfg.feet_air_time_threshold) * first_contact.float(), dim=1
        )
        # Only reward air time when commanded to move
        cmd_magnitude = torch.norm(self._commands[:, :2], dim=1)
        air_time_reward = air_time_reward * (cmd_magnitude > 0.05).float()

        # === UNDESIRED CONTACTS ===
        net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, 0]
        undesired_forces = torch.norm(
            net_contact_forces[:, self._undesired_contact_body_ids], dim=-1
        )
        undesired_contacts = torch.sum(
            (undesired_forces > cfg.undesired_contacts_threshold).float(), dim=1
        )

        # === PER-FOOT CONTACT + SLIP METRICS ===
        foot_forces = torch.norm(net_contact_forces[:, self._feet_ids], dim=-1)
        foot_contact = foot_forces > self._foot_contact_force_threshold
        self._foot_contact_count += foot_contact.float()
        self._foot_contact_force_peak = torch.maximum(self._foot_contact_force_peak, foot_forces)

        air_time_sample = torch.where(first_contact, last_air_time, torch.zeros_like(last_air_time))
        self._foot_air_time_sum += air_time_sample
        self._foot_air_time_sumsq += air_time_sample * air_time_sample
        self._foot_air_time_count += first_contact.float()

        foot_lin_vel_xy = self._robot.data.body_lin_vel_w[:, self._feet_ids, :2]
        foot_slip_speed = torch.linalg.vector_norm(foot_lin_vel_xy, dim=-1)
        slip_sample = foot_slip_speed * foot_contact.float()
        self._foot_slip_speed_sum += slip_sample
        self._foot_slip_speed_count += foot_contact.float()

        # === STABILITY: UPRIGHT ===
        upright = -projected_gravity[:, 2]

        # === HEIGHT METRIC (terrain-relative) ===
        pos_z = self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
        ray_z = self._height_scanner.data.ray_hits_w[..., 2]
        ray_z = torch.where(torch.isfinite(ray_z), ray_z, pos_z)
        height_data = pos_z - ray_z
        current_height = torch.mean(height_data, dim=1)
        target_height = self.cfg.gait.target_height
        height_error = torch.abs(current_height - target_height)
        height_reward = torch.tanh(3.0 * torch.exp(-5.0 * height_error))

        # === BODY CONTACT METRIC ===
        body_contact_penalty = -undesired_contacts

        # === FORWARD MOTION BONUS ===
        # Direct reward for positive vx to bootstrap walking
        # Gate by upright to avoid rewarding forward falling
        forward_motion = cfg.forward_motion_weight * vx * upright.clamp(0.5, 1.0)

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
        }

        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Telemetry
        self._episode_sums["vx_w_mean"] += vx
        self._episode_sums["vy_w_mean"] += torch.abs(vy)
        self._episode_sums["upright_mean"] += upright.clamp(0.0, 1.0)
        self._episode_sums["height_reward"] += height_reward
        self._episode_sums["body_contact_penalty"] += body_contact_penalty
        self._episode_sums["cmd_vx_error"] += torch.abs(vx - cmd_vx)
        self._episode_sums["cmd_vy_error"] += torch.abs(vy - cmd_vy)
        self._episode_sums["cmd_yaw_error"] += torch.abs(wz - cmd_yaw)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate on orientation failure, height, or body contact."""

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        orientation_terminated = (
            self._robot.data.projected_gravity_b[:, 2]
            > self.cfg.termination.orientation_threshold
        )

        # Height termination: use terrain-relative height with warmup period
        height_threshold = self.cfg.termination.height_threshold
        warmup_steps = getattr(self.cfg.termination, 'height_termination_warmup_steps', 0)
        if height_threshold > 0.0:
            pos_z = self._robot.data.root_pos_w[:, 2].unsqueeze(1)
            ray_z = self._height_scanner.data.ray_hits_w[..., 2]
            valid_mask = torch.isfinite(ray_z)
            height_samples = torch.where(valid_mask, pos_z - ray_z, torch.zeros_like(ray_z))
            valid_counts = valid_mask.sum(dim=1)
            current_height = self._robot.data.root_pos_w[:, 2].clone()
            valid_envs = valid_counts > 0
            if torch.any(valid_envs):
                current_height[valid_envs] = (
                    height_samples[valid_envs].sum(dim=1) / valid_counts[valid_envs].float()
                )
            height_terminated = current_height < height_threshold
            # Skip termination during warmup period (sensor initialization)
            if warmup_steps > 0:
                in_warmup = self.episode_length_buf < warmup_steps
                height_terminated = height_terminated & ~in_warmup
        else:
            height_terminated = torch.zeros_like(orientation_terminated)

        # Body contact termination: terminate if body/thigh/shoulder touches ground too hard
        body_contact_threshold = self.cfg.termination.body_contact_threshold
        if body_contact_threshold > 0.0 and body_contact_threshold < float('inf'):
            net_contact_forces = getattr(self._contact_sensor.data, "net_forces_w", None)
            if net_contact_forces is None:
                net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, 0]
            # Sum of absolute Z-force on undesired contact bodies
            undesired_contact_forces = torch.abs(
                net_contact_forces[:, self._undesired_contact_body_ids, 2]
            )
            body_contact_force = undesired_contact_forces.sum(dim=1)
            body_contact_terminated = body_contact_force > body_contact_threshold
        else:
            body_contact_terminated = torch.zeros_like(orientation_terminated)

        # Elbow pose termination: detect via front leg joint angles
        # Joint order: [shoulders(4), thighs(4), calves(4)]
        # Front thighs: indices 4 (fl), 5 (fr); Front calves: indices 8 (fl), 9 (fr)
        elbow_pose_terminated = torch.zeros_like(orientation_terminated)
        if getattr(self.cfg.termination, 'elbow_pose_termination', False):
            joint_pos = self._robot.data.joint_pos
            front_thigh_threshold = self.cfg.termination.front_thigh_threshold
            front_calf_threshold = self.cfg.termination.front_calf_threshold

            # Check if EITHER front leg is in elbow pose (thigh extended AND calf unbent)
            fl_elbow = (joint_pos[:, 4] > front_thigh_threshold) & (joint_pos[:, 8] > front_calf_threshold)
            fr_elbow = (joint_pos[:, 5] > front_thigh_threshold) & (joint_pos[:, 9] > front_calf_threshold)
            elbow_pose_terminated = fl_elbow | fr_elbow

            # Apply warmup to avoid false positives during initialization
            if warmup_steps > 0:
                in_warmup = self.episode_length_buf < warmup_steps
                elbow_pose_terminated = elbow_pose_terminated & ~in_warmup

        terminated = orientation_terminated | height_terminated | body_contact_terminated | elbow_pose_terminated

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset specified environments to the simplified Phase-0 start state."""

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        prev_episode_steps = self.episode_length_buf[env_ids].clone()
        current_root_pos = self._robot.data.root_pos_w[env_ids].clone()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids].zero_()
        self._previous_actions[env_ids].zero_()

        # Reset backlash hysteresis state (Session 37)
        # Set engaged position to ready pose so policy starts fresh
        if self._backlash_enabled and self._engaged_position is not None:
            ready_pose = self._ready_pose.to(dtype=self._engaged_position.dtype)
            self._engaged_position[env_ids] = ready_pose.unsqueeze(0).expand(len(env_ids), -1)

        num = len(env_ids)
        sampled_commands = torch.zeros_like(self._commands[env_ids])

        # Check if variable command mode is enabled
        cmd_cfg = getattr(self.cfg, 'commands', None)
        if cmd_cfg is not None and getattr(cmd_cfg, 'variable_commands', False):
            # Variable command sampling from configured ranges
            # Sample vx uniformly from [vx_min, vx_max]
            sampled_commands[:, 0] = torch.empty(num, device=self.device).uniform_(
                cmd_cfg.vx_min, cmd_cfg.vx_max
            )
            # Sample vy uniformly from [vy_min, vy_max]
            sampled_commands[:, 1] = torch.empty(num, device=self.device).uniform_(
                cmd_cfg.vy_min, cmd_cfg.vy_max
            )
            # Sample yaw uniformly from [yaw_min, yaw_max]
            sampled_commands[:, 2] = torch.empty(num, device=self.device).uniform_(
                cmd_cfg.yaw_min, cmd_cfg.yaw_max
            )

            # Apply zero velocity probability for stopping behavior
            zero_prob = getattr(cmd_cfg, 'zero_velocity_prob', 0.0)
            if zero_prob > 0:
                zero_mask = torch.rand(num, device=self.device) < zero_prob
                sampled_commands[zero_mask, :] = 0.0
        else:
            # Default fixed command sampling (original behavior)
            sampled_commands[:, 0] = torch.empty(num, device=self.device).uniform_(0.25, 0.35)

        # Override for policy logging (consistent commands for evaluation)
        if self._policy_log_dir is not None:
            sampled_commands[:, 0] = 0.4
        self._commands[env_ids] = sampled_commands

        # Reset command timers for dynamic command mode
        self._command_timer[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        num_reset_envs = len(env_ids)
        if num_reset_envs > 0:
            ready_pose = self._ready_pose.to(dtype=joint_pos.dtype).unsqueeze(0).repeat(num_reset_envs, 1)
            joint_pos = ready_pose
            self._robot.data.default_joint_pos[env_ids] = ready_pose

        if hasattr(self._terrain, 'env_origins'):
            if self.cfg.terrain.terrain_type == 'generator' and hasattr(self._terrain, 'terrain_origins'):
                num_cols = self.cfg.terrain.terrain_generator.num_cols
                for env_id in env_ids:
                    env_idx = int(env_id)
                    terrain_level = torch.randint(
                        0, self.cfg.terrain.max_init_terrain_level, (1,), device=self.device
                    ).item()
                    terrain_col = torch.randint(0, num_cols, (1,), device=self.device).item()
                    self._env_terrain_levels[env_idx] = terrain_level
                    self._terrain.env_origins[env_idx, :] = self._terrain.terrain_origins[terrain_level, terrain_col, :]

            origins = self._terrain.env_origins
        else:
            origins = self.scene.env_origins

        if len(env_ids) > 0:
            spacing = getattr(self.cfg.scene, 'env_spacing', 2.0)
            cols = max(int(math.sqrt(self.num_envs)), 1)
            for env_id in env_ids:
                env_idx = int(env_id)
                row = env_idx // cols
                col = env_idx % cols
                origins[env_idx, 0] = row * spacing
                origins[env_idx, 1] = col * spacing

        default_root_state[:, :3] += origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        dr = self.cfg.domain_randomization
        if dr.enable_randomization and dr.randomize_on_reset:
            self._randomize_robot_properties(env_ids)
            self._randomize_physics_materials(env_ids)
            if dr.add_action_delay:
                delay_min, delay_max = dr.action_delay_steps
                self._action_delays[env_ids] = torch.randint(
                    delay_min, delay_max + 1, (len(env_ids),), device=self.device
                )

        # Randomize lin_vel bias per-episode (always when DR enabled)
        # Session 29: Hardware IMU has calibration drift that persists per-episode
        if dr.enable_randomization and getattr(dr, 'add_lin_vel_noise', False):
            bias_std = getattr(dr, 'lin_vel_bias_std', 0.02)
            self._lin_vel_bias[env_ids] = torch.randn(
                len(env_ids), 3, device=self.device
            ) * bias_std

        self._time[env_ids] = 0

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

                valid_env_ids = env_ids[valid]
                x_displacement = current_root_pos[valid, 0] - self._episode_start_pos[valid_env_ids, 0]
                log['Episode_Metric/x_displacement'] = torch.mean(x_displacement)
                log['Episode_Metric/x_displacement_abs'] = torch.mean(torch.abs(x_displacement))

                foot_labels = ("fl", "fr", "bl", "br")
                contact_ratio = self._foot_contact_count[valid_env_ids] / step_counts.unsqueeze(1)
                foot_force_peak = self._foot_contact_force_peak[valid_env_ids]

                air_time_count = torch.clamp(self._foot_air_time_count[valid_env_ids], min=1.0)
                air_time_mean = self._foot_air_time_sum[valid_env_ids] / air_time_count
                air_time_var = self._foot_air_time_sumsq[valid_env_ids] / air_time_count - air_time_mean * air_time_mean
                air_time_std = torch.sqrt(torch.clamp(air_time_var, min=0.0))

                slip_count = torch.clamp(self._foot_slip_speed_count[valid_env_ids], min=1.0)
                slip_mean = self._foot_slip_speed_sum[valid_env_ids] / slip_count

                for i, label in enumerate(foot_labels):
                    log[f'Episode_Metric/foot_contact_ratio_{label}'] = torch.mean(contact_ratio[:, i])
                    log[f'Episode_Metric/foot_contact_force_peak_{label}'] = torch.mean(foot_force_peak[:, i])
                    log[f'Episode_Metric/foot_air_time_mean_{label}'] = torch.mean(air_time_mean[:, i])
                    log[f'Episode_Metric/foot_air_time_std_{label}'] = torch.mean(air_time_std[:, i])
                    log[f'Episode_Metric/foot_slip_speed_mean_{label}'] = torch.mean(slip_mean[:, i])
            for key in [*self._reward_keys, *self._metric_keys]:
                self._episode_sums[key][env_ids] = 0.0
            self._foot_contact_count[env_ids] = 0.0
            self._foot_contact_force_peak[env_ids] = 0.0
            self._foot_air_time_sum[env_ids] = 0.0
            self._foot_air_time_sumsq[env_ids] = 0.0
            self._foot_air_time_count[env_ids] = 0.0
            self._foot_slip_speed_sum[env_ids] = 0.0
            self._foot_slip_speed_count[env_ids] = 0.0

        self._episode_start_pos[env_ids] = default_root_state[:, :3]

        log['Episode_Termination/orientation'] = int(torch.sum(self.reset_terminated[env_ids]).item())
        log['Episode_Termination/time_out'] = int(torch.count_nonzero(self.reset_time_outs[env_ids]).item())

        self.extras['log'] = log

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

        # Add linear velocity noise (indices 0-3)
        # Session 29: Hardware IMU computes lin_vel via accelerometer integration
        # which is inherently noisy and has per-episode calibration drift
        dr = self.cfg.domain_randomization
        if getattr(dr, 'add_lin_vel_noise', False):
            noisy_obs[:, 0:3] = gaussian_noise(
                observations[:, 0:3],
                dr.lin_vel_noise
            )
            # Add per-episode bias (calibration error)
            noisy_obs[:, 0:3] += self._lin_vel_bias

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

    def _update_dynamic_commands(self) -> None:
        """Update velocity commands periodically during episode (Phase 3).

        When dynamic_commands is enabled, commands change every command_change_interval
        seconds instead of only at episode reset. This teaches the policy to respond
        to command changes (e.g., walk -> stop -> walk).

        Uses the same sampling logic as _reset_idx for consistency.
        """
        cmd_cfg = getattr(self.cfg, 'commands', None)
        if cmd_cfg is None or not getattr(cmd_cfg, 'dynamic_commands', False):
            return

        # Get physics timestep (dt * decimation = policy timestep)
        dt = self.cfg.sim.dt * self.cfg.decimation

        # Update command timers
        self._command_timer += dt

        # Check which environments need command updates
        interval = cmd_cfg.command_change_interval
        update_mask = self._command_timer >= interval

        if not update_mask.any():
            return

        # Get indices of environments to update
        update_ids = update_mask.nonzero(as_tuple=False).squeeze(-1)
        num = len(update_ids)

        if num == 0:
            return

        # Apply command change probability
        change_prob = getattr(cmd_cfg, 'command_change_prob', 1.0)
        if change_prob < 1.0:
            prob_mask = torch.rand(num, device=self.device) < change_prob
            update_ids = update_ids[prob_mask]
            num = len(update_ids)
            if num == 0:
                # Reset timers but don't change commands
                self._command_timer[update_mask] = 0.0
                return

        # Sample new commands (same logic as _reset_idx)
        sampled_commands = torch.zeros(num, 3, device=self.device)

        # Sample vx uniformly from [vx_min, vx_max]
        sampled_commands[:, 0] = torch.empty(num, device=self.device).uniform_(
            cmd_cfg.vx_min, cmd_cfg.vx_max
        )
        # Sample vy uniformly from [vy_min, vy_max]
        sampled_commands[:, 1] = torch.empty(num, device=self.device).uniform_(
            cmd_cfg.vy_min, cmd_cfg.vy_max
        )
        # Sample yaw uniformly from [yaw_min, yaw_max]
        sampled_commands[:, 2] = torch.empty(num, device=self.device).uniform_(
            cmd_cfg.yaw_min, cmd_cfg.yaw_max
        )

        # Apply zero velocity probability
        zero_prob = getattr(cmd_cfg, 'zero_velocity_prob', 0.0)
        if zero_prob > 0:
            zero_mask = torch.rand(num, device=self.device) < zero_prob
            sampled_commands[zero_mask, :] = 0.0

        # Update commands and reset timers
        self._commands[update_ids] = sampled_commands
        self._command_timer[update_ids] = 0.0

    def _apply_backlash_hysteresis(self, target: torch.Tensor) -> torch.Tensor:
        """Apply backlash hysteresis model to joint position targets.

        Session 37: Models real servo backlash as dead zone where motor can
        rotate without moving output. Teaches policy to overdrive joints.

        Physical model:
        - Track "engaged position" where gears are meshed
        - Command can be anywhere in [engaged ± half_backlash] without moving output
        - When command exits zone, output follows with half_backlash offset

        Args:
            target: Commanded joint positions [num_envs, 12]

        Returns:
            Effective joint positions after backlash [num_envs, 12]
        """
        if not self._backlash_enabled:
            return target

        # Initialize engaged position on first call
        if self._engaged_position is None:
            self._engaged_position = target.clone()
            return target

        half_backlash = self._backlash_rad / 2.0  # [12]

        # Compute gap between command and current engaged position
        gap = target - self._engaged_position  # [num_envs, 12]

        # Check if command is outside the dead zone for each joint
        outside_zone = torch.abs(gap) > half_backlash  # [num_envs, 12]

        # When outside zone: new engaged = target - sign(gap) * half_backlash
        # This means output lags by half_backlash in direction of motion
        new_engaged = torch.where(
            outside_zone,
            target - torch.sign(gap) * half_backlash,
            self._engaged_position  # Stay where we are if in dead zone
        )

        # Update engaged position
        self._engaged_position = new_engaged

        return self._engaged_position

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
            
            # Shape: [num_envs, num_bodies, 3] - required by Isaac Lab API
            forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

            # Generate random forces for selected environments
            num_forced = apply_force.sum()
            forces[apply_force, 0, :2] = sample_uniform(
                -force_max, force_max, (num_forced, 2), self.device
            )
            forces[apply_force, 0, 2] = sample_uniform(
                force_min, force_max, (num_forced,), self.device
            )

            torques[apply_force, 0] = sample_uniform(
                -torque_max, torque_max, (num_forced, 3), self.device
            )

            # Apply forces to robot base
            self._robot.set_external_force_and_torque(
                forces, torques, body_ids=[self._base_id[0]]
            )

    def __del__(self):
        pass
