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

    cfg: HaroldIsaacLabEnvCfg

    def __init__(self, cfg: HaroldIsaacLabEnvCfg, render_mode: str | None = None, **kwargs):

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

        # Return zero observations for test script simplicity
        obs = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)

        observations = {"policy": obs}

        # Update previous actions
        self._previous_actions.copy_(self._actions)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Push-up playback task returns zero rewards (no RL objective)."""
        zeros = torch.zeros(self.num_envs, device=self.device)

        return zeros

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        # Only time-out termination retained for push-up playback
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None = None) -> None:

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        
        # Fixed spawn position: do not offset by terrain origins; place at scene origin
        default_root_state[:, :3] = torch.tensor((0.0, 0.0, default_root_state[0, 2]), device=self.device)
            
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
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

    def __del__(self):
        pass
