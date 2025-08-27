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
        self._processed_actions = ( self.cfg.action_scale * self._actions ) + self._robot.data.default_joint_pos
        
        # --- Per-joint range scaling for natural action space ---
        # Shoulders get smaller range (0.25) for stability, thighs and calves get larger (0.5)
        self._joint_range = torch.tensor(
            [0.35, 0.35, 0.35, 0.35,  # Shoulders (FL, FR, BL, BR) - increased to aid lateral control
             0.5, 0.5, 0.5, 0.5,       # Thighs (FL, FR, BL, BR)
             0.5, 0.5, 0.5, 0.5],      # Calves (FL, FR, BL, BR)
            device=self.device
        )
        
        # --- Direction multipliers per joint group ---
        # Normalize command space so a positive value means a positive rotation at every joint.
        # With hardware/URDF alignment fixed, all signs are +1. If a future robot needs flips,
        # update these tensors only (the playback values assume +1 signs).
        self._thigh_signs = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        self._calf_signs = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        
        # Commands removed for push-up playback (no velocity tracking)
        self._commands = None

        self._JOINT_ANGLE_MAX = torch.tensor([
            0.5236, 0.5236, 0.5236, 0.5236,  # shoulders
            1.5708, 1.5708, 1.5708, 1.5708,  # thighs
            1.5708, 1.5708, 1.5708, 1.5708   # calves
        ], device=self.device)
        self._JOINT_ANGLE_MIN = -self._JOINT_ANGLE_MAX.clone()

        # --- Push-up playback parameters (match firmware timing at 20 Hz) ---
        # No settle period: start moving from neutral on the first step.
        self._pushup_steps_phase = 24        # 1.2s down or up phase at 20 Hz (1.2 * 20)
        self._pushup_pause_steps = 6         # ~0.3s pause at top/bottom at 20 Hz (0.3 * 20)
        self._pushup_reps = 5
        self._pushup_cycle_steps = (
            self._pushup_steps_phase + self._pushup_pause_steps + self._pushup_steps_phase + self._pushup_pause_steps
        )
        self._pushup_total_steps = self._pushup_reps * self._pushup_cycle_steps
        self._pushup_step_counter = 0

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

        # --- Environment Cloning & Collision Filtering ---
        # Duplicate envs, optionally without copying prim data, and disable collisions with the ground
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # --- Lighting Setup ---
        # Add a dome light to illuminate the scene
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        # Ignore incoming policy actions and drive a push-up trajectory instead
        # Playback runs at policy rate (decimation set to 9 => 20 Hz), matching firmware timing
        self._actions.zero_()

        # Determine current phase of the push-up routine
        step = int(self._pushup_step_counter)

        # Push-up angle keyframes (degrees) under +1 signs using neutral as the top position:
        # - Top (neutral): thighs 0°, calves 0°
        # - Bottom:        thighs +45°, calves -90°
        thigh_deg_top, calf_deg_top = 0.0, 0.0
        thigh_deg_bottom, calf_deg_bottom = 45.0, -90.0

        # Compute step within the repetition cycle starting immediately at step 0
        local = step % self._pushup_cycle_steps

        # Down phase [0, steps_phase): interpolate from top (neutral) to bottom
        if local < self._pushup_steps_phase:
            t = float(local) / float(self._pushup_steps_phase)
            thigh_deg = thigh_deg_top + (thigh_deg_bottom - thigh_deg_top) * t
            calf_deg = calf_deg_top + (calf_deg_bottom - calf_deg_top) * t
        # Bottom pause [steps_phase, steps_phase+pause): hold bottom
        elif local < self._pushup_steps_phase + self._pushup_pause_steps:
            thigh_deg = thigh_deg_bottom
            calf_deg = calf_deg_bottom
        # Up phase [steps_phase+pause, 2*steps_phase+pause): interpolate back to top (neutral)
        elif local < self._pushup_steps_phase + self._pushup_pause_steps + self._pushup_steps_phase:
            up_local = local - (self._pushup_steps_phase + self._pushup_pause_steps)
            t = float(up_local) / float(self._pushup_steps_phase)
            thigh_deg = thigh_deg_bottom + (thigh_deg_top - thigh_deg_bottom) * t
            calf_deg = calf_deg_bottom + (calf_deg_top - calf_deg_bottom) * t
        # Top pause: hold neutral
        else:
            thigh_deg = thigh_deg_top
            calf_deg = calf_deg_top

        # Advance counter until routine completion, then hold athletic stance
        if self._pushup_step_counter < self._pushup_total_steps:
            self._pushup_step_counter += 1

        # Convert degrees to radians
        deg2rad = math.pi / 180.0
        thigh = thigh_deg * deg2rad
        calf = calf_deg * deg2rad

        # Build target joint vector:
        # - Shoulders remain at the default pose
        # - Thighs/Calves receive the per-phase offsets computed above
        target = self._robot.data.default_joint_pos.clone()
        # Apply group direction multipliers (all +1, see __init__) so command signs are uniform
        target[:, 4:8] = target[:, 4:8] + thigh * self._thigh_signs
        target[:, 8:12] = target[:, 8:12] + calf * self._calf_signs

        # Clamp to push-up joint limits
        self._processed_actions = torch.clamp(target, self._JOINT_ANGLE_MIN, self._JOINT_ANGLE_MAX)


    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        obs = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        zeros = torch.zeros(self.num_envs, device=self.device)

        return zeros

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        self._pushup_step_counter = 0

        # Initialize to a fully neutral pose (all joints at 0). Playback now starts immediately.
        self._processed_actions = torch.clamp(self._robot.data.default_joint_pos.clone(),
                                              self._JOINT_ANGLE_MIN,
                                              self._JOINT_ANGLE_MAX)

    def __del__(self):
        pass
