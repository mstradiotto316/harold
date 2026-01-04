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
        
        # --- Direction multipliers per joint group ---
        # Normalize command space so a positive value means a positive rotation at every joint.
        # If a future robot needs flips, update these tensors only (the playback assumes +1 signs).
        self._thigh_signs = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        self._calf_signs = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)
        
        # Commands removed for push-up playback (no velocity tracking)
        self._commands = None

        self._JOINT_ANGLE_MAX = torch.tensor(self.cfg.joint_angle_max, device=self.device)
        self._JOINT_ANGLE_MIN = torch.tensor(self.cfg.joint_angle_min, device=self.device)

        # --- Push-up playback parameters (match firmware timing at 20 Hz) ---
        # Replicate Arduino timing exactly:
        # 1) Neutral hold: ~1.5s (servos reach zero)
        # 2) Pushup start pose hold: ~0.8s
        # 3) Reps: down 1.2s, pause 0.3s, up 1.2s, pause 0.3s, repeated 5 times
        self._neutral_settle_steps = 30      # ~1.5s at 20 Hz (1.5 * 20)
        self._pushup_start_settle_steps = 16     # ~0.8s at 20 Hz (0.8 * 20)
        self._pushup_steps_phase = 24        # 1.2s down or up phase at 20 Hz (1.2 * 20)
        self._pushup_pause_steps = 6         # ~0.3s pause at top/bottom at 20 Hz (0.3 * 20)
        self._pushup_reps = 5
        self._pushup_cycle_steps = (
            self._pushup_steps_phase + self._pushup_pause_steps + self._pushup_steps_phase + self._pushup_pause_steps
        )
        self._pushup_total_steps = (
            self._neutral_settle_steps + self._pushup_start_settle_steps + self._pushup_reps * self._pushup_cycle_steps
        )
        self._pushup_step_counter = 0
        self._physics_step_count = 0

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

        # Ignore incoming policy actions; playback is scripted.
        # We only update the trajectory at control rate inside _apply_action.
        self._actions.zero_()

    def _advance_pushup_and_update_target(self) -> None:

        # Determine current phase of the push-up routine
        step = int(self._pushup_step_counter)

        # Push-up angle keyframes (degrees) with unified +1 joint mapping.
        # Equivalent to Arduino pushup start pose but inverted signs to match +1 mapping:
        # - Top (pushup start pose): thighs +10°, calves -20°
        # - Bottom:               thighs +45° (delta +35°), calves -90° (delta -70°)
        # Calf moves exactly 2x the thigh delta in the opposite direction to counter thigh rotation.
        thigh_deg_top, calf_deg_top = 10.0, -20.0
        thigh_deg_bottom = 45.0  # calf_deg_bottom (-90°) is calculated dynamically below

        # Before reps begin, replicate Arduino holds:
        # a) Hold fully neutral for ~1.5s
        if step < self._neutral_settle_steps:
            target = self._robot.data.default_joint_pos.clone()
            self._processed_actions = torch.clamp(target, self._JOINT_ANGLE_MIN, self._JOINT_ANGLE_MAX)
            if self._pushup_step_counter < self._pushup_total_steps:
                self._pushup_step_counter += 1
            return

        # b) Hold pushup start pose for ~0.8s
        if step < self._neutral_settle_steps + self._pushup_start_settle_steps:
            thigh_deg = thigh_deg_top
            # Enforce calf = calf_top - 2*(thigh - thigh_top)
            calf_deg = calf_deg_top - 2.0 * (thigh_deg - thigh_deg_top)
            # Convert degrees to radians and apply
            deg2rad = math.pi / 180.0
            thigh = thigh_deg * deg2rad
            calf = calf_deg * deg2rad
            target = self._robot.data.default_joint_pos.clone()
            target[:, 4:8] = target[:, 4:8] + thigh * self._thigh_signs
            target[:, 8:12] = target[:, 8:12] + calf * self._calf_signs
            self._processed_actions = torch.clamp(target, self._JOINT_ANGLE_MIN, self._JOINT_ANGLE_MAX)
            if self._pushup_step_counter < self._pushup_total_steps:
                self._pushup_step_counter += 1
            return

        # After holds, compute step within the repetition cycle
        local = (step - self._neutral_settle_steps - self._pushup_start_settle_steps) % self._pushup_cycle_steps

        # Down phase [0, steps_phase): interpolate thigh from top to bottom; derive calf as 2x compensation
        if local < self._pushup_steps_phase:
            t = float(local) / float(self._pushup_steps_phase)
            thigh_deg = thigh_deg_top + (thigh_deg_bottom - thigh_deg_top) * t
            calf_deg = calf_deg_top - 2.0 * (thigh_deg - thigh_deg_top)
        # Bottom pause [steps_phase, steps_phase+pause): hold bottom
        elif local < self._pushup_steps_phase + self._pushup_pause_steps:
            thigh_deg = thigh_deg_bottom
            calf_deg = calf_deg_top - 2.0 * (thigh_deg - thigh_deg_top)
        # Up phase [steps_phase+pause, 2*steps_phase+pause): interpolate back to top (pushup start pose)
        elif local < self._pushup_steps_phase + self._pushup_pause_steps + self._pushup_steps_phase:
            up_local = local - (self._pushup_steps_phase + self._pushup_pause_steps)
            t = float(up_local) / float(self._pushup_steps_phase)
            thigh_deg = thigh_deg_bottom + (thigh_deg_top - thigh_deg_bottom) * t
            calf_deg = calf_deg_top - 2.0 * (thigh_deg - thigh_deg_top)
        # Top pause: hold pushup start pose
        else:
            thigh_deg = thigh_deg_top
            calf_deg = calf_deg_top

        # Advance counter until routine completion, then hold pushup start pose
        if self._pushup_step_counter < self._pushup_total_steps:
            self._pushup_step_counter += 1
        else:
            # Hold pushup start pose indefinitely after finishing reps
            thigh_deg = thigh_deg_top
            calf_deg = calf_deg_top - 2.0 * (thigh_deg - thigh_deg_top)
            deg2rad = math.pi / 180.0
            thigh = thigh_deg * deg2rad
            calf = calf_deg * deg2rad
            target = self._robot.data.default_joint_pos.clone()
            target[:, 4:8] = target[:, 4:8] + thigh * self._thigh_signs
            target[:, 8:12] = target[:, 8:12] + calf * self._calf_signs
            self._processed_actions = torch.clamp(target, self._JOINT_ANGLE_MIN, self._JOINT_ANGLE_MAX)
            return

        # Convert degrees to radians
        deg2rad = math.pi / 180.0
        thigh = thigh_deg * deg2rad
        calf = calf_deg * deg2rad

        # Build target joint vector and apply signs
        target = self._robot.data.default_joint_pos.clone()
        target[:, 4:8] = target[:, 4:8] + thigh * self._thigh_signs
        target[:, 8:12] = target[:, 8:12] + calf * self._calf_signs

        # Clamp to push-up joint limits
        self._processed_actions = torch.clamp(target, self._JOINT_ANGLE_MIN, self._JOINT_ANGLE_MAX)


    def _apply_action(self) -> None:
        # Advance the push-up routine at control rate (20 Hz)
        if (self._physics_step_count % self.cfg.decimation) == 0:
            self._advance_pushup_and_update_target()
            #print(f"Target: {self._processed_actions}")
        self._physics_step_count += 1

        # Send joint targets to robot every physics step
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
        self._physics_step_count = 0

        # Initialize to a fully neutral pose (all joints at 0). Playback now starts immediately.
        self._processed_actions = torch.clamp(self._robot.data.default_joint_pos.clone(),
                                              self._JOINT_ANGLE_MIN,
                                              self._JOINT_ANGLE_MAX)

    def __del__(self):
        pass
