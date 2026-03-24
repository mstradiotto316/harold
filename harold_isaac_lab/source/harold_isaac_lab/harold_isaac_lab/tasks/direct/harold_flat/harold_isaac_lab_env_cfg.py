import math
import sys
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg

from .harold import HAROLD_V4_CFG

_REPO_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "AGENTS.md").exists():
        _REPO_ROOT = _parent
        break
if _REPO_ROOT is None:
    _parents = list(Path(__file__).resolve().parents)
    if len(_parents) > 8:
        _REPO_ROOT = _parents[8]
if _REPO_ROOT and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.policy_config import (
    DEFAULT_ACTION_SCALE,
    FLAT_JOINT_ANGLE_MAX,
    FLAT_JOINT_ANGLE_MIN,
    JOINT_RANGE,
)

# Flat terrain configuration for Harold's locomotion training
HAROLD_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.001,
    slope_threshold=0.0,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=1.0, size=(1.0, 1.0))
    },
    curriculum=False,
    color_scheme="height",
)


@configclass
class RewardsCfg:
    """Spot-aligned reward structure (Session 54).

    No existence rewards (upright, stance_height). Only penalize bad states,
    reward locomotion. Standing earns ~4.2/step, walking earns ~13.0/step.

    Reference: IsaacLab Spot flat config (flat_env_cfg.py, mdp/rewards.py)
    """

    # === TASK REWARDS (velocity tracking) ===
    track_lin_vel_xy_weight: float = 5.0      # Spot: 5.0
    track_lin_vel_xy_std: float = 0.15        # Harold-specific (Spot: 1.0, but Harold's cmd range is 10x smaller)

    track_ang_vel_z_weight: float = 2.0       # Harold-specific (Spot: 5.0, but Harold's yaw range is smaller)
    track_ang_vel_z_std: float = 0.25         # Harold-specific

    # === BASE QUALITY PENALTIES (Spot-aligned, replace existence rewards) ===
    base_orientation_weight: float = 3.0      # Spot: 3.0. Penalizes tilting (replaces +3.0 upright REWARD)
    base_motion_weight: float = 2.0           # Spot: 2.0. Combined vz + omega_xy (replaces -0.0001 lin_vel_z + -0.01 ang_vel_xy)

    # === SMOOTHNESS PENALTIES (Spot-aligned) ===
    dof_torques_weight: float = -5e-4         # Spot: -5e-4. Was -0.0001 (5x increase)
    dof_acc_weight: float = -2.5e-7           # Kept (negligible)
    action_smoothness_weight: float = 1.0     # Spot: 1.0. L2 norm of action diff (replaces -0.01 sum-of-squares)
    shoulder_joint_vel_weight: float = 0.01   # Spot: 1e-2 on hip joints. Harold shoulders = Spot hips.

    # === GAIT REWARDS (Spot-aligned weights) ===
    feet_air_time_weight: float = 5.0         # Spot: 5.0. Was 2.0
    feet_air_time_threshold: float = 0.3      # Spot: 0.3
    continuous_gait_weight: float = 10.0      # Spot: 10.0. Was hardcoded 5.0
    air_time_variance_weight: float = 1.0     # Spot: 1.0. Was hardcoded 0.5

    # === CONTACT PENALTIES ===
    undesired_contacts_weight: float = -1.0   # Penalize body contact
    undesired_contacts_threshold: float = 1.0 # Force threshold (Newtons)
    foot_slip_weight: float = 0.5             # Spot: 0.5. Was hardcoded 0.1

    # === FORWARD MOTION BOOTSTRAP (Harold-specific, Spot has none) ===
    forward_motion_weight: float = 7.0        # Reduced from 7.0. Bootstrap only.

    # === JOINT POSITION REGULARIZATION (Spot original direction) ===
    # Spot: 5x when standing with NO command (keeps tidy when idle).
    # Safe to use Spot direction now that existence rewards are removed.
    joint_pos_weight: float = 0.7             # Spot: 0.7
    joint_pos_stand_still_scale: float = 5.0  # Spot: 5.0
    joint_pos_velocity_threshold: float = 0.1 # Harold-specific (Spot: 0.5)


@configclass
class CommandCfg:
    """Configuration for velocity command sampling.

    Session 36: Pure RL with conservative velocity ranges.
    Commands are sampled per-episode at reset, from uniform distributions.
    """

    # Enable variable command sampling (True for pure RL)
    variable_commands: bool = True

    # Forward velocity range (m/s)
    # Session 36: Reverted to conservative for stability
    vx_min: float = 0.15
    vx_max: float = 0.3

    # Lateral velocity range (m/s)
    vy_min: float = -0.15
    vy_max: float = 0.15

    # Yaw rate range (rad/s) - about ±17 deg/s
    yaw_min: float = -0.30
    yaw_max: float = 0.30

    # Probability of sampling zero velocity (for stopping behavior)
    zero_velocity_prob: float = 0.02  # 2% standing training

    # Dynamic command updates during episode
    dynamic_commands: bool = False
    command_change_interval: float = 10.0  # seconds
    command_change_prob: float = 1.0


@configclass
class GaitCfg:
    """Gait parameters matched to the rough-terrain setup."""

    frequency: float = 2.0
    target_height: float = 0.275


@configclass
class ScriptedGaitCfg:
    """Configuration for scripted walking gait.

    Session 21 RESULT: SUCCESS - Open-loop scripted gait achieves vx=+0.141 m/s
    (141% of 0.1 m/s target!) with stiffness=1200.

    Session 22 RESULT: SUCCESS - Real robot walks forward with this gait.

    Session 34: BACKLASH-TOLERANT UPDATE - Early estimate suggested ~30° servo
    backlash. New measurement (2026-01-03) shows ~10° dead zone; treat older
    notes as historical context.

    Use HAROLD_SCRIPTED_GAIT=1 to enable for physics validation.
    """

    # Master switch - enable via env var HAROLD_SCRIPTED_GAIT=1
    enabled: bool = False

    # Gait parameters - base values HARDWARE-VALIDATED from Session 36 RPi.
    # Source: firmware/scripted_gait_test_1/scripted_gait_test_1.ino
    #   BASE_STANCE_THIGH = -38.15°, BASE_SWING_THIGH = -30.65° (7.5° range)
    #   BASE_STANCE_CALF = 50°, BASE_SWING_CALF = 80° (30° range)
    frequency: float = 0.4   # Hz - Softer gait (longer cycle reduces impact)
    swing_thigh: float = 0.54    # -30.65° → +30.65° in sim coords
    stance_thigh: float = 0.67   # -38.15° → +38.15° in sim coords
    stance_calf: float = -0.87   # 50° → -50° in sim coords (extended)
    swing_calf: float = -1.3963  # 80° → -80° in sim coords (flexed)
    shoulder_amplitude: float = 0.0096  # 0.55 deg in hardware
    thigh_offset_front: float = 0.05    # Bias front thighs forward (rad)
    thigh_offset_back: float = -0.05    # Bias rear thighs back (rad)
    duty_cycle: float = 0.5  # 50% stance / 50% swing to slow the lift/plant


@configclass
class CPGCfg:
    """Central Pattern Generator configuration (open-loop).

    HARDWARE-VALIDATED from Session 36 RPi - these values produce actual
    walking on the real robot with feet lifting off the ground.

    ALIGNMENT APPROACH (Session 36):
    - CPG base trajectory matches hardware scripted gait
    - Open-loop in sim and deployment (policy ignored in CPG mode)

    Enable via: HAROLD_CPG=1 (open-loop playback)
    """

    # Enable open-loop CPG playback
    enabled: bool = False  # Set True via env var HAROLD_CPG=1

    # Base gait parameters - HARDWARE-VALIDATED
    base_frequency: float = 0.4  # Hz - Softer gait (longer cycle reduces impact)
    duty_cycle: float = 0.5      # 50% stance / 50% swing (used by trajectory)

    # Trajectory parameters - base values HARDWARE-VALIDATED from Session 36 RPi
    # Source: firmware/scripted_gait_test_1/scripted_gait_test_1.ino
    # These produce real walking with feet actually lifting!
    swing_thigh: float = 0.54     # -30.65° in hardware → +30.65° in sim
    stance_thigh: float = 0.67    # -38.15° in hardware → +38.15° in sim
    stance_calf: float = -0.87    # 50° in hardware → -50° in sim (extended)
    swing_calf: float = -1.3963   # 80° in hardware → -80° in sim (flexed)
    shoulder_amplitude: float = 0.0096  # 0.55 deg in hardware
    thigh_offset_front: float = 0.05    # Bias front thighs forward (rad)
    thigh_offset_back: float = -0.05    # Bias rear thighs back (rad)


@configclass
class TerminationCfg:
    """Episode termination thresholds (shared with rough terrain)."""

    base_contact_force_threshold: float = math.inf
    undesired_contact_force_threshold: float = math.inf
    orientation_threshold: float = -0.6
    # Height termination: terminate if base height < threshold
    # EXP-002: 10N contact alone wasn't enough - robot stayed low (height=1.76)
    # EXP-003-007: Height termination has issues - scanner returns bad values
    # EXP-008: Disable height termination, rely on height_reward=30.0 to incentivize
    # Spawn height is ~0.24m, elbow pose is ~0.15-0.18m
    height_threshold: float = 0.0
    # Warmup: skip height termination for first N steps after reset (sensor initialization)
    height_termination_warmup_steps: int = 20
    # Body contact termination: terminate if body/thigh/shoulder contact > threshold (N)
    # EXP-002: 10N kept body contact low (-0.04) but didn't prevent elbow pose
    # EXP-013: Root cause - elbow contact ~5N per point, below 10N threshold = undetected
    # Lowering to 3N should make elbow contact visible to the reward system
    body_contact_threshold: float = 3.0

    # Joint-angle termination: detect elbow pose via front leg joint angles
    # EXP-009: thigh>1.0, calf>-0.8 too loose - robot still found elbow pose (height=1.50)
    # EXP-010: Tighter thresholds still didn't work
    # EXP-011: Disable, use low_height_penalty instead
    elbow_pose_termination: bool = False
    front_thigh_threshold: float = 0.85   # Terminate if front thigh > 0.85 rad
    front_calf_threshold: float = -1.0    # AND front calf > -1.0 rad

@configclass
class BacklashCfg:
    """Explicit backlash hysteresis model for sim-to-real transfer.

    Session 37: Early hardware estimate suggested ~30° servo backlash on
    direction reversals (updated measurement: ~10° as of 2026-01-03).
    Previous approach of adding Gaussian noise (std=0.0175) is INCORRECT -
    backlash is hysteresis, not noise.

    This model tracks "engaged position" where gears are meshed. Commands can
    move within a dead zone without affecting output. Only when command exits
    the zone does output follow (with backlash offset).

    Physical behavior:
    - gap = command - engaged_position
    - if |gap| > half_backlash: output moves toward command
    - else: output stays where it was (motor in dead zone)

    This teaches policy to "overdrive" joints to compensate for dead zone.
    """

    # Enable explicit backlash hysteresis modeling
    # Session 37: Disabled to establish CPG baseline, then curriculum
    enable_backlash: bool = False

    # Backlash magnitude in radians
    # Updated measurement (2026-01-03): ~10° dead zone on direction reversals
    # Start with the measured value; increase only if hardware tests contradict.
    backlash_rad: float = math.radians(10)  # 10 degrees

    # Per-joint backlash (optional - use if joints have different backlash)
    # If None, use backlash_rad for all joints
    per_joint_backlash: tuple | None = None

    # Randomize backlash magnitude per episode for robustness
    randomize_backlash: bool = False
    backlash_range: tuple = (0.85, 1.15)  # ±15% variation around backlash_rad


@configclass
class DomainRandomizationCfg:
    """Domain randomization configuration for sim-to-real transfer.

    Active features: per-step sensor noise (IMU, joint, lin_vel) and
    reset state randomization (Spot-style, Session 55).

    Dead features removed in Session 55 cleanup:
    - Physics/robot property randomization (EXP-090: made training worse)
    - Action noise/delays (EXP-154/155/156: hurt learning)
    - External forces (EXP-164: caused instability)
    - Terrain/gravity randomization (never enabled)
    """

    # === MASTER SWITCHES ===
    enable_randomization: bool = True   # Session 28: OPTIMAL for backlash robustness
    randomize_per_step: bool = True     # Session 28: Per-step noise for backlash

    # === SENSOR NOISE CONFIGURATION ===
    # IMU Noise (Body angular velocity and gravity projection)
    add_imu_noise: bool = True                # Add noise to IMU measurements
    imu_angular_velocity_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.01,                             # 0.01 rad/s noise (realistic for MPU6050)
        operation="add"
    )
    imu_gravity_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.05,                             # Small noise for gravity vector
        operation="add"
    )

    # Linear velocity noise (simulates IMU accelerometer integration noise)
    # Session 29: Hardware testing revealed lin_vel is computed via accelerometer
    # integration with 0.95 decay, resulting in noisy/drifting values
    add_lin_vel_noise: bool = True
    lin_vel_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.05,                             # 0.05 m/s noise (hardware shows ~5cm/s drift)
        operation="add"
    )
    # Velocity bias that persists per-episode (simulates calibration error)
    lin_vel_bias_std: float = 0.02            # ±2cm/s per-episode bias

    # Joint Sensor Noise
    # Session 28: Position noise simulates gear backlash (~1-3° in ST3215 servos)
    # - 1° (0.0175 rad): WALKING, vx=0.022 - OPTIMAL (31% better than baseline!)
    add_joint_noise: bool = True              # Session 28: OPTIMAL for backlash robustness
    joint_position_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.0175,
        operation="add"
    )
    joint_velocity_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.05,                             # 0.05 rad/s velocity noise
        operation="add"
    )

    # === RESET STATE RANDOMIZATION (Spot-style, Session 55) ===
    # Robot starts each episode with randomized state instead of all-zeros.
    # Forces the policy to learn locomotion from diverse initial conditions.
    enable_reset_randomization: bool = False

    # Root velocity at reset (m/s, rad/s) — Spot: ±1.5, ±1.0, ±0.5
    # Harold ranges are ~10% of Spot (proportional to speed capability)
    reset_lin_vel_x_range: tuple = (-0.15, 0.15)
    reset_lin_vel_y_range: tuple = (-0.1, 0.1)
    reset_lin_vel_z_range: tuple = (-0.05, 0.05)
    reset_ang_vel_roll_range: tuple = (-0.2, 0.2)
    reset_ang_vel_pitch_range: tuple = (-0.2, 0.2)
    reset_ang_vel_yaw_range: tuple = (-0.3, 0.3)

    # Joint state at reset — Spot: ±0.2 rad pos, ±2.5 rad/s vel
    reset_joint_pos_noise: float = 0.1    # ±rad around ready_pose
    reset_joint_vel_noise: float = 1.0    # ±rad/s

    # Mid-episode velocity pushes — Spot: every 10-15s, ±0.5 m/s
    enable_velocity_pushes: bool = False
    push_interval_range: tuple = (8.0, 12.0)   # seconds
    push_vel_xy_range: float = 0.15             # ±m/s

@configclass
class HaroldIsaacLabEnvCfg(DirectRLEnvCfg):
    # env parameters
    episode_length_s = 30.0
    decimation = 9
    action_scale = 0.5  # Must be literal for autoresearch.py regex rewriting. See common/policy_config.py for canonical default.

    # Space definitions
    # Observation space is always 48D; CPG is open-loop and does not affect policy input size.
    observation_space = 48
    action_space = 12
    state_space = 0

    # Action filtering (EMA low-pass)
    # Session 35: beta=0.40 is optimal (0.50 prevented walking)
    # Lower beta = more smoothing (60% carryover from previous action)
    action_filter_beta: float = 0.2

    # Reward configuration
    rewards = RewardsCfg()

    # Gait configuration
    gait = GaitCfg()

    # Scripted gait configuration (Phase 1 - FAILED, kept for reference)
    scripted_gait = ScriptedGaitCfg()

    # CPG configuration (open-loop playback)
    cpg = CPGCfg()

    # Command configuration (Phase 2 - variable velocity commands)
    commands = CommandCfg()

    # Termination configuration
    termination = TerminationCfg()

    # Domain randomization configuration
    domain_randomization = DomainRandomizationCfg()

    # Backlash hysteresis configuration (Session 37)
    backlash = BacklashCfg()

    # viewer configuration - follow single robot for clear screenshots
    viewer = ViewerCfg(
        eye=(1.0, 0.75, 0.45),    # Close-up view, elevated
        lookat=(0.0, 0.0, 0.12),  # Look at robot center
        origin_type="asset_root", # Follow robot's root position
        asset_name="robot",       # Matches prim_path ".../Robot"
        env_index=0,              # Track environment 0
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt= 1 / 180, #dt= 1 / 360,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.7,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
    )

    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HAROLD_FLAT_TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0.05,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.25, 0.25)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        drift_range=(0.0, 0.0),
    )

    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HAROLD_V4_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    harold_dof_names = [
        "fl_shoulder_joint", 
        "fr_shoulder_joint", 
        "bl_shoulder_joint", 
        "br_shoulder_joint",

        "fl_thigh_joint", 
        "fr_thigh_joint", 
        "bl_thigh_joint", 
        "br_thigh_joint",
        
        "fl_calf_joint", 
        "fr_calf_joint", 
        "bl_calf_joint", 
        "br_calf_joint"
    ]

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,            # 5ms update rate (much higher frequency than 0.05s)
        track_air_time=True             # Enabled for gait-based rewards (EXP-038)
    )

    # === Joint configuration (moved from env implementation) ===
    # Per-joint normalized action ranges (scaled later by action_scale)
    # Order: [shoulders(4), thighs(4), calves(4)]
    joint_range: tuple = JOINT_RANGE

    # Absolute joint angle limits in radians.
    joint_angle_max: tuple = FLAT_JOINT_ANGLE_MAX
    joint_angle_min: tuple = FLAT_JOINT_ANGLE_MIN
