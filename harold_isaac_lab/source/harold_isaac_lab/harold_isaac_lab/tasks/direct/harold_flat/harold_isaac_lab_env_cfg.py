from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.utils.noise import GaussianNoiseCfg
import math

from .harold import HAROLD_V4_CFG
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg

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
    """Reward structure for Phase-0: prioritize standing, then walking.

    EXP-036: Height-based termination to prevent elbow exploit.
    EXP-034/035 showed height_reward=25.0 alone was insufficient.
    Now terminating episodes when height < 0.165m (60% of target).
    """

    # Forward motion rewards - EXP-054: Test gait phase observation
    # Session 13 summary:
    # - EXP-050/051: Higher forward reward → SANITY_FAIL
    # - EXP-052/053: Slip factor experiments → no improvement
    # - EXP-054: Add gait phase (sin/cos) to observation space
    # Hypothesis: Phase signal helps policy coordinate leg movements
    progress_forward_pos: float = 40.0   # Session 23: 50 was worse (vx=0.032), reverting to 40
    progress_forward_neg: float = 10.0   # Keep optimal from EXP-032
    standing_penalty: float = -5.0      # EXP-098: Reverted to -5 (sp=-3 slowed down robot)

    # Stability rewards - EXP-032 optimal values
    # EXP-049 showed reducing these causes instability (SANITY_FAIL)
    upright_reward: float = 10.0        # Keep upright incentive
    height_reward: float = 20.0         # Session 23: 15 was MUCH worse (height=0.57, vx=0.012)

    # Penalties
    # Session 35: Increased torque penalty for smoother motion (was -0.005)
    # -0.02 combined with action_rate may be too strong, trying -0.01
    torque_penalty: float = -0.01       # 2x original for smoother motion

    # Session 35: Action rate penalty - penalizes rapid action changes
    # -0.5 was too strong, -0.1 is optimal (vx=0.017 with all smoothing)
    # Tested: disabling gave vx=0.014, so -0.1 helps
    action_rate_penalty: float = -0.1   # Optimal for smoothness + walking
    lat_vel_penalty: float = 12.0       # Penalize sideways skating
    yaw_rate_penalty: float = 1.0       # Dampen gratuitous spinning
    rear_support_bonus: float = 0.0     # Disabled - was encouraging standing

    height_tolerance: float = 0.02      # |height_error| tolerated before penalty (m)
    height_sigma: float = 0.045         # Controls falloff beyond tolerance (m)

    # EXP-011: Strong negative penalty for low height (instead of termination)
    # Punish being below minimum height to create gradient toward standing
    low_height_penalty: float = -50.0   # Strong negative reward
    low_height_threshold: float = 0.20  # Penalize if height < 0.20m (elbow pose ~0.15-0.18m)

    # EXP-044: Air time rewards DISABLED - they made vx worse in EXP-038-043
    # EXP-038: vx=-0.029, EXP-042: vx=-0.025, EXP-043: vx=-0.004 (all worse than baseline 0.029)
    # Hypothesis: Air time reward interferes with velocity reward gradient
    feet_air_time_reward: float = 0.0   # DISABLED - counterproductive
    optimal_air_time: float = 0.25      # Unused when weight=0

    # EXP-055/056/058: Contact-based diagonal gait reward with forward gating
    # Rewards alternating diagonal pair contacts (FL+BR vs FR+BL) when moving forward
    # EXP-055 (ungated): vx=+0.022 (worse - no direction bias)
    # EXP-056 (gated, weight=5.0): vx=+0.036 (BEST final, 24% better than baseline)
    # EXP-058 (gated, weight=10.0): peak vx=0.061 at 43%, regressed to 0.018
    diagonal_gait_reward: float = 5.0   # Session 23: 10 caused SANITY_FAIL, reverted to 5
    gait_phase_tolerance: float = 0.3   # Phase window where contact is rewarded (0-1 scale)

    # EXP-088/089: Bidirectional gait reward (Spot-style)
    # Spot uses both sync AND async components:
    # - Sync: Diagonal pairs (FL+BR, FR+BL) should have matching contact/air times
    # - Async: Same-side legs (FL vs FR, BL vs BR) should be out of phase
    # EXP-088: Multiplicative (sync*async), weight=2.0 → vx=0.0385 (WORSE)
    # EXP-089: Additive (sync+async), weight=5.0 → vx=0.018 (MUCH WORSE)
    # CONCLUSION: Bidirectional gait reward is counterproductive for Harold
    bidirectional_gait: bool = False  # DISABLED - counterproductive
    bidirectional_gait_weight: float = 5.0
    bidirectional_gait_sigma: float = 0.1
    bidirectional_gait_additive: bool = True

    # EXP-069: Explicit backward penalty to break backward drift attractor
    # Session 16 discovered backward drift is a stable optimum - robot consistently learns
    # to drift backward (vx < 0) regardless of forward gating mechanism.
    # This adds a strong explicit penalty for backward motion beyond progress_forward_neg.
    # The penalty is proportional to backward velocity magnitude.
    backward_motion_penalty: float = 75.0  # EXP-102: 75 is optimal (70 and 80 both caused drift)

    # EXP-080: Velocity threshold bonus - DISABLED, made things worse
    # Robot stuck at ~0.056 m/s, bonus didn't help break plateau
    velocity_threshold_bonus: float = 0.0  # DISABLED - counterproductive
    velocity_bonus_threshold: float = 0.04  # Speed threshold (unused)

    # EXP-083/084/085: Exponential velocity tracking (from AnyMal/Spot reference)
    # EXP-083: σ²=0.25 was too large - vx=0.021 (WORSE than baseline)
    # EXP-084: σ²=0.01 gave vx=0.004 (MUCH WORSE)
    # Analysis: Symmetric exp kernel doesn't incentivize forward motion well
    # EXP-085: DISABLED - return to linear rewards, try bidirectional gait instead
    exp_velocity_tracking: bool = False  # DISABLED - doesn't work for Harold's scale
    exp_velocity_weight: float = 5.0     # Weight for exp tracking reward
    exp_velocity_sigma_sq: float = 0.01  # Variance parameter (σ²)
    exp_velocity_target: float = 0.1     # Target forward velocity (m/s)

    # Command tracking reward - Phase 1 of controllability
    # Unlike exp_velocity_tracking (fixed target), this tracks COMMANDED velocity
    # Uses self._commands[:, 0] as the target, allowing variable speed control
    # Reward: exp(-|vx - cmd_vx|^2 / sigma^2) * weight
    command_tracking_enabled: bool = False  # Enable via this flag
    command_tracking_weight: float = 10.0   # Reward weight (20 was worse in EXP-137: vx=0.005)
    command_tracking_sigma: float = 0.1     # Velocity error tolerance (EXP-138: 0.15 was too permissive, vx=0.003)
    # Replace forward motion bias: when enabled, disable progress_forward rewards
    # and use pure command tracking instead
    command_tracking_replace_forward: bool = True

    # Lateral (vy) command tracking - Session 27: Extend controllability to strafing
    # Uses same exponential kernel as vx tracking: exp(-|vy - cmd_vy|^2 / sigma^2) * weight
    command_tracking_weight_vy: float = 10.0   # Same weight as vx tracking
    command_tracking_sigma_vy: float = 0.1     # Same tolerance as vx tracking

    # Yaw rate command tracking - Session 28: Extend controllability to turning
    # Uses same exponential kernel: exp(-|yaw_rate - cmd_yaw|^2 / sigma^2) * weight
    command_tracking_weight_yaw: float = 10.0   # Same weight as vx/vy tracking
    command_tracking_sigma_yaw: float = 0.2     # Slightly higher tolerance for yaw (rad/s)


@configclass
class CommandCfg:
    """Configuration for velocity command sampling.

    Phase 2 of controllability: Variable command ranges for speed control.
    Commands are sampled per-episode at reset, from uniform distributions.

    Enable via: HAROLD_VAR_CMD=1
    """

    # Enable variable command sampling (otherwise uses fixed 0.25-0.35 range)
    variable_commands: bool = False

    # Forward velocity range (m/s)
    # EXP-131: 0-0.5 with 10% zero was too hard (FAILING)
    # EXP-132: 0.15-0.4 with 5% zero worked
    # EXP-139: 0.15-0.4 with 0% zero achieved WALKING
    # EXP-140: 0.10-0.45 achieved WALKING (vx=0.015) - OPTIMAL
    # EXP-141: 0.05-0.50 achieved WALKING (vx=0.012) - slightly worse
    vx_min: float = 0.10
    vx_max: float = 0.45

    # Lateral velocity range (m/s) - Session 27: Enabled for strafing
    # Conservative range to start (smaller than vx range)
    # Now supported with vy tracking reward and modified lat_vel_penalty
    vy_min: float = -0.15
    vy_max: float = 0.15

    # Yaw rate range (rad/s) - Session 28: Enabled for turning
    # Conservative range to start (about ±17 deg/s)
    # Now supported with yaw tracking reward and modified yaw_rate_penalty
    yaw_min: float = -0.30
    yaw_max: float = 0.30

    # Probability of sampling zero velocity (for stopping behavior)
    # When triggered, sets vx=vy=yaw=0 regardless of ranges
    # EXP-131: 10% was too much; EXP-139: try 0% (always forward)
    zero_velocity_prob: float = 0.0

    # Phase 3: Dynamic command updates during episode
    # When enabled, commands change periodically instead of only at reset
    # Enable via: HAROLD_DYN_CMD=1
    dynamic_commands: bool = False

    # Time between command changes (seconds)
    # At 20 Hz policy rate, 10 seconds = 200 policy steps
    # EXP-134: 5s was too short, robot achieved vx=0.007 (STANDING)
    command_change_interval: float = 10.0

    # Probability of command change at each interval
    # 1.0 = always change, 0.5 = 50% chance to change
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

    Session 34: BACKLASH-TOLERANT UPDATE - Hardware testing (Session 33) revealed
    ~30° servo backlash. Old amplitude (26°) was absorbed. New amplitude (50°)
    should exceed backlash with margin for actual foot lift.

    Use HAROLD_SCRIPTED_GAIT=1 to enable for physics validation.
    """

    # Master switch - enable via env var HAROLD_SCRIPTED_GAIT=1
    enabled: bool = False

    # Gait parameters - Session 34: Backlash-tolerant amplitudes (aligned with CPGCfg)
    # Session 35: Optimal frequency 0.7 Hz (0.5 Hz was worse)
    frequency: float = 0.7  # Hz - Optimal from sweeps
    swing_thigh: float = 0.25    # Session 34: More back during swing (was 0.40)
    stance_thigh: float = 0.95   # Session 34: More forward during stance (was 0.90)
    stance_calf: float = -0.50   # Session 34: More extended stance (was -0.90)
    swing_calf: float = -1.38    # Session 34: Close to limit (was -1.35)
    shoulder_amplitude: float = 0.05  # Unchanged
    duty_cycle: float = 0.6


@configclass
class CPGCfg:
    """Central Pattern Generator configuration for residual learning.

    Session 24: NOW ALIGNED with ScriptedGaitCfg - uses the same duty-cycle based
    trajectory that achieved vx=+0.141 m/s in sim and walks on real hardware.

    Session 34: BACKLASH-TOLERANT UPDATE - Hardware testing (Session 33) revealed
    ~30° servo backlash absorbs direction reversals. Old calf amplitude (26°) was
    entirely absorbed by backlash, causing feet to never lift.

    New design principle: All joint motions > 45° to exceed backlash zone.

    Architecture:
        target_joints = CPG_base_trajectory + policy_output * residual_scale

    When policy_output = 0, robot follows the proven scripted gait trajectory.
    Policy learns to provide balance corrections while CPG provides walking structure.

    Enable via: HAROLD_CPG=1
    """

    # Enable CPG-based action space
    enabled: bool = False  # Set True via env var HAROLD_CPG=1

    # Base gait parameters
    # Session 35: 0.5 Hz was worse (vx=0.009), reverting to optimal 0.7 Hz
    base_frequency: float = 0.7  # Hz - Optimal from previous sweeps
    duty_cycle: float = 0.6      # 60% stance, 40% swing

    # Trajectory parameters - SESSION 34: BACKLASH-TOLERANT AMPLITUDES
    # Hardware has ~30° backlash on direction reversal. All motions must exceed this.
    # Old calf: -1.35 to -0.90 = 26° (absorbed by backlash)
    # New calf: -1.38 to -0.50 = 50° (exceeds backlash with 20° margin)
    swing_thigh: float = 0.25     # Session 34: More back during swing (was 0.40)
    stance_thigh: float = 0.95    # Session 34: More forward during stance (was 0.90, close to 0.9599 limit)
    stance_calf: float = -0.50    # Session 34: More extended during stance (was -0.90)
    swing_calf: float = -1.38     # Session 34: Close to limit -1.3963 (was -1.35)
    shoulder_amplitude: float = 0.05  # Unchanged - shoulders have less backlash

    # Residual scaling - how much the policy can adjust the CPG trajectory
    # EXP-126: 0.15 allowed policy to reverse CPG motion (vx=-0.018)
    # Lowering to 0.05 to limit policy authority - can only fine-tune, not override
    residual_scale: float = 0.05  # EXP-166: 0.08 regressed, 0.05 is optimal


@configclass
class TerminationCfg:
    """Episode termination thresholds (shared with rough terrain)."""

    base_contact_force_threshold: float = math.inf
    undesired_contact_force_threshold: float = math.inf
    orientation_threshold: float = -0.5
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
class DomainRandomizationCfg:
    """Domain randomization configuration for sim-to-real transfer.
    
    To enable domain randomization in training, set:
    ```python
    cfg.domain_randomization.enable_randomization = True
    cfg.domain_randomization.randomize_on_reset = True  # Per-episode randomization
    cfg.domain_randomization.randomize_per_step = True   # Per-step noise
    ```
    
    Provides controlled variation in simulation parameters to improve policy
    robustness when deployed on physical hardware. Parameters are tuned
    specifically for Harold's lightweight construction and servo capabilities.
    
    Randomization Categories:
    - Physics: Material properties affecting contact dynamics
    - Robot: Mass, inertia, and actuator characteristics
    - Sensors: Observation noise modeling real sensor imperfections
    - Actions: Control delays and noise
    - External: Environmental disturbances
    
    All randomization can be toggled on/off for debugging and gradual
    introduction during training curriculum.
    """
    
    # === MASTER SWITCHES ===
    # EXP-090: FULL domain randomization made training HARDER - vx=0.0056 (MUCH WORSE)
    # Robot learned to stand still to cope with uncertainty
    # Session 28: Re-enabled for SENSOR NOISE ONLY to simulate gear backlash (~2°)
    # All physics randomization flags remain False - only sensor noise is active
    enable_randomization: bool = True   # Session 28: OPTIMAL for backlash robustness
    randomize_on_reset: bool = False
    randomize_per_step: bool = True     # Session 28: Per-step noise for backlash
    
    # === PHYSICS RANDOMIZATION ===
    # EXP-090: Domain randomization made training WORSE - DISABLED
    randomize_friction: bool = False  # DISABLED - caused vx=0.0056 (robot stood still)
    friction_range: tuple = (0.5, 0.9)        # Conservative range (was 0.4-1.0)
                                              # Base: 0.7, slight variation

    randomize_restitution: bool = False       # Randomize bounce characteristics
    restitution_range: tuple = (0.0, 0.2)     # Keep low for realistic ground contact

    # === ROBOT PROPERTIES RANDOMIZATION ===
    # EXP-090: Domain randomization made training WORSE - DISABLED
    randomize_mass: bool = False              # DISABLED - caused robot to stand still
    mass_range: tuple = (0.9, 1.1)            # ±10% mass variation (conservative)
                                              # Was ±15%, reduced for stability
    
    randomize_com: bool = False               # Randomize center of mass offsets
    com_offset_range: tuple = (-0.02, 0.02)   # ±2cm COM shift in X/Y/Z
                                              # Small shifts for balance variation
    
    randomize_inertia: bool = False           # Randomize rotational inertia
    inertia_range: tuple = (0.9, 1.1)         # ±10% inertia variation
    
    # === ACTUATOR RANDOMIZATION ===
    randomize_joint_stiffness: bool = False   # Vary joint PD controller stiffness
    stiffness_range: tuple = (150, 250)       # Base: 200, allows ±25% variation
                                              # Models servo response differences
    
    randomize_joint_damping: bool = False     # Vary joint PD controller damping
    damping_range: tuple = (50, 100)          # Base: 75, allows ±33% variation
                                              # Models servo damping characteristics
    
    randomize_effort_limit: bool = False      # Vary maximum joint torques
    effort_limit_range: tuple = (0.8, 1.0)    # 80-100% of nominal torque
                                              # Conservative to prevent servo damage
    
    randomize_joint_limits: bool = False      # Add small variations to joint limits
    joint_limit_noise: float = 0.02           # ±0.02 rad (~1.15°) variation
    
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
    # - 2° (0.035 rad): STANDING, vx=0.007 - too much noise
    # - 1° (0.0175 rad): WALKING, vx=0.022 - OPTIMAL (31% better than baseline!)
    # The noise acts as regularization, improving generalization
    add_joint_noise: bool = True              # Add noise to joint measurements
    joint_position_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.0175,                           # Session 28: 1° optimal for backlash robustness
        operation="add"
    )
    joint_velocity_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.05,                             # 0.05 rad/s velocity noise
        operation="add"
    )
    
    # === ACTION RANDOMIZATION ===
    # Session 29: Testing action noise for sim-to-real transfer
    # EXP-154: 0.5% (0.005) -> STANDING (vx=0.009), hurt training
    # EXP-155: 0.2% (0.002) -> STANDING (vx=0.009), still hurt training
    # CONCLUSION: Action noise hurts learning, disabled
    add_action_noise: bool = False             # DISABLED - hurts learning
    action_noise: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0,
        std=0.002,                            # Not used when disabled
        operation="add"
    )

    # EXP-156: Action delays hurt training -> DISABLED
    # Adding any action-side randomization hurts when observation noise already present
    add_action_delay: bool = False            # DISABLED - hurts learning
    action_delay_steps: tuple = (0, 1)        # Not used when disabled

    # === EXTERNAL DISTURBANCES ===
    # EXP-164: External perturbations FAILED - caused falling and backward drift
    # Even light forces (0.2-0.5N, 0.5% prob) broke training
    apply_external_forces: bool = False       # DISABLED - causes instability
    external_force_probability: float = 0.005 # 0.5% per step (conservative)
    external_force_range: tuple = (0.2, 0.5)  # Light forces (0.2-0.5 N)
    external_torque_range: tuple = (0.02, 0.1) # Light torques (0.02-0.1 Nm)
    
    # === TERRAIN RANDOMIZATION ===
    add_terrain_noise: bool = False           # Add height noise to terrain
    terrain_noise_magnitude: float = 0.01     # ±1cm height variations
    
    # === GRAVITY RANDOMIZATION ===
    randomize_gravity: bool = False               # Vary gravity magnitude and direction
    gravity_magnitude_range: tuple = (9.6, 10.0)  # 9.6-10.0 m/s² (small variation)
    gravity_angle_range: float = 0.05             # ±0.05 rad (~3°) tilt in gravity vector

@configclass
class HaroldIsaacLabEnvCfg(DirectRLEnvCfg):
    # env parameters
    episode_length_s = 30.0
    decimation = 9
    action_scale = 0.5  # Session 23: 0.7 was worse (vx=0.029, contact failing)

    # Space definitions
    # EXP-054: Added 2D gait phase (sin/cos) to help policy coordinate leg movements
    observation_space = 50  # Was 48, now includes gait phase
    action_space = 12
    state_space = 0

    # Action filtering (EMA low-pass)
    # Session 35: beta=0.40 is optimal (0.50 prevented walking)
    # Lower beta = more smoothing (60% carryover from previous action)
    action_filter_beta: float = 0.40

    # Observation clipping (matches deployment clip_obs=5.0)
    # Session 29: Hardware deployment clips normalized obs to ±5.0
    # Training without clipping causes policy to see larger ranges than deployment
    clip_observations: bool = True
    clip_observations_value: float = 5.0      # Match deployment clipping

    # Reward configuration
    rewards = RewardsCfg()

    # Gait configuration
    gait = GaitCfg()

    # Scripted gait configuration (Phase 1 - FAILED, kept for reference)
    scripted_gait = ScriptedGaitCfg()

    # CPG configuration (Phase 2 - structured action space)
    cpg = CPGCfg()

    # Command configuration (Phase 2 - variable velocity commands)
    commands = CommandCfg()

    # Termination configuration
    termination = TerminationCfg()

    # Domain randomization configuration
    domain_randomization = DomainRandomizationCfg()

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
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.15, 0.2, 0.25),  # Dark blue-gray for contrast with white robot
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0.05,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
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
    joint_range: tuple = (
        0.30, 0.30, 0.30, 0.30,
        0.90, 0.90, 0.90, 0.90,
        0.90, 0.90, 0.90, 0.90,
    )

    # === Joint configuration for push-up routine ===
    # Absolute joint angle limits in radians
    # Joint limits (Session 30: aligned with hardware safe limits)
    # Sign convention: thighs/calves are inverted between sim and hardware
    # Hardware limits from deployment/config/hardware.yaml
    # thigh hardware: [-55°, +5°] → sim: [-5°, +55°] = [-0.0873, +0.9599] rad
    # calf hardware: [-5°, +80°] → sim: [-80°, +5°] = [-1.3963, +0.0873] rad
    joint_angle_max: tuple = (
        0.4363, 0.4363, 0.4363, 0.4363,  # shoulders: ±25° (hardware safe limit)
        0.9599, 0.9599, 0.9599, 0.9599,  # thighs: sim +55° (hw -55°)
        0.0873, 0.0873, 0.0873, 0.0873   # calves: sim +5° (hw -5°)
    )
    joint_angle_min: tuple = (
        -0.4363, -0.4363, -0.4363, -0.4363,  # shoulders: ±25°
        -0.0873, -0.0873, -0.0873, -0.0873,  # thighs: sim -5° (hw +5°)
        -1.3963, -1.3963, -1.3963, -1.3963   # calves: sim -80° (hw +80°)
    )
