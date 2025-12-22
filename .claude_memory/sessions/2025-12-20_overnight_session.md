# Overnight Session: 2025-12-20 (00:44 - 02:50)

## Goal
Run autonomous experiments to achieve forward walking from stable standing.

## Experiments Conducted

### EXP-006: High Upright Reward (15.0)
- **Config**: upright=15, forward=100
- **Result**: FAILED - Robot fell by step 8000

### EXP-007: Aggressive Termination
- **Config**: termination_threshold=-0.7 (vs -0.5)
- **Result**: FAILED - Made learning harder, not better

### EXP-008: Fine-tune terrain_62 + Low Forward
- **Config**: Checkpoint + forward=30, upright=20
- **Result**: FAILED - Initial stability degraded by step 8000

### EXP-009: Gait Rewards + Very Low Forward
- **Config**: forward=10, feet_air_time=2, diagonal=3, leg_move=1.5
- **Result**: FAILED - Same degradation pattern

### EXP-010: Stability Only (Zero Forward)
- **Config**: upright=25, height=8, forward=0
- **Result**: PARTIAL SUCCESS - Standing at 5k-10k, fell by 15k

### EXP-011: Stability + Aggressive Termination
- **Config**: Same as EXP-010 + termination=-0.7
- **Result**: FAILED - Termination made learning harder

### EXP-012: Stability + Frequent Checkpoints
- **Config**: upright=25, height=8, forward=0, checkpoint_interval=2500
- **Result**: **SUCCESS** - Stable standing achieved!
- **Best Checkpoint**: agent_5000.pt shows clear standing

### EXP-013: Fine-tune agent_5000 + Minimal Forward
- **Config**: Checkpoint + forward=3.0
- **Result**: FAILED - Even forward=3.0 destabilized standing

### EXP-014: Leg Movement Only (No Forward)
- **Config**: forward=0, leg_movement=2.0, feet_air_time=1.0
- **Result**: FAILED - Gait rewards also destabilize standing

## Key Discoveries

### What Works
- **Stability-only training** produces stable standing (EXP-012)
- Robot CAN learn to stand from scratch with proper rewards

### What Doesn't Work
- ANY forward velocity reward (100, 30, 10, 3, 0) destabilizes
- Gait/leg movement rewards ALSO destabilize
- Aggressive termination makes learning harder
- Fine-tuning from standing checkpoint doesn't preserve stability

### The Core Problem
The standing equilibrium is EXTREMELY fragile. Any gradient toward motion eventually causes falls. The standing behavior is a local minimum that exploration pushes the policy away from.

## Artifacts Created
- **Best checkpoint**: `logs/skrl/harold_direct/2025-12-20_02-18-43_ppo_torch/checkpoints/agent_5000.pt`
- **Experiment logs**: 9 training runs with videos at 2500-step intervals
- **Memory updates**: EXPERIMENTS.md, OBSERVATIONS.md, NEXT_STEPS.md

## Recommended Next Steps
1. **Extended stability training** (100k+ iterations) to build robust standing
2. **Micro-curriculum** adding forward=0.1 increments every 10k steps
3. **Reference motion tracking** (imitation learning) as alternative approach
4. **Robot configuration review** (joint limits, PD gains, CoM)

## Session Duration
~2 hours of autonomous experimentation
