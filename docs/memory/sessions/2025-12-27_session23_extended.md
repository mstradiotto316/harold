# Session 23 Extended: Comprehensive Parameter Optimization

**Date**: 2025-12-27
**Duration**: ~6 hours
**Experiments**: 15+

## Key Findings

### 1. Stiffness Sweep (Optimal: 600)

| Stiffness | Height | vx (final) | vx (peak) | Verdict |
|-----------|--------|------------|-----------|---------|
| 400 | 0.74 | +0.024 | +0.024 | FAILING |
| **600** | **1.54** | **+0.027** | **+0.047** | **STANDING** |
| 800 | 1.66 | +0.012 | +0.041 | STANDING |
| 1000 | 1.93 | -0.012 | +0.058 | STANDING |

**Conclusion**: Stiffness=600 is optimal. Higher stiffness = better height but worse final vx (more regression).

### 2. Training Length Sweep (Optimal: 500-625 iterations)

| Iterations | Height | vx | Verdict |
|------------|--------|------|---------|
| 312 | 0.81 | 0.019 | FAILING |
| **500** | **1.61** | **0.051** | **STANDING** |
| 625 | 1.56 | 0.051 | STANDING |
| 750 | 1.36 | 0.045 | STANDING |
| 1250 | 1.54 | 0.027 | STANDING |

**Conclusion**: 500 iterations is optimal. Longer training causes regression.

### 3. Variance Check

Two 500-iteration runs achieved vx=0.051 and vx=0.050 - highly consistent.

### 4. Parameter Adjustments (All Failed)

| Change | Result | Verdict |
|--------|--------|---------|
| forward_pos=50 (up from 40) | vx=0.032 | WORSE |
| height_reward=15 (down from 20) | vx=0.012, height=0.57 | MUCH WORSE |
| diagonal_gait=10 (up from 5) | ep_len=61, vx=0.007 | SANITY_FAIL |
| action_scale=0.7 (up from 0.5) | vx=0.029, contact=-0.11 | WORSE |
| 16384 envs (up from 8192) | vx=0.007, height=0.54 | MUCH WORSE |

**Conclusion**: Current configuration is at a local optimum. Further tuning doesn't help.

## Best Configuration (Session 23)

```python
# Actuator (harold.py)
stiffness = 600.0
damping = 45.0
effort_limit = 2.8

# Training
iterations = 500
num_envs = 8192

# Rewards (optimal, don't change)
progress_forward_pos = 40.0
height_reward = 20.0
diagonal_gait_reward = 5.0
backward_motion_penalty = 75.0
```

**Result**: vx=0.051 m/s (51% of 0.1 target), height=1.61, STANDING

## The 0.051 Barrier

Session 23 conclusively shows that vx=0.051 is a ceiling with current:
- Reward structure
- Network architecture
- PPO hyperparameters

To break through this barrier, fundamentally different approaches are needed:
1. Different RL algorithm (SAC, TD3)
2. Curriculum learning (progressive difficulty)
3. Reference motion imitation
4. Policy architecture changes (LSTM for gait memory)

## Regression Pattern Analysis

All experiments show consistent pattern:
- Peak vx at 30-50% of training
- Regression to lower vx by 100%

This suggests PPO is destabilizing learned behaviors during continued training.

## Files Modified

- `harold.py`: stiffness=600 (final)
- `harold_isaac_lab_env_cfg.py`: All rewards reverted to optimal defaults

## Experiments Summary

1. Stiffness 400 - FAILING (vx=0.024)
2. Stiffness 600 (full 1250) - STANDING (vx=0.027)
3. Stiffness 800 - STANDING (vx=0.012)
4. Stiffness 1000 - STANDING (vx=-0.012)
5. 625 iterations - STANDING (vx=0.051)
6. 312 iterations - FAILING (vx=0.019)
7. 500 iterations #1 - STANDING (vx=0.051)
8. 500 iterations #2 - STANDING (vx=0.050)
9. 750 iterations - STANDING (vx=0.045)
10. forward_pos=50 - STANDING (vx=0.032)
11. height_reward=15 - FAILING (vx=0.012)
12. diagonal_gait=10 - SANITY_FAIL (vx=0.007)
13. action_scale=0.7 - FAILING (vx=0.029)
14. 16384 envs - FAILING (vx=0.007)
