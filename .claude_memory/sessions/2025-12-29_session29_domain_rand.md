# Session 29: Domain Randomization & Sim-to-Real Alignment

**Date**: 2025-12-29
**Duration**: ~4 hours
**Result**: Lin_vel noise + obs clipping implemented, action noise/delays found counterproductive

## Summary

This session had two phases:
1. **Domain randomization testing** - Action noise/delays hurt training
2. **Sim-to-real alignment** - Lin_vel noise + obs clipping implemented per hardware testing findings

## Experiment Results

### Phase 1: Domain Randomization (Action-Side)

| EXP | Config | vx | Verdict | Notes |
|-----|--------|-----|---------|-------|
| 154 | Action noise 0.5% | 0.007 | STANDING | Action noise hurts training |
| 155 | Action noise 0.2% | 0.009 | STANDING | Still hurts |
| 156 | Action delay 0-1 steps | 0.008 | STANDING | Delays hurt too |

**Key Finding**: Action-side randomization (noise on actions, control delays) hurts training when observation noise is already present. The combination is too much uncertainty for the policy to learn through.

### Phase 2: Sim-to-Real Alignment

| EXP | Config | vx | Verdict | Notes |
|-----|--------|-----|---------|-------|
| 159 | Lin_vel noise + obs clipping | 0.008 | STANDING | Doesn't hurt, same regression |

**Key Finding**: Lin_vel noise and observation clipping don't hurt training (unlike action noise), but they also don't prevent the late-training regression pattern we've seen in all experiments.

## Implementation Changes

### Config Changes (`harold_isaac_lab_env_cfg.py`)

1. **Linear velocity noise** (new):
```python
add_lin_vel_noise: bool = True
lin_vel_noise: GaussianNoiseCfg(std=0.05)  # m/s
lin_vel_bias_std: float = 0.02  # per-episode calibration drift
```

2. **Observation clipping** (new):
```python
clip_observations: bool = True
clip_observations_value: float = 5.0  # raw clip at ±50
```

3. **Action noise/delays** (disabled):
```python
add_action_noise: bool = False   # Hurts training
add_action_delay: bool = False   # Hurts training
```

### Environment Changes (`harold_isaac_lab_env.py`)

1. Added `_lin_vel_bias` buffer initialization
2. Added per-episode bias randomization in `_reset_idx()`
3. Added lin_vel noise application in `_add_observation_noise()`
4. Added observation clipping in `_get_observations()`

## Observations

### Why Action Noise/Delays Hurt

Action-side randomization corrupts the control signal that the policy is trying to learn to produce. When combined with observation noise (which is already making sensing uncertain), the policy can't determine cause-and-effect between actions and outcomes.

Observation noise works as regularization because the policy must learn to be robust to sensing uncertainty while still producing clean, correct actions.

### Regression Pattern

All experiments show the same pattern:
- Peak performance around 40-70% training (vx=0.013-0.015)
- Final regression to vx=0.007-0.009

This is a PPO training dynamics issue, not related to domain randomization.

## Next Steps

1. Continue with joint limit alignment (EXP-160, 161)
2. Consider shorter training (625 iters) to capture peak before regression
3. Export best checkpoints for hardware testing

## Files Modified

| File | Changes |
|------|---------|
| `harold_isaac_lab_env_cfg.py` | Added lin_vel noise, obs clipping config |
| `harold_isaac_lab_env.py` | Implemented noise/clipping application |
| `NEXT_STEPS.md` | Updated with experiment queue |
