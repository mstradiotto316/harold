# Plan: Close Sim-to-Real Gap for Harold Robot

## Overview
Modify Isaac Lab training environment to match hardware deployment behavior, then retrain policy.

## Investigation Findings (Session 28-29)

### IMU Units Comparison
| Observation | Hardware | Isaac Lab | Status |
|-------------|----------|-----------|--------|
| Linear velocity | m/s (integrated accel with 0.95 decay) | m/s (physics engine) | **Different source - add noise** |
| Angular velocity | rad/s | rad/s | ✓ Match |
| Projected gravity | Z-up convention | Z-down convention | ✓ Fixed (sign flip in deployment) |

### Clipping/Preprocessing Comparison
| Component | Training | Deployment | Status |
|-----------|----------|------------|--------|
| Observation clipping | **None** | ±5.0 | **Add to simulation** |
| Action clipping | None before scale | None before scale | ✓ Match |
| Joint limits | ±30°/±90° (symmetric) | ±25°/asymmetric | **Align to hardware** |

### User Decisions
1. ✅ **Add observation clipping to simulation** - Match deployment's ±5.0
2. ✅ **Add noise to simulation linear velocity** - Make policy robust to noisy IMU

---

## Implementation Plan (Execute on Desktop)

### Phase 1: Linear Velocity Noise (Priority: High)

**File: `harold_isaac_lab_env_cfg.py`**

Add to `DomainRandomizationCfg` (after line ~389):
```python
# Linear velocity noise (simulates IMU accelerometer integration noise)
add_lin_vel_noise: bool = True
lin_vel_noise: GaussianNoiseCfg = GaussianNoiseCfg(
    mean=0.0,
    std=0.05,  # 0.05 m/s noise (hardware shows ~5cm/s drift)
    operation="add"
)
# Velocity bias that persists per-episode (simulates calibration error)
lin_vel_bias_std: float = 0.02  # ±2cm/s per-episode bias
```

**File: `harold_isaac_lab_env.py`**

1. Add bias state in `_setup_scene()`:
```python
self._lin_vel_bias = torch.zeros((self.num_envs, 3), device=self.device)
```

2. Randomize bias in `_reset_idx()`:
```python
if self.cfg.domain_randomization.add_lin_vel_noise:
    bias_std = self.cfg.domain_randomization.lin_vel_bias_std
    self._lin_vel_bias[env_ids] = torch.randn((len(env_ids), 3), device=self.device) * bias_std
```

3. Apply noise in `_add_observation_noise()`:
```python
# Add linear velocity noise (indices 0:3)
if self.cfg.domain_randomization.add_lin_vel_noise:
    noisy_obs[:, 0:3] = gaussian_noise(observations[:, 0:3], self.cfg.domain_randomization.lin_vel_noise)
    noisy_obs[:, 0:3] += self._lin_vel_bias
```

### Phase 2: Observation Clipping (Priority: High)

**File: `harold_isaac_lab_env_cfg.py`**

Add to main config class:
```python
# Observation clipping (matches deployment clip_obs=5.0)
clip_observations: bool = True
clip_observations_value: float = 5.0
```

**File: `harold_isaac_lab_env.py`**

Add clipping in `_get_observations()` before return:
```python
# Apply observation clipping if enabled (matches deployment)
if getattr(self.cfg, 'clip_observations', False):
    clip_val = getattr(self.cfg, 'clip_observations_value', 5.0)
    # Note: This clips raw obs; deployment clips post-normalization
    # Use larger raw clip to approximate effect
    obs = torch.clamp(obs, -clip_val * 10, clip_val * 10)
```

**Alternative (more accurate)**: Modify skrl preprocessor or add custom wrapper to clip post-normalization.

### Phase 3: Joint Limit Alignment (Priority: Critical)

**File: `harold_isaac_lab_env_cfg.py`**

Update joint limits to match hardware (lines ~551-560):
```python
# Hardware-aligned joint limits (from deployment safe_limits_deg)
joint_angle_max: tuple = (
    0.4363, 0.4363, 0.4363, 0.4363,  # shoulders +25° (was +30°)
    0.0873, 0.0873, 0.0873, 0.0873,  # thighs +5° (was +90°) - MAJOR
    1.3963, 1.3963, 1.3963, 1.3963   # calves +80° (was +90°)
)
joint_angle_min: tuple = (
    -0.4363, -0.4363, -0.4363, -0.4363,  # shoulders -25°
    -0.9599, -0.9599, -0.9599, -0.9599,  # thighs -55° (was -90°)
    -0.0873, -0.0873, -0.0873, -0.0873   # calves -5° (was -90°) - MAJOR
)
```

### Phase 4: Retrain and Export

```bash
# Train with new sim-to-real aligned config
HAROLD_CPG=1 HAROLD_CMD_TRACK=1 python scripts/harold.py train \
  --hypothesis "Sim-to-real: lin_vel noise, obs clip, hw joint limits" \
  --tags "sim2real,domain_rand" --iterations 2500

# Export new ONNX policy
python deployment_artifacts/export_policy.py --checkpoint <new_checkpoint>
```

---

## Critical Files

| Purpose | Path |
|---------|------|
| Training env | `harold_isaac_lab/.../harold_flat/harold_isaac_lab_env.py` |
| Training config | `harold_isaac_lab/.../harold_flat/harold_isaac_lab_env_cfg.py` |
| Deployment obs | `deployment/inference/observation_builder.py` |
| Deployment action | `deployment/inference/action_converter.py` |
| Hardware IMU | `deployment/drivers/imu_reader_rpi5.py` |

---

## Hardware Deployment Fixes Already Applied (Session 28)

These fixes were made to deployment code on the Pi:
1. ✅ Gravity sign flip (Z-up → Z-down convention)
2. ✅ Default commands changed to 0.3 m/s (matches training)
3. ✅ Observation clipping to ±5.0 added
4. ✅ Removed pre-scaling action clip (matches training)
5. ✅ Added 3-cycle CPG warmup before policy enables

---

## MEMORY SYSTEM UPDATES (Copy to appropriate files)

### Update NEXT_STEPS.md - Replace Priority 0 with:

```markdown
## PRIORITY 0: Sim-to-Real Alignment (Session 29 Investigation)

**Status**: Investigation complete. Implementation required on desktop.

### Background
Hardware deployment testing (Session 28-29) revealed critical sim-to-real gaps causing policy to output extreme values (±60-100) and unstable behavior.

### Root Causes Identified

1. **Linear velocity mismatch**: Hardware IMU uses accelerometer integration (noisy, drifts). Simulation uses perfect physics engine velocity.

2. **Observation clipping mismatch**: Deployment clips normalized observations to ±5.0. Training has NO clipping.

3. **Joint limit mismatch**: Training uses symmetric ±30°/±90° limits. Hardware has asymmetric limits (thigh: -55° to +5°).

### Implementation Tasks

1. **Add lin_vel noise to simulation** (DomainRandomizationCfg):
   - Gaussian noise: std=0.05 m/s
   - Per-episode bias: std=0.02 m/s

2. **Add observation clipping to simulation**:
   - Clip post-normalization to ±5.0 (or approximate with raw clip)

3. **Align joint limits to hardware**:
   - Shoulders: ±25° (was ±30°)
   - Thighs: -55° to +5° (was ±90°) - MAJOR CHANGE
   - Calves: -5° to +80° (was ±90°) - MAJOR CHANGE

4. **Retrain and export**:
   ```bash
   HAROLD_CPG=1 HAROLD_CMD_TRACK=1 python scripts/harold.py train \
     --hypothesis "Sim-to-real: lin_vel noise, obs clip, hw joint limits" \
     --tags "sim2real,domain_rand" --iterations 2500
   ```

### Key Files to Modify
- `harold_isaac_lab_env_cfg.py` - Add noise config, joint limits
- `harold_isaac_lab_env.py` - Implement noise application, clipping

### Reference
Full implementation details in plan file: `.claude/plans/unified-knitting-lagoon.md`
```

### Update Session Log (append to 2025-12-29_session.md):

```markdown
---

## Session 29: Sim-to-Real Investigation

**Date**: 2025-12-29 (evening)
**Platform**: Raspberry Pi 5 (onboard Harold robot)
**Focus**: Investigate sim-to-real gap root causes

### Investigation Summary

Conducted deep investigation into why policy outputs are extreme (±60-100) during deployment.

### Key Findings

#### 1. IMU Units Analysis
- Hardware IMU outputs match Isaac Lab units (m/s, rad/s)
- **BUT**: Hardware computes lin_vel via accelerometer integration with 0.95 decay
- Isaac Lab provides perfect physics velocity
- **Fix**: Add noise + bias to simulation lin_vel

#### 2. Clipping Analysis
- Training: NO observation clipping
- Deployment: Clips normalized obs to ±5.0
- **Fix**: Add clipping to simulation training

#### 3. Joint Limits Analysis
- Training: Symmetric limits (±30° shoulder, ±90° thigh/calf)
- Hardware: Asymmetric (thigh -55° to +5°, calf -5° to +80°)
- **Fix**: Align simulation limits to hardware

### User Decisions
- ✅ Add observation clipping to simulation
- ✅ Add noise to simulation linear velocity

### Deployment Fixes Applied This Session
1. Gravity sign flip (Z-up → Z-down)
2. Default commands: 0.3 m/s
3. Observation clipping: ±5.0
4. Removed pre-scaling action clip
5. Added 3-cycle CPG warmup

### Next Steps
- Implement changes on desktop (Isaac Lab training)
- Retrain policy with sim-to-real aligned config
- Export new ONNX and test on hardware
```
