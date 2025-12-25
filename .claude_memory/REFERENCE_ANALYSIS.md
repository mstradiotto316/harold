# Reference Implementation Analysis

**Created:** 2025-12-24 (Session 18)
**Purpose:** Document findings from analyzing Isaac Lab quadruped examples

---

## Source Files Analyzed

- **AnyMal-C Direct:** `/home/matteo/Desktop/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/`
- **Spot Manager:** `/home/matteo/Desktop/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/`
- **Spot Rewards:** `.../config/spot/mdp/rewards.py` (13 reward functions)
- **Base Velocity Env:** `.../manager_based/locomotion/velocity/velocity_env_cfg.py`

---

## Key Differences from Harold

### 1. Reward Design Philosophy

| Aspect | Reference (AnyMal/Spot) | Harold |
|--------|------------------------|--------|
| **Velocity tracking** | Exponential kernel: `exp(-error²/0.25)` | Linear penalties |
| **Reward magnitudes** | O(1-10) | O(40-50) |
| **Backward penalty** | None (velocity tracking handles it) | Explicit 50.0 penalty |
| **Standing penalty** | None (joint position penalty when still) | Explicit -5.0 |

### 2. Gait Enforcement

**Spot's Approach (Most Sophisticated):**
```python
# Bidirectional: sync within pairs AND async between pairs
synced_feet_pairs = [("FL", "HR"), ("FR", "HL")]  # Diagonals
sync_reward = exp(-(air_time_diff² + contact_time_diff²))
async_reward = exp(-(cross_phase_diff²))
gait_reward = sync_reward × async_reward
```

**Harold's Approach:**
```python
# Unidirectional: only rewards diagonal contact pattern
diagonal_gait_reward = 5.0 × (diagonal_match) × forward_gate
```

### 3. Actuator Models

| Robot | Actuator Type | Features |
|-------|---------------|----------|
| AnyMal | LSTM Network | Learned response from real robot |
| Spot | Delayed PD | 0-8ms random delay |
| Harold | Implicit PD | Ideal, no delays |

### 4. Domain Randomization

| Parameter | Reference Range | Harold (DISABLED) |
|-----------|----------------|-------------------|
| Mass | ±5kg (10%) | ±15% defined but off |
| Friction | 0.3-1.0 | 0.4-1.0 defined but off |
| Random pushes | Every 10-15s | Not implemented |
| COM offset | ±5cm | ±2cm defined but off |

### 5. Network Architecture

| Model | Architecture | Size |
|-------|-------------|------|
| AnyMal | [128, 128, 128] | 33K params |
| Harold | [512, 256, 128] | ~200K params |

### 6. Policy Frequency

| Robot | Policy Hz | Sim Hz | Decimation |
|-------|-----------|--------|------------|
| AnyMal | 50 Hz | 200 Hz | 4 |
| Spot | 50 Hz | 500 Hz | 10 |
| Harold | 20 Hz | 180 Hz | 9 |

---

## Notable Techniques NOT in Harold

### 1. Exponential Velocity Tracking
```python
# Instead of: reward = -weight × (target - actual)²
# Use: reward = weight × exp(-error² / variance)
lin_vel_error_sq = (cmd_vel - actual_vel) ** 2
reward = exp(-lin_vel_error_sq / 0.25)
```

### 2. Feet Slide Penalty
```python
# Penalize foot XY velocity when in contact
foot_xy_vel = foot_velocities[:, :2]  # XY only
is_contact = contact_forces > 0.1
foot_slip_penalty = -0.5 × sum(norm(foot_xy_vel) × is_contact)
```

### 3. Air Time Variance Penalty
```python
# Penalize inconsistent stepping between feet
variance_penalty = -1.0 × (var(last_air_times) + var(last_contact_times))
```

### 4. Joint Position Penalty with Standing Scaling
```python
# Stronger penalty when standing to maintain pose
standing_scale = 5.0 if velocity_cmd < 0.5 else 1.0
joint_pos_penalty = -0.7 × norm(joint_pos - default) × standing_scale
```

### 5. Velocity Ramping
```python
# Scale reward by velocity magnitude (encourages faster walking)
vel_cmd_mag = norm(velocity_command)
scaling = clamp(1.0 + 0.5 × (vel_cmd_mag - 1.0), min=1.0)
velocity_reward *= scaling
```

---

## Recommendations (Priority Order)

### HIGH PRIORITY

1. **Switch to Exponential Velocity Tracking**
   - Provides smooth gradients near zero
   - Eliminates need for explicit backward penalty
   - Standard approach in all reference implementations

2. **Reduce Reward Magnitudes**
   - Current: 40-50
   - Target: 1-10
   - Prevents gradient explosion and reward hacking

3. **Add Feet Slide Penalty**
   - Already partially implemented (slip_factor)
   - Make more explicit like reference

4. **Improve Gait Reward**
   - Add async penalty between diagonal pairs
   - Match Spot's bidirectional approach

5. **Enable Domain Randomization**
   - Infrastructure exists but disabled
   - Start with conservative ranges

### MEDIUM PRIORITY

6. **Reduce Network Size**
   - [128, 128, 128] instead of [512, 256, 128]
   - May be overparameterized

7. **Increase Learning Rate**
   - 1.0e-3 instead of 5.0e-4
   - Reference uses higher LR

8. **Add Actuator Delays**
   - 0-2 timestep random delay
   - Better sim-to-real transfer

### LOW PRIORITY

9. **Increase Policy Frequency**
   - 50 Hz instead of 20 Hz
   - May improve stability

10. **Consider Actuator Network**
    - Train LSTM on real servo response
    - Advanced sim-to-real technique

---

## Scale Mismatch Warning

Harold is **25× lighter** than reference robots:
- AnyMal: ~50 kg
- Spot: ~32 kg
- Harold: ~2 kg

This may require different reward tuning. Lighter robots:
- More sensitive to perturbations
- May benefit from higher control frequency
- Need more conservative action scaling

---

## Files for Reference

When implementing changes, refer to:
- Spot rewards: `.../config/spot/mdp/rewards.py`
- AnyMal env: `.../direct/anymal_c/anymal_c_env.py`
- Velocity tracking: Look for `exp(-error/std)` patterns
