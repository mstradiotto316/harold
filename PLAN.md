# Harold Quadruped RL Training Analysis & Fix Plan

## Problem Summary

During Harold quadruped RL training, we discovered a classic **reward hacking** problem where the robot learned to exploit the reward structure rather than develop proper locomotion behaviors.

### Initial Symptoms
- Linear velocity reward (`track_xy_lin_commands`) plateaus consistently around ~200 total episode reward
- Plateau occurs regardless of reward normalization values (tested: 0.001, 0.0005, 0.00025, 0.0001)
- Robot exhibits "humping" or "rocking" motion instead of walking
- When locomotion occurs, it's shuffling/dragging feet rather than proper stepping
- Poor performance on rough terrain due to inadequate gait patterns

### Investigation Timeline

#### 1. Initial Hypothesis: Reward Normalization Issue
**Tested:** Different exponential reward normalization factors in:
```python
lin_vel_reward = torch.exp(-torch.square(lin_vel_error) / NORMALIZATION_FACTOR)
```

**Result:** All normalization values (0.001 → 0.0001) produced same plateau around 200 reward, indicating the issue wasn't reward sensitivity.

#### 2. Curriculum Learning Investigation
**Finding:** Curriculum alpha scaling was disabled:
```python
# WARNING: Alpha scaling has proven to be unstable and is not used in the final version
self._alpha = 1.0
```

**User Insight:** Enabling curriculum learning hurt training performance because it creates a **non-stationary environment** that violates PPO's core assumptions.

**Analysis:** Changing command difficulty over time (α scaling) causes:
- Value function inaccuracy as task distribution shifts
- Policy gradient bias from changing reward landscape  
- Sample inefficiency as previous experience becomes less relevant
- Forced "unlearning" and relearning of gait patterns

**Conclusion:** Fixed command distribution (no curriculum) is correct approach for PPO.

#### 3. Command Difficulty Analysis
**Initial Theory:** Robot quickly masters limited command range (`-0.3 to 0.3 m/s`) and plateaus due to lack of challenge.

**Reality Check:** Robot was NOT mastering the command range - it was finding a reward hacking strategy.

#### 4. Root Cause Discovery: Reward Hacking
**Breakthrough:** Robot learned to exploit reward structure through oscillatory body movements instead of proper locomotion.

## Root Cause Analysis

### Reward Structure Analysis
From `harold_isaac_lab_env_cfg.py`:

```python
track_xy_lin_commands: float = 600   # MASSIVE weight - dominates everything
feet_air_time: float = 300.0        # High weight but BROKEN implementation
velocity_jitter: float = -30        # Small penalty, easily overwhelmed
torque_penalty: float = -3          # Negligible penalty
```

### The Fatal Flaw: Broken Air Time Reward

**Current Implementation:**
```python
air_time_reward = torch.sum((last_air_time - 0.3) * first_contact, dim=1)
```

**Problems:**
1. **Negative for Normal Walking:** 0.3s air time is too long for a 40cm robot (should be ~0.1-0.2s)
2. **Penalizes Proper Stepping:** Normal gait gets negative rewards
3. **Rewards Keeping Feet Down:** Robot learns to avoid lifting feet

### Why "Humping/Rocking" Wins

The robot discovered an optimal strategy given the broken reward structure:

1. **Rock/oscillate body** → Generates velocity in commanded direction → **+600 weighted rewards**
2. **Keep feet planted** → Avoids negative air time penalties → **No penalty**
3. **Smooth oscillation** → Minimal jitter penalty → **Only -30 penalty**

**Net Result:** Body oscillation massively outperforms proper locomotion in the reward function.

### Reward Balance Problem

With velocity tracking weight at 600 and air time penalty structure, the robot optimized for:
- Maximum velocity tracking reward (dominant signal)
- Minimum air time penalties (avoid stepping)
- Ignore small jitter/torque penalties

This created a **local optimum** in behavior space that's far from the intended locomotion.

## Proposed Solution Framework

The core issue is **misaligned incentives** in the reward structure. We need to restructure rewards to make proper locomotion the optimal strategy.

### Key Principles for Fix
1. **Reward actual stepping patterns** rather than penalizing them
2. **Balance reward magnitudes** so locomotion quality matters as much as velocity tracking
3. **Add explicit anti-cheating mechanisms** to prevent oscillatory hacking
4. **Ensure reward components work synergistically** toward walking goals

---

## Recommended Fixes

### Fix 1: Correct Air Time Reward Implementation
**Problem:** Current reward penalizes normal stepping patterns
**Solution:** Reward optimal air time for small robot, penalize deviations

```python
# Replace broken implementation:
# air_time_reward = torch.sum((last_air_time - 0.3) * first_contact, dim=1)

# With corrected version:
optimal_air_time = 0.15  # Appropriate for 40cm robot (was 0.3s)
air_time_error = torch.abs(last_air_time - optimal_air_time)
air_time_reward = torch.sum(torch.exp(-air_time_error * 10.0) * first_contact, dim=1)
```

**Justification:**
- 0.15s air time is realistic for small quadruped (0.3s was too long)
- Exponential reward curve provides strong signal for optimal stepping
- Positive rewards encourage stepping rather than penalizing it
- Maintains high weight (300) to balance against velocity tracking

### Fix 2: Add Contact Frequency Reward
**Problem:** Robot can get velocity rewards while keeping all feet planted
**Solution:** Explicitly reward proper contact patterns during locomotion

```python
# Require alternating foot contacts to prevent "standing and rocking"
contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids]
feet_in_contact = (torch.norm(contact_forces, dim=-1) > 0.05).float()
contact_count = torch.sum(feet_in_contact, dim=1)

# Reward having 2-3 feet in contact (normal for walking)
contact_reward = torch.exp(-torch.abs(contact_count - 2.5)) * 50.0  # Medium weight
```

**Justification:**
- Normal quadruped walking has 2-3 feet in contact during gait cycle
- Prevents "4 feet planted + body rocking" strategy
- Exponential reward creates strong preference for proper contact patterns
- 50.0 weight provides meaningful signal without overwhelming velocity tracking

### Fix 3: Penalize Non-Locomotory Movement
**Problem:** Body oscillations generate velocity without proper locomotion
**Solution:** Add acceleration-based penalty to discourage rapid body movements

```python
# Add body acceleration penalty to discourage oscillations
body_acc = (self._robot.data.root_lin_vel_b - self._prev_lin_vel) / self.step_dt
oscillation_penalty = torch.norm(body_acc[:, :2], dim=1) * 20.0  # Medium penalty

# Add to rewards dict:
"oscillation_penalty": -oscillation_penalty * self.step_dt
```

**Justification:**
- High body accelerations indicate oscillatory/jerky movement
- Smooth locomotion has lower acceleration profiles
- Penalty magnitude (20.0) significant enough to discourage hacking
- Complements existing jitter penalty with different physics signal

### Fix 4: Velocity Tracking Consistency Requirement
**Problem:** Instantaneous velocity rewards can be gamed with brief movements
**Solution:** Require sustained, consistent movement for full velocity rewards

```python
# Only reward velocity if it's sustained over multiple steps
velocity_consistency = torch.norm(
    self._robot.data.root_lin_vel_b[:, :2] - self._prev_lin_vel, dim=1
)
sustained_movement_factor = torch.exp(-velocity_consistency * 5.0)

# Modify existing reward:
lin_vel_reward = lin_vel_reward * sustained_movement_factor
```

**Justification:**
- Oscillatory movement has high velocity inconsistency between steps
- Proper walking has smoother, more consistent velocity profiles
- Multiplicative factor preserves existing reward structure while adding constraint
- Forces robot to maintain velocity through proper gait rather than brief impulses

### Implementation Priority
1. **Fix 1 (Air Time):** Highest priority - fixes core incentive misalignment
2. **Fix 2 (Contact Patterns):** High priority - directly prevents reward hacking
3. **Fix 3 (Oscillation Penalty):** Medium priority - additional anti-cheating measure
4. **Fix 4 (Consistency):** Lower priority - refinement of existing reward

### Expected Outcomes
- Robot forced to develop proper stepping patterns to achieve optimal air time rewards
- Body rocking/humping becomes suboptimal due to contact and acceleration penalties
- Velocity tracking requires sustained locomotion rather than brief movements
- Overall behavior shifts from reward hacking to genuine quadruped locomotion
