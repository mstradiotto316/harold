# Harold Next Steps

## PRIORITY 1: Model Gear Backlash in Simulation (Overnight Experiment)

**Context from Session 27:** Hardware investigation revealed that the servo "dead zone" is actually **mechanical gear backlash** (~1-3°) in the ST3215 gearbox. This cannot be fixed in software - the policy must learn to be robust to it.

### Experiment Plan

**Goal:** Train a policy that's robust to position uncertainty caused by gear backlash.

**Approach:** Increase joint position observation noise to simulate the uncertainty within the backlash zone.

### Implementation Steps

1. **Baseline run** (if not already established):
   ```bash
   HAROLD_CPG=1 HAROLD_CMD_TRACK=1 python scripts/harold.py train \
     --hypothesis "Baseline before backlash modeling" --iterations 1250
   ```

2. **Backlash modeling via observation noise**:

   Modify `harold_isaac_lab_env_cfg.py`:
   ```python
   # In DomainRandomizationCfg class
   enable_randomization: bool = True
   randomize_per_step: bool = True

   # Increase joint position noise to match ~2° backlash
   add_joint_noise: bool = True
   joint_position_noise: GaussianNoiseCfg = GaussianNoiseCfg(
       mean=0.0,
       std=0.035,  # ~2° in radians (was 0.005)
   )
   ```

3. **Run experiment**:
   ```bash
   HAROLD_CPG=1 HAROLD_CMD_TRACK=1 python scripts/harold.py train \
     --hypothesis "Backlash robustness via 2deg position noise" \
     --tags "backlash,sim2real" --iterations 2500
   ```

4. **Compare with baseline** using `harold compare`

### Alternative Approaches (if observation noise doesn't help)

- **Action noise**: Enable `add_action_noise: bool = True` with `std=0.02`
- **Lower stiffness**: Reduce from 400 to 300 in `harold.py`
- **Custom backlash model**: Implement hysteresis in `_apply_action()`

### Success Criteria

- Policy maintains walking (vx > 0.1 m/s) despite position noise
- More robust/smoother motion than baseline
- Ideally: better sim-to-real transfer when deployed

---

## PRIORITY 2: Add Lateral (vy) Commands

Current implementation only tracks forward velocity (vx). To add lateral commands:

1. Add vy tracking to command_tracking reward:
   ```python
   # In _compute_rewards():
   cmd_vy = self._commands[:, 1]
   vy_error = vy_w - cmd_vy
   vy_tracking = cmd_weight * torch.exp(-torch.square(vy_error / cmd_sigma))
   ```

2. Modify lat_vel_penalty to not penalize commanded motion

---

## PRIORITY 3: Add Yaw Rate Commands

Similar to lateral commands - add yaw tracking and modify yaw_rate_penalty.

---

## Session 27 Hardware Findings

### Servo Register Reference (ST3215)

| Register | Name | Factory | Purpose |
|----------|------|---------|---------|
| 21 | P_COEF | 32 | Proportional gain (higher=stiffer, but oscillates) |
| 22 | D_COEF | 32 | Derivative gain (damping) |
| 26-27 | CW/CCW_DEAD | 1 | Dead zone (already minimal) |
| 48-49 | TORQUE_LIMIT | 1000 | Torque limit (100%) |

### Key Insight

The "dead zone" feel is **gear backlash**, not software dead zone. Servos have ~1-3° of mechanical play where they don't resist movement. This is a hardware limitation that must be modeled in simulation.

### ServoTuning Utility

New firmware utility at `firmware/CalibrationAndSetup/ServoTuning/ServoTuning.ino`:
- `s` - Scan all servos
- `X <id>` - Dump all registers
- `x <id> <reg>` - Read register
- `W <id> <reg> <val>` - Write register

---

## Session 26 Results (for reference)

| EXP | Config | vx | Verdict |
|-----|--------|-----|---------|
| 134 | Dynamic 5s | 0.007 | STANDING |
| 135 | Dynamic 10s | 0.010 | STANDING |
| 139 | zero_prob=0 | 0.011 | WALKING |
| 140 | range 0.10-0.45 | 0.015 | WALKING |
| 141 | range 0.05-0.50 | 0.012 | WALKING |
| 142 | Confirmation | 0.015 | WALKING |
| 143 | 2500 iters | 0.016 | WALKING |

**Best Result**: EXP-143 with 2500 iterations achieved vx=0.016 m/s.

---

## Environment Variables Reference

| Variable | Effect |
|----------|--------|
| `HAROLD_CPG=1` | Enable CPG base trajectory |
| `HAROLD_CMD_TRACK=1` | Enable command tracking reward |
| `HAROLD_VAR_CMD=1` | Enable variable command sampling |
| `HAROLD_DYN_CMD=1` | Enable dynamic command changes (implies VAR_CMD) |
| `HAROLD_SCRIPTED_GAIT=1` | Enable scripted gait (no learning) |

---

## Files Reference

| Purpose | Path |
|---------|------|
| Domain randomization config | `harold_isaac_lab_env_cfg.py` - DomainRandomizationCfg |
| Actuator config | `harold_flat/harold.py` - ImplicitActuatorCfg |
| Session 27 log | `.claude_memory/sessions/2025-12-29_session27_servo_tuning.md` |
