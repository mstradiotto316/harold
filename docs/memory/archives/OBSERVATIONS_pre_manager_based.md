# Pre-Manager-Based Observations (archived 2026-03-30)

These observations cover the direct-env era (2025-12 through 2026-03-28). The direct-env approach
never achieved walking. See `OBSERVATIONS.md` for the manager-based breakthrough.

## 2026-03-23: CRITICAL CORRECTION — ALL historical KEEPs were false positives

**Every historical "walking KEEP" was a false positive.** The robot has NEVER achieved anything remotely approaching real walking behavior. What was labeled as "walking" was actually:
- Body-frame velocity gaming (oscillation producing positive vx_b without actual displacement)
- Controlled-fall-and-reset cycles producing transient forward drift
- Micro-stepping with x_displacement < 0.035m (well below the new 0.1m threshold)

The Session 51 post-mortem (commit feb26a7) added honest evaluation:
- x_displacement hard gate (>=0.1m) correctly rejects all old "KEEPs"
- EXP-532 (x_disp=0.034m) and EXP-571 (x_disp=0.008m) both now correctly FAIL

Session 52 (EXP-688 through EXP-705, 18 experiments) validated the new reward architecture:
- Hybrid 50/50 body/world forward_motion + standstill penalty produces gait patterns (gait_reward=0.93)
- Peak transient vx=0.045 at 4242 seed/16384 envs, but policy regresses with longer training
- **All results at 16384 envs converged to standing** — consistent with Session 50 observation

### Session 52 video reviewer false positives (EXP-709, 712, 713, 714)
- Video reviewers labeled these as "WALKING" but the user confirmed they were NOT walking
- The "cyclic foot lifting and sustained forward translation" was actually micro-movements and drift (~2cm/s)
- x_disp was consistently < 0.02m — the 0.1m hard gate correctly rejected all of them
- **Fixed by separating concerns:** Video uses LOCOMOTION (not WALKING), metrics gate (`cmd_tracking_ratio >= 0.5`) is sole KEEP authority.

## 2026-03-21: Session 51 — Quality Ceiling Confirmed (64 experiments)

### Walking is a TRANSIENT training phenomenon
- The walking behavior is a saddle point, not a stable equilibrium
- At 15min (fast), the policy passes through the walking basin and training stops, capturing it
- At 30min+, the policy continues past the walking basin toward standing (the true equilibrium)

### grad_norm_clip=0.5 is the most impactful PPO discovery
- EXP-554 (15min): vx=0.107 (record!) but ep_len=130 — aggressive lunging, frequent falls
- EXP-560 (30min): vx=0.048, ep_len=295 — genuine stepping, best balance
- Pattern: grad_norm=0.5 produces progressively more conservative strategies as training lengthens

### Pitch penalty is INCOMPATIBLE with forward motion
- Quadratic pitch penalty: STANDING
- Soft threshold pitch penalty: STANDING
- The walking mechanism REQUIRES forward lean. Any pitch penalty removes locomotion.

### Walking is a controlled-fall-and-reset strategy
- Baseline (EXP-514/532): robot steps forward 2-4s, nose-dives, resets, repeats
- Every modification preventing nose-dive also removes forward progress
- Walking is a knife-edge equilibrium: upright=3.0, forward=7.0, air_time=2.0, seed=38, 4096 envs

### 20 experiments on correct baseline — all DISCARD
- Quality ceiling confirmed. No improvement on baseline possible with direct-env.

## 2026-03-20: Session 50 — 16384 envs Standing Attractor

- 16384 envs structurally prevents walking (8 experiments, all standing/degenerate)
- 4096 envs breaks the standing attractor — immediately produces shuffling/stepping
- Diagonal gait alternation reward stabilizes walking (KEEP: EXP-514)
- Forward pitch collapse is the structural failure mode

## 2026-03-18: Coordinate Frame Investigation (RESOLVED)

- Isaac Lab uses (w, x, y, z) quaternion format
- Identity quaternion = (1.0, 0.0, 0.0, 0.0) — correct for Harold
- 180 deg Z rotation was temporarily applied (EXP-429) but reverted — no bug found
- Body-frame velocity computed via quat_apply_inverse

## 2026-03-19: Scoring Consolidation & Session 48

- walk_score had hard gate (ep_len >= 300) — never triggered in 200+ experiments
- Consolidated to single compute_score() with soft gate
- Session 48: alive_bonus=2.0 promising, forward_motion increase causes tipping

## Earlier Sessions

- 2026-03-16: 1024 envs transformative, video review essential, front-leg passivity bottleneck
- 2026-03-15: Body-frame velocity alignment, audit remediation, deployment sign conventions
- See `OBSERVATIONS_2026-01-04_full.md` for hardware CPG baseline findings
