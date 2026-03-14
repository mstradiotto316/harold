# Harold Autoresearch Strategy

This file is the human's control surface. Edit this to steer the autonomous research agent.
The agent reads this at session start and follows the goals, constraints, and priorities below.

## Session Parameters

- max_experiments: 8
- duration_per_experiment: short
- mode: rl

## Research Goals (Priority Order)

1. Achieve consistent forward walking (vx_w_mean > 0.02 m/s)
2. Maintain stability (upright_mean > 0.95, height_reward > 0.6)
3. Develop regular gait pattern (visible stepping in video, no foot dragging)
4. Minimize body contact (body_contact > -0.02)

## Exploration Focus

Start with reward weight tuning (one parameter at a time):
- `forward_motion_weight`: try [2.0, 4.0, 5.0] (current: 3.0)
- `feet_air_time_weight`: try [0.5, 1.5, 2.0] (current: 1.0)
- `track_lin_vel_xy_weight`: try [3.0, 7.0, 10.0] (current: 5.0)
- `action_rate_weight`: try [-0.005, -0.02, -0.03] (current: -0.01)
- `upright_weight`: try [1.0, 3.0, 4.0] (current: 2.0)

If individual improvements are found, try combining the top 2-3 in a single experiment.

## Do Not Change

- `learning_rate` (5e-4 is optimal; 3e-4 causes SANITY_FAIL)
- `action_scale` (0.5 is hardware-validated)
- `action_filter_beta` (0.40 is optimal; 0.50 prevented walking)
- Stiffness / damping / effort_limit (sim-to-real critical, FROZEN)
- Sensor noise settings (1 degree joint noise is optimal per Session 28)
- Network architecture [512, 256, 128] (EXP-092 showed smaller was worse)

## Known Dead Ends

- Domain randomization (full) → robot stands still (EXP-090)
- Action noise (any amount) → hurts learning (EXP-154, EXP-155)
- External perturbations → breaks training (EXP-164)
- Learning rate 3e-4 → SANITY_FAIL (Session 23)
- Action scale 0.7 → vx=0.029, contact failing (Session 23)
- Action filter beta 0.50 → prevented walking (Session 35)
- forward_motion_weight 10.0 → regression (Session 36)

## Scoring Priorities

- vx_w_mean improvement: weight 3x
- upright_mean maintenance: weight 2x
- height_reward maintenance: weight 1.5x
- body_contact improvement: weight 1x
- episode_length maintenance: weight 1x

## Notes

- Try one parameter at a time to isolate effects.
- If a short run is borderline promising (STANDING with vx > 0.005), rerun with standard duration.
- If 3 consecutive experiments are DISCARD, pause and reassess the strategy direction.
- Prefer simplification: if removing a reward term gives equal results, that's a win.
