#!/usr/bin/env python3
"""
Harold 4-Metric Analysis Script

Analyzes TensorBoard logs and outputs only the key metrics to avoid context overflow.
Returns exit codes:
  0 = SUCCESS (all metrics pass - robot walking)
  1 = STANDING (height ok but no forward motion)
  2 = FAILURE (robot not at height - likely fallen)
  3 = ERROR (analysis failed)

Usage:
  python scripts/analyze_run.py [run_name]
  python scripts/analyze_run.py  # analyzes latest run
"""

import os
import sys

def analyze_run(run_path: str | None = None):
    """Analyze a training run using 4-metric protocol."""
    from tensorboard.backend.event_processing import event_accumulator

    log_base = '/home/matteo/Desktop/code_projects/harold/logs/skrl/harold_direct'

    # Find run directory
    if run_path is None:
        runs = sorted([d for d in os.listdir(log_base)
                      if os.path.isdir(os.path.join(log_base, d)) and '2025-' in d])
        if not runs:
            print("ERROR: No runs found")
            return 3
        run_name = runs[-1]
        path = os.path.join(log_base, run_name)
    else:
        if os.path.isabs(run_path):
            path = run_path
            run_name = os.path.basename(run_path)
        else:
            path = os.path.join(log_base, run_path)
            run_name = run_path

    print(f"Analyzing: {run_name}")
    print("-" * 50)

    try:
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
    except Exception as e:
        print(f"ERROR: Failed to load TensorBoard data: {e}")
        return 3

    # Helper to get average of last N samples
    def avg_last(tag: str, n: int = 10) -> float:
        try:
            scalars = ea.Scalars(tag)
            if not scalars:
                return float('nan')
            last_n = scalars[-n:]
            return sum(s.value for s in last_n) / len(last_n)
        except KeyError:
            return float('nan')

    # Get the 4 key metrics
    upright = avg_last('Info / Episode_Metric/upright_mean')
    height_rew = avg_last('Info / Episode_Reward/height_reward')
    body_contact = avg_last('Info / Episode_Reward/body_contact_penalty')
    vx = avg_last('Info / Episode_Metric/vx_w_mean')
    ep_len = avg_last('Episode / Total timesteps (mean)')

    # Print metrics
    print("=== 4-METRIC ANALYSIS ===")
    print(f"upright_mean:         {upright:.4f} (need > 0.9)")
    print(f"height_reward:        {height_rew:.4f} (need > 2.0)")
    print(f"body_contact_penalty: {body_contact:.4f} (need > -0.1)")
    print(f"vx_w_mean:            {vx:.4f} m/s (need > 0.1)")
    print(f"episode_length:       {ep_len:.1f}")
    print("")

    # Diagnosis
    print("=== DIAGNOSIS ===")

    # Check for NaN values
    import math
    if math.isnan(height_rew) or math.isnan(upright):
        print(">>> ERROR: Metrics not available (training too short?)")
        return 3

    if height_rew < 1.0:
        print(">>> FAILURE: Robot NOT at standing height!")
        print("    Likely on elbows or collapsed.")
        return 2

    if body_contact < -0.2:
        print(">>> FAILURE: Body contact detected!")
        print("    Robot lying/dragging on ground.")
        return 2

    if upright < 0.9:
        print(">>> FAILURE: Robot tilted too far")
        print("    Falling sideways or backward.")
        return 2

    if height_rew >= 2.0 and vx >= 0.1 and body_contact >= -0.1:
        print(">>> SUCCESS: WALKING!")
        print("    All 4 metrics pass.")
        return 0

    if height_rew >= 2.0 and vx < 0.05:
        print(">>> STATUS: Standing but not walking")
        print(f"    Forward velocity: {vx:.4f} m/s (too low)")
        print("    Consider increasing forward_reward.")
        return 1

    print(f">>> STATUS: Partial progress")
    print(f"    height_reward: {'OK' if height_rew >= 2.0 else 'LOW'}")
    print(f"    vx_w_mean: {vx:.4f} m/s")
    return 1


if __name__ == "__main__":
    run_path = sys.argv[1] if len(sys.argv) > 1 else None
    exit_code = analyze_run(run_path)
    sys.exit(exit_code)
