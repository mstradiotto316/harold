#!/usr/bin/env python3
"""
Harold Training Validator - Run after EVERY experiment.

This script validates that training is actually working by checking
metrics in a specific order. The MOST IMPORTANT check is episode length -
if robots are dying immediately, all other metrics are meaningless.

Usage:
    python scripts/validate_training.py                    # Validate latest run
    python scripts/validate_training.py <run_name>         # Validate specific run
    python scripts/validate_training.py --list             # List recent runs
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_metrics(run_path: str) -> dict:
    """Extract all 5 key metrics from a TensorBoard run."""
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(run_path)
    ea.Reload()

    def avg_last_n(scalars, n=10):
        if not scalars:
            return None
        vals = [s.value for s in scalars[-n:]]
        return sum(vals) / len(vals)

    def get_scalar(key):
        try:
            return ea.Scalars(key)
        except KeyError:
            return []

    metrics = {
        'episode_length': avg_last_n(get_scalar('Episode / Total timesteps (mean)')),
        'upright_mean': avg_last_n(get_scalar('Info / Episode_Metric/upright_mean')),
        'height_reward': avg_last_n(get_scalar('Info / Episode_Reward/height_reward')),
        'body_contact': avg_last_n(get_scalar('Info / Episode_Reward/body_contact_penalty')),
        'vx_w_mean': avg_last_n(get_scalar('Info / Episode_Metric/vx_w_mean')),
        'data_points': len(get_scalar('Info / Episode_Metric/vx_w_mean')),
    }

    return metrics


def validate_run(run_path: str) -> dict:
    """
    Validate a training run using the 5-metric hierarchical protocol.

    Returns a dict with:
        - 'sanity_pass': bool - Did episode length pass?
        - 'all_pass': bool - Did all metrics pass?
        - 'metrics': dict - All metric values
        - 'diagnosis': str - Human-readable diagnosis
        - 'status': str - 'PASS', 'FAIL', or 'PARTIAL'
    """
    metrics = get_metrics(run_path)
    run_name = os.path.basename(run_path)

    # Define thresholds
    THRESHOLDS = {
        'episode_length': (100, 300),  # (minimum, expected)
        'upright_mean': 0.9,
        'height_reward': 1.2,  # Athletic crouch acceptable (user validated via video)
        'body_contact': -0.1,  # Must be GREATER than this (less negative)
        'vx_w_mean': 0.1,
    }

    results = {
        'run_name': run_name,
        'metrics': metrics,
        'checks': {},
        'sanity_pass': False,
        'all_pass': False,
        'diagnosis': '',
        'status': 'UNKNOWN',
    }

    # Check 1: SANITY - Episode Length (MOST CRITICAL)
    ep_len = metrics.get('episode_length')
    if ep_len is None:
        results['checks']['episode_length'] = ('FAIL', 'No data')
        results['diagnosis'] = 'No episode length data found. Training may not have started.'
        results['status'] = 'FAIL'
        return results
    elif ep_len < THRESHOLDS['episode_length'][0]:
        results['checks']['episode_length'] = ('FAIL', f'{ep_len:.1f} < {THRESHOLDS["episode_length"][0]}')
        results['diagnosis'] = f'CRITICAL: Episodes only {ep_len:.1f} steps! Robots dying immediately. CHECK TERMINATION LOGIC.'
        results['status'] = 'FAIL'
        return results
    elif ep_len < THRESHOLDS['episode_length'][1]:
        results['checks']['episode_length'] = ('WARN', f'{ep_len:.1f} < {THRESHOLDS["episode_length"][1]}')
        results['sanity_pass'] = True
    else:
        results['checks']['episode_length'] = ('PASS', f'{ep_len:.1f}')
        results['sanity_pass'] = True

    # Check 2: Stability - Upright Mean
    upright = metrics.get('upright_mean')
    if upright is None:
        results['checks']['upright_mean'] = ('FAIL', 'No data')
    elif upright < THRESHOLDS['upright_mean']:
        results['checks']['upright_mean'] = ('FAIL', f'{upright:.4f} < {THRESHOLDS["upright_mean"]}')
    else:
        results['checks']['upright_mean'] = ('PASS', f'{upright:.4f}')

    # Check 3: Height - Height Reward
    height = metrics.get('height_reward')
    if height is None:
        results['checks']['height_reward'] = ('FAIL', 'No data')
    elif height < THRESHOLDS['height_reward']:
        results['checks']['height_reward'] = ('FAIL', f'{height:.4f} < {THRESHOLDS["height_reward"]}')
    else:
        results['checks']['height_reward'] = ('PASS', f'{height:.4f}')

    # Check 4: Contact - Body Contact Penalty
    contact = metrics.get('body_contact')
    if contact is None:
        results['checks']['body_contact'] = ('FAIL', 'No data')
    elif contact < THRESHOLDS['body_contact']:
        results['checks']['body_contact'] = ('FAIL', f'{contact:.4f} < {THRESHOLDS["body_contact"]}')
    else:
        results['checks']['body_contact'] = ('PASS', f'{contact:.4f}')

    # Check 5: Walking - Forward Velocity
    vx = metrics.get('vx_w_mean')
    if vx is None:
        results['checks']['vx_w_mean'] = ('FAIL', 'No data')
    elif vx < THRESHOLDS['vx_w_mean']:
        if vx > 0:
            results['checks']['vx_w_mean'] = ('WARN', f'{vx:.4f} < {THRESHOLDS["vx_w_mean"]}')
        else:
            results['checks']['vx_w_mean'] = ('FAIL', f'{vx:.4f}')
    else:
        results['checks']['vx_w_mean'] = ('PASS', f'{vx:.4f}')

    # Determine overall status and diagnosis
    all_pass = all(c[0] == 'PASS' for c in results['checks'].values())
    any_fail = any(c[0] == 'FAIL' for c in results['checks'].values())

    results['all_pass'] = all_pass

    if all_pass:
        results['status'] = 'PASS'
        results['diagnosis'] = 'Robot is WALKING! All metrics pass. Verify with video if needed.'
    elif any_fail:
        results['status'] = 'FAIL'
        # Generate specific diagnosis
        if results['checks'].get('upright_mean', ('',))[0] == 'FAIL':
            results['diagnosis'] = 'Robot is FALLING OVER. Upright metric too low.'
        elif results['checks'].get('height_reward', ('',))[0] == 'FAIL':
            results['diagnosis'] = 'Robot is ON ELBOWS or COLLAPSED. Height reward too low.'
        elif results['checks'].get('body_contact', ('',))[0] == 'FAIL':
            results['diagnosis'] = 'Robot BODY ON GROUND. Body contact penalty too high.'
        elif results['checks'].get('vx_w_mean', ('',))[0] == 'FAIL':
            results['diagnosis'] = 'Robot is STANDING but moving BACKWARD.'
        else:
            results['diagnosis'] = 'Multiple metrics failing. Check individual results.'
    else:
        results['status'] = 'PARTIAL'
        if results['checks'].get('vx_w_mean', ('',))[0] == 'WARN':
            results['diagnosis'] = 'Robot is STANDING but not walking forward yet. Continue training.'
        else:
            results['diagnosis'] = 'Some metrics below optimal. Check warnings.'

    return results


def print_results(results: dict, use_color: bool = True):
    """Pretty-print validation results."""

    def color(text, code):
        if use_color:
            return f'\033[{code}m{text}\033[0m'
        return text

    RED = '91'
    GREEN = '92'
    YELLOW = '93'
    BOLD = '1'

    print()
    print(color('=== HAROLD TRAINING VALIDATOR ===', BOLD))
    print(f"Run: {results['run_name']}")
    print(f"Data points: {results['metrics'].get('data_points', 0)}")
    print()

    # Print each check
    check_names = {
        'episode_length': ('SANITY CHECK', 'Episode Length'),
        'upright_mean': ('Stability', 'Upright Mean'),
        'height_reward': ('Height', 'Height Reward'),
        'body_contact': ('Contact', 'Body Contact Penalty'),
        'vx_w_mean': ('Walking', 'Forward Velocity'),
    }

    for i, (key, (category, name)) in enumerate(check_names.items(), 1):
        status, value = results['checks'].get(key, ('?', '?'))

        if status == 'PASS':
            status_str = color('PASS', GREEN)
        elif status == 'WARN':
            status_str = color('WARN', YELLOW)
        else:
            status_str = color('FAIL', RED)

        prefix = color('[CRITICAL]', RED) if key == 'episode_length' and status == 'FAIL' else f'[{i}/5]'
        print(f"{prefix} {category}: {name}")
        print(f"      Value: {value}")
        print(f"      Status: {status_str}")
        print()

    # Print diagnosis
    print(color('=== DIAGNOSIS ===', BOLD))
    if results['status'] == 'PASS':
        print(color(results['diagnosis'], GREEN))
    elif results['status'] == 'FAIL':
        print(color(results['diagnosis'], RED))
    else:
        print(color(results['diagnosis'], YELLOW))
    print()

    # Print recommendation
    print(color('=== RECOMMENDATION ===', BOLD))
    if not results['sanity_pass']:
        print(color('DO NOT TRUST OTHER METRICS. Fix termination/spawning first.', RED))
        print('Common causes:')
        print('  - Height termination using world Z instead of height above terrain')
        print('  - Robot spawning in invalid pose')
        print('  - Orientation threshold too strict')
    elif results['status'] == 'PASS':
        print('Training is working. Consider:')
        print('  - Increasing training duration')
        print('  - Saving checkpoint for fine-tuning')
    elif results['status'] == 'PARTIAL':
        print('Training is progressing. Continue monitoring.')
    else:
        print('Check the failing metrics and adjust rewards/config accordingly.')

    print()


def list_runs(log_dir: str):
    """List recent training runs."""
    runs = sorted([
        d for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d)) and '2025-' in d
    ])

    print(f"\nRecent runs in {log_dir}:")
    for run in runs[-10:]:
        print(f"  {run}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Validate Harold training runs')
    parser.add_argument('run', nargs='?', help='Run name or path (default: latest)')
    parser.add_argument('--list', action='store_true', help='List recent runs')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    parser.add_argument('--log-dir', default='logs/skrl/harold_direct', help='Log directory')

    args = parser.parse_args()

    # Resolve log directory
    log_dir = args.log_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(PROJECT_ROOT, log_dir)

    if args.list:
        list_runs(log_dir)
        return 0

    # Find run to validate
    if args.run:
        if os.path.isabs(args.run):
            run_path = args.run
        else:
            run_path = os.path.join(log_dir, args.run)
    else:
        # Get latest run
        runs = sorted([
            d for d in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, d)) and '2025-' in d
        ])
        if not runs:
            print("No runs found in", log_dir)
            return 1
        run_path = os.path.join(log_dir, runs[-1])

    if not os.path.exists(run_path):
        print(f"Run not found: {run_path}")
        return 1

    # Validate
    results = validate_run(run_path)
    print_results(results, use_color=not args.no_color)

    # Return exit code based on results
    if not results['sanity_pass']:
        return 2  # Critical failure
    elif results['status'] == 'FAIL':
        return 1  # Failure
    else:
        return 0  # Success or partial


if __name__ == '__main__':
    sys.exit(main())
