#!/home/matteo/Desktop/env_isaaclab/bin/python
"""
Harold Training CLI - Unified observability tool for Harold robot training.

A deep module that provides simple interface for hypothesis-driven experimentation.
Hides TensorBoard complexity behind manifest files and comparison tools.

Usage:
    # Start experiment with metadata
    harold train --hypothesis "Higher backward penalty prevents regression" \
                 --tags "forward_motion,regression"

    # Check current status (state-only, no prescriptive suggestions)
    harold status                   # Current run with metrics
    harold status --json            # Machine-readable output

    # Validate a completed run
    harold validate                 # Latest run
    harold validate EXP-034         # By alias
    harold validate <run_id>        # By directory name

    # List recent runs
    harold runs                     # Last 10 runs with status
    harold runs --hypothesis        # Include hypothesis for each

    # Compare experiments side-by-side (essential for hypothesis-driven workflow)
    harold compare EXP-034 EXP-035  # Specific experiments
    harold compare                  # Last 5 experiments
    harold compare --tag forward_motion  # All with tag

    # Add observations to experiments
    harold note EXP-034 "Robot walked at 40-80% then regressed"

Exit Codes:
    0 = All metrics pass (robot walking)
    1 = Partial (standing but not walking)
    2 = Failing (on elbows, fallen, etc.)
    3 = Sanity failure (episodes too short)
    4 = Not running / no data
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "skrl" / "harold_direct"
INDEX_FILE = LOG_DIR / "experiments_index.json"
PID_FILE = Path("/tmp/harold_train.pid")
LOG_FILE = Path("/tmp/harold_train.log")
ENV_PATH = Path.home() / "Desktop" / "env_isaaclab" / "bin" / "activate"

# Metric thresholds
THRESHOLDS = {
    'episode_length': (100, 300),  # (minimum, expected)
    'upright_mean': 0.9,
    'height_reward': 2.0,
    'body_contact': -0.1,
    'vx_w_mean': 0.1,
}

# Metrics to extract for comparison (in priority order)
COMPARISON_METRICS = [
    'episode_length',
    'upright_mean',
    'height_reward',
    'body_contact',
    'vx_w_mean',
]


def get_latest_run() -> Path | None:
    """Get the most recent training run directory."""
    if not LOG_DIR.exists():
        return None
    runs = sorted([
        d for d in LOG_DIR.iterdir()
        if d.is_dir() and '2025-' in d.name
    ])
    return runs[-1] if runs else None


def get_metrics(run_path: Path) -> dict:
    """Extract metrics from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(str(run_path))
        ea.Reload()
    except Exception:
        return {}

    def avg_last(tag: str, n: int = 10) -> float | None:
        try:
            scalars = ea.Scalars(tag)
            if not scalars:
                return None
            last_n = scalars[-n:]
            return sum(s.value for s in last_n) / len(last_n)
        except KeyError:
            return None

    def get_count(tag: str) -> int:
        try:
            return len(ea.Scalars(tag))
        except KeyError:
            return 0

    return {
        'episode_length': avg_last('Episode / Total timesteps (mean)'),
        'upright_mean': avg_last('Info / Episode_Metric/upright_mean'),
        'height_reward': avg_last('Info / Episode_Reward/height_reward'),
        'body_contact': avg_last('Info / Episode_Reward/body_contact_penalty'),
        'vx_w_mean': avg_last('Info / Episode_Metric/vx_w_mean'),
        'reward_total': avg_last('Reward / Total reward (mean)'),
        'data_points': get_count('Info / Episode_Metric/vx_w_mean'),
    }


# === MANIFEST & INDEX SYSTEM ===
# Deep module: hides TensorBoard complexity behind simple manifest files

def load_index() -> dict:
    """Load the experiments index, creating if needed."""
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {
        'schema_version': 1,
        'experiments': {},
        'next_exp_number': 1,
        'tags': {}
    }


def save_index(index: dict) -> None:
    """Persist experiments index to disk."""
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2))


def get_next_alias(index: dict) -> str:
    """Generate next experiment alias (EXP-001, EXP-002, ...)."""
    num = index.get('next_exp_number', 1)
    return f"EXP-{num:03d}"


def resolve_experiment(exp_id: str) -> Path | None:
    """Resolve experiment alias or directory name to run path.

    Accepts:
      - Alias: "EXP-034"
      - Directory name: "2025-12-21_21-38-22_ppo_torch"
      - Full path: "/home/.../logs/skrl/harold_direct/2025-..."

    Returns None if not found (defines error out of existence).
    """
    if not exp_id:
        return get_latest_run()

    # Full path
    if os.path.isabs(exp_id):
        path = Path(exp_id)
        return path if path.exists() else None

    # Alias lookup
    if exp_id.startswith('EXP-'):
        index = load_index()
        dir_name = index.get('experiments', {}).get(exp_id)
        if dir_name:
            path = LOG_DIR / dir_name
            return path if path.exists() else None
        return None

    # Directory name
    path = LOG_DIR / exp_id
    return path if path.exists() else None


def get_manifest_path(run_path: Path) -> Path:
    """Get path to manifest.json for a run."""
    return run_path / 'manifest.json'


def load_manifest(run_path: Path) -> dict | None:
    """Load manifest from disk, returns None if not found."""
    manifest_path = get_manifest_path(run_path)
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            pass
    return None


def save_manifest(run_path: Path, manifest: dict) -> None:
    """Persist manifest to disk."""
    manifest_path = get_manifest_path(run_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def generate_manifest(run_path: Path) -> dict:
    """Generate manifest from TensorBoard and config files.

    Defines errors out of existence: always returns a valid manifest,
    even if TensorBoard data is incomplete or missing.
    """
    run_name = run_path.name

    # Extract metrics from TensorBoard
    metrics = get_metrics(run_path)

    # Determine verdict
    if metrics.get('episode_length') is not None:
        status, diagnosis, exit_code = get_diagnosis(metrics)
    else:
        status, diagnosis, exit_code = 'NO_DATA', 'No metrics available', 4

    # Check if run is still active
    running, pid, _ = is_training_running()
    if running:
        run_status = 'running'
    elif metrics.get('data_points', 0) > 0:
        run_status = 'completed'
    else:
        run_status = 'unknown'

    # Try to extract start time from directory name
    started_at = None
    try:
        # Format: 2025-12-21_21-38-22_ppo_torch
        date_part = run_name.split('_ppo_')[0]
        started_at = datetime.strptime(date_part, '%Y-%m-%d_%H-%M-%S').isoformat() + 'Z'
    except (ValueError, IndexError):
        started_at = datetime.utcnow().isoformat() + 'Z'

    return {
        'id': run_name,
        'alias': None,  # Will be set when registered in index
        'hypothesis': '',
        'tags': [],
        'started_at': started_at,
        'status': run_status,
        'notes': [],
        'summary': {
            'final': {k: metrics.get(k) for k in COMPARISON_METRICS},
            'verdict': status
        }
    }


def get_or_create_manifest(run_path: Path) -> dict:
    """Load manifest or generate from TensorBoard (lazy caching).

    Deep module: hides all TensorBoard parsing complexity.
    """
    manifest = load_manifest(run_path)
    if manifest is not None:
        # Refresh summary if run was in progress
        if manifest.get('status') == 'running':
            metrics = get_metrics(run_path)
            if metrics.get('data_points', 0) > 0:
                status, _, _ = get_diagnosis(metrics)
                manifest['summary'] = {
                    'final': {k: metrics.get(k) for k in COMPARISON_METRICS},
                    'verdict': status
                }
                # Check if still running
                running, _, _ = is_training_running()
                if not running:
                    manifest['status'] = 'completed'
                save_manifest(run_path, manifest)
        return manifest

    # Generate new manifest
    manifest = generate_manifest(run_path)
    save_manifest(run_path, manifest)
    return manifest


def register_experiment(run_path: Path, hypothesis: str = '', tags: list[str] = None) -> str:
    """Register a new experiment and return its alias."""
    index = load_index()
    alias = get_next_alias(index)

    # Update index
    index['experiments'][alias] = run_path.name
    index['next_exp_number'] = index.get('next_exp_number', 1) + 1

    # Update tag mappings
    if tags:
        for tag in tags:
            if tag not in index['tags']:
                index['tags'][tag] = []
            index['tags'][tag].append(alias)

    save_index(index)

    # Create or update manifest
    manifest = load_manifest(run_path) or generate_manifest(run_path)
    manifest['alias'] = alias
    manifest['hypothesis'] = hypothesis
    manifest['tags'] = tags or []
    save_manifest(run_path, manifest)

    return alias


def get_experiments_by_tag(tag: str) -> list[str]:
    """Get list of experiment aliases with given tag."""
    index = load_index()
    return index.get('tags', {}).get(tag, [])


def get_recent_experiments(n: int = 5) -> list[str]:
    """Get the n most recent experiment aliases."""
    if not LOG_DIR.exists():
        return []

    runs = sorted([
        d for d in LOG_DIR.iterdir()
        if d.is_dir() and '2025-' in d.name
    ])

    # Get aliases for runs that have them, or directory names
    index = load_index()
    reverse_index = {v: k for k, v in index.get('experiments', {}).items()}

    result = []
    for run in runs[-n:]:
        alias = reverse_index.get(run.name)
        result.append(alias if alias else run.name)

    return result


def get_diagnosis(metrics: dict) -> tuple[str, str, int]:
    """Analyze metrics and return state-only diagnosis.

    State-only reporting: describes current state without prescriptive suggestions.
    The agent interprets results and decides next steps.

    Returns:
        status: 'WALKING', 'STANDING', 'FAILING', 'SANITY_FAIL', 'NO_DATA'
        diagnosis: Factual description of current state
        exit_code: 0-4
    """
    ep_len = metrics.get('episode_length')
    height = metrics.get('height_reward')
    contact = metrics.get('body_contact')
    upright = metrics.get('upright_mean')
    vx = metrics.get('vx_w_mean')

    # No data
    if ep_len is None or height is None:
        return 'NO_DATA', 'No metrics available yet', 4

    # Sanity check
    if ep_len < THRESHOLDS['episode_length'][0]:
        return 'SANITY_FAIL', f'Episodes only {ep_len:.0f} steps (threshold: {THRESHOLDS["episode_length"][0]})', 3

    # Failing checks
    if height < THRESHOLDS['height_reward']:
        return 'FAILING', f'Height {height:.2f} below threshold {THRESHOLDS["height_reward"]}', 2

    if contact is not None and contact < THRESHOLDS['body_contact']:
        return 'FAILING', f'Body contact {contact:.2f} below threshold {THRESHOLDS["body_contact"]}', 2

    if upright is not None and upright < THRESHOLDS['upright_mean']:
        return 'FAILING', f'Upright {upright:.2f} below threshold {THRESHOLDS["upright_mean"]}', 2

    # Success checks
    if vx is not None and vx >= THRESHOLDS['vx_w_mean']:
        return 'WALKING', f'Forward velocity {vx:.3f} m/s exceeds threshold {THRESHOLDS["vx_w_mean"]}', 0

    # Partial success
    if vx is not None:
        return 'STANDING', f'Upright and stable, forward velocity {vx:.3f} m/s', 1

    return 'STANDING', 'Upright and stable', 1


def is_training_running() -> tuple[bool, int | None, float | None]:
    """Check if training is running. Returns (running, pid, elapsed_seconds)."""
    if not PID_FILE.exists():
        return False, None, None

    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        os.kill(pid, 0)

        # Get elapsed time
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'etimes='],
            capture_output=True, text=True
        )
        elapsed = float(result.stdout.strip()) if result.returncode == 0 else None
        return True, pid, elapsed
    except (ProcessLookupError, ValueError):
        return False, None, None


def format_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def get_progress(run_path: Path) -> float | None:
    """Estimate training progress from log file."""
    if not LOG_FILE.exists():
        return None

    try:
        # Look for iteration progress in log
        content = LOG_FILE.read_text()
        # Find patterns like "1234/4167" or similar
        import re
        matches = re.findall(r'(\d+)/(\d+)', content)
        if matches:
            current, total = map(int, matches[-1])
            return current / total
    except Exception:
        pass
    return None


# === SUBCOMMANDS ===

def cmd_train(args):
    """Start training in background with optional hypothesis and tags."""
    # Check if already running
    running, pid, _ = is_training_running()
    if running:
        print(f"ERROR: Training already running (PID: {pid})")
        print(f"Check status: harold status")
        print(f"Kill it: kill {pid}")
        return 1

    # Build command
    iterations = args.iterations or 4167  # ~100k timesteps

    cmd = [
        'python', str(PROJECT_ROOT / 'harold_isaac_lab' / 'scripts' / 'skrl' / 'train.py'),
        '--task=Template-Harold-Direct-flat-terrain-v0',
        '--num_envs', '4096',
        '--max_iterations', str(iterations),
        '--headless',
        '--video',
        '--video_interval', '6400',
        '--video_length', '250',
    ]

    if args.checkpoint:
        cmd.extend(['--checkpoint', str(args.checkpoint)])

    # Parse hypothesis and tags
    hypothesis = getattr(args, 'hypothesis', '') or ''
    tags = []
    if hasattr(args, 'tags') and args.tags:
        tags = [t.strip() for t in args.tags.split(',') if t.strip()]

    # Clear old log
    LOG_FILE.write_text('')

    # Run in background with nohup
    print(f"Starting Harold training in background...")
    print(f"  Iterations: {iterations}")
    print(f"  Environments: 4096")
    print(f"  Video recording: enabled")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")

    # Create shell command with source
    shell_cmd = f"source {ENV_PATH} && cd {PROJECT_ROOT} && {' '.join(cmd)} > {LOG_FILE} 2>&1 &"

    process = subprocess.Popen(
        ['bash', '-c', shell_cmd + f" echo $! > {PID_FILE}"],
        cwd=PROJECT_ROOT,
    )
    process.wait()

    # Wait briefly for PID file and run directory
    time.sleep(2)

    if PID_FILE.exists():
        pid_val = PID_FILE.read_text().strip()

        # Find the new run directory and register it
        run_path = get_latest_run()
        if run_path:
            alias = register_experiment(run_path, hypothesis=hypothesis, tags=tags)
            print(f"\n{alias}: {run_path.name}")
            if hypothesis:
                print(f"HYPOTHESIS: {hypothesis}")
            if tags:
                print(f"TAGS: {', '.join(tags)}")
        else:
            print(f"\nTraining started (PID: {pid_val})")

        print(f"\nMonitor: harold status")
        print(f"Logs: tail -f {LOG_FILE}")
        return 0
    else:
        print("ERROR: Failed to start training")
        return 1


def cmd_status(args):
    """Check training status and metrics (state-only reporting)."""
    running, pid, elapsed = is_training_running()
    run_path = get_latest_run()

    # Get manifest for alias and hypothesis
    manifest = None
    if run_path:
        manifest = get_or_create_manifest(run_path)

    if args.json:
        # Machine-readable output
        result = {
            'running': running,
            'pid': pid,
            'elapsed_seconds': elapsed,
            'run_name': run_path.name if run_path else None,
            'alias': manifest.get('alias') if manifest else None,
            'hypothesis': manifest.get('hypothesis') if manifest else None,
        }
        if run_path:
            result['metrics'] = get_metrics(run_path)
            status, diag, code = get_diagnosis(result['metrics'])
            result['status'] = status
            result['diagnosis'] = diag
            result['exit_code'] = code
        print(json.dumps(result, indent=2, default=str))
        return result.get('exit_code', 4)

    # Compact output with alias
    if run_path:
        alias = manifest.get('alias') if manifest else None
        if alias:
            print(f"RUN: {run_path.name} ({alias})")
        else:
            print(f"RUN: {run_path.name}")
        if manifest and manifest.get('hypothesis'):
            print(f"HYPOTHESIS: {manifest['hypothesis']}")
    else:
        print("RUN: (none)")

    # Status line
    if running:
        progress = get_progress(run_path)
        elapsed_str = format_elapsed(elapsed) if elapsed else "?"
        progress_str = f"{int(progress * 100)}%" if progress else "?"
        print(f"STATUS: RUNNING ({progress_str}, {elapsed_str} elapsed)")
    else:
        print("STATUS: NOT RUNNING")

    # Get metrics
    if run_path:
        metrics = get_metrics(run_path)
        reward = metrics.get('reward_total')
        print(f"REWARD: {reward:.1f}" if reward else "REWARD: (no data)")

        # Sanity check
        ep_len = metrics.get('episode_length')
        if ep_len is not None:
            status = 'PASS' if ep_len >= THRESHOLDS['episode_length'][0] else 'FAIL'
            print(f"SANITY: {status} (ep_len={ep_len:.0f})")
        else:
            print("SANITY: (no data)")

        # Standing check
        height = metrics.get('height_reward')
        contact = metrics.get('body_contact')
        if height is not None:
            status = 'PASS' if height >= THRESHOLDS['height_reward'] else 'FAIL'
            contact_str = f", contact={contact:.2f}" if contact is not None else ""
            print(f"STANDING: {status} (height={height:.2f}{contact_str})")
        else:
            print("STANDING: (no data)")

        # Walking check
        vx = metrics.get('vx_w_mean')
        if vx is not None:
            if vx >= THRESHOLDS['vx_w_mean']:
                status = 'PASS'
            elif vx > 0:
                status = 'WARN'
            else:
                status = 'FAIL'
            print(f"WALKING: {status} (vx={vx:.3f}, need >0.1)")
        else:
            print("WALKING: (no data)")

        # Diagnosis (state-only, no NEXT field)
        status, diagnosis, exit_code = get_diagnosis(metrics)
        print(f"VERDICT: {status}")
        print(f"DIAGNOSIS: {diagnosis}")
        return exit_code
    else:
        print("REWARD: (no runs)")
        print("SANITY: (no runs)")
        print("STANDING: (no runs)")
        print("WALKING: (no runs)")
        print("VERDICT: NO_DATA")
        print("DIAGNOSIS: No training runs found")
        return 4


def cmd_validate(args):
    """Validate a completed training run (state-only reporting)."""
    # Find run to validate - support aliases
    run_path = resolve_experiment(args.run) if args.run else get_latest_run()

    if not run_path or not run_path.exists():
        print(f"ERROR: Run not found: {args.run}")
        return 4

    # Get manifest for alias and hypothesis
    manifest = get_or_create_manifest(run_path)
    alias = manifest.get('alias', '')

    if alias:
        print(f"Validating: {run_path.name} ({alias})")
    else:
        print(f"Validating: {run_path.name}")

    if manifest.get('hypothesis'):
        print(f"HYPOTHESIS: {manifest['hypothesis']}")
    print("-" * 50)

    metrics = get_metrics(run_path)
    if not metrics:
        print("ERROR: Could not read metrics")
        return 4

    print(f"Data points: {metrics.get('data_points', 0)}")
    print()

    # Print each metric
    checks = [
        ('episode_length', 'Episode Length', f"> {THRESHOLDS['episode_length'][0]}"),
        ('upright_mean', 'Upright Mean', f"> {THRESHOLDS['upright_mean']}"),
        ('height_reward', 'Height Reward', f"> {THRESHOLDS['height_reward']}"),
        ('body_contact', 'Body Contact', f"> {THRESHOLDS['body_contact']}"),
        ('vx_w_mean', 'Forward Velocity', f"> {THRESHOLDS['vx_w_mean']} m/s"),
    ]

    for key, name, threshold_str in checks:
        val = metrics.get(key)
        if val is None:
            print(f"  {name}: NO DATA")
            continue

        # Determine pass/fail
        if key == 'episode_length':
            passed = val >= THRESHOLDS['episode_length'][0]
        elif key == 'body_contact':
            passed = val >= THRESHOLDS[key]
        else:
            passed = val >= THRESHOLDS.get(key, 0)

        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {val:.4f} ({status}, need {threshold_str})")

    print()
    status, diagnosis, exit_code = get_diagnosis(metrics)
    print(f"VERDICT: {status}")
    print(f"DIAGNOSIS: {diagnosis}")

    return exit_code


def cmd_runs(args):
    """List recent training runs with optional hypothesis display."""
    if not LOG_DIR.exists():
        print("No runs directory found")
        return 0

    runs = sorted([
        d for d in LOG_DIR.iterdir()
        if d.is_dir() and '2025-' in d.name
    ])

    if not runs:
        print("No runs found")
        return 0

    show_hypothesis = getattr(args, 'hypothesis', False)
    index = load_index()
    reverse_index = {v: k for k, v in index.get('experiments', {}).items()}

    print(f"Recent runs in {LOG_DIR}:")
    print("-" * 70)

    # Show last 10 runs with basic status
    for run in runs[-10:]:
        alias = reverse_index.get(run.name, '')
        manifest = load_manifest(run) if show_hypothesis else None

        metrics = get_metrics(run)
        if metrics.get('episode_length'):
            status, _, _ = get_diagnosis(metrics)
            ep_len = metrics.get('episode_length', 0)
            vx = metrics.get('vx_w_mean', 0)

            alias_str = f" ({alias})" if alias else ""
            print(f"  {run.name}{alias_str}  {status:12s}  ep={ep_len:.0f}  vx={vx:.3f}")

            if show_hypothesis and manifest and manifest.get('hypothesis'):
                print(f"    -> {manifest['hypothesis'][:60]}...")
        else:
            alias_str = f" ({alias})" if alias else ""
            print(f"  {run.name}{alias_str}  (no data)")

    return 0


def cmd_compare(args):
    """Compare multiple experiments side-by-side.

    Deep module: hides TensorBoard parsing, manifest loading, and
    metric aggregation behind a simple comparison interface.
    """
    # Resolve which experiments to compare
    if args.experiments:
        exp_ids = args.experiments
    elif args.tag:
        exp_ids = get_experiments_by_tag(args.tag)
        if not exp_ids:
            print(f"No experiments found with tag: {args.tag}")
            return 1
    else:
        exp_ids = get_recent_experiments(5)

    if len(exp_ids) < 2:
        print("Need at least 2 experiments to compare")
        print("Usage: harold compare EXP-034 EXP-035")
        return 1

    # Resolve to run paths and load manifests
    experiments = []
    for exp_id in exp_ids:
        run_path = resolve_experiment(exp_id)
        if run_path:
            manifest = get_or_create_manifest(run_path)
            experiments.append({
                'id': exp_id,
                'path': run_path,
                'manifest': manifest,
            })
        else:
            print(f"Warning: Could not find experiment {exp_id}")

    if len(experiments) < 2:
        print("Not enough valid experiments to compare")
        return 1

    # Header
    exp_labels = [e['manifest'].get('alias') or e['id'] for e in experiments]
    print(f"COMPARISON: {' vs '.join(exp_labels)}")
    print("=" * 70)
    print()

    # Hypotheses
    print("HYPOTHESES:")
    for e in experiments:
        label = e['manifest'].get('alias') or e['id']
        hyp = e['manifest'].get('hypothesis', '(none)')
        print(f"  {label}: {hyp}")
    print()

    # Metrics table
    print("METRICS (final):")
    header = "                   " + "".join(f"{e['manifest'].get('alias') or e['id']:>12}" for e in experiments) + "   threshold"
    print(header)

    for metric in COMPARISON_METRICS:
        row = f"  {metric:17s}"
        for e in experiments:
            val = e['manifest'].get('summary', {}).get('final', {}).get(metric)
            if val is not None:
                row += f"{val:>12.3f}"
            else:
                row += f"{'?':>12}"

        # Add threshold
        threshold = THRESHOLDS.get(metric)
        if isinstance(threshold, tuple):
            threshold = threshold[0]
        if threshold is not None:
            row += f"   > {threshold}"
        print(row)
    print()

    # Verdicts
    print("VERDICT:")
    for e in experiments:
        label = e['manifest'].get('alias') or e['id']
        verdict = e['manifest'].get('summary', {}).get('verdict', 'UNKNOWN')
        print(f"  {label}: {verdict}")

    return 0


def cmd_note(args):
    """Add a note to an experiment."""
    run_path = resolve_experiment(args.experiment)

    if not run_path:
        print(f"ERROR: Experiment not found: {args.experiment}")
        return 1

    manifest = get_or_create_manifest(run_path)

    # Add the note
    note = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'text': args.note
    }
    manifest.setdefault('notes', []).append(note)
    save_manifest(run_path, manifest)

    alias = manifest.get('alias', args.experiment)
    print(f"Note added to {alias}")
    print(f"  {note['text']}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Harold Training CLI - Unified observability tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # train
    train_parser = subparsers.add_parser('train', help='Start training in background')
    train_parser.add_argument('--iterations', type=int, help='Max iterations (default: 4167 = ~100k timesteps)')
    train_parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--hypothesis', type=str, help='Hypothesis being tested (stored with experiment)')
    train_parser.add_argument('--tags', type=str, help='Comma-separated tags for categorization')

    # status
    status_parser = subparsers.add_parser('status', help='Check training status and metrics')
    status_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # validate
    validate_parser = subparsers.add_parser('validate', help='Validate a completed run')
    validate_parser.add_argument('run', nargs='?', help='Run name, alias (EXP-NNN), or path (default: latest)')

    # runs
    runs_parser = subparsers.add_parser('runs', help='List recent training runs')
    runs_parser.add_argument('--hypothesis', action='store_true', help='Show hypothesis for each run')

    # compare
    compare_parser = subparsers.add_parser('compare', help='Compare experiments side-by-side')
    compare_parser.add_argument('experiments', nargs='*', help='Experiment aliases or names to compare')
    compare_parser.add_argument('--tag', type=str, help='Compare all experiments with this tag')

    # note
    note_parser = subparsers.add_parser('note', help='Add a note to an experiment')
    note_parser.add_argument('experiment', help='Experiment alias or name')
    note_parser.add_argument('note', help='Note text to add')

    args = parser.parse_args()

    if args.command == 'train':
        return cmd_train(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'validate':
        return cmd_validate(args)
    elif args.command == 'runs':
        return cmd_runs(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'note':
        return cmd_note(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
