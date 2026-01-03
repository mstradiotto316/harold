#!/home/matteo/Desktop/env_isaaclab/bin/python
"""
Harold Training CLI - Unified observability tool for Harold robot training.

A deep module that provides simple interface for hypothesis-driven experimentation.
Hides TensorBoard complexity behind manifest files and comparison tools.

Usage:
    # Start experiment with metadata
    harold train --hypothesis "Higher backward penalty prevents regression" \
                 --tags "forward_motion,regression"

    # Use duration presets (short/standard/long)
    harold train --duration standard

    # Run scripted gait or CPG modes
    harold train --mode scripted
    harold train --mode cpg

    # Train a different task
    harold train --task rough

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
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "skrl" / "harold_direct"
INDEX_FILE = LOG_DIR / "experiments_index.json"
PID_FILE = Path("/tmp/harold_train.pid")
LOG_FILE = Path("/tmp/harold_train.log")
WATCHDOG_PID_FILE = Path("/tmp/harold_watchdog.pid")
WATCHDOG_LOG_FILE = Path("/tmp/harold_watchdog.log")
WATCHDOG_KILL_MARKER = Path("/tmp/harold_watchdog_killed.json")
ENV_PATH = Path.home() / "Desktop" / "env_isaaclab" / "bin" / "activate"
RUN_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_ppo_")

# Training defaults (single source of truth for run configuration)
TASK_IDS = {
    'flat': 'Template-Harold-Direct-flat-terrain-v0',
    'rough': 'Template-Harold-Direct-rough-terrain-v0',
    'pushup': 'Template-Harold-Direct-pushup-v0',
}
DEFAULT_TASK = 'flat'
TRAINING_DEFAULTS = {
    'num_envs': 8192,
    'video_interval': 3200,
    'video_length': 250,
    'rendering_mode': 'performance',
}
DURATION_PRESETS = {
    'short': 1250,     # ~30 minutes
    'standard': 2500,  # ~60 minutes
    'long': 4167,      # ~100 minutes
}
DEFAULT_DURATION = 'short'
MODE_CHOICES = ('rl', 'cpg', 'scripted')
DATA_POINTS_TAG = 'Info / Episode_Metric/vx_w_mean'

# Memory safety (prevents OOM-induced system hangs)
# Only intervene at truly dangerous levels to avoid interrupting legitimate training
RAM_KILL_THRESHOLD = 95   # Kill only when RAM critically full
SWAP_KILL_THRESHOLD = 70  # Kill only when swap heavily used (thrashing imminent)


# === METRIC SPECIFICATION (Single Source of Truth) ===
# All metric-related code derives from this list. To add a metric, add one line here.

@dataclass
class MetricSpec:
    """Specification for a training metric."""
    key: str                # Internal key used in dicts
    tensorboard_tag: str | tuple[str, ...]    # TensorBoard scalar tag path(s)
    threshold: float        # Pass threshold value
    compare_gt: bool        # True = value > threshold is PASS
    display_name: str       # Human-readable name for output

@dataclass
class AuxMetricSpec:
    """Specification for auxiliary metrics (no thresholds)."""
    key: str
    tensorboard_tag: str | tuple[str, ...]
    display_name: str

METRICS = [
    MetricSpec('episode_length', 'Episode / Total timesteps (mean)', 300, True, 'Episode Length'),  # Session 35: raised from 100 (15s minimum for stable walking)
    MetricSpec('upright_mean', 'Info / Episode_Metric/upright_mean', 0.9, True, 'Upright Mean'),
    MetricSpec('height_reward', ('Info / Episode_Reward/height_reward', 'Info / Episode_Metric/height_reward'), 0.5, True, 'Height Reward'),  # Session 24: lowered from 1.2 (CPG gait has different natural height)
    MetricSpec('body_contact', ('Info / Episode_Reward/body_contact_penalty', 'Info / Episode_Metric/body_contact_penalty'), -0.1, True, 'Body Contact'),
    MetricSpec('vx_w_mean', 'Info / Episode_Metric/vx_w_mean', 0.01, True, 'Forward Velocity'),  # Session 24: lowered from 0.1 (slow controlled gait is acceptable)
]

AUX_METRICS = [
    AuxMetricSpec('x_displacement', 'Info / Episode_Metric/x_displacement', 'X Displacement'),
    AuxMetricSpec('x_displacement_abs', 'Info / Episode_Metric/x_displacement_abs', 'Abs X Displacement'),
]

# Derived lookups (computed once at import time)
METRIC_BY_KEY = {m.key: m for m in METRICS}
METRIC_KEYS = [m.key for m in METRICS]


@dataclass
class TrainingStatus:
    """Status of the training process."""
    running: bool
    pid: int | None = None
    elapsed_seconds: float | None = None


@dataclass
class DiagnosisResult:
    """Result of analyzing training metrics.

    Replaces tuple return from get_diagnosis() - clearer than (str, str, int).
    """
    status: str        # 'WALKING', 'STANDING', 'FAILING', 'SANITY_FAIL', 'NO_DATA'
    diagnosis: str     # Human-readable description
    exit_code: int     # 0=walking, 1=standing, 2=failing, 3=sanity, 4=no data


def get_latest_run() -> Path | None:
    """Get the most recent training run directory."""
    if not LOG_DIR.exists():
        return None
    runs = sorted([
        d for d in LOG_DIR.iterdir()
        if d.is_dir() and RUN_DIR_PATTERN.match(d.name)
    ])
    return runs[-1] if runs else None


def wait_for_new_run(previous: Path | None, timeout_s: int = 600) -> Path | None:
    """Wait for a new run directory to appear after training starts."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        latest = get_latest_run()
        if latest and (previous is None or latest.name != previous.name):
            return latest
        time.sleep(2)
    return get_latest_run()


def get_metrics(run_path: Path) -> dict:
    """Extract metrics from TensorBoard logs.

    Uses METRICS as single source of truth for which metrics to extract.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(str(run_path))
        ea.Reload()
    except Exception:
        return {}

    def _as_tags(tags: str | tuple[str, ...]) -> list[str]:
        if isinstance(tags, (list, tuple)):
            return list(tags)
        return [tags]

    def avg_last(tags: str | tuple[str, ...], n: int = 10) -> float | None:
        for tag in _as_tags(tags):
            try:
                scalars = ea.Scalars(tag)
            except KeyError:
                continue
            if not scalars:
                continue
            last_n = scalars[-n:]
            return sum(s.value for s in last_n) / len(last_n)
        return None

    def get_count(tags: str | tuple[str, ...]) -> int:
        for tag in _as_tags(tags):
            try:
                return len(ea.Scalars(tag))
            except KeyError:
                continue
        return 0

    # Extract metrics from spec lists
    result = {spec.key: avg_last(spec.tensorboard_tag) for spec in METRICS}
    for spec in AUX_METRICS:
        result[spec.key] = avg_last(spec.tensorboard_tag)

    # Add derived metrics (not in METRICS spec)
    result['reward_total'] = avg_last('Reward / Total reward (mean)')
    result['data_points'] = get_count(DATA_POINTS_TAG)

    return result


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


def load_manifest(run_path: Path) -> dict | None:
    """Load manifest from disk, returns None if not found."""
    manifest_path = run_path / 'manifest.json'
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            pass
    return None


def save_manifest(run_path: Path, manifest: dict) -> None:
    """Persist manifest to disk."""
    (run_path / 'manifest.json').write_text(json.dumps(manifest, indent=2))


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
        diag = get_diagnosis(metrics)
    else:
        diag = DiagnosisResult('NO_DATA', 'No metrics available', 4)

    # Check if run is still active
    train_status = is_training_running()
    if train_status.running:
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
            'final': {k: metrics.get(k) for k in METRIC_KEYS},
            'verdict': diag.status
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
                diag = get_diagnosis(metrics)
                manifest['summary'] = {
                    'final': {k: metrics.get(k) for k in METRIC_KEYS},
                    'verdict': diag.status
                }
                # Check if still running
                if not is_training_running().running:
                    manifest['status'] = 'completed'
                save_manifest(run_path, manifest)
        return manifest

    # Generate new manifest
    manifest = generate_manifest(run_path)
    save_manifest(run_path, manifest)
    return manifest


def register_experiment(
    run_path: Path,
    hypothesis: str = '',
    tags: list[str] = None,
    training_config: dict = None
) -> str:
    """Register a new experiment and return its alias.

    Args:
        run_path: Path to the experiment directory
        hypothesis: Hypothesis being tested
        tags: List of tags for categorization
        training_config: Training parameters (num_envs, iterations) for STATUS display
    """
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
    if training_config:
        manifest['training_config'] = training_config
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
        if d.is_dir() and RUN_DIR_PATTERN.match(d.name)
    ])

    # Get aliases for runs that have them, or directory names
    index = load_index()
    reverse_index = {v: k for k, v in index.get('experiments', {}).items()}

    result = []
    for run in runs[-n:]:
        alias = reverse_index.get(run.name)
        result.append(alias if alias else run.name)

    return result


def metric_passes(key: str, value: float | None) -> bool:
    """Check if a metric value passes its threshold."""
    if value is None:
        return False
    spec = METRIC_BY_KEY[key]
    return value >= spec.threshold if spec.compare_gt else value <= spec.threshold


def format_metric_line(key: str, value: float | None, show_threshold: bool = True) -> str:
    """Format a metric for display with pass/fail status.

    Reduces repetition across cmd_status(), cmd_validate(), etc.
    """
    spec = METRIC_BY_KEY[key]
    if value is None:
        return f"  {spec.display_name}: (no data)"

    passed = metric_passes(key, value)
    status = "PASS" if passed else "FAIL"
    cmp = ">" if spec.compare_gt else "<"

    if show_threshold:
        return f"  {spec.display_name}: {value:.4f} ({status}, need {cmp} {spec.threshold})"
    else:
        return f"  {spec.display_name}: {value:.4f} ({status})"


def get_diagnosis(metrics: dict) -> DiagnosisResult:
    """Analyze metrics and return state-only diagnosis.

    State-only reporting: describes current state without prescriptive suggestions.
    The agent interprets results and decides next steps.
    """
    ep_len = metrics.get('episode_length')
    height = metrics.get('height_reward')
    contact = metrics.get('body_contact')
    upright = metrics.get('upright_mean')
    vx = metrics.get('vx_w_mean')

    # No data
    if ep_len is None or height is None:
        return DiagnosisResult('NO_DATA', 'No metrics available yet', 4)

    # Sanity check (episode length)
    ep_spec = METRIC_BY_KEY['episode_length']
    if not metric_passes('episode_length', ep_len):
        return DiagnosisResult('SANITY_FAIL', f'Episodes only {ep_len:.0f} steps (threshold: {ep_spec.threshold})', 3)

    # Failing checks
    height_spec = METRIC_BY_KEY['height_reward']
    if not metric_passes('height_reward', height):
        return DiagnosisResult('FAILING', f'Height {height:.2f} below threshold {height_spec.threshold}', 2)

    contact_spec = METRIC_BY_KEY['body_contact']
    if contact is not None and not metric_passes('body_contact', contact):
        return DiagnosisResult('FAILING', f'Body contact {contact:.2f} below threshold {contact_spec.threshold}', 2)

    upright_spec = METRIC_BY_KEY['upright_mean']
    if upright is not None and not metric_passes('upright_mean', upright):
        return DiagnosisResult('FAILING', f'Upright {upright:.2f} below threshold {upright_spec.threshold}', 2)

    # Success checks
    vx_spec = METRIC_BY_KEY['vx_w_mean']
    if vx is not None and metric_passes('vx_w_mean', vx):
        return DiagnosisResult('WALKING', f'Forward velocity {vx:.3f} m/s exceeds threshold {vx_spec.threshold}', 0)

    # Partial success
    if vx is not None:
        return DiagnosisResult('STANDING', f'Upright and stable, forward velocity {vx:.3f} m/s', 1)

    return DiagnosisResult('STANDING', 'Upright and stable', 1)


def is_training_running() -> TrainingStatus:
    """Check if training is running. Returns TrainingStatus with running, pid, elapsed."""
    if not PID_FILE.exists():
        return TrainingStatus(running=False)

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
        return TrainingStatus(running=True, pid=pid, elapsed_seconds=elapsed)
    except (ProcessLookupError, ValueError):
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
        return TrainingStatus(running=False)


def get_kill_info() -> dict | None:
    """Check if training was killed by watchdog. Returns kill info or None."""
    if not WATCHDOG_KILL_MARKER.exists():
        return None
    try:
        return json.loads(WATCHDOG_KILL_MARKER.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def clear_kill_marker() -> None:
    """Clear the watchdog kill marker (called when starting new training)."""
    if WATCHDOG_KILL_MARKER.exists():
        WATCHDOG_KILL_MARKER.unlink()


def find_training_processes() -> list[dict]:
    """Find all Isaac Lab training processes (including orphans)."""
    processes = []
    try:
        result = subprocess.run(
            ['pgrep', '-af', 'train.py.*Template-Harold'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    pid = int(parts[0])
                    cmd = parts[1] if len(parts) > 1 else ''
                    # Get elapsed time
                    ps_result = subprocess.run(
                        ['ps', '-p', str(pid), '-o', 'etimes='],
                        capture_output=True, text=True
                    )
                    elapsed = float(ps_result.stdout.strip()) if ps_result.returncode == 0 else 0
                    processes.append({'pid': pid, 'cmd': cmd, 'elapsed': elapsed})
    except Exception:
        pass
    return processes


def kill_training(pid: int) -> bool:
    """Kill a training process and its children."""
    try:
        # Try to kill the process group first
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)

        # Wait briefly then force kill if needed
        time.sleep(2)
        try:
            os.kill(pid, 0)  # Check if still alive
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Already dead

        return True
    except Exception as e:
        print(f"Error killing PID {pid}: {e}")
        return False


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


def get_progress(run_path: Path) -> tuple[float | None, int | None, int | None]:
    """Estimate training progress from log file.

    Returns: (progress_fraction, current_iteration, total_iterations)
    """
    if not LOG_FILE.exists():
        return None, None, None

    try:
        # Look for iteration progress in log
        content = LOG_FILE.read_text()
        # Find patterns like "1234/4167" or similar
        import re
        matches = re.findall(r'(\d+)/(\d+)', content)
        if matches:
            current, total = map(int, matches[-1])
            return current / total, current, total
    except Exception:
        pass
    return None, None, None


def get_training_rate() -> tuple[float | None, float | None]:
    """Parse iterations per second from training log.

    Looks for tqdm-style output like '17.02it/s' or '6.31it/s'.
    Returns: (current_rate, average_rate) or (None, None) if not found.
    """
    if not LOG_FILE.exists():
        return None, None

    try:
        import re
        content = LOG_FILE.read_text()
        # Match patterns like "17.02it/s" or "6.31it/s"
        matches = re.findall(r'(\d+\.?\d*)it/s', content)
        if matches:
            rates = [float(r) for r in matches]
            current = rates[-1]
            avg = sum(rates) / len(rates)
            return current, avg
    except Exception:
        pass
    return None, None


# === SUBCOMMANDS ===

def build_train_command(
    num_envs: int,
    iterations: int,
    task_id: str,
    checkpoint: str | None = None,
) -> list[str]:
    """Build the Isaac Lab training command.

    Encapsulates command construction and benchmark-based defaults.
    """
    # Benchmark results (2025-12-25, 64GB RAM):
    #   4096 envs: 18.0 it/s, 1.77M samples/s, GPU 4.3GB, RAM  8GB
    #   6144 envs: 15.2 it/s, 2.25M samples/s, GPU 5.0GB, RAM  9GB
    #   8192 envs: 12.9 it/s, 2.54M samples/s, GPU 5.6GB, RAM  9GB  <- DEFAULT
    #  10000 envs: 11.3 it/s, 2.71M samples/s, GPU 6.0GB, RAM 10GB
    #  12000 envs: 10.6 it/s, 3.05M samples/s, GPU 6.4GB, RAM 10GB
    #  16384 envs:  8.7 it/s, 3.43M samples/s, GPU 7.6GB, RAM 11GB  <- MAX THROUGHPUT
    cmd = [
        'python', str(PROJECT_ROOT / 'harold_isaac_lab' / 'scripts' / 'skrl' / 'train.py'),
        f'--task={task_id}',
        '--num_envs', str(num_envs),
        '--max_iterations', str(iterations),
        '--headless',
        '--rendering_mode', TRAINING_DEFAULTS['rendering_mode'],
        '--video',
        '--video_interval', str(TRAINING_DEFAULTS['video_interval']),
        '--video_length', str(TRAINING_DEFAULTS['video_length']),
    ]
    if checkpoint:
        cmd.extend(['--checkpoint', str(checkpoint)])
    return cmd


def start_watchdog(pid: str) -> bool:
    """Start memory watchdog for the given training PID.

    Returns True if started successfully.
    """
    watchdog_script = PROJECT_ROOT / 'scripts' / 'memory_watchdog.py'
    if not watchdog_script.exists():
        return False

    watchdog_cmd = (
        f"python {watchdog_script} --pid {pid} "
        f"--ram-kill {RAM_KILL_THRESHOLD} --swap-kill {SWAP_KILL_THRESHOLD} "
        f"> {WATCHDOG_LOG_FILE} 2>&1 & echo $! > {WATCHDOG_PID_FILE}"
    )
    subprocess.run(['bash', '-c', watchdog_cmd])
    return True


def cmd_train(args):
    """Start training in background with optional hypothesis and tags."""
    # Check if already running
    train_status = is_training_running()
    if train_status.running:
        print(f"ERROR: Training already running (PID: {train_status.pid})")
        print(f"Check status: harold status")
        print(f"Kill it: kill {train_status.pid}")
        return 1

    existing_processes = find_training_processes()
    if existing_processes:
        print("ERROR: Found existing training processes not tracked by PID file.")
        for proc in existing_processes:
            elapsed_str = format_elapsed(proc['elapsed'])
            print(f"  PID {proc['pid']} (running {elapsed_str})")
        print("Run: harold stop")
        return 1

    # Parse arguments with defaults
    task_key = getattr(args, 'task', DEFAULT_TASK) or DEFAULT_TASK
    if task_key not in TASK_IDS:
        print(f"ERROR: Unknown task '{task_key}'. Valid: {', '.join(TASK_IDS.keys())}")
        return 1
    task_id = TASK_IDS[task_key]

    if args.iterations and args.duration:
        print("ERROR: Use either --iterations or --duration, not both.")
        return 1

    if args.iterations:
        iterations = args.iterations
        duration_label = None
    else:
        duration_label = args.duration or DEFAULT_DURATION
        iterations = DURATION_PRESETS[duration_label]

    if getattr(args, 'num_envs', None) is None:
        num_envs = 1 if task_key == 'pushup' else TRAINING_DEFAULTS['num_envs']
    else:
        num_envs = args.num_envs

    hypothesis = getattr(args, 'hypothesis', '') or ''
    tags = [t.strip() for t in args.tags.split(',') if t.strip()] if getattr(args, 'tags', None) else []

    mode = args.mode

    # Capture latest run before launch (used to detect new run directory)
    previous_run = get_latest_run()

    # Build command
    cmd = build_train_command(num_envs, iterations, task_id, args.checkpoint)

    # Clear old log and kill marker
    LOG_FILE.write_text('')
    clear_kill_marker()

    # Print startup info
    print(f"Starting Harold training in background...")
    print(f"  Task: {task_key}")
    if duration_label:
        print(f"  Duration: {duration_label} ({iterations} iterations)")
    else:
        print(f"  Iterations: {iterations}")
    print(f"  Environments: {num_envs}")
    print(f"  Video recording: enabled")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Mode: {mode}")
    if getattr(args, 'gait_scale', None) is not None:
        print(f"  Gait scale: {args.gait_scale}")

    # Launch training in background
    # Build environment variable prefix (explicitly overrides inherited env)
    env_vars = {
        "HAROLD_CPG": "1" if mode == "cpg" else "0",
        "HAROLD_SCRIPTED_GAIT": "1" if mode == "scripted" else "0",
    }
    if getattr(args, 'gait_scale', None) is not None:
        env_vars["HAROLD_GAIT_AMP_SCALE"] = str(args.gait_scale)
    else:
        env_vars["HAROLD_GAIT_AMP_SCALE"] = ""
    env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())
    if env_prefix:
        env_prefix += " "

    shell_cmd = f"source {ENV_PATH} && cd {PROJECT_ROOT} && {env_prefix}{' '.join(cmd)} > {LOG_FILE} 2>&1 &"
    process = subprocess.Popen(
        ['bash', '-c', shell_cmd + f" echo $! > {PID_FILE}"],
        cwd=PROJECT_ROOT,
    )
    process.wait()
    time.sleep(2)

    if not PID_FILE.exists():
        print("ERROR: Failed to start training")
        return 1

    pid_val = PID_FILE.read_text().strip()

    # Start memory watchdog
    if not getattr(args, 'no_watchdog', False) and not getattr(args, 'no-watchdog', False):
        if start_watchdog(pid_val):
            print(f"  Memory watchdog: active (kills at RAM>{RAM_KILL_THRESHOLD}% or Swap>{SWAP_KILL_THRESHOLD}%)")

    # Register experiment with training config (wait for new run directory)
    run_path = wait_for_new_run(previous_run)
    if run_path:
        training_config = {
            'num_envs': num_envs,
            'iterations': iterations,
            'task': task_key,
            'mode': mode,
        }
        if duration_label:
            training_config['duration'] = duration_label
        if getattr(args, 'gait_scale', None) is not None:
            training_config['gait_scale'] = args.gait_scale
        alias = register_experiment(
            run_path,
            hypothesis=hypothesis,
            tags=tags,
            training_config=training_config
        )
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


def cmd_status(args):
    """Check training status and metrics (state-only reporting)."""
    train_status = is_training_running()
    run_path = get_latest_run()

    # Get manifest for alias and hypothesis
    manifest = None
    if run_path:
        manifest = get_or_create_manifest(run_path)

    if args.json:
        # Machine-readable output
        progress, current_iter, total_iter = get_progress(run_path) if run_path else (None, None, None)
        result = {
            'running': train_status.running,
            'pid': train_status.pid,
            'elapsed_seconds': train_status.elapsed_seconds,
            'run_name': run_path.name if run_path else None,
            'alias': manifest.get('alias') if manifest else None,
            'hypothesis': manifest.get('hypothesis') if manifest else None,
            'training_config': manifest.get('training_config') if manifest else None,
            'progress': progress,
            'current_iteration': current_iter,
            'total_iterations': total_iter,
            'iterations_per_second': get_training_rate()[0] if train_status.running else None,
            'iterations_per_second_avg': get_training_rate()[1] if train_status.running else None,
            'killed_by_watchdog': None,
            'orphan_pids': [],
        }
        # Check if watchdog killed training
        if not train_status.running:
            kill_info = get_kill_info()
            if kill_info:
                result['killed_by_watchdog'] = kill_info
            else:
                result['orphan_pids'] = [p['pid'] for p in find_training_processes()]
        if run_path:
            result['metrics'] = get_metrics(run_path)
            diag = get_diagnosis(result['metrics'])
            result['status'] = diag.status
            result['diagnosis'] = diag.diagnosis
            result['exit_code'] = diag.exit_code
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
        if manifest and manifest.get('training_config'):
            cfg = manifest['training_config']
            config_parts = []
            if cfg.get('task'):
                config_parts.append(f"task={cfg['task']}")
            if cfg.get('mode'):
                config_parts.append(f"mode={cfg['mode']}")
            if cfg.get('duration'):
                config_parts.append(f"duration={cfg['duration']}")
            if cfg.get('gait_scale') is not None:
                config_parts.append(f"gait_scale={cfg['gait_scale']}")
            if config_parts:
                print(f"CONFIG: {', '.join(config_parts)}")
    else:
        print("RUN: (none)")

    # Status line with enhanced info
    if train_status.running:
        progress, current_iter, total_iter = get_progress(run_path)
        elapsed_str = format_elapsed(train_status.elapsed_seconds) if train_status.elapsed_seconds else "?"
        progress_str = f"{int(progress * 100)}%" if progress else "?"

        # Get training rate and config
        current_rate, avg_rate = get_training_rate()
        if current_rate and avg_rate:
            rate_str = f"{current_rate:.1f} it/s (avg {avg_rate:.1f})"
        elif current_rate:
            rate_str = f"{current_rate:.1f} it/s"
        else:
            rate_str = "? it/s"

        # Get num_envs from manifest
        training_config = manifest.get('training_config', {}) if manifest else {}
        num_envs = training_config.get('num_envs')
        envs_str = f"{num_envs} envs" if num_envs else ""

        # Build status line
        parts = [progress_str, elapsed_str + " elapsed", rate_str]
        if envs_str:
            parts.append(envs_str)
        print(f"STATUS: RUNNING ({', '.join(parts)})")
    else:
        # Check if watchdog killed training
        kill_info = get_kill_info()
        if kill_info:
            reason = kill_info.get('reason', 'unknown')
            ram_pct = kill_info.get('ram_percent', 0)
            swap_pct = kill_info.get('swap_percent', 0)
            if reason == 'swap_pressure':
                print(f"STATUS: KILLED_BY_WATCHDOG (swap={swap_pct:.0f}%, ram={ram_pct:.0f}%)")
            elif reason == 'ram_pressure':
                print(f"STATUS: KILLED_BY_WATCHDOG (ram={ram_pct:.0f}%, swap={swap_pct:.0f}%)")
            else:
                print(f"STATUS: KILLED_BY_WATCHDOG ({reason})")
        else:
            orphans = find_training_processes()
            if orphans:
                pids = ", ".join(str(p['pid']) for p in orphans)
                print(f"STATUS: ORPHAN (pids={pids})")
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
            status = 'PASS' if metric_passes('episode_length', ep_len) else 'FAIL'
            print(f"SANITY: {status} (ep_len={ep_len:.0f})")
        else:
            print("SANITY: (no data)")

        # Standing check
        height = metrics.get('height_reward')
        contact = metrics.get('body_contact')
        if height is not None:
            status = 'PASS' if metric_passes('height_reward', height) else 'FAIL'
            contact_str = f", contact={contact:.2f}" if contact is not None else ""
            print(f"STANDING: {status} (height={height:.2f}{contact_str})")
        else:
            print("STANDING: (no data)")

        # Walking check
        vx = metrics.get('vx_w_mean')
        vx_spec = METRIC_BY_KEY['vx_w_mean']
        if vx is not None:
            if metric_passes('vx_w_mean', vx):
                status = 'PASS'
            elif vx > 0:
                status = 'WARN'
            else:
                status = 'FAIL'
            print(f"WALKING: {status} (vx={vx:.3f}, need >{vx_spec.threshold})")
        else:
            print("WALKING: (no data)")

        x_disp = metrics.get('x_displacement')
        x_disp_abs = metrics.get('x_displacement_abs')
        if x_disp is not None:
            if x_disp_abs is not None:
                print(f"DISPLACEMENT: x={x_disp:.3f} (|x|={x_disp_abs:.3f})")
            else:
                print(f"DISPLACEMENT: x={x_disp:.3f}")

        # Diagnosis (state-only, no NEXT field)
        diag = get_diagnosis(metrics)
        print(f"VERDICT: {diag.status}")
        print(f"DIAGNOSIS: {diag.diagnosis}")
        return diag.exit_code
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

    # Print each metric (uses METRICS as single source of truth)
    for spec in METRICS:
        val = metrics.get(spec.key)
        print(format_metric_line(spec.key, val))

    if AUX_METRICS:
        print()
        print("AUX METRICS:")
        for spec in AUX_METRICS:
            val = metrics.get(spec.key)
            if val is None:
                print(f"  {spec.display_name}: (no data)")
            else:
                print(f"  {spec.display_name}: {val:.4f}")

    print()
    diag = get_diagnosis(metrics)
    print(f"VERDICT: {diag.status}")
    print(f"DIAGNOSIS: {diag.diagnosis}")

    return diag.exit_code


def cmd_runs(args):
    """List recent training runs with optional hypothesis display."""
    if not LOG_DIR.exists():
        print("No runs directory found")
        return 0

    runs = sorted([
        d for d in LOG_DIR.iterdir()
        if d.is_dir() and RUN_DIR_PATTERN.match(d.name)
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
            diag = get_diagnosis(metrics)
            ep_len = metrics.get('episode_length', 0)
            vx = metrics.get('vx_w_mean', 0)

            alias_str = f" ({alias})" if alias else ""
            print(f"  {run.name}{alias_str}  {diag.status:12s}  ep={ep_len:.0f}  vx={vx:.3f}")

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

    for spec in METRICS:
        row = f"  {spec.key:17s}"
        for e in experiments:
            val = e['manifest'].get('summary', {}).get('final', {}).get(spec.key)
            if val is not None:
                row += f"{val:>12.3f}"
            else:
                row += f"{'?':>12}"

        # Add threshold from spec
        cmp = ">" if spec.compare_gt else "<"
        row += f"   {cmp} {spec.threshold}"
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


def cmd_stop(args):
    """Stop training and cleanup processes."""
    processes = find_training_processes()

    if not processes:
        print("No training processes found")
        # Clean up stale PID files
        if PID_FILE.exists():
            PID_FILE.unlink()
            print("Cleaned up stale PID file")
        if WATCHDOG_PID_FILE.exists():
            WATCHDOG_PID_FILE.unlink()
            print("Cleaned up stale watchdog PID file")
        return 0

    print(f"Found {len(processes)} training process(es):")
    for p in processes:
        elapsed_str = format_elapsed(p['elapsed'])
        print(f"  PID {p['pid']} (running {elapsed_str})")

    # Kill all training processes
    killed = 0
    for p in processes:
        if kill_training(p['pid']):
            print(f"  Killed PID {p['pid']}")
            killed += 1

    # Also kill watchdog if running
    if WATCHDOG_PID_FILE.exists():
        try:
            watchdog_pid = int(WATCHDOG_PID_FILE.read_text().strip())
            os.kill(watchdog_pid, signal.SIGTERM)
            print(f"  Killed watchdog PID {watchdog_pid}")
        except (ProcessLookupError, ValueError):
            pass
        WATCHDOG_PID_FILE.unlink()

    # Clean up PID files
    if PID_FILE.exists():
        PID_FILE.unlink()

    print(f"\nStopped {killed} process(es)")
    return 0


def cmd_ps(args):
    """List all training processes (including orphans)."""
    processes = find_training_processes()

    # Also check PID file
    tracked_pid = None
    if PID_FILE.exists():
        try:
            tracked_pid = int(PID_FILE.read_text().strip())
        except ValueError:
            pass

    if not processes:
        print("No training processes running")
        if tracked_pid:
            print(f"  (stale PID file points to {tracked_pid})")
        return 0

    print(f"Training processes ({len(processes)}):")
    for p in processes:
        elapsed_str = format_elapsed(p['elapsed'])
        tracked = " (tracked)" if tracked_pid and p['pid'] == tracked_pid else ""
        orphan = " [ORPHAN]" if tracked_pid and p['pid'] != tracked_pid else ""
        if not tracked_pid:
            orphan = " [ORPHAN - no PID file]"
        print(f"  PID {p['pid']:>7} | {elapsed_str:>10} |{tracked}{orphan}")

    # Check for watchdog
    if WATCHDOG_PID_FILE.exists():
        try:
            watchdog_pid = int(WATCHDOG_PID_FILE.read_text().strip())
            os.kill(watchdog_pid, 0)
            print(f"\nWatchdog: PID {watchdog_pid} (active)")
        except (ProcessLookupError, ValueError):
            print(f"\nWatchdog: stale PID file")

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
    train_parser.add_argument('--task', choices=sorted(TASK_IDS.keys()), default=DEFAULT_TASK, help='Task to train (default: flat)')
    train_parser.add_argument('--duration', choices=sorted(DURATION_PRESETS.keys()), help='Duration preset: short (~30m), standard (~60m), long (~100m) (default: short)')
    train_parser.add_argument('--iterations', type=int, help='Max iterations (advanced override)')
    train_parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--hypothesis', type=str, help='Hypothesis being tested (stored with experiment)')
    train_parser.add_argument('--tags', type=str, help='Comma-separated tags for categorization')
    train_parser.add_argument('--no-watchdog', action='store_true', help='Disable memory watchdog (not recommended)')
    train_parser.add_argument('--num-envs', type=int, default=None, help='Number of environments (advanced override; default: 8192, pushup: 1)')
    train_parser.add_argument('--mode', choices=MODE_CHOICES, default='rl', help='Control mode: rl, cpg, scripted (default: rl)')
    train_parser.add_argument('--gait-scale', type=float, help='Scale scripted/CPG gait amplitude (diagnostic)')

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

    # stop
    stop_parser = subparsers.add_parser('stop', help='Stop training and cleanup processes')

    # ps
    ps_parser = subparsers.add_parser('ps', help='List all training processes (including orphans)')

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
    elif args.command == 'stop':
        return cmd_stop(args)
    elif args.command == 'ps':
        return cmd_ps(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
