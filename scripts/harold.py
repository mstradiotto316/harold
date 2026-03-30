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

    # Run scripted gait or open-loop CPG playback (no learning)
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
ISAACLAB_PYTHON = ENV_PATH.parent / "python"
RUN_DIR_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_ppo_|EXP-\d+)")

# Training defaults (single source of truth for run configuration)
TASK_IDS = {
    'flat': 'Template-Harold-Direct-flat-terrain-v0',
    'rough': 'Template-Harold-Direct-rough-terrain-v0',
    'pushup': 'Template-Harold-Direct-pushup-v0',
    'sim_flat_v1': 'Template-Spot-Direct-sim-flat-v1',
    'sim_flat_v2': 'Template-Harold-Direct-sim-flat-v2',
    'harold_mgr': 'Harold-Velocity-Flat-v0',
}
DEFAULT_TASK = 'harold_mgr'
TRAINING_DEFAULTS = {
    'num_envs': 4096,   # Session 54: matched Spot reference. Was 16384.
    'video_length': 250,
    'rendering_mode': 'balanced',
}
DURATION_PRESETS = {
    'fast': 625,       # ~15 minutes (screening)
    'short': 1250,     # ~30 minutes (confirmation)
    'standard': 2500,  # ~60 minutes
    'long': 4167,      # ~100 minutes
}
DEFAULT_DURATION = 'short'
MODE_CHOICES = ('rl', 'cpg', 'scripted')
DATA_POINTS_TAG = 'Info / Episode_Metric/vx_w_mean'

# Map task keys to their env_cfg files (for reading action_scale, etc.)
_TASK_ENV_CFG_PATHS = {
    'flat': PROJECT_ROOT / "harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_flat/harold_isaac_lab_env_cfg.py",
    'rough': PROJECT_ROOT / "harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_rough/harold_isaac_lab_env_cfg.py",
    'pushup': PROJECT_ROOT / "harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/harold_pushup/harold_isaac_lab_env_cfg.py",
    'sim_flat_v1': PROJECT_ROOT / "harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/sim_flat_v1/sim_flat_v1_env_cfg.py",
    'sim_flat_v2': PROJECT_ROOT / "harold_isaac_lab/source/harold_isaac_lab/harold_isaac_lab/tasks/direct/sim_flat_v2/sim_flat_v2_env_cfg.py",
}
_ACTION_SCALE_RE = re.compile(r'^\s*action_scale\s*=\s*([0-9.eE+-]+)', re.MULTILINE)


def read_action_scale(task_key: str) -> float | None:
    """Read action_scale from the env_cfg for the given task, or None if unreadable."""
    cfg_path = _TASK_ENV_CFG_PATHS.get(task_key)
    if cfg_path and cfg_path.exists():
        m = _ACTION_SCALE_RE.search(cfg_path.read_text(encoding="utf-8"))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None

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
    MetricSpec('episode_length', 'Episode / Total timesteps (mean)', 200, True, 'Episode Length'),  # Session 51 post-mortem: lowered from 300 to 200 (SANITY_FAIL gate only, not walking indicator)
    MetricSpec('upright_mean', 'Info / Episode_Metric/upright_mean', 0.9, True, 'Upright Mean'),
    MetricSpec('height_reward', ('Info / Episode_Reward/height_reward', 'Info / Episode_Metric/height_reward'), 0.5, True, 'Height Reward'),  # Session 24: lowered from 1.2 (CPG gait has different natural height)
    MetricSpec('body_contact', ('Info / Episode_Reward/body_contact_penalty', 'Info / Episode_Metric/body_contact_penalty'), -0.1, True, 'Body Contact'),
    MetricSpec('vx_w_mean', 'Info / Episode_Metric/vx_w_mean', 0.05, True, 'Forward Velocity'),  # Session 51 post-mortem: raised from 0.01 (1cm/s trivially satisfied by drift/falling)
    MetricSpec('cmd_tracking_ratio', 'Info / Episode_Metric/cmd_tracking_ratio', 0.5, True, 'Cmd Tracking Ratio'),  # Session 52 fix: robot must cover ≥50% of commanded distance
]

AUX_METRICS = [
    AuxMetricSpec('x_displacement', 'Info / Episode_Metric/x_displacement', 'X Displacement'),
    AuxMetricSpec('x_displacement_abs', 'Info / Episode_Metric/x_displacement_abs', 'Abs X Displacement'),
    AuxMetricSpec(
        'term_orientation',
        ('Info / Episode_Termination/orientation', 'Info / Episode_Metric/termination_orientation'),
        'Term: Orientation',
    ),
    AuxMetricSpec(
        'term_height',
        ('Info / Episode_Termination/height', 'Info / Episode_Metric/termination_height'),
        'Term: Height',
    ),
    AuxMetricSpec(
        'term_body_contact',
        ('Info / Episode_Termination/body_contact', 'Info / Episode_Termination/contact', 'Info / Episode_Metric/termination_body_contact', 'Info / Episode_Metric/termination_contact'),
        'Term: Body Contact',
    ),
    AuxMetricSpec(
        'term_elbow_pose',
        ('Info / Episode_Termination/elbow_pose', 'Info / Episode_Metric/termination_elbow_pose'),
        'Term: Elbow Pose',
    ),
    AuxMetricSpec(
        'term_timeout',
        ('Info / Episode_Termination/time_out', 'Info / Episode_Metric/termination_time_out'),
        'Term: Timeout',
    ),
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
    """Get the most recent training run directory (by modification time)."""
    if not LOG_DIR.exists():
        return None
    runs = sorted(
        [d for d in LOG_DIR.iterdir() if d.is_dir() and RUN_DIR_PATTERN.match(d.name)],
        key=lambda d: d.stat().st_mtime,
    )
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
    training_config: dict = None,
    alias: str | None = None,
) -> str:
    """Register a new experiment and return its alias.

    Args:
        run_path: Path to the experiment directory
        hypothesis: Hypothesis being tested
        tags: List of tags for categorization
        training_config: Training parameters (num_envs, iterations) for STATUS display
        alias: Pre-computed alias (e.g. from pre-launch). If None, generates next.
    """
    index = load_index()
    if alias is None:
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

    runs = sorted(
        [d for d in LOG_DIR.iterdir() if d.is_dir() and RUN_DIR_PATTERN.match(d.name)],
        key=lambda d: d.stat().st_mtime,
    )

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

    # Success checks: require BOTH cmd_tracking_ratio AND vx_w_mean to pass
    tracking = metrics.get('cmd_tracking_ratio')

    vx_pass = vx is not None and metric_passes('vx_w_mean', vx)
    tracking_pass = tracking is not None and metric_passes('cmd_tracking_ratio', tracking)

    if vx_pass and tracking_pass:
        return DiagnosisResult('WALKING', f'Forward velocity {vx:.3f} m/s, cmd tracking ratio {tracking:.3f} — both above thresholds', 0)

    # Partial success
    if vx is not None and tracking is not None:
        return DiagnosisResult('STANDING', f'Upright and stable, vx={vx:.3f} m/s, tracking={tracking:.3f}', 1)
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
    video: bool = False,
    video_interval: int = 2000,
    video_length: int | None = None,
) -> list[str]:
    """Build the Isaac Lab training command.

    Encapsulates command construction and benchmark-based defaults.
    """
    # Benchmark results (2026-03-19, RTX 4080 16GB, 63GB RAM):
    # Training runs WITHOUT video (video captured post-hoc via `harold record`):
    #   1024 envs: 18.1 it/s, 0.45M samples/s, GPU  6.2GB, RAM  8.6GB
    #   4096 envs: 16.0 it/s, 1.58M samples/s, GPU  7.2GB, RAM  9.3GB  <- DEFAULT
    #   8192 envs: 11.5 it/s, 2.26M samples/s, GPU  8.3GB, RAM 10.3GB
    #  16384 envs:  7.3 it/s, 2.88M samples/s, GPU 10.3GB, RAM 12.3GB  <- standing policy (batch too large)
    #  24576 envs:  5.4 it/s, 3.18M samples/s, GPU 12.3GB, RAM 14.6GB
    cmd = [
        str(ISAACLAB_PYTHON), str(PROJECT_ROOT / 'harold_isaac_lab' / 'scripts' / 'skrl' / 'train.py'),
        f'--task={task_id}',
        '--num_envs', str(num_envs),
        '--max_iterations', str(iterations),
        '--headless',
        '--rendering_mode', TRAINING_DEFAULTS['rendering_mode'],
    ]
    if checkpoint:
        cmd.extend(['--checkpoint', str(checkpoint)])
    if video:
        cmd.extend(['--video', '--video_interval', str(video_interval)])
        if video_length is not None:
            cmd.extend(['--video_length', str(video_length)])
        else:
            cmd.extend(['--video_length', str(TRAINING_DEFAULTS['video_length'])])
    return cmd


def start_watchdog(pid: str) -> bool:
    """Start memory watchdog for the given training PID.

    Returns True if started successfully.
    """
    watchdog_script = PROJECT_ROOT / 'scripts' / 'memory_watchdog.py'
    if not watchdog_script.exists():
        return False

    with open(WATCHDOG_LOG_FILE, 'w', encoding='utf-8') as watchdog_log:
        process = subprocess.Popen(
            [
                str(ISAACLAB_PYTHON),
                str(watchdog_script),
                '--pid',
                str(pid),
                '--ram-kill',
                str(RAM_KILL_THRESHOLD),
                '--swap-kill',
                str(SWAP_KILL_THRESHOLD),
            ],
            cwd=PROJECT_ROOT,
            stdout=watchdog_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    WATCHDOG_PID_FILE.write_text(str(process.pid))
    return True


def cmd_train(args):
    """Start training in background with optional hypothesis and tags."""
    # Validate arguments BEFORE stopping any existing training.
    # This prevents killing an in-flight experiment when the new invocation
    # would fail validation (e.g., --iterations together with --duration).
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
        if task_key == 'pushup':
            num_envs = 1
        elif task_key == 'harold_mgr':
            num_envs = 2048
        else:
            num_envs = TRAINING_DEFAULTS['num_envs']
    else:
        num_envs = args.num_envs

    hypothesis = getattr(args, 'hypothesis', '') or ''
    tags = [t.strip() for t in args.tags.split(',') if t.strip()] if getattr(args, 'tags', None) else []

    mode = args.mode

    # Build command (validates interpreter path, etc.)
    video = getattr(args, 'video', False)
    video_interval = getattr(args, 'video_interval', 2000)
    video_length = getattr(args, 'video_length', None)
    cmd = build_train_command(num_envs, iterations, task_id, args.checkpoint,
                              video=video, video_interval=video_interval,
                              video_length=video_length)

    # Reject concurrent launches instead of killing in-flight work.
    train_status = is_training_running()
    if train_status.running:
        print(
            "ERROR: Training is already running "
            f"(PID: {train_status.pid}). Stop it explicitly with `harold stop` before starting a new run."
        )
        return 1

    existing_processes = find_training_processes()
    if existing_processes:
        orphan_pids = ", ".join(str(proc["pid"]) for proc in existing_processes)
        print(
            "ERROR: Found existing Harold training process(es) "
            f"({orphan_pids}). Run `harold stop` to clean them up before starting a new run."
        )
        return 1

    # Capture latest run before launch (used to detect new run directory)
    previous_run = get_latest_run()

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
    if video:
        print(f"  Video: inline (every {video_interval} steps, {video_length or TRAINING_DEFAULTS['video_length']} steps/clip)")
    else:
        print(f"  Video: post-hoc (harold record)")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Mode: {mode}")
    if getattr(args, 'gait_scale', None) is not None:
        print(f"  Gait scale: {args.gait_scale}")

    # Pre-compute experiment alias so the run directory is named EXP-XXX
    index = load_index()
    pre_alias = get_next_alias(index)

    # Launch training in background
    env_vars = {
        "HAROLD_CPG": "1" if mode == "cpg" else "0",
        "HAROLD_SCRIPTED_GAIT": "1" if mode == "scripted" else "0",
        "HAROLD_EXPERIMENT_NAME": pre_alias,
    }
    if getattr(args, 'gait_scale', None) is not None:
        env_vars["HAROLD_GAIT_AMP_SCALE"] = str(args.gait_scale)
    else:
        env_vars["HAROLD_GAIT_AMP_SCALE"] = ""
    child_env = os.environ.copy()
    child_env.update(env_vars)

    with open(LOG_FILE, 'w', encoding='utf-8') as train_log:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=child_env,
            stdout=train_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    PID_FILE.write_text(str(process.pid))
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
        action_scale_val = read_action_scale(task_key)
        if action_scale_val is not None:
            training_config['action_scale'] = action_scale_val
        alias = register_experiment(
            run_path,
            hypothesis=hypothesis,
            tags=tags,
            training_config=training_config,
            alias=pre_alias,
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
        print(json.dumps(result, indent=2, default=str))
        return 0

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

        # Walking check: requires both vx and x_displacement
        vx = metrics.get('vx_w_mean')
        x_disp = metrics.get('x_displacement')
        vx_spec = METRIC_BY_KEY['vx_w_mean']
        x_disp_spec = METRIC_BY_KEY.get('x_displacement')
        vx_pass = vx is not None and metric_passes('vx_w_mean', vx)
        x_disp_pass = x_disp is not None and x_disp_spec is not None and metric_passes('x_displacement', x_disp)
        if vx is not None and x_disp is not None and x_disp_spec is not None:
            if vx_pass and x_disp_pass:
                status = 'PASS'
            elif vx > 0 or (x_disp is not None and x_disp > 0):
                status = 'WARN'
            else:
                status = 'FAIL'
            print(f"WALKING: {status} (vx={vx:.3f} need >{vx_spec.threshold}, x_disp={x_disp:.3f} need >{x_disp_spec.threshold})")
        elif vx is not None:
            status = 'PASS' if vx_pass else ('WARN' if vx > 0 else 'FAIL')
            print(f"WALKING: {status} (vx={vx:.3f}, need >{vx_spec.threshold})")
        else:
            print("WALKING: (no data)")

        x_disp_abs = metrics.get('x_displacement_abs')
        if x_disp is not None and x_disp_abs is not None:
            print(f"DISPLACEMENT: x={x_disp:.3f} (|x|={x_disp_abs:.3f})")
        elif x_disp is not None:
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

    runs = sorted(
        [d for d in LOG_DIR.iterdir() if d.is_dir() and RUN_DIR_PATTERN.match(d.name)],
        key=lambda d: d.stat().st_mtime,
    )

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


def cmd_snapshot_config(args):
    """Dump current training config as JSON for autoresearch integration."""
    try:
        from autoresearch import load_baseline_config
        config = load_baseline_config()
    except ImportError:
        # Direct import from scripts directory
        import importlib.util
        spec = importlib.util.spec_from_file_location("autoresearch", Path(__file__).parent / "autoresearch.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        config = mod.load_baseline_config()
    print(json.dumps(config, indent=2, default=str))
    return 0


def find_best_checkpoint(run_path: Path) -> Path | None:
    """Find the best checkpoint in a run directory.

    Priority: best_agent.pt > highest-numbered agent_*.pt.
    """
    ckpt_dir = run_path / "checkpoints"
    if not ckpt_dir.exists():
        return None
    best = ckpt_dir / "best_agent.pt"
    if best.exists():
        return best
    numbered = sorted(ckpt_dir.glob("agent_*.pt"))
    return numbered[-1] if numbered else None


def build_record_command(
    run_path: Path,
    task_id: str,
    checkpoint: Path,
    video_length: int,
) -> list[str]:
    """Build command to invoke record.py for post-hoc video recording."""
    output_dir = run_path / "videos" / "record"
    return [
        str(ISAACLAB_PYTHON), str(PROJECT_ROOT / 'harold_isaac_lab' / 'scripts' / 'skrl' / 'record.py'),
        f'--task={task_id}',
        '--checkpoint', str(checkpoint),
        '--output_dir', str(output_dir),
        '--video_length', str(video_length),
    ]


def cmd_record(args):
    """Record multi-camera video from a trained checkpoint."""
    run_path = resolve_experiment(args.run) if args.run else get_latest_run()
    if not run_path or not run_path.exists():
        print("ERROR: No run found")
        return 1

    manifest = get_or_create_manifest(run_path)
    tc = manifest.get('training_config', {})
    task_key = tc.get('task', manifest.get('task', DEFAULT_TASK))
    task_id = TASK_IDS.get(task_key, TASK_IDS[DEFAULT_TASK])

    # find checkpoint
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint}")
            return 1
    else:
        checkpoint = find_best_checkpoint(run_path)
        if not checkpoint:
            print(f"ERROR: No checkpoint found in {run_path / 'checkpoints'}")
            return 1

    video_length = args.video_length or TRAINING_DEFAULTS['video_length']
    print(f"Recording video from {manifest.get('alias', run_path.name)}")
    print(f"  Checkpoint: {checkpoint.name}")
    print(f"  Video length: {video_length} steps")

    cmd = build_record_command(run_path, task_id, checkpoint, video_length)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("ERROR: Video recording failed")
        return 1

    # verify output
    video_dir = run_path / "videos" / "record"
    cam_names = ["side", "front", "top", "iso", "main"]
    found = [c for c in cam_names if list(video_dir.glob(f"rl-video-step-0-{c}.mp4"))]
    if found:
        print(f"  Recorded {len(found)} camera views: {', '.join(found)}")
        for cam in found:
            vids = sorted(video_dir.glob(f"rl-video-step-0-{cam}.mp4"))
            if vids:
                print(f"    {vids[0].name}")
    else:
        print("WARNING: No video files found after recording")
        return 1

    return 0


def _extract_step_number(filename: str) -> int:
    """Extract the step number from a video filename like 'rl-video-step-3200-side.mp4'."""
    m = re.search(r'rl-video-step-(\d+)', filename)
    return int(m.group(1)) if m else -1


def _extract_frames_from_video(video: Path, out_dir: Path, fps: int) -> list:
    """Run ffmpeg to extract frames from a single video. Returns list of frame paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
         str(out_dir / "frame_%04d.jpg")],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"WARNING: ffmpeg failed for {video.name}: {result.stderr[-200:]}")
        return []
    return sorted(out_dir.glob("frame_*.jpg"))


def cmd_frames(args):
    """Extract frames from the latest training video(s) for analysis.

    Supports both multi-camera videos (rl-video-step-N-{side,front,top,iso}.mp4)
    and legacy single-camera videos (rl-video-step-N.mp4).
    """
    run_path = resolve_experiment(args.run) if args.run else get_latest_run()
    if not run_path or not run_path.exists():
        print("ERROR: No run found")
        return 1

    # Check record/ first (post-hoc recordings), fall back to train/ (legacy training-time videos)
    video_dir = run_path / "videos" / "record"
    if not video_dir.exists():
        video_dir = run_path / "videos" / "train"
    if not video_dir.exists():
        print(f"ERROR: No videos directory found in {run_path / 'videos'}")
        return 1

    fps = args.fps or 2
    out_dir = Path("/tmp/harold_review_frames")

    # Clean and recreate output directory
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    manifest = get_or_create_manifest(run_path)
    cam_names = ["side", "front", "top", "iso", "main"]

    # Detect multi-camera videos (look for side camera as sentinel)
    multi_cam_videos = sorted(video_dir.glob("rl-video-step-*-side.mp4"),
                              key=lambda p: _extract_step_number(p.name))
    if multi_cam_videos:
        # Multi-camera mode
        latest = multi_cam_videos[-1]
        step = _extract_step_number(latest.name)

        cameras = {}
        total_frames = 0
        for cam in cam_names:
            video = video_dir / f"rl-video-step-{step}-{cam}.mp4"
            if not video.exists():
                continue
            cam_dir = out_dir / cam
            frames = _extract_frames_from_video(video, cam_dir, fps)
            cameras[cam] = {
                "video": str(video),
                "video_name": video.name,
                "num_frames": len(frames),
                "frame_dir": str(cam_dir),
                "frames": [str(f) for f in frames],
            }
            total_frames += len(frames)

        output = {
            "run_name": run_path.name,
            "alias": manifest.get("alias", ""),
            "hypothesis": manifest.get("hypothesis", ""),
            "step": step,
            "fps": fps,
            "multi_camera": True,
            "cameras": cameras,
            "frame_dir": str(out_dir),
        }

        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(f"Extracted frames at {fps}fps from step {step} ({len(cameras)} cameras)")
            for cam, info in cameras.items():
                print(f"  {cam}: {info['num_frames']} frames → {info['frame_dir']}/frame_*.jpg")
            print(f"  Run: {run_path.name} ({manifest.get('alias', '')})")
    else:
        # Legacy single-video mode
        videos = sorted(video_dir.glob("rl-video-step-*.mp4"),
                        key=lambda p: _extract_step_number(p.name))
        if not videos:
            print("ERROR: No training videos found")
            return 1

        video = videos[-1]
        frames = _extract_frames_from_video(video, out_dir, fps)
        if not frames:
            return 1

        output = {
            "run_name": run_path.name,
            "alias": manifest.get("alias", ""),
            "hypothesis": manifest.get("hypothesis", ""),
            "video": str(video),
            "video_name": video.name,
            "step": _extract_step_number(video.name),
            "fps": fps,
            "multi_camera": False,
            "num_frames": len(frames),
            "frame_dir": str(out_dir),
            "frames": [str(f) for f in frames],
        }

        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(f"Extracted {len(frames)} frames at {fps}fps from {video.name}")
            print(f"  Run: {run_path.name} ({manifest.get('alias', '')})")
            print(f"  Frames: {out_dir}/frame_*.jpg")

    return 0


def cmd_log(args):
    """Show training log output for debugging."""
    if not LOG_FILE.exists():
        print("No training log found")
        return 1

    lines = LOG_FILE.read_text().splitlines()

    if args.grep:
        lines = [l for l in lines if re.search(args.grep, l, re.IGNORECASE)]

    tail_n = args.tail or 20
    lines = lines[-tail_n:]

    for line in lines:
        print(line)
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
    train_parser.add_argument('--duration', choices=sorted(DURATION_PRESETS.keys()), help='Duration preset: fast (~15m), short (~30m), standard (~60m), long (~100m) (default: short)')
    train_parser.add_argument('--iterations', type=int, help='Max iterations (advanced override)')
    train_parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--hypothesis', type=str, help='Hypothesis being tested (stored with experiment)')
    train_parser.add_argument('--tags', type=str, help='Comma-separated tags for categorization')
    train_parser.add_argument('--no-watchdog', action='store_true', help='Disable memory watchdog (not recommended)')
    train_parser.add_argument('--num-envs', type=int, default=None, help='Number of environments (advanced override; default: 16384, pushup: 1)')
    train_parser.add_argument('--mode', choices=MODE_CHOICES, default='rl', help='Control mode: rl, cpg (open-loop), scripted (default: rl)')
    train_parser.add_argument('--gait-scale', type=float, help='Scale scripted/CPG gait amplitude (diagnostic)')
    train_parser.add_argument('--video', action='store_true', help='Record video during training (inline, not post-hoc)')
    train_parser.add_argument('--video-interval', type=int, default=2000, help='Steps between video recordings (default: 2000)')
    train_parser.add_argument('--video-length', type=int, default=None, help='Video clip length in steps (default: 250)')

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

    # frames
    frames_parser = subparsers.add_parser('frames', help='Extract frames from latest training video for review')
    frames_parser.add_argument('run', nargs='?', help='Run name or alias (default: latest)')
    frames_parser.add_argument('--fps', type=int, default=2, help='Frame rate for extraction (default: 2)')
    frames_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # log
    log_parser = subparsers.add_parser('log', help='Show training log output (for debugging)')
    log_parser.add_argument('--grep', type=str, help='Filter log lines by pattern')
    log_parser.add_argument('--tail', type=int, help='Number of lines to show (default: 20)')

    # record
    record_parser = subparsers.add_parser('record', help='Record multi-camera video from trained checkpoint')
    record_parser.add_argument('run', nargs='?', help='Run name or alias (default: latest)')
    record_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint (default: best in run)')
    record_parser.add_argument('--video-length', type=int, help='Steps to record (default: 250)')

    # snapshot-config
    subparsers.add_parser('snapshot-config', help='Dump current training config as JSON (for autoresearch)')

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
    elif args.command == 'record':
        return cmd_record(args)
    elif args.command == 'frames':
        return cmd_frames(args)
    elif args.command == 'log':
        return cmd_log(args)
    elif args.command == 'snapshot-config':
        return cmd_snapshot_config(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
