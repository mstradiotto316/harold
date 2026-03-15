#!/usr/bin/env python3
"""
Harold Autoresearch - Helper functions for autonomous RL experiment loops.

This is NOT a standalone daemon. Claude Code is the agent. It calls these
functions during autonomous sessions to propose, apply, evaluate, and log
config changes.

Usage (by Claude Code agent):
    python scripts/autoresearch.py load-baseline
    python scripts/autoresearch.py load-registry
    python scripts/autoresearch.py apply '{"forward_motion_weight": 5.0}'
    python scripts/autoresearch.py revert
    python scripts/autoresearch.py score '{"vx_w_mean": 0.02, "upright_mean": 0.96, ...}'
    python scripts/autoresearch.py log '{"exp_alias": "EXP-228", ...}'
    python scripts/autoresearch.py history
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

ENV_CFG_PATH = (
    PROJECT_ROOT
    / "harold_isaac_lab"
    / "source"
    / "harold_isaac_lab"
    / "harold_isaac_lab"
    / "tasks"
    / "direct"
    / "harold_flat"
    / "harold_isaac_lab_env_cfg.py"
)

PPO_CFG_PATH = (
    PROJECT_ROOT
    / "harold_isaac_lab"
    / "source"
    / "harold_isaac_lab"
    / "harold_isaac_lab"
    / "tasks"
    / "direct"
    / "harold_flat"
    / "agents"
    / "skrl_ppo_cfg.yaml"
)

TRAIN_ENV_PATH = (
    PROJECT_ROOT
    / "harold_isaac_lab"
    / "source"
    / "harold_isaac_lab"
    / "harold_isaac_lab"
    / "tasks"
    / "direct"
    / "harold_flat"
    / "train_env.py"
)

BASELINE_SNAPSHOT_PATH = PROJECT_ROOT / ".autoresearch_baseline.json"

REGISTRY_PATH = PROJECT_ROOT / "docs" / "autoresearch" / "PARAMETER_REGISTRY.md"
RESULTS_PATH = PROJECT_ROOT / "docs" / "autoresearch" / "results.tsv"
SESSION_STATE_PATH = PROJECT_ROOT / "docs" / "autoresearch" / "session_state.json"

# Patterns for extracting values from env_cfg.py
# Matches lines like: `parameter_name: float = 5.0` or `parameter_name = 5.0`
ENV_CFG_FLOAT_PATTERN = re.compile(
    r"^(\s*{param}\s*(?::\s*float\s*)?=\s*)([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(.*)"
)
ENV_CFG_BOOL_PATTERN = re.compile(
    r"^(\s*{param}\s*(?::\s*bool\s*)?=\s*)(True|False)(.*)"
)
ENV_CFG_INT_PATTERN = re.compile(
    r"^(\s*{param}\s*(?::\s*int\s*)?=\s*)(\d+)(.*)"
)


# ── Registry ────────────────────────────────────────────────────────────────

def load_parameter_registry() -> dict:
    """Parse PARAMETER_REGISTRY.md into a dict of {param: {category, current, range, file, notes}}."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Parameter registry not found: {REGISTRY_PATH}")

    registry = {}
    text = REGISTRY_PATH.read_text()

    # Parse markdown tables: | param | category | current | range | notes |
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| Parameter") or line.startswith("|---"):
            continue

        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]  # remove empty from leading/trailing |

        if len(cells) < 5:
            continue

        param = cells[0].strip("`")
        category = cells[1]
        current_str = cells[2]
        range_str = cells[3]
        notes = cells[4] if len(cells) > 4 else ""

        # Skip header-like rows
        if category not in ("FROZEN", "CONSTRAINED", "TUNABLE"):
            continue

        # Parse current value
        current = _parse_value(current_str)

        # Parse range [min, max] or -
        param_range = None
        range_match = re.match(r"\[(.+),\s*(.+)\]", range_str)
        if range_match:
            try:
                param_range = (_parse_value(range_match.group(1)), _parse_value(range_match.group(2)))
            except (ValueError, TypeError):
                param_range = None

        registry[param] = {
            "category": category,
            "current": current,
            "range": param_range,
            "notes": notes,
        }

    return registry


def _parse_value(s: str):
    """Parse a string into float, int, bool, or return as string."""
    s = s.strip()
    if s in ("True", "False"):
        return s == "True"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


# ── Baseline Config ─────────────────────────────────────────────────────────

def load_baseline_config() -> dict:
    """Extract all tunable parameter values from env_cfg.py and skrl_ppo_cfg.yaml."""
    config = {}

    # Parse env_cfg.py
    config.update(_extract_env_cfg_values())

    # Parse PPO YAML
    config.update(_extract_ppo_cfg_values())

    return config


def _extract_env_cfg_values() -> dict:
    """Extract float/bool/int assignments from env_cfg.py dataclass fields."""
    values = {}
    text = ENV_CFG_PATH.read_text()

    # Target parameters from the registry
    registry = load_parameter_registry()
    env_params = [p for p in registry if p not in _PPO_PARAMS]

    for param in env_params:
        for line in text.splitlines():
            # Match float assignment
            float_pat = re.compile(
                rf"^\s*{re.escape(param)}\s*(?::\s*\w+\s*)?=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
            )
            m = float_pat.match(line)
            if m:
                values[param] = _parse_value(m.group(1))
                break

            # Match bool assignment
            bool_pat = re.compile(rf"^\s*{re.escape(param)}\s*(?::\s*\w+\s*)?=\s*(True|False)")
            m = bool_pat.match(line)
            if m:
                values[param] = m.group(1) == "True"
                break

    return values


# PPO params that live in the YAML file
_PPO_PARAMS = {
    "learning_rate", "rollouts", "learning_epochs", "mini_batches",
    "discount_factor", "lambda", "ratio_clip", "value_clip",
    "grad_norm_clip", "entropy_loss_scale", "value_loss_scale",
    "rewards_shaper_scale", "min_log_std", "seed", "timesteps",
    "policy_layers", "value_layers",
}


def _extract_ppo_cfg_values() -> dict:
    """Extract values from skrl_ppo_cfg.yaml."""
    try:
        import yaml
    except ImportError:
        # Fallback: regex extraction
        return _extract_ppo_cfg_regex()

    text = PPO_CFG_PATH.read_text()
    data = yaml.safe_load(text)

    values = {}
    agent = data.get("agent", {})
    models = data.get("models", {})
    trainer = data.get("trainer", {})

    # Direct agent params
    for key in ["rollouts", "learning_epochs", "mini_batches", "discount_factor",
                "lambda", "learning_rate", "grad_norm_clip", "ratio_clip",
                "value_clip", "entropy_loss_scale", "value_loss_scale",
                "rewards_shaper_scale"]:
        if key in agent:
            values[key] = agent[key]

    # Seed
    if "seed" in data:
        values["seed"] = data["seed"]

    # Timesteps from trainer
    if "timesteps" in trainer:
        values["timesteps"] = trainer["timesteps"]

    # Model params
    policy = models.get("policy", {})
    if "min_log_std" in policy:
        values["min_log_std"] = policy["min_log_std"]

    # Network layers
    for net in policy.get("network", []):
        if net.get("name") == "net" and "layers" in net:
            values["policy_layers"] = net["layers"]
    value_model = models.get("value", {})
    for net in value_model.get("network", []):
        if net.get("name") == "net" and "layers" in net:
            values["value_layers"] = net["layers"]

    return values


def _extract_ppo_cfg_regex() -> dict:
    """Fallback regex extraction for PPO YAML when pyyaml is not available."""
    values = {}
    text = PPO_CFG_PATH.read_text()
    for line in text.splitlines():
        for key in _PPO_PARAMS - {"policy_layers", "value_layers"}:
            m = re.match(rf"\s*{re.escape(key)}\s*:\s*([^\s#]+)", line)
            if m:
                values[key] = _parse_value(m.group(1))
    return values


# ── Baseline Snapshot ────────────────────────────────────────────────────────

def _save_baseline_snapshot() -> None:
    """Snapshot mutable config files before mutation.

    Saves file contents + git HEAD SHA so revert_change() can restore the
    exact pre-experiment state even after a git commit.
    """
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True
    ).stdout.strip()

    snapshot = {
        "head_sha": head_sha,
        "files": {}
    }
    for path in [ENV_CFG_PATH, PPO_CFG_PATH, TRAIN_ENV_PATH]:
        if path.exists():
            snapshot["files"][str(path)] = path.read_text()

    BASELINE_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")


# ── Apply / Revert Changes ──────────────────────────────────────────────────

def apply_change(delta: dict) -> list:
    """Apply config changes. Returns list of modified files.

    delta: {param_name: new_value, ...}

    Validates against parameter registry before applying.
    Raises ValueError for FROZEN params or out-of-range values.
    """
    # Auto-snapshot baseline before first mutation in this experiment cycle
    if not BASELINE_SNAPSHOT_PATH.exists():
        _save_baseline_snapshot()

    registry = load_parameter_registry()
    modified_files = set()

    for param, new_value in delta.items():
        # Validate against registry
        if param not in registry:
            raise ValueError(f"Unknown parameter: {param}")

        entry = registry[param]

        if entry["category"] == "FROZEN":
            raise ValueError(
                f"FROZEN parameter '{param}' cannot be modified. "
                f"Reason: {entry.get('notes', 'sim-to-real critical')}"
            )

        if entry["category"] == "CONSTRAINED" and entry["range"] is not None:
            lo, hi = entry["range"]
            if isinstance(new_value, (int, float)) and isinstance(lo, (int, float)):
                if new_value < lo or new_value > hi:
                    raise ValueError(
                        f"Parameter '{param}' value {new_value} outside "
                        f"CONSTRAINED range [{lo}, {hi}]"
                    )

        # Apply to the appropriate file
        if param in _PPO_PARAMS:
            _apply_ppo_change(param, new_value)
            modified_files.add(str(PPO_CFG_PATH))
        else:
            _apply_env_cfg_change(param, new_value)
            modified_files.add(str(ENV_CFG_PATH))

    # Validate modified files still parse
    for f in modified_files:
        _validate_file(f)

    return sorted(modified_files)


def _apply_env_cfg_change(param: str, value) -> None:
    """Replace a parameter value in env_cfg.py using targeted text replacement."""
    text = ENV_CFG_PATH.read_text()
    lines = text.splitlines()
    found = False

    for i, line in enumerate(lines):
        # Match the parameter assignment
        if isinstance(value, bool):
            pat = re.compile(
                rf"^(\s*{re.escape(param)}\s*(?::\s*\w+\s*)?=\s*)(True|False)(.*)"
            )
            m = pat.match(line)
            if m:
                lines[i] = f"{m.group(1)}{value}{m.group(3)}"
                found = True
                break
        elif isinstance(value, (int, float)):
            pat = re.compile(
                rf"^(\s*{re.escape(param)}\s*(?::\s*\w+\s*)?=\s*)([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(.*)"
            )
            m = pat.match(line)
            if m:
                # Format the value appropriately
                if isinstance(value, float):
                    formatted = _format_float(value)
                else:
                    formatted = str(value)
                lines[i] = f"{m.group(1)}{formatted}{m.group(3)}"
                found = True
                break

    if not found:
        raise ValueError(f"Could not find parameter '{param}' in {ENV_CFG_PATH.name}")

    ENV_CFG_PATH.write_text("\n".join(lines) + "\n")


def _format_float(value: float) -> str:
    """Format a float to match the existing code style."""
    if abs(value) < 1e-4 and value != 0:
        return f"{value:.1e}"
    elif value == int(value) and abs(value) < 1000:
        return f"{value:.1f}"
    else:
        return str(value)


def _apply_ppo_change(param: str, value) -> None:
    """Replace a parameter value in skrl_ppo_cfg.yaml."""
    text = PPO_CFG_PATH.read_text()
    lines = text.splitlines()
    found = False

    # Handle special nested params
    yaml_key = param  # Most params map directly

    for i, line in enumerate(lines):
        pat = re.compile(rf"^(\s*{re.escape(yaml_key)}\s*:\s*)([^\s#]+)(.*)")
        m = pat.match(line)
        if m:
            if isinstance(value, float):
                formatted = _format_float(value)
            else:
                formatted = str(value)
            lines[i] = f"{m.group(1)}{formatted}{m.group(3)}"
            found = True
            break

    if not found:
        raise ValueError(f"Could not find parameter '{param}' in {PPO_CFG_PATH.name}")

    PPO_CFG_PATH.write_text("\n".join(lines) + "\n")


def revert_change(revert_all: bool = False) -> None:
    """Revert config files to the pre-experiment baseline state.

    If a baseline snapshot exists (saved by apply_change), restores file
    contents from the snapshot — works even after git commit.
    Falls back to git checkout if no snapshot exists (backward compatible).

    Args:
        revert_all: If True, also revert train_env.py (default: config only).
    """
    if BASELINE_SNAPSHOT_PATH.exists():
        snapshot = json.loads(BASELINE_SNAPSHOT_PATH.read_text())
        files_to_revert = [str(ENV_CFG_PATH), str(PPO_CFG_PATH)]
        if revert_all:
            files_to_revert.append(str(TRAIN_ENV_PATH))

        for filepath in files_to_revert:
            if filepath in snapshot["files"]:
                Path(filepath).write_text(snapshot["files"][filepath])

        BASELINE_SNAPSHOT_PATH.unlink()
    else:
        files = [str(ENV_CFG_PATH), str(PPO_CFG_PATH)]
        if revert_all:
            files.append(str(TRAIN_ENV_PATH))
        subprocess.run(["git", "checkout", "--"] + files, cwd=PROJECT_ROOT, check=True)


def _validate_file(filepath: str) -> None:
    """Validate a modified file still parses correctly."""
    if filepath.endswith(".py"):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')",
             filepath],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise ValueError(f"Python syntax error in {filepath}:\n{result.stderr}")
    elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
        try:
            import yaml
            yaml.safe_load(Path(filepath).read_text())
        except Exception as e:
            raise ValueError(f"YAML parse error in {filepath}: {e}")


# ── Scoring ─────────────────────────────────────────────────────────────────

import math


def compute_walk_score(metrics: dict) -> float:
    """Single scalar 0-100 for walking quality.

    Designed to be:
    - 0 if the robot is broken (sanity fail, fallen, on elbows)
    - Monotonically increasing with forward velocity once gates pass
    - Exploit-proof (elbow exploit, body dragging both gate to 0)

    Formula:
        walk_score = gate * tanh(vx / 0.05) * 100

    Args:
        metrics: dict with keys: episode_length, upright_mean, height_reward,
                 body_contact, vx_w_mean

    Returns:
        float score 0-100
    """
    ep_len = metrics.get("episode_length", 0)

    # Hard gate: robot must survive
    if ep_len < 300:
        return 0.0

    upright = metrics.get("upright_mean", 0)
    height = metrics.get("height_reward", 0)
    contact = metrics.get("body_contact", 0)
    vx = metrics.get("vx_w_mean", 0)

    # Gating factors (0-1): prevent exploit modes
    upright_gate = min(1.0, max(0.0, (upright - 0.85) / 0.10))
    height_gate = min(1.0, max(0.0, (height - 0.3) / 0.3))
    contact_gate = min(1.0, max(0.0, (contact + 0.3) / 0.3))

    gate = min(upright_gate, height_gate, contact_gate)

    # Primary signal: forward velocity (tanh saturates at ~0.15 m/s)
    vx_score = math.tanh(max(0.0, vx) / 0.05)

    return round(gate * vx_score * 100.0, 1)


def compute_progress_score(metrics: dict) -> float:
    """Incremental progress score 0-100 with soft survival gate.

    Same formula as walk_score but replaces the hard ep_len >= 300 cutoff
    with tanh(ep_len / 150), so short-lived experiments still get partial
    credit. Converges to walk_score as ep_len grows.

    Formula:
        progress_score = survival * posture * velocity * 100
        survival = tanh(ep_len / 150)
        posture  = min(upright_gate, height_gate, contact_gate)
        velocity = tanh(max(0, vx) / 0.05)
    """
    ep_len = metrics.get("episode_length", 0)
    upright = metrics.get("upright_mean", 0)
    height = metrics.get("height_reward", 0)
    contact = metrics.get("body_contact", 0)
    vx = metrics.get("vx_w_mean", 0)

    # Soft survival gate (tanh ramp)
    survival = math.tanh(ep_len / 150.0)

    # Posture gates (same as walk_score)
    upright_gate = min(1.0, max(0.0, (upright - 0.85) / 0.10))
    height_gate = min(1.0, max(0.0, (height - 0.3) / 0.3))
    contact_gate = min(1.0, max(0.0, (contact + 0.3) / 0.3))
    posture = min(upright_gate, height_gate, contact_gate)

    # Velocity (same as walk_score)
    velocity = math.tanh(max(0.0, vx) / 0.05)

    return round(survival * posture * velocity * 100.0, 1)


# Backward compatibility alias
def compute_score(metrics: dict, weights: dict = None) -> float:
    """Legacy scoring function. Delegates to compute_walk_score."""
    return compute_walk_score(metrics)


# ── Results Logging ─────────────────────────────────────────────────────────

RESULTS_COLUMNS = [
    "exp_alias", "timestamp", "hypothesis", "changed_params", "duration_min",
    "verdict", "vx", "upright", "height", "contact", "ep_len",
    "walk_score", "progress_score", "decision", "notes",
]


def _scores_from_entry(entry: dict) -> tuple[float, float]:
    """Compute walk_score and progress_score from an entry's metric fields."""
    metrics = {
        "episode_length": float(entry.get("ep_len", 0) or 0),
        "upright_mean": float(entry.get("upright", 0) or 0),
        "height_reward": float(entry.get("height", 0) or 0),
        "body_contact": float(entry.get("contact", 0) or 0),
        "vx_w_mean": float(entry.get("vx", 0) or 0),
    }
    return compute_walk_score(metrics), compute_progress_score(metrics)


def log_result(entry: dict) -> None:
    """Append a result row to docs/autoresearch/results.tsv.

    Auto-computes walk_score and progress_score from the entry's metric
    fields, overriding any caller-provided values for consistency.
    """
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Always recompute scores from stored metrics (prevents drift)
    ws, ps = _scores_from_entry(entry)
    entry["walk_score"] = str(ws)
    entry["progress_score"] = str(ps)

    write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0

    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter="\t",
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(entry)


def load_results_history() -> list:
    """Read results.tsv into a list of dicts."""
    if not RESULTS_PATH.exists():
        return []

    with open(RESULTS_PATH, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


# ── Recompute Scores ───────────────────────────────────────────────────────

def recompute_scores() -> int:
    """Recompute walk_score and progress_score for all rows in results.tsv.

    Reads the file, recalculates both scores from stored metric fields,
    and writes back. Adds progress_score column if missing.
    Returns number of rows updated.
    """
    if not RESULTS_PATH.exists():
        print("No results.tsv found")
        return 0

    rows = []
    with open(RESULTS_PATH, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        existing_fields = reader.fieldnames or []
        rows = list(reader)

    updated = 0
    for row in rows:
        ws, ps = _scores_from_entry(row)
        row["walk_score"] = str(ws)
        row["progress_score"] = str(ps)
        updated += 1

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Recomputed scores for {updated} rows")
    return updated


# ── Backfill ───────────────────────────────────────────────────────────────

def backfill_results() -> int:
    """Scan existing experiment directories and backfill results.tsv."""
    log_dir = PROJECT_ROOT / "logs" / "skrl" / "harold_direct"
    if not log_dir.exists():
        print("No logs directory found")
        return 0

    # Try to import tensorboard for metrics extraction
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard not installed, cannot extract metrics")
        return 1

    existing = {r.get("exp_alias") for r in load_results_history()}
    added = 0

    for run_dir in sorted(log_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text())
        alias = manifest.get("alias", "")

        if alias in existing:
            continue

        # Find tensorboard events
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue

        # Extract metrics
        try:
            ea = EventAccumulator(str(run_dir))
            ea.Reload()
            metrics = {}
            tag_map = {
                "episode_length": "Episode / Total timesteps (mean)",
                "upright_mean": "Info / Episode_Metric/upright_mean",
                "height_reward": "Info / Episode_Metric/height_reward",
                "body_contact": "Info / Episode_Metric/body_contact_penalty",
                "vx_w_mean": "Info / Episode_Metric/vx_w_mean",
            }
            for key, tag in tag_map.items():
                try:
                    events = ea.Scalars(tag)
                    if events:
                        vals = [e.value for e in events[-10:]]
                        metrics[key] = sum(vals) / len(vals)
                except Exception:
                    pass

            if not metrics:
                continue

            score = compute_walk_score(metrics)
            hypothesis = manifest.get("hypothesis", "")
            timestamp = manifest.get("started_at") or manifest.get("created", "")

            entry = {
                "exp_alias": alias,
                "timestamp": timestamp,
                "hypothesis": hypothesis,
                "changed_params": "",
                "duration_min": "",
                "verdict": "backfill",
                "vx": f"{metrics.get('vx_w_mean', 0):.4f}",
                "upright": f"{metrics.get('upright_mean', 0):.4f}",
                "height": f"{metrics.get('height_reward', 0):.4f}",
                "contact": f"{metrics.get('body_contact', 0):.4f}",
                "ep_len": f"{metrics.get('episode_length', 0):.1f}",
                "walk_score": str(score),
                "decision": "backfill",
                "notes": f"backfilled from {run_dir.name}",
            }
            log_result(entry)
            added += 1
            print(f"  {alias}: walk_score={score}")
        except Exception as e:
            print(f"  {alias}: error - {e}")

    print(f"\nBackfilled {added} experiments")
    return 0


# ── Session State ──────────────────────────────────────────────────────────

def save_state(state_update: dict) -> None:
    """Save/update session state to session_state.json.

    Merges state_update into existing state. Auto-populates session_start
    and experiments_run from results.tsv if not provided.
    """
    # Load existing state or start fresh
    existing = load_state()

    if not existing.get("session_start"):
        existing["session_start"] = datetime.now(timezone.utc).isoformat()

    # Count experiments from results.tsv and auto-detect last_exp
    history = load_results_history()
    existing["experiments_run"] = len(history)

    if history and "last_exp" not in state_update:
        last_alias = history[-1].get("exp_alias", "")
        if last_alias:
            existing["last_exp"] = last_alias

    # Merge update
    existing.update(state_update)

    SESSION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_STATE_PATH.write_text(json.dumps(existing, indent=2) + "\n")


def load_state() -> dict:
    """Load session state from session_state.json. Returns empty dict if none."""
    if not SESSION_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SESSION_STATE_PATH.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ── CLI Interface ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Harold Autoresearch helpers")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("load-baseline", help="Print current config as JSON")
    sub.add_parser("load-registry", help="Print parameter registry as JSON")
    sub.add_parser("history", help="Print results.tsv as JSON")

    apply_p = sub.add_parser("apply", help="Apply config changes")
    apply_p.add_argument("delta", help="JSON dict of {param: value}")

    revert_p = sub.add_parser("revert", help="Revert config files to baseline")
    revert_p.add_argument("--all", action="store_true", dest="revert_all",
                          help="Also revert train_env.py (default: config only)")
    sub.add_parser("backfill", help="Backfill results.tsv from existing experiment logs")
    sub.add_parser("state", help="Print current session state")

    save_state_p = sub.add_parser("save-state", help="Save/update session state")
    save_state_p.add_argument("state_json", help="JSON dict of state fields to save/update")

    score_p = sub.add_parser("score", help="Compute quantitative score")
    score_p.add_argument("metrics", help="JSON dict of metrics")

    log_p = sub.add_parser("log", help="Append result to results.tsv")
    log_p.add_argument("entry", help="JSON dict of result entry")

    sub.add_parser("recompute-scores", help="Recompute walk_score and progress_score for all rows")

    args = parser.parse_args()

    if args.command == "load-baseline":
        config = load_baseline_config()
        print(json.dumps(config, indent=2, default=str))

    elif args.command == "load-registry":
        registry = load_parameter_registry()
        print(json.dumps(registry, indent=2, default=str))

    elif args.command == "apply":
        delta = json.loads(args.delta)
        modified = apply_change(delta)
        print(json.dumps({"modified_files": modified}))

    elif args.command == "revert":
        had_snapshot = BASELINE_SNAPSHOT_PATH.exists()
        revert_change(revert_all=args.revert_all)
        source = "baseline snapshot" if had_snapshot else "git HEAD"
        print(f"Reverted to {source}")

    elif args.command == "score":
        metrics = json.loads(args.metrics)
        ws = compute_walk_score(metrics)
        ps = compute_progress_score(metrics)
        print(json.dumps({"walk_score": ws, "progress_score": ps}))

    elif args.command == "log":
        entry = json.loads(args.entry)
        log_result(entry)
        print(f"Logged {entry.get('exp_alias', 'unknown')}")

    elif args.command == "history":
        history = load_results_history()
        print(json.dumps(history, indent=2))

    elif args.command == "recompute-scores":
        recompute_scores()

    elif args.command == "backfill":
        return backfill_results()

    elif args.command == "state":
        state = load_state()
        if state:
            print(json.dumps(state, indent=2))
        else:
            print("No session state found (fresh session)")

    elif args.command == "save-state":
        state_update = json.loads(args.state_json)
        save_state(state_update)
        print(f"Session state saved to {SESSION_STATE_PATH}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
