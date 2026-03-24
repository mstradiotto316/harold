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


# ── Results Logging ─────────────────────────────────────────────────────────

RESULTS_COLUMNS = [
    "exp_alias", "timestamp", "hypothesis", "changed_params", "duration_min",
    "vx", "upright", "height", "contact", "ep_len",
    "video_verdict", "decision", "notes",
]


def log_result(entry: dict) -> None:
    """Append a result row to docs/autoresearch/results.tsv."""
    vv = (entry.get("video_verdict") or "").strip().upper()
    valid_verdicts = {"LOCOMOTION", "STEPPING", "STANDING", "FALLING", "DEGENERATE"}
    if vv not in valid_verdicts:
        print(f"ERROR: video_verdict must be one of {sorted(valid_verdicts)}, got: {vv!r}")
        print("       Run 'harold record' + video review before logging.")
        sys.exit(1)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

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

            hypothesis = manifest.get("hypothesis", "")
            timestamp = manifest.get("started_at") or manifest.get("created", "")

            vx_val = metrics.get('vx_w_mean', 0)
            entry = {
                "exp_alias": alias,
                "timestamp": timestamp,
                "hypothesis": hypothesis,
                "changed_params": "",
                "duration_min": "",
                "vx": f"{vx_val:.4f}",
                "upright": f"{metrics.get('upright_mean', 0):.4f}",
                "height": f"{metrics.get('height_reward', 0):.4f}",
                "contact": f"{metrics.get('body_contact', 0):.4f}",
                "ep_len": f"{metrics.get('episode_length', 0):.1f}",
                "video_verdict": "",
                "decision": "backfill",
                "notes": f"backfilled from {run_dir.name}",
            }
            log_result(entry)
            added += 1
            print(f"  {alias}: vx={vx_val:.4f}")
        except Exception as e:
            print(f"  {alias}: error - {e}")

    print(f"\nBackfilled {added} experiments")
    return 0


# ── Changed Params Parsing ─────────────────────────────────────────────────

# Map parameter names to search axes for plateau detection / synthesis
_PARAM_AXIS = {}
_REWARD_PARAMS = {
    "track_lin_vel_xy_weight", "track_lin_vel_xy_std",
    "track_ang_vel_z_weight", "track_ang_vel_z_std",
    "base_orientation_weight", "base_motion_weight",
    "dof_torques_weight", "dof_acc_weight", "action_smoothness_weight",
    "feet_air_time_weight", "feet_air_time_threshold",
    "continuous_gait_weight", "air_time_variance_weight",
    "foot_slip_weight", "shoulder_joint_vel_weight",
    "undesired_contacts_weight", "undesired_contacts_threshold",
    "forward_motion_weight",
    "joint_pos_weight", "joint_pos_stand_still_scale", "joint_pos_velocity_threshold",
}
_COMMAND_PARAMS = {
    "vx_min", "vx_max", "vy_min", "vy_max", "yaw_min", "yaw_max",
    "zero_velocity_prob", "command_change_interval",
}
_TERMINATION_PARAMS = {
    "orientation_threshold", "height_threshold", "body_contact_threshold",
    "elbow_pose_termination",
}
_DOMAIN_RAND_PARAMS = {
    "enable_randomization", "add_imu_noise", "add_joint_noise",
    "add_lin_vel_noise", "randomize_friction", "randomize_mass",
    "add_action_noise", "apply_external_forces",
}
_ENV_LEVEL_PARAMS = {"episode_length_s", "action_scale", "action_filter_beta"}

for _p in _REWARD_PARAMS:
    _PARAM_AXIS[_p] = "reward_weights"
for _p in _COMMAND_PARAMS:
    _PARAM_AXIS[_p] = "command_ranges"
for _p in _TERMINATION_PARAMS:
    _PARAM_AXIS[_p] = "termination"
for _p in _DOMAIN_RAND_PARAMS:
    _PARAM_AXIS[_p] = "domain_randomization"
for _p in _ENV_LEVEL_PARAMS:
    _PARAM_AXIS[_p] = "env_params"
for _p in _PPO_PARAMS:
    _PARAM_AXIS[_p] = "ppo"


def parse_changed_params(s: str) -> list[dict]:
    """Parse the changed_params TSV column into structured dicts.

    Formats handled:
        'param:old->new'
        'param:old->new, param2:old->new'
        'train_env.py: description'  (free-text code change)
        'duration:fast->short'

    Returns list of dicts:
        [{"param": str, "old": val, "new": val, "type": "config"|"code"|"duration"}]
    """
    if not s or not s.strip():
        return []

    s = s.strip()

    # Free-text code change (train_env.py)
    if s.startswith("train_env.py:"):
        return [{"param": "train_env.py", "old": None, "new": None,
                 "type": "code", "text": s}]

    entries = []
    # Split on comma, but be careful with spaces
    parts = [p.strip() for p in s.split(",")]

    for part in parts:
        m = re.match(r"^(.+?):(.+)->(.+)$", part.strip())
        if m:
            param = m.group(1).strip()
            old_raw = m.group(2).strip()
            new_raw = m.group(3).strip()

            if param == "duration":
                entries.append({"param": "duration", "old": old_raw,
                                "new": new_raw, "type": "duration"})
            else:
                entries.append({
                    "param": param,
                    "old": _try_parse_numeric(old_raw),
                    "new": _try_parse_numeric(new_raw),
                    "type": "config",
                })
        elif part.strip():
            # Unrecognized format — treat as free-text
            entries.append({"param": "unknown", "old": None, "new": None,
                            "type": "code", "text": part.strip()})

    return entries


def _try_parse_numeric(s: str):
    """Try to parse as float/int/bool, else return string."""
    if s in ("True", "False"):
        return s == "True"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def _categorize_experiment(row: dict) -> str:
    """Categorize an experiment row by its primary search axis."""
    changes = parse_changed_params(row.get("changed_params", ""))
    if not changes:
        return "unknown"

    for ch in changes:
        if ch["type"] == "code":
            return "code_changes"
        if ch["type"] == "duration":
            continue
        param = ch["param"]
        if param == "seed":
            return "seed"
        if param in _PARAM_AXIS:
            return _PARAM_AXIS[param]

    # If only duration changes
    if all(ch["type"] == "duration" for ch in changes):
        return "duration"

    return "unknown"


# ── Experiment Similarity Detection ────────────────────────────────────────

def check_similarity(proposed: dict, hypothesis: str = "") -> dict:
    """Check if a proposed change is similar to a previously failed experiment.

    Args:
        proposed: dict of {param: value} for config changes
        hypothesis: text description for code-change similarity

    Returns:
        {"warnings": [str, ...], "similar_count": int}
    """
    history = load_results_history()
    warnings = []

    for row_idx, row in enumerate(history):
        decision = (row.get("decision") or "").strip().upper()
        if decision not in ("DISCARD",):
            continue

        changes = parse_changed_params(row.get("changed_params", ""))

        # Check config param similarity
        for ch in changes:
            if ch["type"] != "config":
                continue
            param = ch["param"]
            if param not in proposed:
                continue

            proposed_val = proposed[param]
            hist_val = ch["new"]

            # Compare values
            if isinstance(proposed_val, (int, float)) and isinstance(hist_val, (int, float)):
                if hist_val == 0:
                    similar = (proposed_val == 0)
                else:
                    similar = abs(proposed_val - hist_val) / abs(hist_val) < 0.20
            else:
                similar = (str(proposed_val) == str(hist_val))

            if similar:
                alias = row.get("exp_alias", f"row {row_idx + 1}")
                vx = row.get("vx", "?")
                warnings.append(
                    f"Similar to DISCARD {alias} "
                    f"({ch['param']}:{ch['old']}->{ch['new']}, vx={vx})"
                )

        # Check hypothesis text similarity for code changes
        if hypothesis:
            row_hyp = (row.get("hypothesis") or "").lower()
            row_params = (row.get("changed_params") or "").lower()
            # Extract keywords from hypothesis (3+ char words)
            keywords = [w for w in re.findall(r"[a-z_]{3,}", hypothesis.lower())]
            matching = sum(1 for kw in keywords
                          if kw in row_hyp or kw in row_params)
            if keywords and matching >= len(keywords) * 0.6:
                alias = row.get("exp_alias", f"row {row_idx + 1}")
                vx = row.get("vx", "?")
                warnings.append(
                    f"Similar hypothesis to DISCARD {alias}: "
                    f"\"{row.get('hypothesis', '?')[:80]}\" (vx={vx})"
                )

    return {"warnings": warnings, "similar_count": len(warnings)}


# ── Plateau Detection ──────────────────────────────────────────────────────

def detect_plateau(window: int = 10) -> dict:
    """Detect optimization plateaus beyond simple consecutive-DISCARD counting.

    Returns structured report with plateau status, axes tried, and suggestions.
    """
    history = load_results_history()
    if not history:
        return {"is_plateau": False, "reason": "no history"}

    # Find last KEEP row
    last_keep_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if (history[i].get("decision") or "").strip().upper() == "KEEP":
            last_keep_idx = i
            break

    # Find best-ever vx (across all rows)
    best_vx = -999.0
    best_vx_alias = ""
    best_vx_row = -1
    for i, row in enumerate(history):
        try:
            vx = float(row.get("vx", 0) or 0)
        except (ValueError, TypeError):
            continue
        if vx > best_vx:
            best_vx = vx
            best_vx_alias = row.get("exp_alias", f"row {i + 1}")
            best_vx_row = i

    experiments_since_keep = len(history) - 1 - last_keep_idx if last_keep_idx >= 0 else len(history)

    # Categorize experiments since last KEEP by axis
    axes_tried = {}
    if last_keep_idx >= 0:
        for row in history[last_keep_idx + 1:]:
            axis = _categorize_experiment(row)
            axes_tried[axis] = axes_tried.get(axis, 0) + 1

    # Check if recent experiments are within noise of best
    recent_rows = history[-window:]

    # Video verdict distribution for recent experiments
    recent_verdicts = {}
    for row in recent_rows:
        vv = (row.get("video_verdict") or "").strip().upper()
        if vv:
            recent_verdicts[vv] = recent_verdicts.get(vv, 0) + 1

    recent_best_vx = -999.0
    recent_best_row = ""
    for row in recent_rows:
        try:
            vx = float(row.get("vx", 0) or 0)
        except (ValueError, TypeError):
            continue
        if vx > recent_best_vx:
            recent_best_vx = vx
            recent_best_row = row.get("exp_alias", "?")

    noise_plateau = (best_vx > 0 and recent_best_vx > 0 and
                     abs(recent_best_vx - best_vx) / best_vx < 0.10)

    # Find untried axes
    all_axes = {"reward_weights", "command_ranges", "termination",
                "domain_randomization", "env_params", "ppo", "code_changes",
                "duration", "seed"}
    tried_axes = set(axes_tried.keys()) - {"unknown"}
    untried_axes = sorted(all_axes - tried_axes)

    # Determine plateau
    is_plateau = (experiments_since_keep >= 5 and noise_plateau) or experiments_since_keep >= 15

    # Generate suggestion
    suggestion = ""
    if is_plateau:
        if untried_axes:
            suggestion = f"Untried axes: {', '.join(untried_axes)}. Try one of these."
        elif "code_changes" not in tried_axes:
            suggestion = "All config axes exhausted. Try train_env.py code changes."
        else:
            suggestion = ("All axes tried. Consider: combination experiments, "
                          "longer training duration, or different seed.")

    return {
        "is_plateau": is_plateau,
        "experiments_since_improvement": experiments_since_keep,
        "best_ever_vx": round(best_vx, 4),
        "best_ever_alias": best_vx_alias,
        "best_ever_row": best_vx_row + 1,
        "recent_best_vx": round(recent_best_vx, 4),
        "recent_best_alias": recent_best_row,
        "axes_tried_since_keep": axes_tried,
        "untried_axes": untried_axes,
        "recent_video_verdicts": recent_verdicts,
        "suggestion": suggestion,
    }


# ── Cross-Session Synthesis ────────────────────────────────────────────────

def synthesize_history() -> dict:
    """Programmatic pattern extraction from the full experiment history.

    Returns:
        - parameter_sensitivity: per-param stats (best value, direction, outcome)
        - winning_config: accumulated KEEP changes
        - untried_params: registry params never changed in any experiment
        - interaction_pairs: params that co-occurred in KEEP experiments
    """
    history = load_results_history()
    if not history:
        return {"error": "no history"}

    # Parse all changed_params, group by parameter
    param_experiments = {}  # param -> [{value, vx, decision, alias, row}]
    for i, row in enumerate(history):
        changes = parse_changed_params(row.get("changed_params", ""))
        decision = (row.get("decision") or "").strip().upper()
        try:
            vx = float(row.get("vx", 0) or 0)
        except (ValueError, TypeError):
            vx = 0.0

        for ch in changes:
            if ch["type"] != "config":
                continue
            param = ch["param"]
            if param not in param_experiments:
                param_experiments[param] = []
            param_experiments[param].append({
                "new_value": ch["new"],
                "old_value": ch["old"],
                "vx": vx,
                "decision": decision,
                "alias": row.get("exp_alias", f"row {i + 1}"),
            })

    # Compute per-param sensitivity
    sensitivity = {}
    for param, exps in param_experiments.items():
        if not exps:
            continue
        best_exp = max(exps, key=lambda e: e["vx"])
        worst_exp = min(exps, key=lambda e: e["vx"])
        keeps = [e for e in exps if e["decision"] == "KEEP"]
        discards = [e for e in exps if e["decision"] == "DISCARD"]

        entry = {
            "experiments": len(exps),
            "best_value": best_exp["new_value"],
            "best_vx": round(best_exp["vx"], 4),
            "best_alias": best_exp["alias"],
            "worst_vx": round(worst_exp["vx"], 4),
            "keeps": len(keeps),
            "discards": len(discards),
        }

        # Detect direction (for numeric params with 2+ experiments)
        numeric_exps = [e for e in exps
                        if isinstance(e["new_value"], (int, float))]
        if len(numeric_exps) >= 2:
            sorted_by_val = sorted(numeric_exps, key=lambda e: e["new_value"])
            sorted_by_vx = sorted(numeric_exps, key=lambda e: e["vx"], reverse=True)
            best_val = sorted_by_vx[0]["new_value"]
            old_val = sorted_by_val[0].get("old_value")
            if isinstance(old_val, (int, float)) and isinstance(best_val, (int, float)):
                if best_val > old_val:
                    entry["best_direction"] = "increase"
                elif best_val < old_val:
                    entry["best_direction"] = "decrease"

            # Diminishing returns: 3+ values, improvement shrinking
            if len(numeric_exps) >= 3:
                vals_and_vx = sorted(
                    [(e["new_value"], e["vx"]) for e in numeric_exps],
                    key=lambda x: x[0]
                )
                # Check if best vx is at an intermediate value (not extremes)
                best_idx = max(range(len(vals_and_vx)), key=lambda j: vals_and_vx[j][1])
                if 0 < best_idx < len(vals_and_vx) - 1:
                    entry["diminishing_returns"] = True

        sensitivity[param] = entry

    # Winning config: accumulate all KEEP changes in order
    winning_config = {}
    for row in history:
        if (row.get("decision") or "").strip().upper() != "KEEP":
            continue
        changes = parse_changed_params(row.get("changed_params", ""))
        for ch in changes:
            if ch["type"] == "config" and ch["new"] is not None:
                winning_config[ch["param"]] = ch["new"]

    # Untried params: registry params never in any experiment
    tried_params = set(param_experiments.keys())
    try:
        registry = load_parameter_registry()
        all_tunable = {p for p, meta in registry.items()
                       if meta["category"] in ("TUNABLE", "CONSTRAINED")}
        untried = sorted(all_tunable - tried_params)
    except FileNotFoundError:
        untried = []

    # Interaction pairs: params that co-occurred in KEEP experiments
    interactions = []
    for row in history:
        if (row.get("decision") or "").strip().upper() != "KEEP":
            continue
        changes = parse_changed_params(row.get("changed_params", ""))
        config_params = [ch["param"] for ch in changes if ch["type"] == "config"]
        if len(config_params) >= 2:
            interactions.append({
                "params": config_params,
                "alias": row.get("exp_alias", "?"),
                "vx": float(row.get("vx", 0) or 0),
            })

    return {
        "parameter_sensitivity": sensitivity,
        "winning_config": winning_config,
        "untried_params": untried,
        "interaction_pairs": interactions,
    }


# ── Combination Generator ─────────────────────────────────────────────────

def suggest_combinations(top_n: int = 3) -> list[dict]:
    """Generate combination experiment proposals from near-miss DISCARDs.

    Finds DISCARD experiments where at least one metric beat the current
    baseline, then proposes non-conflicting pairs.

    Returns list of proposals with sources and rationale.
    """
    history = load_results_history()
    if not history:
        return []

    # Find current baseline (last KEEP)
    baseline = None
    for row in reversed(history):
        if (row.get("decision") or "").strip().upper() == "KEEP":
            baseline = row
            break
    if not baseline:
        return []

    baseline_vx = float(baseline.get("vx", 0) or 0)
    baseline_upright = float(baseline.get("upright", 0) or 0)
    baseline_ep_len = float(baseline.get("ep_len", 0) or 0)

    # Accumulate current KEEP config
    keep_config = {}
    for row in history:
        if (row.get("decision") or "").strip().upper() != "KEEP":
            continue
        for ch in parse_changed_params(row.get("changed_params", "")):
            if ch["type"] == "config" and ch["new"] is not None:
                keep_config[ch["param"]] = ch["new"]

    # Find near-miss DISCARDs: beat baseline on at least one metric
    near_misses = []
    for i, row in enumerate(history):
        if (row.get("decision") or "").strip().upper() != "DISCARD":
            continue

        changes = parse_changed_params(row.get("changed_params", ""))
        config_changes = [ch for ch in changes if ch["type"] == "config"]
        if not config_changes:
            continue

        # Skip if the change is already in the current KEEP config
        novel_changes = []
        for ch in config_changes:
            if ch["param"] in keep_config and keep_config[ch["param"]] == ch["new"]:
                continue
            novel_changes.append(ch)
        if not novel_changes:
            continue

        try:
            vx = float(row.get("vx", 0) or 0)
            upright = float(row.get("upright", 0) or 0)
            ep_len = float(row.get("ep_len", 0) or 0)
        except (ValueError, TypeError):
            continue

        improvements = []
        if vx > baseline_vx * 0.8:  # Within 20% of baseline vx
            improvements.append(f"vx={vx:.3f}")
        if upright > baseline_upright:
            improvements.append(f"upright={upright:.3f}")
        if ep_len > baseline_ep_len:
            improvements.append(f"ep_len={ep_len:.0f}")

        if improvements:
            near_misses.append({
                "row": i + 1,
                "alias": row.get("exp_alias", f"row {i + 1}"),
                "changes": novel_changes,
                "improvements": improvements,
                "vx": vx,
                "upright": upright,
                "ep_len": ep_len,
            })

    # Generate non-conflicting pairs
    proposals = []
    for i in range(len(near_misses)):
        for j in range(i + 1, len(near_misses)):
            a, b = near_misses[i], near_misses[j]
            a_params = {ch["param"] for ch in a["changes"]}
            b_params = {ch["param"] for ch in b["changes"]}

            # Skip if they modify the same parameter
            if a_params & b_params:
                continue

            # Score: sum of metric improvements
            score = (a["vx"] + b["vx"]) + (a["upright"] + b["upright"]) * 0.1

            changes_desc = []
            for ch in a["changes"] + b["changes"]:
                changes_desc.append(f"{ch['param']}={ch['new']}")

            proposals.append({
                "changes": {ch["param"]: ch["new"]
                            for ch in a["changes"] + b["changes"]},
                "changes_desc": " + ".join(changes_desc),
                "sources": [
                    f"{a['alias']} ({', '.join(a['improvements'])})",
                    f"{b['alias']} ({', '.join(b['improvements'])})",
                ],
                "score": round(score, 4),
                "conflicts": [],
            })

    # Sort by score descending, take top N
    proposals.sort(key=lambda p: p["score"], reverse=True)
    return proposals[:top_n]


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

    log_p = sub.add_parser("log", help="Append result to results.tsv")
    log_p.add_argument("entry", help="JSON dict of result entry")

    sim_p = sub.add_parser("check-similarity", help="Check if proposed change is similar to past failures")
    sim_p.add_argument("delta", help="JSON dict of {param: value}")
    sim_p.add_argument("--hypothesis", default="", help="Text hypothesis for code-change similarity")

    sub.add_parser("detect-plateau", help="Detect optimization plateaus")

    sub.add_parser("synthesize", help="Extract patterns from experiment history")

    combo_p = sub.add_parser("suggest-combinations", help="Suggest combination experiments from near-misses")
    combo_p.add_argument("--top", type=int, default=3, help="Number of proposals (default: 3)")

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

    elif args.command == "log":
        entry = json.loads(args.entry)
        log_result(entry)
        print(f"Logged {entry.get('exp_alias', 'unknown')}")

    elif args.command == "history":
        history = load_results_history()
        print(json.dumps(history, indent=2))

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

    elif args.command == "check-similarity":
        proposed = json.loads(args.delta)
        result = check_similarity(proposed, hypothesis=args.hypothesis)
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"WARNING: {w}")
            print(f"\n{result['similar_count']} similar DISCARD experiment(s) found.")
        else:
            print("No similar past failures found. Proceed.")
        print(json.dumps(result))

    elif args.command == "detect-plateau":
        result = detect_plateau()
        if result.get("is_plateau"):
            print("PLATEAU DETECTED:")
        else:
            print("No plateau detected:")
        print(f"  experiments_since_improvement: {result.get('experiments_since_improvement', 0)}")
        print(f"  best_ever_vx: {result.get('best_ever_vx', 0)} ({result.get('best_ever_alias', '?')})")
        print(f"  recent_best_vx: {result.get('recent_best_vx', 0)} ({result.get('recent_best_alias', '?')})")
        if result.get("axes_tried_since_keep"):
            axes = ", ".join(f"{k}({v})" for k, v in result["axes_tried_since_keep"].items())
            print(f"  axes_tried_since_keep: {axes}")
        if result.get("untried_axes"):
            print(f"  untried_axes: {result['untried_axes']}")
        if result.get("recent_video_verdicts"):
            verdicts = ", ".join(f"{k}({v})" for k, v in result["recent_video_verdicts"].items())
            print(f"  recent_video_verdicts: {verdicts}")
        if result.get("suggestion"):
            print(f"  suggestion: {result['suggestion']}")
        print(json.dumps(result))

    elif args.command == "synthesize":
        result = synthesize_history()
        sens = result.get("parameter_sensitivity", {})
        if sens:
            print("PARAMETER SENSITIVITY:")
            for param, info in sorted(sens.items(), key=lambda x: -x[1].get("experiments", 0)):
                direction = info.get("best_direction", "?")
                dr = " (DIMINISHING RETURNS)" if info.get("diminishing_returns") else ""
                print(f"  {param}: {info['experiments']} exps, "
                      f"best={info['best_value']} (vx={info['best_vx']}), "
                      f"direction={direction}{dr}, "
                      f"keeps={info['keeps']}/discards={info['discards']}")
        wc = result.get("winning_config", {})
        if wc:
            print(f"\nWINNING CONFIG (accumulated KEEPs): {json.dumps(wc)}")
        untried = result.get("untried_params", [])
        if untried:
            print(f"\nUNTRIED PARAMETERS: {', '.join(untried)}")
        pairs = result.get("interaction_pairs", [])
        if pairs:
            print("\nINTERACTION PAIRS:")
            for p in pairs:
                print(f"  {' + '.join(p['params'])} ({p['alias']}, vx={p['vx']:.3f})")
        print(json.dumps(result))

    elif args.command == "suggest-combinations":
        proposals = suggest_combinations(top_n=args.top)
        if not proposals:
            print("No combination proposals found (need near-miss DISCARDs).")
        else:
            for i, p in enumerate(proposals, 1):
                print(f"\nPROPOSAL {i}: {p['changes_desc']}")
                for src in p["sources"]:
                    print(f"  Source: {src}")
                print(f"  Score: {p['score']}")
        print(json.dumps(proposals))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
