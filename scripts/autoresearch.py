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

REGISTRY_PATH = PROJECT_ROOT / "docs" / "autoresearch" / "PARAMETER_REGISTRY.md"
RESULTS_PATH = PROJECT_ROOT / "docs" / "autoresearch" / "results.tsv"

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


# ── Apply / Revert Changes ──────────────────────────────────────────────────

def apply_change(delta: dict) -> list:
    """Apply config changes. Returns list of modified files.

    delta: {param_name: new_value, ...}

    Validates against parameter registry before applying.
    Raises ValueError for FROZEN params or out-of-range values.
    """
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


def revert_change(files: list = None) -> None:
    """Revert config files to the last git commit state."""
    if files is None:
        files = [str(ENV_CFG_PATH), str(PPO_CFG_PATH)]
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

DEFAULT_WEIGHTS = {
    "episode_length": 1.0,
    "upright": 2.0,
    "height": 1.5,
    "contact": 1.0,
    "vx": 3.0,
}


def compute_score(metrics: dict, weights: dict = None) -> float:
    """Compute a 0-100 quantitative score from the 5-metric protocol.

    Args:
        metrics: dict with keys from harold validate (episode_length, upright_mean,
                 height_reward, body_contact, vx_w_mean)
        weights: optional override for metric weights (from strategy.md scoring priorities)

    Returns:
        float score 0-100
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    ep_len = metrics.get("episode_length", 0)

    # Sanity gate
    if ep_len < 300:
        return 0.0

    scores = {}

    # Episode length: 300=0, 600=1
    scores["episode_length"] = min(1.0, max(0.0, (ep_len - 300) / 300))

    # Upright: 0.85=0, 0.98=1
    upright = metrics.get("upright_mean", 0)
    scores["upright"] = min(1.0, max(0.0, (upright - 0.85) / 0.13))

    # Height: 0.4=0, 0.8=1
    height = metrics.get("height_reward", 0)
    scores["height"] = min(1.0, max(0.0, (height - 0.4) / 0.4))

    # Contact: -0.5=0, 0.0=1
    contact = metrics.get("body_contact", -1)
    scores["contact"] = min(1.0, max(0.0, (contact + 0.5) / 0.5))

    # Forward velocity: 0=0, 0.03=1
    vx = metrics.get("vx_w_mean", 0)
    scores["vx"] = min(1.0, max(0.0, vx / 0.03))

    total_weight = sum(weights.get(k, 1.0) for k in scores)
    weighted_score = sum(scores[k] * weights.get(k, 1.0) for k in scores) / total_weight

    return round(weighted_score * 100, 1)


# ── Results Logging ─────────────────────────────────────────────────────────

RESULTS_COLUMNS = [
    "exp_alias", "timestamp", "hypothesis", "changed_params", "duration_min",
    "verdict", "vx", "upright", "height", "contact", "ep_len",
    "quant_score", "qual_score", "composite", "decision", "gait_type", "video_notes",
]


def log_result(entry: dict) -> None:
    """Append a result row to docs/autoresearch/results.tsv."""
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


# ── CLI Interface ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Harold Autoresearch helpers")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("load-baseline", help="Print current config as JSON")
    sub.add_parser("load-registry", help="Print parameter registry as JSON")
    sub.add_parser("history", help="Print results.tsv as JSON")

    apply_p = sub.add_parser("apply", help="Apply config changes")
    apply_p.add_argument("delta", help="JSON dict of {param: value}")

    sub.add_parser("revert", help="Revert config files to git HEAD")

    score_p = sub.add_parser("score", help="Compute quantitative score")
    score_p.add_argument("metrics", help="JSON dict of metrics")

    log_p = sub.add_parser("log", help="Append result to results.tsv")
    log_p.add_argument("entry", help="JSON dict of result entry")

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
        revert_change()
        print("Reverted to git HEAD")

    elif args.command == "score":
        metrics = json.loads(args.metrics)
        s = compute_score(metrics)
        print(json.dumps({"score": s}))

    elif args.command == "history":
        history = load_results_history()
        print(json.dumps(history, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
