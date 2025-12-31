# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Record observations and actions from a trained policy for ONNX validation.

This script runs a trained policy and records both raw and normalized observations
along with the policy's action outputs. The data can then be used to validate
that the ONNX export produces identical outputs.

Usage:
    python record_episode.py --checkpoint logs/skrl/harold_direct/2025-12-30_07-30-54_ppo_torch/checkpoints/best_agent.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record observations/actions from trained policy")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Template-Harold-Direct-flat-terrain-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_steps", type=int, default=200, help="Number of steps to record")
parser.add_argument("--output", type=str, default=None, help="Output JSON path")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
import os
import torch
import numpy as np
from pathlib import Path

import skrl
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import harold_isaac_lab.tasks  # noqa: F401


def main():
    """Record episode data from trained policy."""
    # Parse environment config
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )

    # Load experiment config
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_ppo_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for skrl
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Configure runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    # Load checkpoint
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    runner.agent.load(checkpoint_path)
    runner.agent.set_running_mode("eval")

    # Extract running stats from agent's state preprocessor
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    running_mean = checkpoint["state_preprocessor"]["running_mean"].numpy().tolist()
    running_var = checkpoint["state_preprocessor"]["running_variance"].numpy().tolist()

    print(f"[INFO] Running mean shape: {len(running_mean)}")
    print(f"[INFO] Running var shape: {len(running_var)}")

    # Record data
    timesteps = []
    obs, _ = env.reset()

    print(f"\n[INFO] Recording {args_cli.num_steps} timesteps...")

    with torch.inference_mode():
        for step in range(args_cli.num_steps):
            # Get raw observation (before normalization)
            obs_raw = obs.cpu().numpy()[0].tolist()

            # Apply running stats normalization (same as skrl)
            mean = torch.tensor(running_mean, device=obs.device)
            var = torch.tensor(running_var, device=obs.device)
            eps = 1e-8
            obs_normalized = (obs - mean) / torch.sqrt(var + eps)
            obs_normalized = torch.clamp(obs_normalized, -5.0, 5.0)  # Clip like deployment
            obs_norm_list = obs_normalized.cpu().numpy()[0].tolist()

            # Get action from policy
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            action_list = actions.cpu().numpy()[0].tolist()

            # Store timestep data
            timesteps.append({
                "step": step,
                "obs_raw": obs_raw,
                "obs_normalized": obs_norm_list,
                "action": action_list,
            })

            # Step environment
            obs, _, _, _, _ = env.step(actions)

            if step % 50 == 0:
                print(f"  Step {step}/{args_cli.num_steps}")

    # Prepare output
    output_data = {
        "metadata": {
            "checkpoint": checkpoint_path,
            "num_steps": args_cli.num_steps,
            "task": args_cli.task,
            "obs_dim": len(running_mean),
            "action_dim": 12,
            "running_mean": running_mean,
            "running_var": running_var,
        },
        "timesteps": timesteps,
    }

    # Save to JSON
    if args_cli.output:
        output_path = Path(args_cli.output)
    else:
        output_path = Path("deployment/validation/sim_episode.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[INFO] Saved {len(timesteps)} timesteps to: {output_path}")
    print(f"[INFO] File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Print sample data for verification
    print("\n[INFO] Sample data (step 0):")
    print(f"  obs_raw[:5]: {timesteps[0]['obs_raw'][:5]}")
    print(f"  obs_normalized[:5]: {timesteps[0]['obs_normalized'][:5]}")
    print(f"  action[:4]: {timesteps[0]['action'][:4]}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
