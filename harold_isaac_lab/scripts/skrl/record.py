# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Post-hoc multi-camera video recording from a trained checkpoint.

Records a short episode using MultiCameraRecordVideo (4 cameras: side, front,
top, iso) from a saved checkpoint at 1 env.  Designed to decouple training
throughput from video recording overhead.

Usage:
    python record.py --task Template-Harold-Direct-flat-terrain-v0 \
        --checkpoint logs/skrl/.../checkpoints/best_agent.pt \
        --output_dir logs/skrl/.../videos/train
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record multi-camera video from a trained checkpoint.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory for output video files.")
parser.add_argument("--video_length", type=int, default=250, help="Number of steps to record (default: 250).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1).")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras for video recording
args_cli.enable_cameras = True
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import skrl
from packaging import version

SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import harold_isaac_lab.tasks  # noqa: F401
from harold_isaac_lab.common.multi_camera_video import MultiCameraRecordVideo


def main():
    """Record multi-camera video from a trained checkpoint."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=False
    )

    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_ppo_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # create isaac environment with rgb_array render mode
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for multi-camera video recording
    video_kwargs = {
        "video_folder": args_cli.output_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
    }
    print("[INFO] Recording multi-camera video from checkpoint.")
    print_dict(video_kwargs, nesting=4)
    env = MultiCameraRecordVideo(env, **video_kwargs)

    # wrap for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure runner (no logging, no checkpoints)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    # load checkpoint
    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # reset and run for video_length steps
    obs, _ = env.reset()
    timestep = 0
    with torch.inference_mode():
        while timestep < args_cli.video_length:
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
            timestep += 1

    print(f"[INFO] Recorded {timestep} steps to {args_cli.output_dir}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
