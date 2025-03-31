

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--export_policy", action="store_true", help="Export policy instead of playing")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper


def export_policy(agent, env, export_path="/home/matteo/Desktop/Harold_V5/policy"):
    """Exports policy to ONNX and action_config using pickle, for MLP policies (no RNN) - CORRECTED DEVICE ISSUE"""
    import os
    import torch
    import pickle  # Import pickle

    # Create export directory
    os.makedirs(export_path, exist_ok=True)

    # 1. Get device from network
    device = next(agent.model.a2c_network.parameters()).device

    # **2. Move running_mean_std to CPU before export - FIX DEVICE MISMATCH**
    agent.model.running_mean_std.cpu() # Move running_mean_std to CPU! - NEW!


    # 3. Create wrapper with device awareness - INCORPORATE OBSERVATION NORMALIZATION
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, network, device, model): # Pass 'model' to wrapper
            super().__init__()
            self.network = network
            self.device = device
            self._model = model # Store model

        def forward(self, input_tensor):
            # Ensure input is on correct device and NORMALIZE OBSERVATIONS - NEW!
            input_dict = {'obs': input_tensor.to(self.device)} # Create input dictionary
            input_dict['obs'] = self._model.norm_obs(input_dict['obs']) # Apply Observation Normalization - CRITICAL!
            rnn_out = self.network(input_dict)[0] # Get network output
            return rnn_out # Return ONLY 'mu' (mean action)

    # 4. Move network to CPU for export (Jetson will use CPU)
    original_network = agent.model.a2c_network.cpu()
    wrapped_network = ONNXWrapper(original_network, device='cpu', model=agent.model) # Pass 'model' to wrapper
    wrapped_network.eval()

    # 5. Create dummy input on CPU - SIMPLIFIED for MLP - NO RNN STATES
    dummy_input = torch.randn(1, 50).cpu()  # Match observation dimensions - NO RNN STATES NEEDED FOR MLP


    # 6. Export to ONNX - SIMPLIFIED for MLP - NO RNN INPUT/OUTPUT
    onnx_path = os.path.join(export_path, "harold_policy.onnx")
    torch.onnx.export(
        wrapped_network,
        dummy_input, # Pass dummy input tensor directly - SIMPLIFIED for MLP
        onnx_path,
        input_names=['obs'], # Input name is just 'obs' - SIMPLIFIED for MLP
        output_names=['action'], # Output name is 'action' now (mu only)
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        },
        opset_version=13,
        do_constant_folding=True
    )

    # 7. Save config using pickle.dump() - No changes needed here
    config_path = os.path.join(export_path, "action_config.pt")
    action_config = {
        'action_scale': env.cfg.action_scale,
        'default_joint_pos': env.cfg.robot.init_state.joint_pos
    }
    with open(config_path, 'wb') as f: # Open in binary write mode 'wb'
        pickle.dump(action_config, f) # Use pickle.dump to save

    print(f"Successfully exported ONNX policy to {onnx_path} (MLP Policy Export - No RNN - DEVICE FIX)") # Updated message
    print(f"Successfully exported action config to {config_path} (using pickle)")


def main():

    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)

    # USE THIS TO EXPORT THE POLICY ==============================
    if args_cli.export_policy:
        export_policy(agent, env.unwrapped)
        return  # Exit after exporting

    agent.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.

    while simulation_app.is_running():

        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=True)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            
            # Print joint positions
            if isinstance(obs, dict):
                # Print shape to debug
                print("Observation shape:", obs["policy"].shape)
                # Extract joint positions (adjust indices if needed)
                joint_positions = obs["policy"][11:23]  # Remove the first dimension indexing
                print("Joint Positions:", joint_positions)
            
            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

