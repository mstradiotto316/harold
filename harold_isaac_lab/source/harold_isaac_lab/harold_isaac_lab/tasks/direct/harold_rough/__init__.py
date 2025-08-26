import gymnasium as gym

from . import agents

# Register Gym environment for Rough Terrain (copy of flat for now)
gym.register(
    id="Template-Harold-Direct-rough-terrain-v0",
    entry_point=f"{__name__}.harold_isaac_lab_env:HaroldIsaacLabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.harold_isaac_lab_env_cfg:HaroldIsaacLabEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

