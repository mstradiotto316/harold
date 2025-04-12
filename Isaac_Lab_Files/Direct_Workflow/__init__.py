# THIS FILE SHOULD BE PLACED IN:
# IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/harold_v3/__init__.py

"""
Harold Direct Workflow environment.
"""

import gymnasium as gym

from . import agents
from .harold_env_cfg import HaroldEnvCfg
from .harold_env import HaroldEnv

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Harold-Direct-v3",
    entry_point="omni.isaac.lab_tasks.direct.harold_v3:HaroldEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HaroldEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HaroldPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

