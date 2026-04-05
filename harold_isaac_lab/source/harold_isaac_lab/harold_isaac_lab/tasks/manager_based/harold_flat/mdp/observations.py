"""Harold-specific observation terms for sim-to-real transfer.

The hardware IMU (MPU6050) estimates linear velocity via dead-reckoning
integration of accelerometer data with exponential decay. This produces
a noisy, oscillatory, and physically unrealistic signal. Training a
"velocity-blind" policy is standard practice for low-cost quadrupeds
without ground-truth velocity sensors.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def zero_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return zeros in place of base linear velocity.

    Forces the policy to walk without relying on velocity feedback,
    which is unreliable on hardware (MPU6050 dead-reckoning drift).
    The velocity tracking reward still uses the true physics velocity.
    """
    asset = env.scene[asset_cfg.name]
    return torch.zeros(asset.data.root_lin_vel_b.shape, device=env.device)
