"""CPG (Central Pattern Generator) trajectory generator.

Generates the base walking trajectory for Harold robot.
Uses the shared CPG kernel in common/cpg_math.py for sim-to-real match.

Gait Pattern:
    - Diagonal trot: FL+BR alternate with FR+BL
    - 0.5 Hz frequency (2 second cycle)
    - Duty-cycle aware stance/swing split (stance holds calf, swing flexes knee)
    - Thigh and calf are phased to shorten the leg during swing for clearance
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import cpg_math


@dataclass
class CPGConfig:
    """CPG parameters - must match simulation exactly."""
    frequency_hz: float = 0.5
    duty_cycle: float = 0.6
    swing_thigh: float = 0.54
    stance_thigh: float = 0.67
    stance_calf: float = -0.87
    swing_calf: float = -1.3963
    shoulder_amplitude: float = 0.0096
    thigh_offset_front: float = 0.0
    thigh_offset_back: float = 0.0

    @classmethod
    def from_yaml(cls, path: Path) -> "CPGConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        traj = data.get("trajectory", {})
        return cls(
            frequency_hz=data.get("frequency_hz", 0.5),
            duty_cycle=data.get("duty_cycle", 0.6),
            swing_thigh=traj.get("swing_thigh", 0.54),
            stance_thigh=traj.get("stance_thigh", 0.67),
            stance_calf=traj.get("stance_calf", -0.87),
            swing_calf=traj.get("swing_calf", -1.3963),
            shoulder_amplitude=traj.get("shoulder_amplitude", 0.0096),
            thigh_offset_front=traj.get("thigh_offset_front", 0.0),
            thigh_offset_back=traj.get("thigh_offset_back", 0.0),
        )


class CPGGenerator:
    """Generates CPG base trajectory for walking gait.

    Usage:
        cpg = CPGGenerator()
        targets = cpg.compute(time)  # Returns 12D joint targets
        phase = cpg.phase            # Current phase [0, 1)
    """

    def __init__(self, config: CPGConfig | None = None):
        self.cfg = config or CPGConfig()
        self._phase = 0.0
        self._ops = cpg_math.numpy_ops(np)

    @property
    def phase(self) -> float:
        """Current gait phase [0, 1)."""
        return self._phase

    def compute(self, time: float) -> np.ndarray:
        """Compute CPG trajectory at given time.

        Args:
            time: Elapsed time in seconds since gait start

        Returns:
            12D numpy array of joint targets (radians)
            Order: [shoulders(4), thighs(4), calves(4)]
                   Each group: [FL, FR, BL, BR]
        """
        # Compute phase from time
        self._phase = (time * self.cfg.frequency_hz) % 1.0

        phase = np.array(self._phase, dtype=np.float32)
        targets = cpg_math.compute_cpg_targets(phase, self.cfg, self._ops)
        return targets.astype(np.float32)

    def get_phase_sin_cos(self) -> tuple[float, float]:
        """Get sin/cos of current phase for observation."""
        return math.sin(2 * math.pi * self._phase), math.cos(2 * math.pi * self._phase)


if __name__ == "__main__":
    # Test CPG generator
    import matplotlib.pyplot as plt

    cpg = CPGGenerator()

    # Generate one full cycle
    times = np.linspace(0, 2.0, 100)  # 2 seconds = 1 cycle at 0.5 Hz
    trajectories = np.array([cpg.compute(t) for t in times])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot shoulders
    axes[0].plot(times, trajectories[:, 0], label='FL')
    axes[0].plot(times, trajectories[:, 1], label='FR')
    axes[0].plot(times, trajectories[:, 2], label='BL')
    axes[0].plot(times, trajectories[:, 3], label='BR')
    axes[0].set_ylabel('Shoulder (rad)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot thighs
    axes[1].plot(times, trajectories[:, 4], label='FL')
    axes[1].plot(times, trajectories[:, 5], label='FR')
    axes[1].plot(times, trajectories[:, 6], label='BL')
    axes[1].plot(times, trajectories[:, 7], label='BR')
    axes[1].set_ylabel('Thigh (rad)')
    axes[1].legend()
    axes[1].grid(True)

    # Plot calves
    axes[2].plot(times, trajectories[:, 8], label='FL')
    axes[2].plot(times, trajectories[:, 9], label='FR')
    axes[2].plot(times, trajectories[:, 10], label='BL')
    axes[2].plot(times, trajectories[:, 11], label='BR')
    axes[2].set_ylabel('Calf (rad)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle('CPG Trajectory (1 cycle)')
    plt.tight_layout()
    plt.savefig('cpg_trajectory.png')
    print("Saved cpg_trajectory.png")
