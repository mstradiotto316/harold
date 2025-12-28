"""CPG (Central Pattern Generator) trajectory generator.

Generates the base walking trajectory for Harold robot.
Ported from simulation (harold_isaac_lab_env.py) for exact sim-to-real match.

Gait Pattern:
    - Diagonal trot: FL+BR alternate with FR+BL
    - 0.5 Hz frequency (2 second cycle)
    - 60% stance, 40% swing (duty cycle)
"""
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class CPGConfig:
    """CPG parameters - must match simulation exactly."""
    frequency_hz: float = 0.5
    duty_cycle: float = 0.6
    swing_thigh: float = 0.40
    stance_thigh: float = 0.90
    stance_calf: float = -0.90
    swing_calf: float = -1.40
    shoulder_amplitude: float = 0.05
    residual_scale: float = 0.05

    @classmethod
    def from_yaml(cls, path: Path) -> "CPGConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        traj = data.get("trajectory", {})
        return cls(
            frequency_hz=data.get("frequency_hz", 0.5),
            duty_cycle=data.get("duty_cycle", 0.6),
            swing_thigh=traj.get("swing_thigh", 0.40),
            stance_thigh=traj.get("stance_thigh", 0.90),
            stance_calf=traj.get("stance_calf", -0.90),
            swing_calf=traj.get("swing_calf", -1.40),
            shoulder_amplitude=traj.get("shoulder_amplitude", 0.05),
            residual_scale=data.get("residual_scale", 0.05),
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

        # Phase for each diagonal pair
        phase_fl_br = self._phase
        phase_fr_bl = (self._phase + 0.5) % 1.0

        # Compute leg trajectories
        thigh_fl, calf_fl = self._compute_leg_trajectory(phase_fl_br)
        thigh_br, calf_br = self._compute_leg_trajectory(phase_fl_br)
        thigh_fr, calf_fr = self._compute_leg_trajectory(phase_fr_bl)
        thigh_bl, calf_bl = self._compute_leg_trajectory(phase_fr_bl)

        # Shoulder oscillation for balance
        amp = self.cfg.shoulder_amplitude
        shoulder_fl = amp * math.sin(2 * math.pi * phase_fl_br)
        shoulder_br = -amp * math.sin(2 * math.pi * phase_fl_br)
        shoulder_fr = -amp * math.sin(2 * math.pi * phase_fr_bl)
        shoulder_bl = amp * math.sin(2 * math.pi * phase_fr_bl)

        # Assemble joint targets [shoulders(4), thighs(4), calves(4)]
        targets = np.zeros(12, dtype=np.float32)

        # Shoulders (indices 0-3): FL, FR, BL, BR
        targets[0] = shoulder_fl
        targets[1] = shoulder_fr
        targets[2] = shoulder_bl
        targets[3] = shoulder_br

        # Thighs (indices 4-7): FL, FR, BL, BR
        targets[4] = thigh_fl
        targets[5] = thigh_fr
        targets[6] = thigh_bl
        targets[7] = thigh_br

        # Calves (indices 8-11): FL, FR, BL, BR
        targets[8] = calf_fl
        targets[9] = calf_fr
        targets[10] = calf_bl
        targets[11] = calf_br

        return targets

    def _compute_leg_trajectory(self, phase: float) -> tuple[float, float]:
        """Compute smooth leg trajectory based on gait phase.

        Args:
            phase: Gait phase for this leg [0, 1)

        Returns:
            Tuple of (thigh_angle, calf_angle) in radians

        Trajectory Design:
            - Stance (0 to duty_cycle): Leg on ground, gradually extends backward
            - Swing (duty_cycle to 1): Leg in air, flexes forward then extends down
            - Cosine interpolation for smooth transitions
        """
        duty = self.cfg.duty_cycle
        in_stance = phase < duty

        if in_stance:
            # Stance phase progress (0 to 1 within stance)
            stance_progress = phase / duty

            # Thigh: starts at swing_thigh (just landed), extends to stance_thigh (pushes back)
            thigh = self.cfg.swing_thigh + (self.cfg.stance_thigh - self.cfg.swing_thigh) * (
                (1 - math.cos(stance_progress * math.pi)) / 2
            )

            # Calf: constant during stance (ground contact)
            calf = self.cfg.stance_calf
        else:
            # Swing phase progress (0 to 1 within swing)
            swing_progress = (phase - duty) / (1.0 - duty)

            # Thigh: starts at stance_thigh (just lifted), flexes to swing_thigh (recovery)
            thigh = self.cfg.stance_thigh + (self.cfg.swing_thigh - self.cfg.stance_thigh) * (
                (1 - math.cos(swing_progress * math.pi)) / 2
            )

            # Calf: parabolic lift profile (up then down)
            swing_lift = math.sin(swing_progress * math.pi)  # 0 -> 1 -> 0
            calf = self.cfg.stance_calf + (self.cfg.swing_calf - self.cfg.stance_calf) * swing_lift

        return thigh, calf

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
