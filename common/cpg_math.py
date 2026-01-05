"""Shared CPG math kernel for sim and hardware."""
from dataclasses import dataclass
from typing import Callable
import math


@dataclass(frozen=True)
class CPGOps:
    """Backend operations for the CPG kernel."""
    sin: Callable
    cos: Callable
    where: Callable
    full_like: Callable
    zeros_like: Callable
    zeros: Callable
    as_like: Callable


def numpy_ops(np_module) -> CPGOps:
    """Create numpy-backed ops without importing numpy here."""
    def as_like(x, value):
        dtype = x.dtype if hasattr(x, "dtype") else np_module.float32
        return np_module.array(value, dtype=dtype)

    def zeros(shape, like):
        dtype = like.dtype if hasattr(like, "dtype") else np_module.float32
        return np_module.zeros(shape, dtype=dtype)

    return CPGOps(
        sin=np_module.sin,
        cos=np_module.cos,
        where=np_module.where,
        full_like=np_module.full_like,
        zeros_like=np_module.zeros_like,
        zeros=zeros,
        as_like=as_like,
    )


def torch_ops(torch_module) -> CPGOps:
    """Create torch-backed ops without importing torch here."""
    def as_like(x, value):
        return x.new_tensor(value)

    def zeros(shape, like):
        return torch_module.zeros(shape, device=like.device, dtype=like.dtype)

    return CPGOps(
        sin=torch_module.sin,
        cos=torch_module.cos,
        where=torch_module.where,
        full_like=torch_module.full_like,
        zeros_like=torch_module.zeros_like,
        zeros=zeros,
        as_like=as_like,
    )


def _smoothstep(x):
    """Smoothstep interpolation (0..1)."""
    return x * x * (3.0 - 2.0 * x)


def compute_leg_trajectory(phase, cfg, ops: CPGOps):
    """Compute thigh/calf trajectory for a given gait phase.

    Args:
        phase: Scalar or tensor/array of phases in [0, 1)
        cfg: Object with duty_cycle, swing_thigh, stance_thigh, stance_calf, swing_calf
        ops: Backend operations (numpy/torch)
    """
    duty = min(max(float(cfg.duty_cycle), 0.05), 0.95)
    duty_t = ops.as_like(phase, duty)
    zero = ops.zeros_like(phase)

    stance_mask = phase < duty_t
    stance_phase = ops.where(stance_mask, phase / duty_t, zero)
    swing_phase = ops.where(stance_mask, zero, (phase - duty_t) / (1.0 - duty_t))

    stance_blend = _smoothstep(stance_phase)
    swing_blend = _smoothstep(swing_phase)

    thigh_stance = cfg.swing_thigh + (cfg.stance_thigh - cfg.swing_thigh) * stance_blend
    thigh_swing = cfg.stance_thigh + (cfg.swing_thigh - cfg.stance_thigh) * swing_blend
    thigh = ops.where(stance_mask, thigh_stance, thigh_swing)

    calf_stance = ops.full_like(phase, cfg.stance_calf)
    two_pi = ops.as_like(phase, 2.0 * math.pi)
    lift = 0.5 - 0.5 * ops.cos(two_pi * swing_phase)
    calf_swing = cfg.stance_calf + (cfg.swing_calf - cfg.stance_calf) * lift
    calf = ops.where(stance_mask, calf_stance, calf_swing)

    return thigh, calf


def compute_cpg_targets(phase, cfg, ops: CPGOps):
    """Compute 12D joint targets for the diagonal-trot gait.

    Args:
        phase: Scalar or tensor/array of phases in [0, 1)
        cfg: Object with trajectory parameters (see CPGConfig/ScriptedGaitCfg)
        ops: Backend operations (numpy/torch)
    """
    phase_fl_br = phase
    phase_fr_bl = (phase + 0.5) % 1.0

    thigh_fl, calf_fl = compute_leg_trajectory(phase_fl_br, cfg, ops)
    thigh_br, calf_br = compute_leg_trajectory(phase_fl_br, cfg, ops)
    thigh_fr, calf_fr = compute_leg_trajectory(phase_fr_bl, cfg, ops)
    thigh_bl, calf_bl = compute_leg_trajectory(phase_fr_bl, cfg, ops)

    thigh_fl = thigh_fl + cfg.thigh_offset_front
    thigh_fr = thigh_fr + cfg.thigh_offset_front
    thigh_bl = thigh_bl + cfg.thigh_offset_back
    thigh_br = thigh_br + cfg.thigh_offset_back

    two_pi = ops.as_like(phase, 2.0 * math.pi)
    amp = cfg.shoulder_amplitude
    shoulder_fl = amp * ops.sin(two_pi * phase_fl_br)
    shoulder_br = amp * ops.sin(two_pi * phase_fl_br)
    shoulder_fr = amp * ops.sin(two_pi * phase_fr_bl)
    shoulder_bl = amp * ops.sin(two_pi * phase_fr_bl)

    phase_shape = getattr(phase, "shape", ())
    if len(phase_shape) == 0:
        targets = ops.zeros((12,), like=phase)
        targets[0] = shoulder_fl
        targets[1] = shoulder_fr
        targets[2] = shoulder_bl
        targets[3] = shoulder_br
        targets[4] = thigh_fl
        targets[5] = thigh_fr
        targets[6] = thigh_bl
        targets[7] = thigh_br
        targets[8] = calf_fl
        targets[9] = calf_fr
        targets[10] = calf_bl
        targets[11] = calf_br
        return targets

    targets = ops.zeros((phase_shape[0], 12), like=phase)
    targets[:, 0] = shoulder_fl
    targets[:, 1] = shoulder_fr
    targets[:, 2] = shoulder_bl
    targets[:, 3] = shoulder_br
    targets[:, 4] = thigh_fl
    targets[:, 5] = thigh_fr
    targets[:, 6] = thigh_bl
    targets[:, 7] = thigh_br
    targets[:, 8] = calf_fl
    targets[:, 9] = calf_fr
    targets[:, 10] = calf_bl
    targets[:, 11] = calf_br
    return targets
