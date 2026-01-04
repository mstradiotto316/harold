#!/usr/bin/env python3
"""Log scripted/CPG gait playback from Isaac Lab for sim-to-real comparison."""
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

import numpy as np

JOINT_NAMES = [
    "fl_sh", "fr_sh", "bl_sh", "br_sh",
    "fl_th", "fr_th", "bl_th", "br_th",
    "fl_ca", "fr_ca", "bl_ca", "br_ca",
]


@dataclass
class ImuModel:
    accel_noise_std: float = 0.0
    gyro_noise_std: float = 0.0
    vel_decay: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._lin_vel = np.zeros(3, dtype=np.float32)
        self._prev_lin_vel = None
        self._last_gyro = np.zeros(3, dtype=np.float32)
        self._last_proj_g = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def reset(self) -> None:
        self._lin_vel = np.zeros(3, dtype=np.float32)
        self._prev_lin_vel = None
        self._last_gyro = np.zeros(3, dtype=np.float32)
        self._last_proj_g = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def update(self, lin_vel_b: np.ndarray, ang_vel_b: np.ndarray, proj_g_hw: np.ndarray, dt: float) -> None:
        if self._prev_lin_vel is None:
            lin_acc_b = np.zeros(3, dtype=np.float32)
        else:
            lin_acc_b = (lin_vel_b - self._prev_lin_vel) / max(dt, 1e-6)
        self._prev_lin_vel = lin_vel_b

        accel_g = proj_g_hw + lin_acc_b / 9.81
        if self.accel_noise_std > 0.0:
            accel_g = accel_g + self._rng.normal(0.0, self.accel_noise_std, size=3)

        accel_norm = float(np.linalg.norm(accel_g))
        if accel_norm > 0.1:
            self._last_proj_g = (accel_g / accel_norm).astype(np.float32)
        else:
            self._last_proj_g = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        gyro = ang_vel_b
        if self.gyro_noise_std > 0.0:
            gyro = gyro + self._rng.normal(0.0, self.gyro_noise_std, size=3)
        self._last_gyro = gyro.astype(np.float32)

        dt = min(dt, 0.1)
        accel_linear = accel_g - np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._lin_vel = self.vel_decay * self._lin_vel + accel_linear * 9.81 * dt

    @property
    def lin_vel(self) -> np.ndarray:
        return self._lin_vel

    @property
    def gyro(self) -> np.ndarray:
        return self._last_gyro

    @property
    def proj_g(self) -> np.ndarray:
        return self._last_proj_g


def _set_if_attr(obj, name, value) -> None:
    if value is None:
        return
    if hasattr(obj, name):
        setattr(obj, name, value)


def _apply_cpg_yaml(env_cfg, cpg_yaml: Path) -> dict:
    import yaml

    if not cpg_yaml.exists():
        return {}

    with cpg_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    traj = data.get("trajectory", {}) or {}
    freq = data.get("frequency_hz")
    duty = data.get("duty_cycle")

    for cfg in (env_cfg.cpg, env_cfg.scripted_gait):
        _set_if_attr(cfg, "swing_thigh", traj.get("swing_thigh"))
        _set_if_attr(cfg, "stance_thigh", traj.get("stance_thigh"))
        _set_if_attr(cfg, "stance_calf", traj.get("stance_calf"))
        _set_if_attr(cfg, "swing_calf", traj.get("swing_calf"))
        _set_if_attr(cfg, "shoulder_amplitude", traj.get("shoulder_amplitude"))
        _set_if_attr(cfg, "stride_scale", traj.get("stride_scale"))
        _set_if_attr(cfg, "calf_lift_scale", traj.get("calf_lift_scale"))
        _set_if_attr(cfg, "stride_scale_front", traj.get("stride_scale_front"))
        _set_if_attr(cfg, "stride_scale_back", traj.get("stride_scale_back"))
        _set_if_attr(cfg, "calf_lift_scale_front", traj.get("calf_lift_scale_front"))
        _set_if_attr(cfg, "calf_lift_scale_back", traj.get("calf_lift_scale_back"))
        _set_if_attr(cfg, "thigh_offset_front", traj.get("thigh_offset_front"))
        _set_if_attr(cfg, "thigh_offset_back", traj.get("thigh_offset_back"))
        _set_if_attr(cfg, "duty_cycle", duty)

    _set_if_attr(env_cfg.cpg, "base_frequency", freq)
    _set_if_attr(env_cfg.scripted_gait, "frequency", freq)
    _set_if_attr(env_cfg.cpg, "residual_scale", data.get("residual_scale"))

    control = data.get("control", {}) or {}
    _set_if_attr(env_cfg, "action_filter_beta", control.get("action_filter_beta"))

    return data


def _configure_commands(env_cfg, vx: float, vy: float, yaw: float) -> None:
    cfg = env_cfg.commands
    cfg.variable_commands = False
    cfg.dynamic_commands = False
    cfg.zero_velocity_prob = 0.0
    cfg.vx_min = vx
    cfg.vx_max = vx
    cfg.vy_min = vy
    cfg.vy_max = vy
    cfg.yaw_min = yaw
    cfg.yaw_max = yaw


def main() -> None:
    parser = argparse.ArgumentParser(description="Log CPG/scripted gait playback from Isaac Lab.")
    parser.add_argument("--task", default="Template-Harold-Direct-flat-terrain-v0", help="Gym task ID")
    parser.add_argument("--mode", choices=("cpg", "scripted"), default="cpg", help="Playback mode")
    parser.add_argument("--duration", type=float, default=20.0, help="Playback duration in seconds")
    parser.add_argument("--log-rate-hz", type=float, default=5.0, help="Logging rate (Hz)")
    parser.add_argument("--cmd-vx", type=float, default=0.3, help="Commanded vx (m/s)")
    parser.add_argument("--cmd-vy", type=float, default=0.0, help="Commanded vy (m/s)")
    parser.add_argument("--cmd-yaw", type=float, default=0.0, help="Commanded yaw rate (rad/s)")
    parser.add_argument("--cpg-yaml", type=Path, default=Path("deployment/config/cpg.yaml"), help="CPG config YAML")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--enable-domain-rand", action="store_true", help="Enable domain randomization")
    parser.add_argument("--imu-mode", choices=("raw", "hw"), default="raw", help="IMU logging mode")
    parser.add_argument("--imu-vel-decay", type=float, default=0.95, help="IMU velocity decay (hw mode)")
    parser.add_argument("--imu-accel-noise", type=float, default=0.0, help="IMU accel noise std (g, hw mode)")
    parser.add_argument("--imu-gyro-noise", type=float, default=0.0, help="IMU gyro noise std (rad/s, hw mode)")
    parser.add_argument("--imu-seed", type=int, default=0, help="IMU noise RNG seed")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab_tasks  # noqa: F401
    import harold_isaac_lab.tasks  # noqa: F401

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    env_cfg.domain_randomization.enable_randomization = args_cli.enable_domain_rand

    if args_cli.mode == "cpg":
        env_cfg.cpg.enabled = True
        env_cfg.scripted_gait.enabled = False
        env_cfg.observation_space = 50
    else:
        env_cfg.cpg.enabled = False
        env_cfg.scripted_gait.enabled = True
        env_cfg.observation_space = 48

    _configure_commands(env_cfg, args_cli.cmd_vx, args_cli.cmd_vy, args_cli.cmd_yaw)
    cpg_yaml = _apply_cpg_yaml(env_cfg, args_cli.cpg_yaml)

    env = gym.make(args_cli.task, cfg=env_cfg)

    if env.unwrapped.num_envs < 1:
        raise RuntimeError("Environment did not initialize any environments.")

    if env.unwrapped.num_envs != args_cli.num_envs:
        print(f"[WARN] Requested num_envs={args_cli.num_envs}, got {env.unwrapped.num_envs}")

    output_path = args_cli.output
    if output_path is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = Path("deployment/validation/sim_logs") / f"sim_{args_cli.mode}_{stamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = output_path.with_suffix(".json")

    step_dt = getattr(env.unwrapped, "step_dt", None)
    if step_dt is None:
        step_dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    log_stride = max(1, int(round((1.0 / args_cli.log_rate_hz) / step_dt)))
    total_steps = int(args_cli.duration / step_dt)

    header = [
        "sim_time",
        "step",
        "mode",
        "cpg_phase",
        "cmd_vx",
        "cmd_vy",
        "cmd_yaw",
    ]
    header += [f"pos_{name}" for name in JOINT_NAMES]
    header += [f"cmd_pos_{name}" for name in JOINT_NAMES]
    header += [f"sim_torque_{name}" for name in JOINT_NAMES]
    header += [
        "imu_valid",
        "imu_lin_vel_x", "imu_lin_vel_y", "imu_lin_vel_z",
        "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
        "imu_proj_g_x", "imu_proj_g_y", "imu_proj_g_z",
    ]

    env.reset()

    action_dim = env.action_space.shape[0]
    actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)

    imu_model = ImuModel(
        accel_noise_std=args_cli.imu_accel_noise,
        gyro_noise_std=args_cli.imu_gyro_noise,
        vel_decay=args_cli.imu_vel_decay,
        seed=args_cli.imu_seed,
    )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with torch.inference_mode():
            for step in range(total_steps):
                env.step(actions)

                robot = env.unwrapped._robot
                lin_vel_b = robot.data.root_lin_vel_b[0].detach().cpu().numpy()
                ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
                proj_g = robot.data.projected_gravity_b[0].detach().cpu().numpy()
                proj_g_hw = np.array([-proj_g[0], -proj_g[1], -proj_g[2]], dtype=np.float32)

                if args_cli.imu_mode == "hw":
                    imu_model.update(lin_vel_b, ang_vel_b, proj_g_hw, step_dt)

                if step % log_stride != 0:
                    continue

                joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
                cmd_pos = env.unwrapped._processed_actions[0].detach().cpu().tolist()
                torque = robot.data.applied_torque[0].detach().cpu().tolist()

                cmd = env.unwrapped._commands[0].detach().cpu().tolist()
                if args_cli.mode == "scripted":
                    phase_value = (env.unwrapped._time[0].item() * env_cfg.scripted_gait.frequency) % 1.0
                else:
                    phase_value = env.unwrapped._cpg_phase[0].item()

                if args_cli.imu_mode == "hw":
                    imu_lin_vel = imu_model.lin_vel.tolist()
                    imu_gyro = imu_model.gyro.tolist()
                    imu_proj_g = imu_model.proj_g.tolist()
                else:
                    imu_lin_vel = lin_vel_b.tolist()
                    imu_gyro = ang_vel_b.tolist()
                    imu_proj_g = proj_g_hw.tolist()

                row = [
                    float(env.unwrapped._time[0].item()),
                    int(step),
                    "CPG_ONLY" if args_cli.mode == "cpg" else "SCRIPTED",
                    float(phase_value),
                    float(cmd[0]),
                    float(cmd[1]),
                    float(cmd[2]),
                ]
                row += joint_pos
                row += cmd_pos
                row += torque
                row += [1] + imu_lin_vel + imu_gyro + imu_proj_g
                writer.writerow(row)

    actuator_cfg = env_cfg.robot.actuators["all_joints"]
    metadata = {
        "mode": args_cli.mode,
        "duration_s": args_cli.duration,
        "log_rate_hz": args_cli.log_rate_hz,
        "step_dt": step_dt,
        "num_envs": env.unwrapped.num_envs,
        "command": {"vx": args_cli.cmd_vx, "vy": args_cli.cmd_vy, "yaw": args_cli.cmd_yaw},
        "cpg_yaml": str(args_cli.cpg_yaml),
        "cpg_yaml_data": cpg_yaml,
        "domain_randomization": {
            "enabled": bool(args_cli.enable_domain_rand),
        },
        "actuators": {
            "stiffness": float(actuator_cfg.stiffness),
            "damping": float(actuator_cfg.damping),
            "effort_limit_sim": float(actuator_cfg.effort_limit_sim),
        },
        "imu_model": {
            "mode": args_cli.imu_mode,
            "vel_decay": float(args_cli.imu_vel_decay),
            "accel_noise_std_g": float(args_cli.imu_accel_noise),
            "gyro_noise_std": float(args_cli.imu_gyro_noise),
            "seed": int(args_cli.imu_seed),
        },
        "note": "sim_torque_* columns are applied joint torques (N*m). imu_proj_g_* uses Z-up convention to match hardware logs.",
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    env.close()
    simulation_app.close()

    print(f"[OK] Wrote {output_path}")
    print(f"[OK] Wrote {meta_path}")


if __name__ == "__main__":
    main()
