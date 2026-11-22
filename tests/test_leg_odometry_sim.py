"""Exercise the deployed velocity estimator on a deterministic policy rollout.

Run with the desktop Isaac Lab Python and --headless --checkpoint <best_agent.pt>.
The policy sees its training observations by default. Use --odometry-feedback to
feed the estimate back into the policy. Servo load is approximated from simulated
actuator torque. The threshold for forward velocity is MAE <0.15 m/s and r >0.7.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

COMMANDS = [
    (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.0),
    (-0.2, 0.0, 0.0), (0.0, 0.15, 0.0), (0.0, -0.15, 0.0),
    (0.0, 0.0, 0.5), (0.0, 0.0, -0.5),
]
TASK = "Harold-Velocity-Flat-v0"
ROOT = Path(__file__).resolve().parents[1]


def evaluate(args):
    import gymnasium as gym
    import torch
    import yaml
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils.runner.torch import Runner
    import harold_isaac_lab.tasks  # noqa: F401

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "deployment"))
    from inference.leg_odometry import LEG_NAMES, LegOdometry

    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    expected_names = [f"{leg}_{joint}_joint" for joint in ("shoulder", "thigh", "calf") for leg in LEG_NAMES]
    if args.firmware_clamps:
        from deployment.validation.firmware_limits import manager_joint_target_limits

        lower, upper = manager_joint_target_limits()
        cfg.actions.joint_pos.clip = {
            name: (float(lo), float(hi)) for name, lo, hi in zip(expected_names, lower, upper)
        }
    env = gym.make(TASK, cfg=cfg)
    try:
        manager = env.unwrapped
        robot = manager.scene["robot"]
        command = manager.command_manager.get_command("base_velocity")
        print("Policy articulation order:", robot.joint_names, flush=True)
        if robot.joint_names != expected_names:
            raise RuntimeError(f"Policy and deployment articulation orders differ: expected {expected_names}")
        checkpoint = args.checkpoint.resolve()
        with (checkpoint.parents[1] / "params/agent.yaml").open() as stream:
            agent_cfg = yaml.safe_load(stream)
        agent_cfg["agent"]["experiment"].update(write_interval=0, checkpoint_interval=0)
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
        agent = Runner(wrapped, agent_cfg).agent
        agent.load(str(checkpoint))
        agent.set_running_mode("eval")
        onnx = None
        if args.onnx:
            import onnxruntime as ort

            onnx = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        odoms = [LegOdometry() for _ in range(args.num_envs)]
        estimates = np.zeros((args.num_envs, 3), dtype=np.float32)
        obs, _ = wrapped.reset()
        q_prev = robot.data.joint_pos.cpu().numpy().copy()
        dq = np.zeros_like(q_prev)
        age = np.zeros(args.num_envs, dtype=int)
        measurements, predictions = [], []
        segment_samples = [[] for _ in COMMANDS]
        falls = np.zeros(len(COMMANDS), dtype=int)
        saturated = np.zeros(12, dtype=int)
        with torch.inference_mode():
            for step in range(args.num_steps):
                phase = step * len(COMMANDS) // args.num_steps
                cmd = torch.tensor(COMMANDS[phase], device=manager.device)
                command[:] = cmd
                obs[:, 9:12] = cmd
                if args.odometry_feedback:
                    obs[:, :3] = torch.as_tensor(estimates, device=manager.device)
                if onnx is None:
                    sampled, _, extras = agent.act(obs, timestep=step, timesteps=args.num_steps)
                    action = extras.get("mean_actions", sampled)
                else:
                    output = onnx.run(["mean"], {onnx.get_inputs()[0].name: obs.cpu().numpy()})[0]
                    action = torch.as_tensor(output, device=manager.device)
                obs, _, terminated, truncated, _ = wrapped.step(action)
                if args.firmware_clamps:
                    # Targets equal an endpoint when the firmware model clips a joint.
                    target = manager.action_manager.get_term("joint_pos").processed_actions.cpu().numpy()
                    saturated += np.sum((target <= lower) | (target >= upper), axis=0)
                done = (terminated | truncated).cpu().numpy().reshape(-1)
                falls[phase] += int(terminated.sum())
                q = robot.data.joint_pos.cpu().numpy()
                dq = 0.5 * ((q - q_prev) / manager.step_dt) + 0.5 * dq
                q_prev = q.copy()
                loads = (1000 * robot.data.applied_torque / robot.data.joint_effort_limits).cpu().numpy()
                angular = robot.data.root_ang_vel_b.cpu().numpy()
                truth = robot.data.root_lin_vel_b.cpu().numpy()
                age += 1
                for i, odom in enumerate(odoms):
                    if done[i]:
                        odom.reset()
                        estimates[i] = dq[i] = 0
                        age[i] = 0
                        continue
                    estimates[i] = odom.update(q[i], dq[i], loads[i], angular_velocity=angular[i])
                    if age[i] > 10:
                        measurements.append(truth[i].copy())
                        predictions.append(estimates[i].copy())
                        # Report settled tracking after 0.5 simulated seconds.
                        phase_start = (phase * args.num_steps + len(COMMANDS) - 1) // len(COMMANDS)
                        if step - phase_start >= 10:
                            segment_samples[phase].append([*truth[i, :2], angular[i, 2]])
        print(f"Policy {args.onnx or checkpoint}, seed {args.seed}, odometry feedback {args.odometry_feedback}")
        if args.firmware_clamps:
            print("Target saturation (%) in articulation order:",
                  (100 * saturated / (args.num_steps * args.num_envs)).round(1))
        for cmd, samples, fall_count in zip(COMMANDS, segment_samples, falls):
            mean = np.mean(samples, axis=0).round(3) if samples else "unavailable"
            print(f"command {cmd} measured [vx, vy, yaw rate] {mean}  n={len(samples)} falls={fall_count}")
        if not measurements:
            raise RuntimeError("No post-warmup velocity samples")
        truth = np.asarray(measurements)
        est = np.asarray(predictions)
        mae = np.abs(truth - est).mean(axis=0)
        corr = np.corrcoef(truth[:, 0], est[:, 0])[0, 1]
        print(f"Samples {len(est)}, XYZ MAE (m/s) {mae.round(4)}, VX correlation {corr:.4f}", flush=True)
        return 0 if mae[0] < 0.15 and corr > 0.7 else 1
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=540)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--odometry-feedback", action="store_true")
    parser.add_argument("--firmware-clamps", action="store_true", help="Apply the firmware target envelope in simulation")
    parser.add_argument("--onnx", type=Path, help="Evaluate an exported ONNX mean action on the same observations")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_steps < 12 * len(COMMANDS) or args.num_envs < 1:
        parser.error("Require at least 12 steps per command and at least one environment")
    app = AppLauncher(args).app
    try:
        return evaluate(args)
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
