"""Verify simulated actuator behavior against ST-3215 servo datasheet specs.

Loads the Harold articulation in a minimal Isaac Lab scene and compares
simulated joint behavior against the FeeTech ST-3215-C018 servo specifications
from docs/ST-3215.pdf.

Usage:
    ./isaaclab.sh -p tests/test_servo_sim_accuracy.py

ST-3215 Key Specs (from datasheet):
    Stall torque:     30 kg.cm = 2.94 Nm @ 12V (±10%)
    No-load speed:    0.222 sec/60° = 270°/s = 4.71 rad/s @ 12V (±10%)
    Rated torque:     10 kg.cm = 0.98 Nm
    Stall current:    2.7 A
    Rated current:    900 mA
    Backlash:         ≤0.5° (0.0087 rad)
    Resolution:       0.088° (12-bit, 4096 positions)
    Gear ratio:       1/345
    Over-load prot:   >80% stall for 2s triggers shutdown
    Over-temp prot:   >70°C shuts down torque output
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

# Isaac Lab imports (must be run via isaaclab.sh)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ST-3215 servo sim verification")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext

# Harold robot config
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[0].parent))
from harold_isaac_lab.source.harold_isaac_lab.harold_isaac_lab.tasks.manager_based.harold_flat.harold import HAROLD_V4_CFG


# ── ST-3215 Datasheet Constants ──────────────────────────────────────────────

ST3215_STALL_TORQUE_NM = 2.94       # 30 kg.cm @ 12V
ST3215_STALL_TORQUE_TOL = 0.10      # ±10%
ST3215_NO_LOAD_SPEED_RADS = 4.71    # 270°/s @ 12V
ST3215_NO_LOAD_SPEED_TOL = 0.10     # ±10%
ST3215_RATED_TORQUE_NM = 0.98       # 10 kg.cm
ST3215_BACKLASH_RAD = 0.0087        # ≤0.5°
ST3215_RESOLUTION_RAD = 0.00154     # 0.088° (360°/4096)
ST3215_OVERLOAD_THRESHOLD = 0.80    # 80% of stall torque

# Sim config values (from harold.py)
SIM_EFFORT_LIMIT = 2.8              # effort_limit_sim (Nm)
SIM_VELOCITY_LIMIT = 4.29           # velocity_limit_sim (rad/s)
SIM_STIFFNESS = 40.0                # PD Kp
SIM_DAMPING = 0.5                   # PD Kd
SIM_DT = 0.002                      # physics timestep (500 Hz)


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_result(name: str, expected, actual, unit: str = "", tolerance: float = 0.10):
    """Print a comparison result with PASS/FAIL status."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        error_pct = abs(actual - expected) / abs(expected) * 100 if expected != 0 else 0
        status = "PASS" if error_pct <= tolerance * 100 else "FAIL"
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}")
        print(f"    Expected: {expected:.4f} {unit}")
        print(f"    Actual:   {actual:.4f} {unit}")
        print(f"    Error:    {error_pct:.1f}% (tolerance: ±{tolerance*100:.0f}%)")
        print(f"    Status:   {status}")
    else:
        print(f"  ℹ {name}")
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")
        print(f"    Status:   INFO")
    return status if isinstance(expected, (int, float)) else "INFO"


def create_scene() -> tuple[SimulationContext, Articulation]:
    """Create a minimal scene with Harold for testing."""
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    # Harold robot
    robot_cfg = HAROLD_V4_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    return sim, robot


def test_max_torque(sim: SimulationContext, robot: Articulation) -> str:
    """Test that sim enforces effort_limit_sim and compare to ST-3215 stall torque.

    Commands a joint far from its current position to saturate the PD controller.
    The resulting applied torque should be clamped at effort_limit_sim.
    """
    print_header("TEST 1: Maximum Torque")

    # Reset robot to default pose
    robot.reset()

    # Command a large position error on the FL thigh joint (index 4)
    # This should saturate the PD controller at effort_limit
    joint_idx = 4  # FL thigh
    default_pos = robot.data.default_joint_pos.clone()
    target_pos = default_pos.clone()
    target_pos[:, joint_idx] += 1.0  # 1 rad step = large error

    # Step physics for 10 steps to let torque build up
    max_torque = 0.0
    for _ in range(10):
        robot.set_joint_position_target(target_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        torque = robot.data.applied_torque[:, joint_idx].abs().max().item()
        max_torque = max(max_torque, torque)

    # Check against sim limit
    s1 = print_result(
        "Sim torque clamp (effort_limit_sim)",
        SIM_EFFORT_LIMIT, max_torque, "Nm", tolerance=0.05,
    )

    # Compare sim limit to hardware spec
    s2 = print_result(
        "Sim limit vs ST-3215 stall torque",
        ST3215_STALL_TORQUE_NM, SIM_EFFORT_LIMIT, "Nm", tolerance=ST3215_STALL_TORQUE_TOL,
    )

    print(f"\n  Note: Sim uses {SIM_EFFORT_LIMIT} Nm = {SIM_EFFORT_LIMIT/ST3215_STALL_TORQUE_NM*100:.0f}% of ST-3215 stall torque")
    print(f"  This is intentional — provides 5% safety margin for hardware protection.")

    return "FAIL" if "FAIL" in (s1, s2) else "PASS"


def test_max_velocity(sim: SimulationContext, robot: Articulation) -> str:
    """Test that sim enforces velocity_limit_sim and compare to ST-3215 no-load speed.

    Commands a large step input and measures peak joint velocity.
    """
    print_header("TEST 2: Maximum Velocity")

    robot.reset()

    joint_idx = 4  # FL thigh
    default_pos = robot.data.default_joint_pos.clone()
    target_pos = default_pos.clone()
    target_pos[:, joint_idx] += 0.8  # Large step to hit max speed

    max_vel = 0.0
    for _ in range(200):  # 200 steps @ 500Hz = 0.4s
        robot.set_joint_position_target(target_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        vel = robot.data.joint_vel[:, joint_idx].abs().max().item()
        max_vel = max(max_vel, vel)

    s1 = print_result(
        "Peak sim velocity vs velocity_limit_sim",
        SIM_VELOCITY_LIMIT, max_vel, "rad/s", tolerance=0.15,
    )

    s2 = print_result(
        "Sim velocity limit vs ST-3215 no-load speed",
        ST3215_NO_LOAD_SPEED_RADS, SIM_VELOCITY_LIMIT, "rad/s", tolerance=ST3215_NO_LOAD_SPEED_TOL,
    )

    print(f"\n  Note: Sim velocity limit {SIM_VELOCITY_LIMIT} rad/s = {SIM_VELOCITY_LIMIT/ST3215_NO_LOAD_SPEED_RADS*100:.0f}% of ST-3215 no-load speed")

    return "FAIL" if "FAIL" in (s1, s2) else "PASS"


def test_position_tracking(sim: SimulationContext, robot: Articulation) -> str:
    """Test position tracking accuracy with a sinusoidal trajectory.

    Commands a 0.5 Hz sinusoidal trajectory on the FL thigh joint and measures
    tracking error. Compares against expected PD response.
    """
    print_header("TEST 3: Position Tracking (Sinusoidal)")

    robot.reset()

    joint_idx = 4  # FL thigh
    default_pos = robot.data.default_joint_pos.clone()
    freq = 0.5  # Hz
    amplitude = 0.3  # rad
    duration = 4.0  # seconds (2 full cycles)
    n_steps = int(duration / SIM_DT)

    errors = []
    for step in range(n_steps):
        t = step * SIM_DT
        target_offset = amplitude * math.sin(2 * math.pi * freq * t)

        target_pos = default_pos.clone()
        target_pos[:, joint_idx] += target_offset

        robot.set_joint_position_target(target_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        actual_pos = robot.data.joint_pos[:, joint_idx].mean().item()
        target_val = default_pos[:, joint_idx].mean().item() + target_offset
        errors.append(abs(actual_pos - target_val))

    # Skip first 0.5s (transient)
    steady_errors = errors[int(0.5 / SIM_DT):]
    mean_error = sum(steady_errors) / len(steady_errors)
    max_error = max(steady_errors)
    rms_error = (sum(e**2 for e in steady_errors) / len(steady_errors)) ** 0.5

    print(f"  Trajectory: {amplitude} rad amplitude, {freq} Hz, {duration}s")
    print(f"  Mean tracking error: {mean_error:.4f} rad ({math.degrees(mean_error):.2f}°)")
    print(f"  Max tracking error:  {max_error:.4f} rad ({math.degrees(max_error):.2f}°)")
    print(f"  RMS tracking error:  {rms_error:.4f} rad ({math.degrees(rms_error):.2f}°)")

    # With Kp=40, Kd=0.5, and 500Hz physics, tracking should be reasonable
    # Allow up to 0.1 rad (5.7°) mean error as acceptable
    status = "PASS" if mean_error < 0.1 else "FAIL"
    print(f"  Status: {status} (mean error {'<' if status == 'PASS' else '>'} 0.1 rad threshold)")

    return status


def test_torque_speed_relationship(sim: SimulationContext, robot: Articulation) -> str:
    """Document the torque-speed relationship in sim vs real servo.

    Real servos have a linear torque-speed curve: max torque at zero speed,
    zero torque at max speed. The sim's PD controller may not match this.
    """
    print_header("TEST 4: Torque-Speed Relationship")

    robot.reset()
    joint_idx = 4

    print("  Testing torque at various speeds...")
    print(f"  {'Speed (rad/s)':>15} {'Torque (Nm)':>15} {'% Stall':>10} {'% Max Speed':>12}")
    print(f"  {'-'*15} {'-'*15} {'-'*10} {'-'*12}")

    # Real ST-3215 linear model: torque = stall_torque * (1 - speed/no_load_speed)
    results = []
    for target_speed_frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        robot.reset()

        # Apply constant velocity by repeatedly stepping with a moving target
        default_pos = robot.data.default_joint_pos.clone()
        target_speed = target_speed_frac * ST3215_NO_LOAD_SPEED_RADS

        torques = []
        speeds = []
        for step in range(500):
            t = step * SIM_DT
            target_pos = default_pos.clone()
            target_pos[:, joint_idx] += target_speed * t  # Ramp

            robot.set_joint_position_target(target_pos)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

            if step > 100:  # Skip transient
                torques.append(robot.data.applied_torque[:, joint_idx].abs().mean().item())
                speeds.append(robot.data.joint_vel[:, joint_idx].abs().mean().item())

        avg_torque = sum(torques) / len(torques) if torques else 0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        expected_torque = ST3215_STALL_TORQUE_NM * (1.0 - target_speed_frac)

        results.append((avg_speed, avg_torque, expected_torque))
        print(f"  {avg_speed:15.2f} {avg_torque:15.3f} {avg_torque/ST3215_STALL_TORQUE_NM*100:9.1f}% {avg_speed/ST3215_NO_LOAD_SPEED_RADS*100:11.1f}%")

    print(f"\n  Note: Real ST-3215 has a linear torque-speed curve.")
    print(f"  The sim PD controller does not model this — torque depends on")
    print(f"  position error and velocity error, not a motor electrical model.")
    print(f"  This is a known sim-to-real gap.")

    return "INFO"


def print_spec_summary():
    """Print a summary comparison table of all specs."""
    print_header("SPEC COMPARISON SUMMARY")

    specs = [
        ("Stall torque", f"{ST3215_STALL_TORQUE_NM} Nm", f"{SIM_EFFORT_LIMIT} Nm",
         f"{SIM_EFFORT_LIMIT/ST3215_STALL_TORQUE_NM*100:.0f}%", "Intentional 5% margin"),
        ("No-load speed", f"{ST3215_NO_LOAD_SPEED_RADS} rad/s", f"{SIM_VELOCITY_LIMIT} rad/s",
         f"{SIM_VELOCITY_LIMIT/ST3215_NO_LOAD_SPEED_RADS*100:.0f}%", "9% conservative"),
        ("Rated torque", f"{ST3215_RATED_TORQUE_NM} Nm", "Not modeled",
         "N/A", "Sim allows full stall continuously"),
        ("Backlash", f"{ST3215_BACKLASH_RAD} rad (0.5°)", "Not modeled",
         "N/A", "Gap — real servo has ≤0.5° backlash"),
        ("Resolution", f"{ST3215_RESOLUTION_RAD} rad (0.088°)", "Continuous",
         "N/A", "Negligible — sim is higher fidelity"),
        ("Over-load prot", ">80% stall for 2s", "Not modeled",
         "N/A", "Gap — real servo shuts down under sustained load"),
        ("Over-temp prot", ">70°C", "Not modeled",
         "N/A", "Gap — no thermal model in sim"),
        ("Torque-speed curve", "Linear (motor model)", "PD controller",
         "N/A", "Gap — sim PD ≠ real DC motor + gearbox"),
    ]

    print(f"  {'Parameter':<20} {'ST-3215 Spec':<20} {'Sim Value':<20} {'Match':<8} {'Notes'}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*8} {'-'*40}")
    for name, spec, sim, match, notes in specs:
        print(f"  {name:<20} {spec:<20} {sim:<20} {match:<8} {notes}")


def main():
    sim, robot = create_scene()

    results = {}
    results["max_torque"] = test_max_torque(sim, robot)
    results["max_velocity"] = test_max_velocity(sim, robot)
    results["position_tracking"] = test_position_tracking(sim, robot)
    results["torque_speed"] = test_torque_speed_relationship(sim, robot)

    print_spec_summary()

    print_header("RESULTS")
    for name, status in results.items():
        symbol = {"PASS": "✓", "FAIL": "✗", "INFO": "ℹ"}[status]
        print(f"  {symbol} {name}: {status}")

    n_fail = sum(1 for s in results.values() if s == "FAIL")
    if n_fail > 0:
        print(f"\n  {n_fail} test(s) FAILED")
    else:
        print(f"\n  All tests passed or informational")

    simulation_app.close()


if __name__ == "__main__":
    main()
