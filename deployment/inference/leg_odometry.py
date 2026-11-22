"""Leg odometry for Harold — velocity estimation from joint encoders.

Computes body velocity by tracking foot positions via forward kinematics.
During stance (foot on ground), the foot is stationary in the world frame,
so body velocity is the negative sum of foot motion relative to the body and
the rotational velocity of the foot about the body origin.

This replaces the previous velocity-blind approach (zeros at obs[0:3])
and provides the same signal the sim policy sees (ground-truth velocity + noise).
"""

from __future__ import annotations

import numpy as np


# --- Harold kinematic constants (from USD/URDF, Isaac Lab coordinate frame) ---
# Coordinate frame: X = forward, Y = left, Z = up

# Body center → shoulder joint offsets [x, y, z] (meters)
# From harold_8_joints.csv (USD localPos0 for shoulder joints)
HIP_OFFSETS = {
    "fl": np.array([0.1855, 0.0479, 0.0177]),
    "fr": np.array([0.1855, -0.0479, 0.0177]),
    "bl": np.array([-0.2308, 0.0479, 0.0177]),
    "br": np.array([-0.2308, -0.0479, 0.0177]),
}

# Shoulder → thigh joint offset in shoulder frame (before shoulder rotation)
# From USD: fl_thigh localPos0 = (0.023, 0.0161, 0.0114)
# Small offset — shoulder is mostly a roll joint with compact bracket
SHOULDER_TO_THIGH = {
    "fl": np.array([0.023, 0.0161, 0.0114]),
    "fr": np.array([0.023, -0.0161, 0.0114]),
    "bl": np.array([0.023, 0.0161, 0.0114]),
    "br": np.array([0.023, -0.0161, 0.0114]),
}

# Thigh length: distance from thigh joint to calf joint (meters)
# From USD: fl_calf localPos0 Z = -0.1031
THIGH_LENGTH = 0.1031

# Calf length: distance from calf joint to foot contact point (meters)
# Computed from standing height geometry (init_state.pos z=0.20m, default pose)
CALF_LENGTH = 0.1294

# Joint ordering: [FL, FR, BL, BR] for each category
# Indices into the 12D joint position array
SHOULDER_IDX = [0, 1, 2, 3]  # fl_sh, fr_sh, bl_sh, br_sh
THIGH_IDX = [4, 5, 6, 7]     # fl_th, fr_th, bl_th, br_th
CALF_IDX = [8, 9, 10, 11]    # fl_ca, fr_ca, bl_ca, br_ca

LEG_NAMES = ["fl", "fr", "bl", "br"]

# Stance detection: servo load threshold (0-1000 scale from ST3215)
# Legs bearing weight typically have load > 150
DEFAULT_LOAD_THRESHOLD = 150

# Finite difference step for numerical Jacobian
_FK_EPS = 1e-4


def _rot_x(angle: float) -> np.ndarray:
    """Rotation matrix around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def _rot_y(angle: float) -> np.ndarray:
    """Rotation matrix around Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def foot_position(
    leg: str,
    shoulder_angle: float,
    thigh_angle: float,
    calf_angle: float,
) -> np.ndarray:
    """Compute foot position in body frame for one leg.

    Args:
        leg: Leg name ("fl", "fr", "bl", "br")
        shoulder_angle: Shoulder joint angle (rad, RL convention)
        thigh_angle: Thigh joint angle (rad, RL convention)
        calf_angle: Calf joint angle (rad, RL convention)

    Returns:
        3D foot position in body frame [x, y, z] (meters)
    """
    # The USD uses right-handed +Y pitch axes. At neutral, a positive
    # thigh rotation moves a downward-pointing link toward body -X.
    side = 1 if leg.endswith("l") else -1
    calf_origin = np.array([-0.0026, -side * 0.0008, -THIGH_LENGTH])
    # Approximate the contact point at the middle of the distal foot pad.
    foot_tip = np.array([0.00075, side * 0.0228, -CALF_LENGTH])
    thigh_to_foot = _rot_y(thigh_angle) @ (calf_origin + _rot_y(calf_angle) @ foot_tip)
    return HIP_OFFSETS[leg] + _rot_x(shoulder_angle) @ (SHOULDER_TO_THIGH[leg] + thigh_to_foot)


def foot_jacobian(
    leg: str,
    shoulder_angle: float,
    thigh_angle: float,
    calf_angle: float,
) -> np.ndarray:
    """Compute 3x3 Jacobian of foot position w.r.t. joint angles (numerical).

    J[i, j] = d(foot_pos[i]) / d(joint_angle[j])
    Columns: [shoulder, thigh, calf]

    Args:
        leg: Leg name
        shoulder_angle, thigh_angle, calf_angle: Joint angles (rad)

    Returns:
        3x3 Jacobian matrix
    """
    angles = np.array([shoulder_angle, thigh_angle, calf_angle], dtype=float)
    J = np.zeros((3, 3))

    p0 = foot_position(leg, *angles)
    for j in range(3):
        angles_plus = angles.copy()
        angles_plus[j] += _FK_EPS
        p_plus = foot_position(leg, *angles_plus)
        J[:, j] = (p_plus - p0) / _FK_EPS

    return J


class LegOdometry:
    """Estimates body velocity from joint encoder readings via leg kinematics.

    During stance phase, the foot is stationary on the ground.
    Gyro compensation removes the contribution of body rotation at the foot.
    Averages across all detected stance legs for robustness.

    Usage:
        odom = LegOdometry()
        vel = odom.update(joint_pos_rl, joint_vel_rl, servo_loads)
    """

    def __init__(
        self,
        load_threshold: float = DEFAULT_LOAD_THRESHOLD,
        filter_alpha: float = 0.4,
        min_stance_legs: int = 1,
    ):
        """
        Args:
            load_threshold: Servo load above which a leg is considered in stance.
            filter_alpha: Low-pass filter coefficient for velocity output.
                          0 retains the previous estimate and 1 uses the raw measurement.
            min_stance_legs: Minimum legs in stance for a valid estimate.
                            If fewer, returns the previous filtered estimate.
        """
        self.load_threshold = load_threshold
        self.filter_alpha = filter_alpha
        self.min_stance_legs = min_stance_legs

        # Filtered velocity output (body frame, [vx, vy, vz])
        self._vel_filtered = np.zeros(3, dtype=np.float32)

    def update(
        self,
        joint_pos_rl: np.ndarray,
        joint_vel_rl: np.ndarray,
        servo_loads: np.ndarray | None = None,
        angular_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute body velocity estimate from joint state.

        Args:
            joint_pos_rl: 12D joint positions in RL convention (absolute, NOT relative).
            joint_vel_rl: 12D joint velocities in RL convention (rad/s).
            angular_velocity: Body-frame gyro reading in rad/s.
            servo_loads: 12D servo load values (0-1000). If None, assumes all legs
                        are in stance (fallback when load data unavailable).

        Returns:
            3D body velocity in body frame [vx, vy, vz] (m/s).
        """
        vel_estimates = []

        for i, leg in enumerate(LEG_NAMES):
            # Check if this leg is in stance
            if servo_loads is not None:
                # Use thigh + calf loads as stance indicator (shoulder load is less informative)
                thigh_load = abs(servo_loads[THIGH_IDX[i]])
                calf_load = abs(servo_loads[CALF_IDX[i]])
                leg_load = max(thigh_load, calf_load)
                if leg_load < self.load_threshold:
                    continue  # Swing leg — skip

            # Get joint angles and velocities for this leg
            sh_angle = joint_pos_rl[SHOULDER_IDX[i]]
            th_angle = joint_pos_rl[THIGH_IDX[i]]
            ca_angle = joint_pos_rl[CALF_IDX[i]]

            sh_vel = joint_vel_rl[SHOULDER_IDX[i]]
            th_vel = joint_vel_rl[THIGH_IDX[i]]
            ca_vel = joint_vel_rl[CALF_IDX[i]]

            q_dot = np.array([sh_vel, th_vel, ca_vel])

            # Compute Jacobian
            J = foot_jacobian(leg, sh_angle, th_angle, ca_angle)

            # Body velocity = -(J * q_dot) during stance
            # (foot is stationary, so body moves opposite to foot velocity in body frame)
            v_body = -(J @ q_dot)
            if angular_velocity is not None:
                v_body -= np.cross(angular_velocity, foot_position(leg, sh_angle, th_angle, ca_angle))
            vel_estimates.append(v_body)

        if len(vel_estimates) >= self.min_stance_legs:
            # Average across stance legs
            raw_vel = np.mean(vel_estimates, axis=0).astype(np.float32)
            # Low-pass filter
            self._vel_filtered = (
                self.filter_alpha * raw_vel
                + (1 - self.filter_alpha) * self._vel_filtered
            )
        # If insufficient stance legs, keep previous filtered estimate

        return self._vel_filtered.copy()

    def reset(self) -> None:
        """Reset velocity estimate to zero."""
        self._vel_filtered = np.zeros(3, dtype=np.float32)
