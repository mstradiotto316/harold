import numpy as np

from deployment.inference.leg_odometry import LegOdometry, foot_jacobian, foot_position


def test_positive_usd_pitch_moves_extended_foot_backwards():
    jac = foot_jacobian("fl", 0, 0, 0)
    # A positive right-handed Y rotation of a downward link has negative X speed.
    np.testing.assert_allclose(jac[0, 1:], [-0.1031 - 0.1294, -0.1294], atol=1e-5)


def test_body_velocity_during_rotation_about_stationary_foot():
    q = np.zeros(12)
    load = np.zeros(12)
    load[4] = 300  # Only the front left leg bears weight.
    omega = np.array([0.0, 0.0, 0.8])
    estimate = LegOdometry(filter_alpha=1).update(q, q, load, angular_velocity=omega)
    foot = foot_position("fl", 0, 0, 0)
    np.testing.assert_allclose(estimate, [omega[2] * foot[1], -omega[2] * foot[0], 0])
