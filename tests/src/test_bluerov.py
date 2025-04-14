import numpy as np
import pytest
from numpy.testing import assert_allclose

# Assuming the project root is added to PYTHONPATH or using pytest features
# If running pytest from the root directory, this should work.
from src.bluerov import ssa, bluerov_compute
from src.parameters import m, Z_wd, g, F_bouy

# --- Tests for ssa (smallest signed angle) ---

@pytest.mark.parametrize("angle, expected", [
    (0, 0),
    (np.pi, -np.pi), # Corrected: pi maps to -pi
    (-np.pi, -np.pi), # Note: -pi maps to -pi, not pi
    (np.pi + 0.1, -np.pi + 0.1),
    (-np.pi - 0.1, np.pi - 0.1),
    (2 * np.pi, 0),
    (-2 * np.pi, 0),
    (3 * np.pi, -np.pi), # Corrected: 3*pi maps to -pi
    (-3 * np.pi, -np.pi),
    (1.5 * np.pi, -0.5 * np.pi),
    (-1.5 * np.pi, 0.5 * np.pi),
])
def test_ssa(angle, expected):
    """Tests the ssa function for various angles."""
    assert_allclose(ssa(angle), expected, atol=1e-7)

# --- Tests for bluerov_compute ---

def test_bluerov_compute_zero_input_zero_velocity():
    """
    Tests bluerov_compute with zero initial velocities and zero control inputs.
    The only expected motion is vertical acceleration due to gravity/buoyancy mismatch.
    """
    t = 0.0
    # State: [x, y, z, psi, u, v, w, r]
    x_ = np.zeros(8, dtype=np.float64)
    # Input: [X, Y, Z, M_z]
    u_ = np.zeros(4, dtype=np.float64)

    # Calculate expected w_dot
    # w_d = 1 / (m - Z_wd) * (Z + (Z_w + Z_wc * abs(w)) * w + m * g - F_bouy)
    # With Z=0, w=0:
    expected_w_dot = 1 / (m - Z_wd) * (m * g - F_bouy)

    # Expected state derivative: [x_d, y_d, z_d, psi_d, u_d, v_d, w_d, r_d]
    expected_x_dot = np.array([
        0.0,  # x_d = cos(0)*0 - sin(0)*0
        0.0,  # y_d = sin(0)*0 + cos(0)*0
        0.0,  # z_d = w = 0
        0.0,  # psi_d = r = 0
        0.0,  # u_d (X=0, v=0, r=0, u=0)
        0.0,  # v_d (Y=0, u=0, r=0, v=0)
        expected_w_dot, # w_d
        0.0   # r_d (M_z=0, u=0, v=0, r=0)
    ], dtype=np.float64)

    # Compute actual state derivative
    actual_x_dot = bluerov_compute(t, x_, u_)

    # Assert shape and values
    assert actual_x_dot.shape == (8,)
    assert_allclose(actual_x_dot, expected_x_dot, atol=1e-7, rtol=1e-7)

def test_bluerov_compute_forward_thrust_initial_velocity():
    """
    Tests bluerov_compute with initial forward velocity and forward thrust.
    Expects acceleration in u, and potentially some coupling effects if v or r were non-zero.
    """
    t = 0.0
    # State: [x, y, z, psi, u, v, w, r]
    initial_u = 1.0
    x_ = np.array([0.0, 0.0, 0.0, 0.0, initial_u, 0.0, 0.0, 0.0], dtype=np.float64)
    # Input: [X, Y, Z, M_z]
    thrust_X = 10.0 # Example forward thrust
    u_ = np.array([thrust_X, 0.0, 0.0, 0.0], dtype=np.float64)

    # Import necessary parameters for calculation
    from src.parameters import m, X_ud, Y_vd, Z_wd, I_zz, N_rd, X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc, N_r, N_rc, g, F_bouy

    # Calculate expected derivatives based on the equations in bluerov.py
    # eta_dot
    expected_x_d = np.cos(x_[3]) * x_[4] - np.sin(x_[3]) * x_[5] # cos(0)*1 - sin(0)*0 = 1
    expected_y_d = np.sin(x_[3]) * x_[4] + np.cos(x_[3]) * x_[5] # sin(0)*1 + cos(0)*0 = 0
    expected_z_d = x_[6] # 0
    expected_psi_d = x_[7] # 0

    # nu_dot
    # u_d = 1 / (m - X_ud) * (X + (m - Y_vd) * v * r + (X_u + X_uc * abs(u)) * u)
    expected_u_d = 1 / (m - X_ud) * (thrust_X + (m - Y_vd) * 0 * 0 + (X_u + X_uc * abs(initial_u)) * initial_u)
    # v_d = 1 / (m - Y_vd) * (Y - (m - X_ud) * u * r + (Y_v + Y_vc * abs(v)) * v)
    expected_v_d = 1 / (m - Y_vd) * (0 - (m - X_ud) * initial_u * 0 + (Y_v + Y_vc * abs(0)) * 0)
    # w_d = 1 / (m - Z_wd) * (Z + (Z_w + Z_wc * abs(w)) * w + m * g - F_bouy)
    expected_w_d = 1 / (m - Z_wd) * (0 + (Z_w + Z_wc * abs(0)) * 0 + m * g - F_bouy)
    # r_d = 1 / (I_zz - N_rd) * (M_z - (X_ud - Y_vd) * u * v + (N_r + N_rc * abs(r)) * r)
    expected_r_d = 1 / (I_zz - N_rd) * (0 - (X_ud - Y_vd) * initial_u * 0 + (N_r + N_rc * abs(0)) * 0)


    expected_x_dot = np.array([
        expected_x_d, expected_y_d, expected_z_d, expected_psi_d,
        expected_u_d, expected_v_d, expected_w_d, expected_r_d
    ], dtype=np.float64)

    # Compute actual state derivative
    actual_x_dot = bluerov_compute(t, x_, u_)

    # Assert shape and values
    assert actual_x_dot.shape == (8,)
    assert_allclose(actual_x_dot, expected_x_dot, atol=1e-7, rtol=1e-7)
