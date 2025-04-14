import torch
import pytest
import numpy as np # For np.pi

# Assuming the project root is added to PYTHONPATH or using pytest features
from src.bluerov_torch import ssa, bluerov_compute
from src.parameters import m, Z_wd, g, F_bouy

# Use float64 for better precision in comparisons
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tests for ssa (smallest signed angle) ---

@pytest.mark.parametrize("angle_val, expected_val", [
    (0, 0),
    (np.pi, -np.pi), # Matches numpy version behavior
    (-np.pi, -np.pi),
    (np.pi + 0.1, -np.pi + 0.1),
    (-np.pi - 0.1, np.pi - 0.1),
    (2 * np.pi, 0),
    (-2 * np.pi, 0),
    (3 * np.pi, -np.pi), # Matches numpy version behavior
    (-3 * np.pi, -np.pi),
    (1.5 * np.pi, -0.5 * np.pi),
    (-1.5 * np.pi, 0.5 * np.pi),
])
def test_ssa_torch(angle_val, expected_val):
    """Tests the PyTorch ssa function."""
    angle = torch.tensor([angle_val], dtype=torch.float64, device=device) # Ensure input is float64
    expected = torch.tensor([expected_val], dtype=torch.float64, device=device) # Ensure expected is float64
    assert torch.allclose(ssa(angle), expected, atol=1e-7)

# --- Tests for bluerov_compute (PyTorch version) ---

def test_bluerov_compute_torch_zero_input_zero_velocity():
    """
    Tests PyTorch bluerov_compute with zero initial velocities and zero control inputs.
    State uses [x, y, z, cos(psi), sin(psi), u, v, w, r] format.
    """
    t = 0.0 # Not used in bluerov_compute, but kept for consistency
    # State: [x, y, z, cos(psi), sin(psi), u, v, w, r]
    # psi=0 -> cos(psi)=1, sin(psi)=0
    x_ = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    # Input: [X, Y, Z, M_z]
    u_ = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)

    # Calculate expected w_dot (same as numpy version)
    expected_w_dot = 1 / (m - Z_wd) * (m * g - F_bouy)

    # Expected state derivative: [x_d, y_d, z_d, cos(psi)_d, sin(psi)_d, u_d, v_d, w_d, r_d]
    expected_x_dot = torch.tensor([[
        0.0,  # x_d = cos(psi)*u - sin(psi)*v = 1*0 - 0*0
        0.0,  # y_d = sin(psi)*u + cos(psi)*v = 0*0 + 1*0
        0.0,  # z_d = w = 0
        0.0,  # cos(psi)_d = -sin(psi)*r = -0*0
        0.0,  # sin(psi)_d = cos(psi)*r = 1*0
        0.0,  # u_d (X=0, v=0, r=0, u=0)
        0.0,  # v_d (Y=0, u=0, r=0, v=0)
        expected_w_dot, # w_d
        0.0   # r_d (M_z=0, u=0, v=0, r=0)
    ]], device=device)

    # Compute actual state derivative
    actual_x_dot = bluerov_compute(t, x_, u_)

    # Assert shape and values
    assert actual_x_dot.shape == (1, 9)
    assert torch.allclose(actual_x_dot, expected_x_dot, atol=1e-7, rtol=1e-7)

def test_bluerov_compute_torch_forward_thrust_initial_velocity():
    """
    Tests PyTorch bluerov_compute with initial forward velocity and forward thrust.
    State uses [x, y, z, cos(psi), sin(psi), u, v, w, r] format.
    """
    t = 0.0
    # State: [x, y, z, cos(psi), sin(psi), u, v, w, r]
    initial_u = 1.0
    # psi=0 -> cos(psi)=1, sin(psi)=0
    x_ = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, initial_u, 0.0, 0.0, 0.0]], device=device)
    # Input: [X, Y, Z, M_z]
    thrust_X = 10.0 # Example forward thrust
    u_ = torch.tensor([[thrust_X, 0.0, 0.0, 0.0]], device=device)

    # Import necessary parameters for calculation
    from src.parameters import m, X_ud, Y_vd, Z_wd, I_zz, N_rd, X_u, X_uc, Y_v, Y_vc, Z_w, Z_wc, N_r, N_rc, g, F_bouy

    # Calculate expected derivatives (similar to numpy version, but for 9 states)
    # eta_dot
    expected_x_d = x_[0, 3] * x_[0, 5] - x_[0, 4] * x_[0, 6] # cos(psi)*u - sin(psi)*v = 1*1 - 0*0 = 1
    expected_y_d = x_[0, 4] * x_[0, 5] + x_[0, 3] * x_[0, 6] # sin(psi)*u + cos(psi)*v = 0*1 + 1*0 = 0
    expected_z_d = x_[0, 7] # w = 0
    expected_cos_psi_d = -x_[0, 4] * x_[0, 8] # -sin(psi)*r = -0*0 = 0
    expected_sin_psi_d = x_[0, 3] * x_[0, 8] # cos(psi)*r = 1*0 = 0

    # nu_dot (same calculations as numpy version)
    expected_u_d = 1 / (m - X_ud) * (thrust_X + (m - Y_vd) * 0 * 0 + (X_u + X_uc * abs(initial_u)) * initial_u)
    expected_v_d = 1 / (m - Y_vd) * (0 - (m - X_ud) * initial_u * 0 + (Y_v + Y_vc * abs(0)) * 0)
    expected_w_d = 1 / (m - Z_wd) * (0 + (Z_w + Z_wc * abs(0)) * 0 + m * g - F_bouy)
    expected_r_d = 1 / (I_zz - N_rd) * (0 - (X_ud - Y_vd) * initial_u * 0 + (N_r + N_rc * abs(0)) * 0)

    expected_x_dot = torch.tensor([[
        expected_x_d, expected_y_d, expected_z_d, expected_cos_psi_d, expected_sin_psi_d,
        expected_u_d, expected_v_d, expected_w_d, expected_r_d
    ]], device=device)

    # Compute actual state derivative
    actual_x_dot = bluerov_compute(t, x_, u_)

    # Assert shape and values
    assert actual_x_dot.shape == (1, 9)
    assert torch.allclose(actual_x_dot, expected_x_dot, atol=1e-7, rtol=1e-7)
