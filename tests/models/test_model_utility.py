import torch
import pytest

# Assuming the project root is added to PYTHONPATH or using pytest features
from models.model_utility import (
    convert_input_data,
    convert_output_data,
    convert_input_collocation
)

# Use float64 for consistency and precision
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Test Data Setup ---
N_BATCH = 2
N_SEQ = 10
N_X = 9 # State dimension (x, y, z, cos, sin, u, v, w, r)
N_U = 4 # Control dimension
N_COLL = 3 # Number of collocation points

# Dummy data tensors
@pytest.fixture
def dummy_data():
    X = torch.randn(N_BATCH, N_SEQ, N_X, device=device)
    U = torch.randn(N_BATCH, N_SEQ, N_U, device=device)
    time = torch.linspace(0, 1, N_SEQ, device=device).repeat(N_BATCH, 1).unsqueeze(-1)
    t_coll = torch.rand(N_BATCH, N_SEQ, N_COLL, device=device) * 0.1 # Small random times
    return X, U, time, t_coll

# --- Tests for convert_input_data ---

def test_convert_input_data_shape(dummy_data):
    """Tests the output shape of convert_input_data."""
    X, U, time, _ = dummy_data
    N_in_expected = N_X + N_U + 1
    Z, n_batch, n_seq, n_x = convert_input_data(X, U, time)

    assert Z.shape == (N_BATCH * N_SEQ, N_in_expected)
    assert n_batch == N_BATCH
    assert n_seq == N_SEQ
    assert n_x == N_X
    assert Z.requires_grad is True # Check if grad is enabled

def test_convert_input_data_values(dummy_data):
    """Tests if the values in the converted tensor Z are correct."""
    X, U, time, _ = dummy_data
    N_in_expected = N_X + N_U + 1
    Z, _, _, _ = convert_input_data(X, U, time)

    # Reshape Z back to check values
    Z_reshaped = Z.view(N_BATCH, N_SEQ, N_in_expected)

    # Check concatenation order and values
    assert torch.allclose(Z_reshaped[..., :N_X], X)
    assert torch.allclose(Z_reshaped[..., N_X:N_X+N_U], U)
    assert torch.allclose(Z_reshaped[..., N_X+N_U:], time)

# --- Tests for convert_output_data ---

def test_convert_output_data_shape(dummy_data):
    """Tests the output shape of convert_output_data."""
    X, _, _, _ = dummy_data # Use X shape as reference for output
    N_out = N_X
    X_hat_flat = torch.randn(N_BATCH * N_SEQ, N_out, device=device)

    X_hat_reshaped = convert_output_data(X_hat_flat, N_BATCH, N_SEQ, N_out)

    assert X_hat_reshaped.shape == (N_BATCH, N_SEQ, N_out)

def test_convert_output_data_values(dummy_data):
    """Tests if reshaping back and forth preserves values."""
    X, _, _, _ = dummy_data # Use X shape as reference for output
    N_out = N_X
    X_hat_flat = torch.randn(N_BATCH * N_SEQ, N_out, device=device)

    X_hat_reshaped = convert_output_data(X_hat_flat, N_BATCH, N_SEQ, N_out)
    # Flatten again to compare
    X_hat_flat_again = X_hat_reshaped.view(N_BATCH * N_SEQ, N_out)

    assert torch.allclose(X_hat_flat_again, X_hat_flat)


# --- Tests for convert_input_collocation ---

def test_convert_input_collocation_shapes(dummy_data):
    """Tests the output shapes of convert_input_collocation."""
    X, U, _, t_coll = dummy_data
    N_in_expected = N_X + N_U + 1

    Z_coll, U_coll_expanded = convert_input_collocation(X, U, t_coll)

    # Expected shape for Z_coll: [N_BATCH * N_SEQ * N_COLL, N_in_expected]
    assert Z_coll.shape == (N_BATCH * N_SEQ * N_COLL, N_in_expected)
    # Expected shape for U_coll_expanded: [N_BATCH, N_SEQ, N_COLL, N_U]
    assert U_coll_expanded.shape == (N_BATCH, N_SEQ, N_COLL, N_U)
    assert Z_coll.requires_grad is True # Check if grad is enabled

def test_convert_input_collocation_values(dummy_data):
    """Tests if the values in the converted collocation tensors are correct."""
    X, U, _, t_coll = dummy_data
    N_in_expected = N_X + N_U + 1

    Z_coll, U_coll_expanded = convert_input_collocation(X, U, t_coll)

    # Reshape Z_coll back to check values
    Z_coll_reshaped = Z_coll.view(N_BATCH, N_SEQ, N_COLL, N_in_expected)

    # Check expanded U values
    assert torch.allclose(U_coll_expanded, U.unsqueeze(2).expand(-1, -1, N_COLL, -1))

    # Check concatenation order and values in Z_coll_reshaped
    # Z = torch.cat((X_coll, U_coll, t_coll.unsqueeze(3)), dim=3)
    X_coll_expected = X.unsqueeze(2).expand(-1, -1, N_COLL, -1)
    assert torch.allclose(Z_coll_reshaped[..., :N_X], X_coll_expected)
    assert torch.allclose(Z_coll_reshaped[..., N_X:N_X+N_U], U_coll_expanded)
    assert torch.allclose(Z_coll_reshaped[..., N_X+N_U:], t_coll.unsqueeze(3))


# --- Tests for DNN Model ---
from models.model_utility import DNN
from torch.nn import Softplus

# Dummy model parameters
N_IN_DNN = N_X + N_U + 1
N_OUT_DNN = N_X
N_H_DNN = [16, 16] # Smaller hidden layers for faster testing
N_LAYER_DNN = len(N_H_DNN)

@pytest.fixture
def dummy_model():
    """Provides a dummy DNN model instance."""
    model = DNN(
        N_in=N_IN_DNN,
        N_out=N_OUT_DNN,
        N_h=N_H_DNN,
        N_layer=N_LAYER_DNN,
        activation=Softplus
    ).to(device)
    model.eval() # Set to eval mode for testing consistency
    return model

def test_dnn_initialization(dummy_model):
    """Tests if the DNN model initializes correctly."""
    assert isinstance(dummy_model, DNN)
    # Check if layers exist (basic check)
    assert hasattr(dummy_model, 'layers')
    assert len(dummy_model.layers) > 0

def test_dnn_forward_pass_shape(dummy_data, dummy_model):
    """Tests the output shape of the DNN forward pass."""
    X, U, time, _ = dummy_data
    Z, n_batch, n_seq, n_x = convert_input_data(X, U, time)

    with torch.no_grad(): # Disable gradients for simple forward pass test
        X_hat = dummy_model(Z)

    # Output shape should be [N_BATCH * N_SEQ, N_OUT_DNN]
    assert X_hat.shape == (N_BATCH * N_SEQ, N_OUT_DNN)
    assert X_hat.dtype == torch.float64 # Check default dtype is used


# --- Tests for Loss Functions ---
from models.model_utility import (
    data_loss_fn,
    physics_loss_fn,
    rollout_loss_fn,
    initial_condition_loss,
    compute_time_derivatives
)
# Need bluerov_compute for physics loss calculation inside physics_loss_fn
from src.bluerov_torch import bluerov_compute as bluerov_compute_torch

def test_compute_time_derivatives_shape(dummy_data, dummy_model):
    """Tests the output shapes of compute_time_derivatives."""
    X, U, time, _ = dummy_data
    Z, _, _, _ = convert_input_data(X, U, time)
    N_in = Z.shape[1]

    X_hat, dX_hat_dt = compute_time_derivatives(Z, N_in, dummy_model)

    assert X_hat.shape == (N_BATCH * N_SEQ, N_OUT_DNN)
    assert dX_hat_dt.shape == (N_BATCH * N_SEQ, N_OUT_DNN)
    assert X_hat.dtype == torch.float64
    assert dX_hat_dt.dtype == torch.float64

def test_data_loss_fn(dummy_data, dummy_model):
    """Tests data_loss_fn returns a non-negative scalar tensor."""
    X, U, time, _ = dummy_data
    loss = data_loss_fn(dummy_model, X, U, time, device)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([]) # Scalar
    assert loss.item() >= 0.0

def test_initial_condition_loss(dummy_data, dummy_model):
    """Tests initial_condition_loss returns a non-negative scalar tensor."""
    X, U, time, _ = dummy_data
    # Use only the first time step for IC loss typically
    loss = initial_condition_loss(dummy_model, X[:, 0:1, :], U[:, 0:1, :], time[:, 0:1, :], device)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([]) # Scalar
    assert loss.item() >= 0.0

def test_physics_loss_fn(dummy_data, dummy_model):
    """Tests physics_loss_fn returns a non-negative scalar tensor."""
    X, U, _, t_coll = dummy_data
    # Need to mock or provide bluerov_compute if it's not imported/available
    # Assuming bluerov_compute_torch is available from src.bluerov_torch
    loss = physics_loss_fn(dummy_model, X, U, t_coll, device)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([]) # Scalar
    # Physics loss can theoretically be negative if derivatives match poorly,
    # but MSE ensures it's >= 0
    assert loss.item() >= 0.0

def test_rollout_loss_fn(dummy_data, dummy_model):
    """Tests rollout_loss_fn returns non-negative scalar tensors."""
    X, U, time, t_coll = dummy_data
    N_roll = 5 # Shorter rollout for testing
    pinn = True # Test with physics loss enabled

    l_roll, l_phy = rollout_loss_fn(dummy_model, X, U, time, N_roll, device, t_coll, pinn)

    assert isinstance(l_roll, torch.Tensor)
    assert l_roll.shape == torch.Size([]) # Scalar
    assert l_roll.item() >= 0.0

    assert isinstance(l_phy, torch.Tensor)
    assert l_phy.shape == torch.Size([]) # Scalar
    assert l_phy.item() >= 0.0

def test_rollout_loss_fn_no_pinn(dummy_data, dummy_model):
    """Tests rollout_loss_fn without PINN loss."""
    X, U, time, t_coll = dummy_data
    N_roll = 5
    pinn = False # Test with physics loss disabled

    l_roll, l_phy = rollout_loss_fn(dummy_model, X, U, time, N_roll, device, t_coll, pinn)

    assert isinstance(l_roll, torch.Tensor)
    assert l_roll.shape == torch.Size([]) # Scalar
    assert l_roll.item() >= 0.0

    # Physics loss should be zero when pinn=False
    assert isinstance(l_phy, int) or isinstance(l_phy, float) # Should return 0
    assert l_phy == 0.0
