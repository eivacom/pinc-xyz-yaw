import numpy as np
import pytest
import torch

# Assuming the project root is added to PYTHONPATH or using pytest features
from data.data_utility import random_x0, random_input, TrajectoryDataset

# --- Tests for random_x0 ---

def test_random_x0_output_shape():
    """Tests if random_x0 returns the correct shape."""
    intervals = [1.0] * 8
    x_0 = random_x0(intervals)
    assert x_0.shape == (8,)

def test_random_x0_intervals():
    """Tests if random_x0 respects the given intervals."""
    intervals = [0.1, 0.2, 0.3, np.pi/2, 0.4, 0.5, 0.6, 0.7]
    for _ in range(100): # Run multiple times due to randomness
        x_0 = random_x0(intervals)
        # Check bounds (allow for floating point inaccuracies slightly)
        assert -0.101 <= x_0[0] <= 0.101 # x
        assert -0.201 <= x_0[1] <= 0.201 # y
        assert -0.301 <= x_0[2] <= 0.301 # z
        assert -np.pi/2 - 0.01 <= x_0[3] <= np.pi/2 + 0.01 # psi
        assert -0.401 <= x_0[4] <= 0.401 # u
        assert -0.501 <= x_0[5] <= 0.501 # v
        assert 0 <= x_0[6] <= 0.601 # w (vertical velocity) should be non-negative
        assert -0.701 <= x_0[7] <= 0.701 # r

def test_random_x0_invalid_intervals():
    """Tests if random_x0 raises ValueError for incorrect interval length."""
    with pytest.raises(ValueError, match="intervals must have length 8."):
        random_x0([1.0] * 7)
    with pytest.raises(ValueError, match="intervals must have length 8."):
        random_x0([1.0] * 9)

# --- Tests for random_input ---

@pytest.mark.parametrize("input_type", ['noise', 'sine', 'noise_x', 'sine_x'])
def test_random_input_output_shape(input_type):
    """Tests if random_input returns the correct shape."""
    N = 50
    N_u = 4
    t = np.linspace(0, 5, N)
    U = random_input(t, N_u, input_type)
    assert U.shape == (N, N_u)

def test_random_input_invalid_type():
    """Tests if random_input raises ValueError for invalid input_type."""
    N = 50
    N_u = 4
    t = np.linspace(0, 5, N)
    with pytest.raises(ValueError, match="Invalid input_type"):
        random_input(t, N_u, 'invalid_type')

# --- Tests for TrajectoryDataset (will be added after setting up fixtures) ---
# TODO: Add tests for TrajectoryDataset using fixtures for dummy data files.
