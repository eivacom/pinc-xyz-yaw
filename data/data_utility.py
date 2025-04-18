import numpy as np
import torch
import os
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    Custom PyTorch Dataset for loading trajectory data.
    Assumes data is stored in .pt files (X.pt, U.pt, t_coll.pt, t.pt)
    within the specified directory (data_dir).
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        try:
            self.X = torch.load(os.path.join(data_dir, 'X.pt'))
            self.U = torch.load(os.path.join(data_dir, 'U.pt'))
            self.t_coll = torch.load(os.path.join(data_dir, 't_coll.pt'))
            # Load time vector - assuming it's the same for all trajectories in the set
            # If time varies per trajectory, this needs adjustment.
            self.time = torch.load(os.path.join(data_dir, 't.pt'))
            # Ensure time has the correct shape [N_traj, N_seq, 1] if needed by model_utility
            # Assuming time is [N_seq], we might need to expand it later or handle in __getitem__
            # For now, let's assume model_utility functions can handle a single time vector or adapt.
            # A common pattern is to repeat the time vector for each batch item.
            # Let's reshape t_coll to match X and U's first dimension (N_traj)
            # The create_data script saves t_coll as [N_traj, N, N_coll_tot]
            # and time as [N,]. Let's adjust time to be [N_traj, N, 1]

            N_traj = self.X.shape[0]
            N_seq = self.X.shape[1]
            # Reshape time: [N_seq] -> [1, N_seq, 1] -> [N_traj, N_seq, 1]
            self.time = self.time.unsqueeze(0).unsqueeze(-1).expand(N_traj, -1, -1)

            # Basic validation
            assert self.X.shape[0] == self.U.shape[0] == self.t_coll.shape[0] == self.time.shape[0], "Mismatch in number of trajectories"
            assert self.X.shape[1] == self.U.shape[1] == self.t_coll.shape[1] == self.time.shape[1], "Mismatch in sequence length"

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading data from {data_dir}. Missing file: {e.filename}")
        except Exception as e:
            raise RuntimeError(f"Error loading or processing data from {data_dir}: {e}")

    def __len__(self):
        """Returns the number of trajectories in the dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx):
        """Returns a single trajectory tuple (X, U, t_coll, time)."""
        # Note: time returned here is shaped [N_seq, 1] for the specific trajectory idx
        return self.X[idx], self.U[idx], self.t_coll[idx], self.time[idx]


def random_input(t, N_u, input_type='noise', params=None):
    """
    Generates input signals for the system based on the specified type.

    Args:
        t (ndarray): Time array of shape (N,).
        N_u (int): Number of control inputs.
        input_type (str): Type of input signal ('noise', 'sine', 'noise_x', 'sine_x', 'line', 'circle', 'figure8').
        params (dict, optional): Parameters for specific trajectories (e.g., speed, radius). Defaults to None.

    Returns:
        U (ndarray): Input signal array of shape (N, N_u).
    """
    N = len(t)
    U = np.zeros((N, N_u))
    if params is None:
        params = {}

    if input_type in ['noise', 'noise_x']:
        # Generate noise input
        u_signs = np.random.choice([-1, 1], size=N_u)
        u_offsets = np.random.normal(0, 0.5, size=N_u)
        m_indices = np.arange(N)
        half_N = N // 2
        
        # Ramp up and down profiles
        ramp_up = (u_signs * m_indices[:half_N, None] / half_N - 0.5 * u_signs + u_offsets)
        ramp_down = (u_signs * (N - m_indices[half_N:, None]) / half_N - 0.5 * u_signs + u_offsets)
        U_noise = np.vstack((ramp_up, ramp_down))
        
        # Alternative code with random pulses like pseudo random trajectories. Can be used in future work for more exciting signals
        '''for n in range(N_u):
            u_sign = np.random.choice([-1, 1])
            for m in range(N):
                if m%4==0:
                    u = np.random.normal(0, 0.5, 1)
                if m < 0.5*N:
                    U_noise[m, n] = u_sign*m/(0.5*N) - 0.5*u_sign + u
                else:
                    U_noise[m, n] = u_sign*(N-m)/(0.5*N) - 0.5*u_sign + u'''
        
        if input_type == 'noise':
            U += U_noise
        elif input_type == 'noise_x':
            U[:, 0] += U_noise[:, 0]
    
    elif input_type in ['sine', 'sine_x']:
        # Generate sine input
        sine_freq = np.random.uniform(0.01, 0.2, N_u)
        sine_phase = np.random.uniform(0, 2 * np.pi, N_u)
        sine_amp = 3  # Scalar amplitude
        U_sine = sine_amp * np.sin(2 * np.pi * sine_freq * t[:, None] + sine_phase)
        
        if input_type == 'sine':
            U += U_sine
        elif input_type == 'sine_x':
            U[:, 0] += U_sine[:, 0]

    elif input_type == 'line':
        # Constant forward thrust for a straight line
        forward_thrust = params.get('forward_thrust', 5.0) # Default thrust
        U[:, 0] = forward_thrust

    elif input_type == 'circle':
        # Constant forward thrust and constant yaw moment for a circle
        forward_thrust = params.get('forward_thrust', 5.0)
        yaw_moment = params.get('yaw_moment', 0.5) # Default moment
        U[:, 0] = forward_thrust
        U[:, 3] = yaw_moment

    elif input_type == 'figure8':
        # Constant forward thrust and sinusoidal yaw moment for a figure 8
        forward_thrust = params.get('forward_thrust', 5.0)
        yaw_amplitude = params.get('yaw_amplitude', 1.0) # Default amplitude
        yaw_frequency = params.get('yaw_frequency', 0.2) # Default frequency (adjust based on T_tot)
        U[:, 0] = forward_thrust
        U[:, 3] = yaw_amplitude * np.sin(2 * np.pi * yaw_frequency * t)

    else:
        raise ValueError(f"Invalid input_type '{input_type}'. Must be 'noise', 'sine', 'noise_x', 'sine_x', 'line', 'circle', or 'figure8'.")

    return U

def random_x0(intervals):
    """
    Generates a random initial state vector x_0 based on specified intervals.
    
    Args:
        intervals (list or ndarray): Intervals for each state variable. Should have length 8.
        
    Returns:
        x_0 (ndarray): Initial state vector of shape (8,).
    """
    intervals = np.asarray(intervals)
    if intervals.shape[0] != 8:
        raise ValueError("intervals must have length 8.")
    
    # Generate random positions
    p = np.random.uniform(-intervals[0:3], intervals[0:3])
    
    # Generate random heading angle psi
    psi = np.random.uniform(-intervals[3], intervals[3])
    
    # Generate random velocities
    v = np.random.uniform(-intervals[4:7], intervals[4:7])
    v[2] = np.abs(v[2])  # Only dive (positive vertical velocity)
    
    # Generate random angular velocity w_r
    w_r = np.random.uniform(-intervals[7], intervals[7])
    
    # Concatenate all state variables
    x_0 = np.concatenate((p, [psi], v, [w_r]))
    
    return x_0
