import os
import numpy as np
import torch
import control as ct
from scipy.stats.qmc import LatinHypercube

from ..src.bluerov import bluerov
from .data_utility import random_input, random_x0

# Set seed to ensure reproducibility
np.random.seed(0)

# Define the ROV system globally
# Using NonlinearIOSystem instead of nlsys
rov_sys = ct.NonlinearIOSystem(
    bluerov, None, inputs=('X', 'Y', 'Z', 'M_z'),
    outputs=('x', 'y', 'z', 'psi', 'u', 'v', 'w', 'r'),
    states=('x', 'y', 'z', 'psi', 'u', 'v', 'w', 'r'),
    name='bluerov_system' # Name is often required or good practice
)

def create_data(
    N_traj, input_type, params=None, T_tot=5.2, dt=0.08, N_x=9, N_u=4,
    N_coll=0, fixed_coll_points=None, intervals=None
):
    """
    Generates trajectories for the BlueROV simulation.

    Args:
        N_traj (int): Number of trajectories to generate.
        input_type (str): Type of input ('noise', 'sine', 'line', 'circle', 'figure8', etc.).
        params (dict, optional): Parameters for specific trajectories. Defaults to None.
        T_tot (float): Total time of trajectory.
        dt (float): Time step.
        N_x (int): Number of state variables.
        N_u (int): Number of control inputs.
        N_coll (int): Number of collocation points.
        fixed_coll_points (list or None): Fixed collocation points.
        intervals (list or None): Intervals for random initial conditions.

    Returns:
        X (np.ndarray): State trajectories of shape (N_traj, N, N_x).
        U (np.ndarray): Control inputs of shape (N_traj, N, N_u).
        t (np.ndarray): Time array of shape (N,).
        t_coll (np.ndarray): Collocation times of shape (N_traj, N, N_coll_tot).
    """
    if intervals is None:
        intervals = [0.0] * 8  # Ensure intervals have correct length
    if fixed_coll_points is None:
        fixed_coll_points = [dt]

    N = int(T_tot / dt)  # Number of points in a trajectory
    print(N)
    N_coll_tot = N_coll + len(fixed_coll_points)
    t = np.linspace(0, T_tot, N, dtype=np.float32)

    U = np.zeros((N_traj, N, N_u), dtype=np.float32)
    X = np.zeros((N_traj, N, N_x), dtype=np.float32)
    t_coll = np.zeros((N_traj, N, N_coll_tot), dtype=np.float32)

    # Initialize Latin Hypercube Sampler
    if N_coll > 0:
        lhs_sampler = LatinHypercube(1, seed=0)

    for n in range(N_traj):
        print(f"Generating trajectory {n + 1}/{N_traj}")

        # Generate random initial state
        x_0 = random_x0(intervals)

        # Generate input sequence based on type and params
        U[n, :] = random_input(t, N_u, input_type, params=params)

        # Apply general adjustments ONLY if not a specific trajectory type
        # (Assume specific types handle their own scaling/logic)
        if input_type not in ['line', 'circle', 'figure8']:
            U[n, :, 1] *= 0.1   # Scale Y input
            U[n, :, 2] = 5 * np.abs(U[n, :, 2])  # Only diving (Z input positive)
            U[n, :, -1] *= 0.05  # Scale M_z (yaw) input
        # Note: For line/circle/figure8, the random_input function now sets U directly.
        # We might need to refine this logic if those trajectories also need scaling.

        # Simulate the system
        _, x = ct.input_output_response(
            rov_sys, t, U[n, :].T, x_0
        )
        x = x.T  # Transpose to match dimensions

        # Store states
        X[n, :, :3] = x[:, :3]  # x, y, z
        X[n, :, 3] = np.cos(x[:, 3])  # cos(psi)
        X[n, :, 4] = np.sin(x[:, 3])  # sin(psi)
        X[n, :, 5:] = x[:, 4:]  # u, v, w, r

        # Generate collocation times
        for m in range(N):
            if N_coll > 0:
                random_times = dt * lhs_sampler.random(n=N_coll).flatten()
                coll_times = np.sort(np.concatenate((random_times, fixed_coll_points)))
            else:
                coll_times = np.array(fixed_coll_points)
            t_coll[n, m, :] = coll_times

    return X, U, t, t_coll

def main():
    dt = 0.08
    T_tot = 5.2  # Longer trajectories for longer predictions
    N_x = 9
    N_u = 4
    N_coll = 0

    paths = ['training_set', 'dev_set', 'test_set_interp', 'test_set_extrap']
    no_trajs = [400, 1000, 1000, 1000]  # Number of trajectories for each set
    dts = [dt, dt, dt-0.02, dt+0.02]
    T_tots = [T_tot, T_tot, 3.9, 6.5]
    # Define intervals for initial conditions
    intervals_dict = {
        'training_set': [1.0, 1.0, 1.0, np.pi, 1.0, 0.0, 0.1, 0.0],
        'dev_set': [0.0, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0, 0.0],
        'test_set_interp': [0.0, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0, 0.0],
        'test_set_extrap': [0.0, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0, 0.0]
    }
    # more complex intervals for initial condition
    # intervals_dict = {
    #     'training_set': [1.0, 1.0, 1.0, np.pi, 1.0, 1.0, 1.0, 0.5],
    #     'dev_set': [1.0, 1.0, 1.0, np.pi, 1.0, 1.0, 1.0, 0.5],
    #     'test_set_interp': [1.0, 1.0, 1.0, np.pi, 1.0, 1.0, 1.0, 0.5],
    #     'test_set_extrap': [1.0, 1.0, 1.0, np.pi, 1.0, 1.0, 1.0, 0.5]
    # }
    
    #Define input types for each dataset
    input_type_dict = {
        'training_set': 'noise',
        'dev_set': 'sine',
        'test_set_interp': 'sine',
        'test_set_extrap': 'sine',
        # Add new trajectory types
        'test_set_line': 'line',
        'test_set_circle': 'circle',
        'test_set_figure8': 'figure8'
    }

    # Parameters for specific trajectories (using defaults from data_utility for now)
    trajectory_params = {
        'line': {'forward_thrust': 5.0},
        'circle': {'forward_thrust': 5.0, 'yaw_moment': 0.5},
        'figure8': {'forward_thrust': 5.0, 'yaw_amplitude': 1.0, 'yaw_frequency': 0.2} # Adjust freq based on T_tot if needed
    }

    # Add new paths and parameters for specific trajectories
    paths.extend(['test_set_line', 'test_set_circle', 'test_set_figure8'])
    no_trajs.extend([10, 10, 10]) # Generate fewer trajectories for these specific tests
    dts.extend([dt, dt, dt]) # Use default dt
    T_tots.extend([T_tot, T_tot, T_tot]) # Use default T_tot

    # Use zero intervals for deterministic starting points for new trajectories
    intervals_dict['test_set_line'] = [0.0] * 8
    intervals_dict['test_set_circle'] = [0.0] * 8
    intervals_dict['test_set_figure8'] = [0.0] * 8


    for path, n_traj, dt_, T_tot_ in zip(paths, no_trajs, dts, T_tots):
        print(f"\n--- Generating dataset: {path} ---")
        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Created directory: {path}")

        intervals = intervals_dict.get(path, [0.0] * 8) # Default to zero intervals if not specified
        input_type = input_type_dict.get(path, 'sine') # Default to sine if not specified
        params = trajectory_params.get(input_type, None) # Get specific params if applicable

        print(f"Parameters: N_traj={n_traj}, input_type='{input_type}', T_tot={T_tot_}, dt={dt_}, intervals={intervals}, params={params}")

        X, U, t, t_coll = create_data(
            N_traj=n_traj,
            input_type=input_type,
            params=params, # Pass trajectory specific parameters
            T_tot=T_tot_,
            dt=dt_,
            N_x=N_x,
            N_u=N_u,
            N_coll=N_coll,
            fixed_coll_points=[dt_],
            intervals=intervals
        )

        # Save data
        torch.save(torch.from_numpy(t), os.path.join(path, 't.pt'))
        torch.save(torch.from_numpy(U), os.path.join(path, 'U.pt'))
        torch.save(torch.from_numpy(X), os.path.join(path, 'X.pt'))
        torch.save(torch.from_numpy(t_coll), os.path.join(path, 't_coll.pt'))

if __name__ == '__main__':
    main()
