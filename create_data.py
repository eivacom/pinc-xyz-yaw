import os
import numpy as np
import torch
import control as ct
from scipy.stats.qmc import LatinHypercube

from bluerov import bluerov
from data_utility import random_input, random_x0

# Set seed to ensure reproducibility 
np.random.seed(0)

# Define the ROV system globally
rov_sys = ct.nlsys(
    bluerov, None, inputs=('X', 'Y', 'Z', 'M_z'),
    outputs=('x', 'y', 'z', 'psi', 'u', 'v', 'w', 'r'),
    states=('x', 'y', 'z', 'psi', 'u', 'v', 'w', 'r')
)

def create_data(
    N_traj, input_type, T_tot=5.2, dt=0.08, N_x=9, N_u=4,
    N_coll=0, fixed_coll_points=None, intervals=None
):
    """
    Generates trajectories for the BlueROV simulation.

    Args:
        N_traj (int): Number of trajectories to generate.
        input_type (str): Type of input ('noise', 'sine', etc.).
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

        # Generate random input sequence
        U[n, :] = random_input(t, N_u, input_type)
        # Adjust control inputs
        U[n, :, 1] *= 0.1   # Scale Y input
        U[n, :, 2] = 5 * np.abs(U[n, :, 2])  # Only diving (Z input positive)
        U[n, :, -1] *= 0.05  # Scale M_z (yaw) input

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
        'test_set': 'sine'
    }

    for path, n_traj, dt_, T_tot_ in zip(paths, no_trajs, dts, T_tots):
        if not os.path.exists(path):
            os.mkdir(path)

        intervals = intervals_dict.get(path, [0.0] * 8)
        input_type = input_type_dict.get(path, 'sine')

        X, U, t, t_coll = create_data(
            N_traj=n_traj,
            input_type=input_type,
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
