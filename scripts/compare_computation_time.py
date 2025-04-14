import torch
import time
import numpy as np
import control
import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing project modules
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.model_utility import DNN, TrajectoryDataset, convert_input_data, convert_output_data, Softplus
from src.bluerov import bluerov # Import the correct function

# Define the DNN architecture MATCHING THE SAVED MODELS (_best_dev_l_0, _best_dev_l_993)
# Based on scripts/pi_dnn.py and state_dict errors
N_out_model = 9  # Output size of the saved models
hidden_size_model = 32 # Hidden layer size of the saved models
N_layer_model = 4  # Number of hidden layers in the saved models
# Input size: 9 states (assumed u,v,w,p,q,r,x,y,z) + 4 controls + 1 time = 14
N_in_model = 9 + 4 + 1
hidden_sizes_model = [hidden_size_model] * N_layer_model # List of hidden layer sizes

# The saved X.pt files contain 9 states: [x, y, z, cos(psi), sin(psi), u, v, w, r]
# The simulation function bluerov uses 8 states: [x, y, z, psi, u, v, w, r]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def time_pinn_rollout(model_path, X_gt, U_gt, t_eval, model_arch):
    """Times the PINN forward passes over a full trajectory duration."""
    print(f"Timing PINN forward passes for full trajectory: {model_path}")
    # X_gt, U_gt, t_eval are now passed directly
    model = model_arch.to(device)
    try:
        # Load model weights, ensure weights_only=True for security unless payload is trusted
        # Set weights_only=False as fallback if needed, based on how models were saved.
        try:
             model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except: # Broad except to catch potential pickle errors with weights_only=True
             print(f"Warning: Loading model {model_path} with weights_only=True failed. Attempting with weights_only=False.")
             model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)) # Fallback
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

    model.eval()

    # Data is already loaded and passed as arguments
    x0_gt = X_gt[0, :].unsqueeze(0).to(device) # Initial 9 states (1, 9)
    U_traj = U_gt.to(device) # Control inputs (T, 4)
    # t_eval is already a numpy array

    # Store predictions (using the model's output size)
    X_pred_model = torch.zeros_like(X_gt).to(device) # Shape (T, 9)
    X_pred_model[0, :] = x0_gt # Use the initial 9 states

    # Warm-up one pass
    x_warmup = x0_gt
    u_warmup = U_traj[0, :].unsqueeze(0)
    t_warmup = torch.tensor([[t_eval[0]]], dtype=torch.float32).to(device)
    input_warmup = torch.cat((x_warmup, u_warmup, t_warmup), dim=1)
    _ = model(input_warmup)
    if device == torch.device('cuda'):
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        # Loop through the trajectory duration, performing forward passes
        for i in range(len(t_eval) - 1):
            # Get current 9-state from ground truth for input prep
            x_curr_gt = X_gt[i, :].unsqueeze(0).to(device) # Shape (1, 9)
            u_curr = U_traj[i, :].unsqueeze(0) # Shape (1, 4)
            time_curr = torch.tensor([[t_eval[i]]], dtype=torch.float32).to(device) # Shape (1, 1)

            # Prepare model input: [9 states from GT, 4 controls, 1 time]
            model_input = torch.cat((x_curr_gt, u_curr, time_curr), dim=1)

            # Predict *next* 9 states (we don't use the output, just time the pass)
            x_next_model = model(model_input) # Expected shape (1, 9)

            # Store prediction (optional, mainly timing the loop)
            # Ensure shape is correct before storing if needed
            if x_next_model.shape == (1, N_out_model):
                 X_pred_model[i+1, :] = x_next_model
            elif x_next_model.numel() == N_out_model:
                 X_pred_model[i+1, :] = x_next_model.view(1, N_out_model)
            # No state update needed as we use GT for next input

    if device == torch.device('cuda'):
        torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"PINN Full Trajectory Forward Passes Time: {elapsed_time:.4f} seconds")
    return elapsed_time # Return time in seconds

# --- RK4 Implementation ---
def rk4_step(f, t, x, u, dt):
    """Performs one RK4 integration step."""
    k1 = f(t, x, u)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, u)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, u)
    k4 = f(t + dt, x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Wrapper for bluerov dynamics to match RK4 signature f(t, x, u)
def bluerov_dynamics_wrapper(t, x, u):
    # bluerov function expects (t, x_, u_) where u_ are the 4 control inputs
    # It ignores the params argument when called as bluerov(t, x, u, None)
    return bluerov(t, x, u, None)
# --- End RK4 Implementation ---

def time_simulation_rollout(X_gt, U_gt, t_eval):
    """Times a full RK4 rollout simulation for the trajectory duration."""
    print("Timing Simulation model (Full RK4 Rollout)")
    # X_gt, U_gt, t_eval are now passed directly

    # Extract the 8 states needed for bluerov simulation: [x, y, z, psi, u, v, w, r]
    x0_gt_np = X_gt[0, :].numpy() # Initial 9 states as numpy
    x_sim = x0_gt_np[0]
    y_sim = x0_gt_np[1]
    z_sim = x0_gt_np[2]
    cos_psi_sim = x0_gt_np[3]
    sin_psi_sim = x0_gt_np[4]
    u_sim = x0_gt_np[5]
    v_sim = x0_gt_np[6]
    w_sim = x0_gt_np[7]
    r_sim = x0_gt_np[8]
    psi_sim = np.arctan2(sin_psi_sim, cos_psi_sim)
    # Construct the 8-state initial condition for the simulation step
    x0_np = np.array([x_sim, y_sim, z_sim, psi_sim, u_sim, v_sim, w_sim, r_sim])

    # Use passed t_eval (already numpy array)
    t_eval_np = t_eval
    U_traj_np = U_gt.numpy() # Shape (T, 4)
    x_sim_rk4 = np.zeros((len(t_eval_np), 8)) # Store 8-state simulation results
    x_sim_rk4[0, :] = x0_np # Use the calculated 8-state initial condition

    # Warm-up one step
    dt_warmup = t_eval_np[1] - t_eval_np[0]
    _ = rk4_step(bluerov_dynamics_wrapper, t_eval_np[0], x_sim_rk4[0, :], U_traj_np[0, :], dt_warmup)

    start_time = time.time()
    # Perform closed-loop RK4 simulation for the full duration
    for i in range(len(t_eval_np) - 1):
        dt = t_eval_np[i+1] - t_eval_np[i]
        x_sim_rk4[i+1, :] = rk4_step(
            bluerov_dynamics_wrapper,
            t_eval_np[i],
            x_sim_rk4[i, :], # Use previous simulation state
            U_traj_np[i, :], # Use control input for this step
            dt
        )
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Simulation Full RK4 Rollout Time: {elapsed_time:.4f} seconds")
    return elapsed_time # Return time in seconds

def main(args):
    # --- Load Dataset ---
    dataset_path = project_root / args.dataset_dir
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found at {dataset_path}")
        return
    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = TrajectoryDataset(dataset_path)
        # Extract data for the first trajectory
        X_gt_traj, U_gt_traj, _, t_gt_traj = dataset[0]
        t_eval_main = t_gt_traj.numpy().flatten() # Get t_eval in main scope
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return

    # --- Define Model Architecture ---
    # Use the architecture matching the saved models
    model_arch = DNN(N_in=N_in_model, N_out=N_out_model, N_h=hidden_sizes_model, N_layer=N_layer_model, activation=Softplus)

    # --- Find All Models in models/ directory ---
    models_dir = project_root / "models"
    models_to_test = []
    for item in models_dir.iterdir():
        # Include files, exclude the utility script and cache directories
        if item.is_file() and item.name != "model_utility.py":
            models_to_test.append(item)
        elif item.is_dir() and item.name == "__pycache__":
            continue # Skip cache directory
        elif item.is_dir(): # Handle potential unexpected subdirectories if needed
             print(f"Warning: Skipping unexpected directory in models/: {item.name}")


    if not models_to_test:
        print("Error: No model files found in models/ directory.")
        return

    print(f"Found {len(models_to_test)} model files to test.")

    results = {}

    # --- Time All PINN Models (Full Trajectory Duration) ---
    for model_path in models_to_test:
        model_name = model_path.name
        print("-" * 20) # Separator for clarity
        # Pass extracted trajectory data to the timing function
        timing_sec_pinn = time_pinn_rollout(str(model_path), X_gt_traj, U_gt_traj, t_eval_main, model_arch)
        if timing_sec_pinn is not None:
            results[f"PINN: {model_name}"] = timing_sec_pinn # Store time in seconds
        else:
             print(f"Skipping results for model {model_name} due to timing error.")
    print("-" * 20) # Separator

    # --- Time Simulation Model (Full Trajectory Duration) ---
    # Pass extracted trajectory data to the timing function
    sim_timing_sec = time_simulation_rollout(X_gt_traj, U_gt_traj, t_eval_main)
    if sim_timing_sec is not None:
        results["Simulation (RK4 Rollout)"] = sim_timing_sec # Store time in seconds

    # --- Format and Save Results ---
    if not results:
        print("No timing results were generated.")
        return

    output_path = project_root / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving comparison results to: {output_path}")

    with open(output_path, 'w') as f:
        f.write("# Computational Time Comparison: Full Trajectory Duration\n\n")
        f.write(f"Comparison performed over the full duration of the first trajectory from dataset: `{args.dataset_dir}`\n")
        # Use t_eval_main which is defined in this scope
        f.write(f"Trajectory Duration: {t_eval_main[-1]:.2f} seconds ({len(t_eval_main)} steps)\n")
        f.write(f"Using device: `{device}`\n\n")
        f.write("## Description of Tests:\n")
        f.write("- **PINN Model Time:** Measures the total wall-clock time to perform sequential forward passes of the neural network for each step in the trajectory duration. Input for each step uses the ground truth state from the previous step.\n")
        f.write("- **Simulation Time:** Measures the total wall-clock time to perform a closed-loop simulation using RK4 integration for the entire trajectory duration.\n\n")
        f.write("## Results:\n")
        f.write("| Method                                                                             | Total Time (seconds) |\n")
        f.write("|------------------------------------------------------------------------------------|----------------------|\n")
        # Sort results for consistent output, putting simulation last
        sorted_results = sorted(results.items(), key=lambda item: item[0].startswith("Simulation"))
        for name, timing_sec in sorted_results:
            # Extract model name for PINN entries for cleaner display
            display_name = name.replace("models/", "") if name.startswith("PINN:") else name
            f.write(f"| {display_name:<82} | {timing_sec:.4f}             |\n")

    print("Comparison complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare computational time of PINN models and simulation.")
    parser.add_argument('--dataset_dir', type=str, default='test_set_line',
                        help='Directory containing the test dataset (relative to project root)')
    parser.add_argument('--output_file', type=str, default='results/computation_time_comparison.md',
                        help='Output Markdown file path (relative to project root)')
    args = parser.parse_args()
    main(args)
