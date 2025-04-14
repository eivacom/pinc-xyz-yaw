#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import seaborn as sns
sns.set()

from data.data_utility import TrajectoryDataset # Use the dataset class
from models.model_utility import DNN, Softplus, convert_input_data, convert_output_data # Import necessary components
from src.bluerov_torch import * # Import BlueROV dynamics if needed for physics loss (optional here)

# --- Configuration ---
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
# plt.rc('text', usetex=True) # Disable LaTeX rendering if dvipng is not installed
plt.rcParams.update({'font.size': 14})

# --- Model Definition ---
# Define model parameters (should match the trained models)
N_in = 14 # 9 states + 4 controls + 1 time
N_out = 9 # 9 states
N_h = [32, 32, 32, 32]
N_layer = len(N_h)
activation = Softplus

# --- Plotting Helper ---
def rescale_ax(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_scale = abs(x_max - x_min)
    y_scale = abs(y_max - y_min)
    if not x_scale or not y_scale: return # Avoid division by zero

    if x_scale > y_scale:
        y_center = (y_min + y_max) / 2
        y_new_half = x_scale / 2 * (ax.get_position().height / ax.get_position().width)
        ax.set_ylim([y_center - y_new_half, y_center + y_new_half])
    else:
        x_center = (x_min + x_max) / 2
        x_new_half = y_scale / 2 * (ax.get_position().width / ax.get_position().height)
        ax.set_xlim([x_center - x_new_half, x_center + x_new_half])

# --- Evaluation Function ---
def evaluate_rollout(model_path, dataset_path, output_base_dir):
    """
    Performs rollout evaluation for a given model and dataset.
    """
    model_name = os.path.basename(model_path)
    dataset_name = os.path.basename(dataset_path)
    output_dir = os.path.join(output_base_dir, dataset_name, model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Evaluating Model: {model_name} on Dataset: {dataset_name} ---")
    print(f"Outputting results to: {output_dir}")

    # --- Model Loading ---
    print("Loading model...")
    model = DNN(N_in=N_in, N_out=N_out, N_h=N_h, N_layer=N_layer, activation=activation).to(device)
    try:
        # Use weights_only=True for security
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return # Skip evaluation if model loading fails

    # --- Data Loading ---
    print("Loading data...")
    try:
        dataset = TrajectoryDataset(dataset_path)
        # Use a DataLoader for batching, though we might only use the first batch/trajectory
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) # Evaluate one trajectory at a time
        X_0, U_0, t_coll_0, time_0 = next(iter(dataloader)) # Get the first trajectory
        print(f"Data loaded. Trajectory shape: X={X_0.shape}, U={U_0.shape}")
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return # Skip evaluation if data loading fails

    # --- Rollout Prediction ---
    print("Performing rollout prediction...")
    N_batch, N_seq, N_x_dim = X_0.shape
    dt = dataset.time[0, 1, 0] - dataset.time[0, 0, 0] # Infer dt from data
    dt = dt.item() # Convert to float
    t = dataset.time[0, :, 0].cpu().numpy() # Time vector for plotting

    X_0_preds_rollout = torch.zeros_like(X_0).to(device)
    X_current_rollout = X_0[:, 0, :].unsqueeze(1).to(device) # Start with initial state

    rollout_steps = N_seq - 1
    print(f"Rollout steps: {rollout_steps}, dt: {dt:.4f}")
    for i in range(rollout_steps):
        U_step = U_0[:, i, :].unsqueeze(1).to(device)
        time_step = time_0[:, i, :].unsqueeze(1).to(device) # Use time from loaded data

        Z_step, N_batch_step, N_seq_step, N_x_step = convert_input_data(X_current_rollout, U_step, time_step)
        with torch.no_grad(): # Disable gradient calculation for evaluation
             Z_hat_step = model(Z_step)

        X_next_rollout = convert_output_data(Z_hat_step, N_batch_step, N_seq_step, N_x_step)
        X_0_preds_rollout[:, i+1, :] = X_next_rollout.squeeze(1)
        X_current_rollout = X_next_rollout

    # Detach for plotting
    X_0_np = X_0.cpu().numpy()
    X_0_preds_rollout_np = X_0_preds_rollout.detach().cpu().numpy()

    # Extract states for plotting (using the first trajectory in the batch, index 0)
    plot_idx = 0
    x_gt = X_0_np[plot_idx, :, 0]
    y_gt = X_0_np[plot_idx, :, 1]
    z_gt = X_0_np[plot_idx, :, 2]
    x_rollout = X_0_preds_rollout_np[plot_idx, :, 0]
    y_rollout = X_0_preds_rollout_np[plot_idx, :, 1]
    z_rollout = X_0_preds_rollout_np[plot_idx, :, 2]

    # --- Rollout Prediction Plots ---
    print("Plotting rollout prediction results...")
    # 2D Projections
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6)) # Wider figure

    ax0.plot(x_gt[1:], y_gt[1:], label="Ground Truth", marker='.', linestyle='-', markersize=4)
    ax0.plot(x_rollout[1:], y_rollout[1:], label="Rollout Pred.", marker='x', linestyle='--', markersize=4)
    ax0.scatter(x_gt[0], y_gt[0], label="Start", c='g', s=50, zorder=5)
    ax0.set_xlabel("x [m]")
    ax0.set_ylabel("y [m]")
    ax0.set_title("XY Projection")
    rescale_ax(ax0)
    ax0.grid(True)

    ax1.plot(x_gt[1:], z_gt[1:], label="Ground Truth", marker='.', linestyle='-', markersize=4)
    ax1.plot(x_rollout[1:], z_rollout[1:], label="Rollout Pred.", marker='x', linestyle='--', markersize=4)
    ax1.scatter(x_gt[0], z_gt[0], label="Start", c='g', s=50, zorder=5)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("z [m]")
    ax1.set_title("XZ Projection")
    rescale_ax(ax1)
    ax1.grid(True)

    ax2.plot(y_gt[1:], z_gt[1:], label="Ground Truth", marker='.', linestyle='-', markersize=4)
    ax2.plot(y_rollout[1:], z_rollout[1:], label="Rollout Pred.", marker='x', linestyle='--', markersize=4)
    ax2.scatter(y_gt[0], z_gt[0], label="Start", c='g', s=50, zorder=5)
    ax2.set_xlabel("y [m]")
    ax2.set_ylabel("z [m]")
    ax2.set_title("YZ Projection")
    rescale_ax(ax2)
    ax2.grid(True)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05)) # Adjust legend position
    fig.suptitle(f"Rollout: {model_name} on {dataset_name} - Projections", y=1.02) # Adjust title position
    plt.tight_layout(rect=[0, 0.05, 1, 0.98]) # Adjust layout
    plt.savefig(os.path.join(output_dir, "rollout_projections.pdf"))
    plt.close(fig)

    # 3D Plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.plot(x_gt[1:], y_gt[1:], z_gt[1:], label="Ground Truth", marker='.', linestyle='-', markersize=4)
    ax.plot(x_rollout[1:], y_rollout[1:], z_rollout[1:], label="Rollout Pred.", marker='x', linestyle='--', markersize=4)
    ax.scatter(x_gt[0], y_gt[0], z_gt[0], label="Start", c='g', s=50, zorder=5)
    ax.legend()
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_zlabel("$z$ [m]")
    ax.set_title(f"Rollout: {model_name} on {dataset_name} - 3D Trajectory")
    ax.set_box_aspect([np.ptp(x_gt), np.ptp(y_gt), np.ptp(z_gt)]) # Aspect ratio based on data range
    # rescale_ax(ax) # Rescale might not be ideal for 3D
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rollout_trajectory_3d.pdf"))
    plt.close(fig)

    # Position Error Norm Plot
    t_rollout_plot = t[1:] # Time vector for plotting rollout results
    e_pos = np.sqrt((x_gt[1:] - x_rollout[1:])**2 + (y_gt[1:] - y_rollout[1:])**2 + (z_gt[1:] - z_rollout[1:])**2)
    plt.figure(figsize=(8, 4))
    plt.plot(t_rollout_plot, e_pos)
    plt.xlabel("Time (s)")
    plt.ylabel("$||\mathbf{e}_{pos}||_2$ [m]")
    plt.title(f"Rollout: {model_name} on {dataset_name} - Position Error Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rollout_pos_error_norm.pdf"))
    plt.close()

    # --- IVP Calculation ---
    print("Calculating Interval of Valid Prediction (IVP)...")
    ivp_sum = 0.0
    e_thr = 0.05 # Threshold for valid prediction
    # Use torch tensors for calculation
    X_0_tensor = X_0.to(device)
    X_0_preds_rollout_tensor = X_0_preds_rollout.to(device)

    # Calculate for the single trajectory in the batch
    error_norm_n = torch.sqrt(
        (X_0_tensor[0, 1:, 0] - X_0_preds_rollout_tensor[0, 1:, 0])**2 +
        (X_0_tensor[0, 1:, 1] - X_0_preds_rollout_tensor[0, 1:, 1])**2 +
        (X_0_tensor[0, 1:, 2] - X_0_preds_rollout_tensor[0, 1:, 2])**2
    )
    valid_indices = torch.nonzero(error_norm_n <= e_thr)
    ivp = torch.tensor(0.0) # Default to 0
    if len(valid_indices) > 0:
        last_valid_index = valid_indices[-1][0]
        ivp = dt * (last_valid_index + 1) # +1 because index is 0-based

    print(f"IVP (e_thr={e_thr}): {ivp.item():.4f} s")

    # --- Save Numerical Results ---
    results_summary_path = os.path.join(output_dir, "summary_results.txt")
    with open(results_summary_path, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Average IVP (e_thr={e_thr}): {ivp.item():.4f} s\n")
        # Add other metrics if calculated (e.g., final position error)
        final_pos_error = e_pos[-1] if len(e_pos) > 0 else float('nan')
        f.write(f"Final Position Error: {final_pos_error:.4f} m\n")
    print(f"Numerical results saved to {results_summary_path}")
    print(f"--- Finished evaluation for {model_name} on {dataset_name} ---")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple models on multiple datasets.")
    parser.add_argument('--models', nargs='+', default=[
        "models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_0",
        "models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_50",
        "models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_993"
        ], help='List of model paths to evaluate.')
    parser.add_argument('--datasets', nargs='+', default=[
        "test_set_line",
        "test_set_circle",
        "test_set_figure8"
        ], help='List of dataset directories to evaluate on.')
    parser.add_argument('--output_dir', type=str, default="results/multi_rollout_results",
                        help='Base directory to save evaluation results.')

    args = parser.parse_args()

    print("Starting multi-rollout evaluation...")
    print(f"Models to evaluate: {args.models}")
    print(f"Datasets to evaluate on: {args.datasets}")
    print(f"Output directory: {args.output_dir}")

    for model_p in args.models:
        if not os.path.exists(model_p):
            print(f"Warning: Model file not found: {model_p}. Skipping.")
            continue
        for dataset_p in args.datasets:
            if not os.path.isdir(dataset_p):
                 print(f"Warning: Dataset directory not found: {dataset_p}. Skipping.")
                 continue
            evaluate_rollout(model_p, dataset_p, args.output_dir)

    print("\n--- Multi-rollout evaluation script finished. ---")
