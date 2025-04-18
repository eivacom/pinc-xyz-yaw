import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Using absolute imports relative to the project root
from models.model_utility import get_data_sets, convert_input_data, convert_output_data, DNN # Import DNN definition
from data.data_utility import TrajectoryDataset # Import dataset class

def rescale_ax(ax):
    """Rescales axes to be equal."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_scale = abs(x_max - x_min)
    y_scale = abs(y_max - y_min)

    if hasattr(ax, 'get_zlim'): # Handle 3D axes
        z_min, z_max = ax.get_zlim()
        z_scale = abs(z_max - z_min)
        max_scale = max(x_scale, y_scale, z_scale)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        ax.set_xlim([x_center - max_scale / 2, x_center + max_scale / 2])
        ax.set_ylim([y_center - max_scale / 2, y_center + max_scale / 2])
        ax.set_zlim([z_center - max_scale / 2, z_center + max_scale / 2])

    else: # Handle 2D axes
        if x_scale > y_scale:
            y_center = (y_min + y_max) / 2
            y_new_half = x_scale / 2
            ax.set_ylim([y_center - y_new_half, y_center + y_new_half])
        else:
            x_center = (x_min + x_max) / 2
            x_new_half = y_scale / 2
            ax.set_xlim([x_center - x_new_half, x_center + x_new_half])


def analyze_rollout(model_path, data_size=100, rollout_steps=65, output_dir="results/plots"):
    """
    Performs multi-step rollout prediction and compares with ground truth.
    """
    sns.set()
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

    # Disable LaTeX rendering
    plt.rc('text', usetex=False)
    plt.rcParams.update({'font.size': 14})

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load Model
    try:
        # Need to load the state dict into an instance of the DNN class
        # First, determine model parameters (assuming they are consistent with training script)
        # These might need adjustment if the saved model differs significantly
        N_x = 9 # Number of states
        N_u = 4 # Number of controls
        N_in = N_x + N_u + 1 # Input size: state + control + time
        N_h = [32, 32, 32, 32] # Hidden layers
        N_layer = len(N_h)
        from torch.nn import Softplus # Assuming Softplus was the activation
        activation = Softplus # Assuming Softplus was the activation

        # Load the entire model object directly
        model = torch.load(model_path, map_location=device)
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    # Load Data (using dev set for consistency with notebook)
    try:
        _, dev_dataloader, _, _ = get_data_sets(data_size, dev_path='dev_set') # Ensure using dev_set
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Get a batch from the development set
    try:
        X_0_batch, U_0_batch, _, time_0_batch = next(iter(dev_dataloader))
        # Use only the first trajectory in the batch for plotting, like in the notebook
        X_0 = X_0_batch[0].unsqueeze(0).to(device) # Keep batch dim for model input
        U_0 = U_0_batch[0].unsqueeze(0).to(device)
        time_0 = time_0_batch[0].unsqueeze(0).to(device)
        dt = time_0[0, 0, 0].item() # Get dt from the data
        print(f"Using dt: {dt}")
    except StopIteration:
        print("Error: Development dataloader is empty.")
        return None, None
    except Exception as e:
        print(f"Error getting data batch: {e}")
        return None, None

    # Perform Rollout Prediction
    N_seq = X_0.shape[1]
    actual_rollout_steps = min(rollout_steps, N_seq - 1) # Ensure rollout doesn't exceed sequence length
    print(f"Performing rollout for {actual_rollout_steps} steps...")

    X_0_preds = torch.zeros(1, actual_rollout_steps + 1, N_x).to(device) # Store initial state + predictions
    X_hat = X_0[:, 0, :].unsqueeze(1) # Start with the initial state (t=0)
    X_0_preds[:, 0, :] = X_hat.squeeze(1)

    with torch.no_grad():
        for i in range(actual_rollout_steps):
            # Prepare input for the current step
            # Use predicted state X_hat, control U_0 at step i, time_0 at step i
            current_U = U_0[:, i, :].unsqueeze(1)
            current_time = time_0[:, i, :].unsqueeze(1) # Time relative to the start of the step? Or absolute time? Assuming relative dt.
            # The model expects time input, let's use dt for each step prediction
            step_time = torch.full_like(current_time, dt)

            Z_step, N_batch_step, N_seq_step, N_x_step = convert_input_data(X_hat, current_U, step_time)
            Z_hat_step = model(Z_step)
            X_hat = convert_output_data(Z_hat_step, N_batch_step, N_seq_step, N_x_step)
            X_0_preds[:, i+1, :] = X_hat.squeeze(1) # Store the prediction for the next state

    # Convert to numpy for plotting
    X_0_cpu = X_0[0].cpu().numpy() # Ground truth for the first trajectory
    X_0_preds_cpu = X_0_preds[0].cpu().numpy() # Predictions for the first trajectory

    # --- Plotting ---
    print("Generating rollout plots...")
    t_rollout = np.linspace(0, actual_rollout_steps * dt, actual_rollout_steps + 1)

    # Plot XY
    fig_xy, ax_xy = plt.subplots()
    ax_xy.plot(X_0_cpu[1:actual_rollout_steps+1, 0], X_0_cpu[1:actual_rollout_steps+1, 1], label='Ground Truth', marker='o', linestyle='-')
    ax_xy.plot(X_0_preds_cpu[1:, 0], X_0_preds_cpu[1:, 1], label='Prediction (Rollout)', marker='x', linestyle='--')
    ax_xy.scatter(X_0_cpu[0, 0], X_0_cpu[0, 1], label='Initial Point', c='g', s=50, zorder=5)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.legend()
    ax_xy.set_title(f"XY Rollout ({actual_rollout_steps} steps)")
    rescale_ax(ax_xy)
    fig_xy.savefig(os.path.join(output_dir, "rollout_xy.pdf"))

    # Plot XZ
    fig_xz, ax_xz = plt.subplots()
    ax_xz.plot(X_0_cpu[1:actual_rollout_steps+1, 0], X_0_cpu[1:actual_rollout_steps+1, 2], label='Ground Truth', marker='o', linestyle='-')
    ax_xz.plot(X_0_preds_cpu[1:, 0], X_0_preds_cpu[1:, 2], label='Prediction (Rollout)', marker='x', linestyle='--')
    ax_xz.scatter(X_0_cpu[0, 0], X_0_cpu[0, 2], label='Initial Point', c='g', s=50, zorder=5)
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.legend()
    ax_xz.set_title(f"XZ Rollout ({actual_rollout_steps} steps)")
    rescale_ax(ax_xz)
    fig_xz.savefig(os.path.join(output_dir, "rollout_xz.pdf"))

    # Plot YZ
    fig_yz, ax_yz = plt.subplots()
    ax_yz.plot(X_0_cpu[1:actual_rollout_steps+1, 1], X_0_cpu[1:actual_rollout_steps+1, 2], label='Ground Truth', marker='o', linestyle='-')
    ax_yz.plot(X_0_preds_cpu[1:, 1], X_0_preds_cpu[1:, 2], label='Prediction (Rollout)', marker='x', linestyle='--')
    ax_yz.scatter(X_0_cpu[0, 1], X_0_cpu[0, 2], label='Initial Point', c='g', s=50, zorder=5)
    ax_yz.set_xlabel("y [m]")
    ax_yz.set_ylabel("z [m]")
    ax_yz.legend()
    ax_yz.set_title(f"YZ Rollout ({actual_rollout_steps} steps)")
    rescale_ax(ax_yz)
    fig_yz.savefig(os.path.join(output_dir, "rollout_yz.pdf"))

    # Plot Psi (Orientation)
    cos_psi_gt = X_0_cpu[:, 3]
    sin_psi_gt = X_0_cpu[:, 4]
    psi_gt = np.arctan2(sin_psi_gt, cos_psi_gt)

    cos_psi_pred = X_0_preds_cpu[:, 3]
    sin_psi_pred = X_0_preds_cpu[:, 4]
    psi_pred = np.arctan2(sin_psi_pred, cos_psi_pred)

    fig_psi, ax_psi = plt.subplots()
    ax_psi.plot(t_rollout[1:], psi_gt[1:actual_rollout_steps+1], label='Ground Truth', marker='o', linestyle='-')
    ax_psi.plot(t_rollout[1:], psi_pred[1:], label='Prediction (Rollout)', marker='x', linestyle='--')
    ax_psi.scatter(t_rollout[0], psi_gt[0], label='Initial Point', c='g', s=50, zorder=5) # Use psi_gt[0] for initial point
    ax_psi.set_xlabel("Time (s)")
    ax_psi.set_ylabel("Psi (rad)")
    ax_psi.legend()
    ax_psi.set_title(f"Psi Rollout ({actual_rollout_steps} steps)")
    fig_psi.savefig(os.path.join(output_dir, "rollout_psi.pdf"))

    # Plot Velocities (u, v, w, r)
    vel_labels = ['u', 'v', 'w', 'r']
    vel_indices = [5, 6, 7, 8]
    for label, idx in zip(vel_labels, vel_indices):
        fig_vel, ax_vel = plt.subplots()
        ax_vel.plot(t_rollout[1:], X_0_cpu[1:actual_rollout_steps+1, idx], label='Ground Truth', marker='o', linestyle='-')
        ax_vel.plot(t_rollout[1:], X_0_preds_cpu[1:, idx], label='Prediction (Rollout)', marker='x', linestyle='--')
        ax_vel.scatter(t_rollout[0], X_0_cpu[0, idx], label='Initial Point', c='g', s=50, zorder=5)
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel(f"{label} (m/s or rad/s)")
        ax_vel.legend()
        ax_vel.set_title(f"{label} Rollout ({actual_rollout_steps} steps)")
        fig_vel.savefig(os.path.join(output_dir, f"rollout_{label}.pdf"))

    # Plot Position Error (Euclidean Distance)
    pos_error = np.sqrt(np.sum((X_0_cpu[1:actual_rollout_steps+1, :3] - X_0_preds_cpu[1:, :3])**2, axis=1))
    fig_err, ax_err = plt.subplots()
    ax_err.plot(t_rollout[1:], pos_error)
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error ||e_pos||_2 [m]")
    ax_err.set_title(f"Rollout Position Error ({actual_rollout_steps} steps)")
    ax_err.grid(True)
    fig_err.savefig(os.path.join(output_dir, "rollout_pos_error.pdf"))

    print(f"Saved rollout plots to {output_dir}")
    plt.close('all')

    # Calculate IVP metric (from notebook)
    e_thr = 0.05
    ivp_sum = 0.0
    # Need to run rollout for the whole batch to calculate average IVP
    # Re-running rollout for the batch (less efficient, but simpler for now)
    print("Calculating IVP metric for the batch...")
    # Reload the batch data to ensure t_coll_batch is defined in this scope
    try:
        X_0_batch, U_0_batch, t_coll_batch, time_0_batch = next(iter(dev_dataloader))
    except StopIteration:
        print("Error: Could not get another batch for IVP calculation.")
        return ivp_metric, actual_rollout_steps # Return previously calculated results if any

    # Move data to device
    X_0_batch, U_0_batch, t_coll_batch, time_0_batch = X_0_batch.to(device), U_0_batch.to(device), t_coll_batch.to(device), time_0_batch.to(device)
    N_batch_full = X_0_batch.shape[0]
    X_batch_preds = torch.zeros(N_batch_full, actual_rollout_steps + 1, N_x).to(device)
    X_hat_batch = X_0_batch[:, 0, :].unsqueeze(1)
    X_batch_preds[:, 0, :] = X_hat_batch.squeeze(1)

    with torch.no_grad():
        for i in range(actual_rollout_steps):
            current_U_batch = U_0_batch[:, i, :].unsqueeze(1)
            step_time_batch = torch.full_like(time_0_batch[:, i, :].unsqueeze(1), dt)
            Z_step_batch, N_b, N_s, N_x_s = convert_input_data(X_hat_batch, current_U_batch, step_time_batch)
            Z_hat_step_batch = model(Z_step_batch)
            X_hat_batch = convert_output_data(Z_hat_step_batch, N_b, N_s, N_x_s)
            X_batch_preds[:, i+1, :] = X_hat_batch.squeeze(1)

    # Calculate error for the batch
    pos_error_batch = torch.sqrt(torch.sum((X_0_batch[:, 1:actual_rollout_steps+1, :3] - X_batch_preds[:, 1:, :3])**2, dim=2))
    for n in range(N_batch_full):
        # Find the last index where error is below threshold
        valid_indices = torch.nonzero(pos_error_batch[n] <= e_thr)
        if len(valid_indices) > 0:
            last_valid_step_index = valid_indices[-1][0] # Index relative to steps 1 to actual_rollout_steps
            ivp_sum += dt * (last_valid_step_index + 1) # +1 because index is 0-based
        # If error is always above threshold, contribution is 0

    ivp_metric = ivp_sum / N_batch_full if N_batch_full > 0 else 0.0
    print(f"Average Interval of Validity Prediction (IVP) @ {e_thr}m error: {ivp_metric:.4f} seconds")


    return ivp_metric, actual_rollout_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze multi-step rollout performance of a trained BlueROV model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2",
        help="Path to the trained model file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Directory to save the output plots."
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=100, # Use a smaller batch for faster IVP calculation
        help="Batch size of the dataset to use for IVP calculation."
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=65, # Match notebook
        help="Number of steps for rollout prediction."
    )

    args = parser.parse_args()

    analyze_rollout(
        model_path=args.model_path,
        data_size=args.data_size,
        rollout_steps=args.rollout_steps,
        output_dir=args.output_dir
    )

    print("\nRollout analysis script finished.")
