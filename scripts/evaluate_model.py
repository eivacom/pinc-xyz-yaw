import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse # Added argparse for command-line arguments

# Assuming these imports exist and are correctly structured
# If errors occur, we may need to adjust paths based on execution context
# Using absolute imports relative to the project root
from models.model_utility import get_data_sets, convert_input_data, convert_output_data, rollout_loss_fn, physics_loss_fn
# Removed the unnecessary import of BlueROVTorch and the relative import attempt

# Updated default model path and added output_dir argument
def evaluate_model(model_path="models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2",
                   data_size=1000,
                   plot_results=False,
                   save_plots=True, # Default to saving plots now
                   output_dir="results/plots"):
    """
    Loads a pre-trained model and evaluates it on a development dataset.
    Optionally plots results and saves them to a specified directory.
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

    # Disable LaTeX rendering to avoid dependency issues
    plt.rc('text', usetex=False)
    # plt.rc('text', usetex=True) # Original line
    plt.rcParams.update({'font.size': 14})

    # Create output directory if it doesn't exist
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load Model
    try:
        model = torch.load(model_path, map_location=device)
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Data
    try:
        _, dev_dataloader, _, _ = get_data_sets(data_size)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Get a batch from the development set
    try:
        X_0, U_0, t_coll_0, time_0 = next(iter(dev_dataloader))
    except StopIteration:
        print("Error: Development dataloader is empty.")
        return
    except Exception as e:
        print(f"Error getting data batch: {e}")
        return

    # Prepare data and run model inference
    X_0, U_0, t_coll_0, time_0 = X_0.to(device), U_0.to(device), t_coll_0.to(device), time_0.to(device)
    Z_coll, N_batch, N_seq, N_x = convert_input_data(X_0, U_0, time_0)

    with torch.no_grad(): # Disable gradient calculations for inference
        X_hat_flat = model(Z_coll)

    N_total = Z_coll.shape[0]
    N_in = Z_coll.shape[1]
    N_out = X_hat_flat.shape[1]
    N_u = U_0.shape[-1]

    X_hat = X_hat_flat.view(N_batch, N_seq, N_x)
    X_hat_cpu = X_hat.detach().cpu().numpy()
    X_0_cpu = X_0.detach().cpu().numpy()
    U_0_cpu = U_0.detach().cpu().numpy()
    time_0_cpu = time_0.detach().cpu().numpy()

    # Calculate Metrics
    mse = ((X_0_cpu[:, 1:] - X_hat_cpu[:, :-1])**2).mean()
    print(f"One-step prediction MSE: {mse:.6f}")

    try:
        with torch.no_grad():
            rollout_loss_val, _ = rollout_loss_fn(model, X_0, U_0, time_0, 10, device, t_coll_0, False, 0.0)
            physics_loss_val = physics_loss_fn(model, X_0, U_0, t_coll_0, device, 0.0)
        print(f"Rollout Loss (10 steps, log10): {(100*torch.log10(rollout_loss_val)).round()/100:.2f}")
        print(f"Physics Loss (log10): {(100*torch.log10(physics_loss_val)).round()/100:.2f}")
    except Exception as e:
        print(f"Error calculating custom losses: {e}")


    # --- Plotting (Optional) ---
    if plot_results or save_plots:
        print("Generating plots...")
        dt = time_0_cpu[0, 0, 0] # Assuming constant dt
        t = np.linspace(0, dt * N_seq, N_seq)

        # Plot Inputs
        plt.figure()
        plt.plot(np.arange(U_0_cpu[0, :, 0].shape[0]) * dt, U_0_cpu[0, :, 0])
        plt.plot(np.arange(U_0_cpu[0, :, 1].shape[0]) * dt, U_0_cpu[0, :, 1])
        plt.plot(np.arange(U_0_cpu[0, :, 2].shape[0]) * dt, U_0_cpu[0, :, 2])
        plt.plot(np.arange(U_0_cpu[0, :, 3].shape[0]) * dt, U_0_cpu[0, :, 3])
        plt.legend(["$X$", "$Y$", "$Z$", "$M_z$"])
        plt.xlabel("time (s)")
        plt.title("Input trajectories: development set sample")
        plt.xlim([0, dt * N_seq])
        plt.tight_layout()
        if save_plots:
            save_path = os.path.join(output_dir, "input_development.pdf")
            plt.savefig(save_path)
            print(f"Saved {save_path}")

        # Plot Trajectories (2D projections and 3D)
        plot_trajectories(X_0_cpu[0], X_hat_cpu[0], save_plots, output_dir)

        # Plot MSE per state over time
        plot_mse_over_time(t, X_0_cpu[0], X_hat_cpu[0], save_plots, output_dir)

        if plot_results:
            plt.show()
        else:
            plt.close('all') # Close figures if only saving


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


def plot_trajectories(X_0_sample, X_hat_sample, save_plots=False, output_dir="."):
    """Plots 2D and 3D trajectories for ground truth and prediction."""
    fig_2d, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

    # X-Y Plane
    ax0.plot(X_0_sample[1:, 0], X_0_sample[1:, 1], label='Ground Truth')
    ax0.plot(X_hat_sample[:-1, 0], X_hat_sample[:-1, 1], label='Prediction', linestyle='--')
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    rescale_ax(ax0)
    ax0.legend()

    # X-Z Plane
    ax1.plot(X_0_sample[1:, 0], X_0_sample[1:, 2])
    ax1.plot(X_hat_sample[:-1, 0], X_hat_sample[:-1, 2], linestyle='--')
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    rescale_ax(ax1)

    # Y-Z Plane
    ax2.plot(X_0_sample[1:, 1], X_0_sample[1:, 2])
    ax2.plot(X_hat_sample[:-1, 1], X_hat_sample[:-1, 2], linestyle='--')
    ax2.set_xlabel("y")
    ax2.set_ylabel("z")
    rescale_ax(ax2)

    fig_2d.suptitle("Trajectory Projections (Ground Truth vs. Prediction)")
    fig_2d.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # 3D Plot
    fig_3d = plt.figure(figsize=(8, 8))
    ax_3d = fig_3d.add_subplot(projection='3d')
    ax_3d.plot(X_0_sample[1:, 0], X_0_sample[1:, 1], X_0_sample[1:, 2], label='Ground Truth')
    ax_3d.plot(X_hat_sample[:-1, 0], X_hat_sample[:-1, 1], X_hat_sample[:-1, 2], label='Prediction', linestyle='--')
    ax_3d.scatter(X_0_sample[0, 0], X_0_sample[0, 1], X_0_sample[0, 2], c='g', marker='o', s=50, label='Start')
    ax_3d.scatter(X_0_sample[-1, 0], X_0_sample[-1, 1], X_0_sample[-1, 2], c='r', marker='x', s=50, label='End (GT)')
    ax_3d.set_xlabel("$x$ [m]")
    ax_3d.set_ylabel("$y$ [m]")
    ax_3d.set_zlabel("$z$ [m]")
    ax_3d.legend()
    ax_3d.set_title("3D Trajectory (Ground Truth vs. Prediction)")
    # Attempt to set aspect ratio after plotting
    try:
        ax_3d.set_box_aspect([np.ptp(ax_3d.get_xlim()), np.ptp(ax_3d.get_ylim()), np.ptp(ax_3d.get_zlim())])
        rescale_ax(ax_3d) # Rescale after setting aspect
    except AttributeError:
        print("Warning: Could not set 3D aspect ratio automatically. Manual adjustment might be needed.")


    if save_plots:
        fig_2d.savefig(os.path.join(output_dir, "trajectory_projections.pdf"))
        fig_3d.savefig(os.path.join(output_dir, "trajectory_3d.pdf"))
        print(f"Saved trajectory plots to {output_dir}")


def plot_mse_over_time(t, X_0_sample, X_hat_sample, save_plots=False, output_dir="."):
    """Plots the Mean Squared Error for each state variable over time."""
    t_eval = t[:-1] # Time steps for comparison

    # Position MSE
    plt.figure()
    plt.plot(t_eval, (X_0_sample[1:, 0] - X_hat_sample[:-1, 0])**2, label='x')
    plt.plot(t_eval, (X_0_sample[1:, 1] - X_hat_sample[:-1, 1])**2, label='y')
    plt.plot(t_eval, (X_0_sample[1:, 2] - X_hat_sample[:-1, 2])**2, label='z')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position MSE")
    plt.title("MSE for Position States")
    plt.grid(True)
    if save_plots:
        plt.savefig(os.path.join(output_dir, "mse_position.pdf"))

    # Orientation (Psi) MSE
    cos_psi = X_0_sample[1:, 3]
    sin_psi = X_0_sample[1:, 4]
    cos_psi_hat = X_hat_sample[:-1, 3]
    sin_psi_hat = X_hat_sample[:-1, 4]
    psi = np.arctan2(sin_psi, cos_psi)
    psi_hat = np.arctan2(sin_psi_hat, cos_psi_hat)
    # Handle angle wrapping issues for MSE calculation if necessary
    psi_diff = psi - psi_hat
    psi_mse = np.minimum(psi_diff**2, (psi_diff + 2*np.pi)**2)
    psi_mse = np.minimum(psi_mse, (psi_diff - 2*np.pi)**2)

    plt.figure()
    plt.plot(t_eval, psi_mse)
    plt.xlabel("Time (s)")
    plt.ylabel("Psi MSE")
    plt.title("MSE for Orientation (Psi)")
    plt.grid(True)
    if save_plots:
        plt.savefig(os.path.join(output_dir, "mse_psi.pdf"))

    # Linear Velocity MSE
    plt.figure()
    plt.plot(t_eval, (X_0_sample[1:, 5] - X_hat_sample[:-1, 5])**2, label='u')
    plt.plot(t_eval, (X_0_sample[1:, 6] - X_hat_sample[:-1, 6])**2, label='v')
    plt.plot(t_eval, (X_0_sample[1:, 7] - X_hat_sample[:-1, 7])**2, label='w')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Linear Velocity MSE")
    plt.title("MSE for Linear Velocity States")
    plt.grid(True)
    if save_plots:
        plt.savefig(os.path.join(output_dir, "mse_linear_velocity.pdf"))

    # Angular Velocity MSE
    plt.figure()
    plt.plot(t_eval, (X_0_sample[1:, 8] - X_hat_sample[:-1, 8])**2, label='r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity MSE")
    plt.title("MSE for Angular Velocity State (r)")
    plt.grid(True)
    if save_plots:
        plt.savefig(os.path.join(output_dir, "mse_angular_velocity.pdf"))
        print(f"Saved MSE plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained BlueROV model.")
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
        default=1000,
        help="Size of the dataset to use for evaluation."
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively instead of just saving."
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_size=args.data_size,
        plot_results=args.show_plots,
        save_plots=not args.show_plots, # Save if not showing interactively
        output_dir=args.output_dir
    )

    print("\nEvaluation script finished.")
    # Note: Long-range prediction/rollout code from the notebook is not included here
    # as its purpose and integration were less clear for a standard evaluation script.
    # It could be added as a separate function if needed.
