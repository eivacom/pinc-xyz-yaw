#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib.pyplot as plt
import numpy as np
import os # Added import

import seaborn as sns
sns.set()

from models.model_utility import DNN, Softplus, get_data_sets, convert_input_data, convert_output_data, rollout_loss_fn, physics_loss_fn # Import specific classes/functions
from src.bluerov_torch import*

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

# Define and create output directory
output_dir = "results/paper_results"
os.makedirs(output_dir, exist_ok=True)
print(f"Ensured output directory exists: {output_dir}")

# --- Model Definition and Loading ---
print("Defining and loading model...")
# Define model parameters based on training script
N_in = 14 # 9 states + 4 controls + 1 time
N_out = 9 # 9 states
N_h = [32, 32, 32, 32]
N_layer = len(N_h)
activation = Softplus

# Instantiate the model
model = DNN(N_in=N_in, N_out=N_out, N_h=N_h, N_layer=N_layer, activation=activation).to(device)

# Load the saved state dictionary
model_path = "models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_993"
print(f"Loading state dict from: {model_path}")
# Use weights_only=True for security if loading state_dict, though it might be less critical here
# as we are loading into a defined structure. Let's keep it for now.
state_dict = torch.load(model_path, map_location=device, weights_only=True) 
model.load_state_dict(state_dict)
model.eval() # Set model to evaluation mode
print("Model loaded successfully.")

# --- Data Loading ---
print("Loading data...")
train_dataloader, dev_dataloader, _, _ = get_data_sets(1000) # Batch size might differ from training, adjust if needed
print("Data loaded.")

# --- Single-Step Prediction ---
print("Performing single-step prediction...")
X_0, U_0, t_coll_0, time_0 = next(iter(dev_dataloader))
Z_coll, N_batch, N_seq, N_x = convert_input_data(X_0.to(device), U_0.to(device), time_0.to(device))
X_hat_single_step = model(Z_coll) # Renamed for clarity

# Reshape and detach
N_total = Z_coll.shape[0]
N_in = Z_coll.shape[1]
N_out = X_hat_single_step.shape[1]
N_batch = X_0.shape[0]
N_x = X_0.shape[2]
N_u = U_0.shape[-1]
N_seq = X_0.shape[1]

X_hat_single_step = X_hat_single_step.view(N_batch, N_seq, N_x)
X_hat_single_step_np = X_hat_single_step.detach().cpu().numpy() # Keep tensor for loss, use numpy for plots
X_0_np = X_0.cpu().numpy() # Use numpy for plots
U_0_np = U_0.cpu().numpy()
time_0_np = time_0.cpu().numpy()

dt = 0.08 # Assuming constant dt from context
t = np.linspace(0, dt*(N_seq-1), N_seq) # Time vector for full sequence
t_pred = t[:-1] # Time vector for predictions (N_seq-1 points)

# Plot Input Trajectories
print("Plotting input trajectories...")
plt.figure(figsize=(8, 4))
plt.plot(t * time_0_np[0,0], U_0_np[0, :, 0], label="$F_x$") # Use full time vector
plt.plot(t * time_0_np[0,0], U_0_np[0, :, 1], label="$F_y$")
plt.plot(t * time_0_np[0,0], U_0_np[0, :, 2], label="$F_z$")
plt.plot(t * time_0_np[0,0], U_0_np[0, :, 3], label="$M_z$")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Force/Moment")
plt.title("Input Trajectories (Development Set Sample)")
plt.xlim([0, dt*(N_seq-1)]) # Adjust xlim based on actual time
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "input_development.pdf"))
plt.close()

# Calculate Single-Step MSE (Example)
# Ensure both tensors are on the same device before calculation
single_step_mse = ((X_0.to(device)[:, 1:] - X_hat_single_step[:, :-1])**2).mean().item()
print(f"Overall Single-Step MSE: {single_step_mse:.6f}")

# --- Loss Calculation ---
print("Calculating losses...")
rollout_loss = (100*torch.log10(rollout_loss_fn(model, X_0.to(device), U_0.to(device), time_0.to(device), 10, device, t_coll_0.to(device), False, 0.0)[0])).round()/100
print(f"Rollout Loss (10 steps, dB): {rollout_loss.item():.2f}")

physics_loss = (100*torch.log10(physics_loss_fn(model, X_0.to(device), U_0.to(device), t_coll_0.to(device), device, 0.0))).round()/100
print(f"Physics Loss (dB): {physics_loss.item():.2f}")

# --- Plotting Helper ---
def rescale_ax(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_scale = abs(x_max - x_min)
    y_scale = abs(y_max - y_min)
    if not x_scale or not y_scale: return # Avoid division by zero if limits are the same
    
    if x_scale > y_scale:
        y_center = (y_min + y_max) / 2
        y_new_half = x_scale / 2 * (ax.get_position().height / ax.get_position().width) # Adjust for aspect ratio
        ax.set_ylim([y_center - y_new_half, y_center + y_new_half])
    else:
        x_center = (x_min + x_max) / 2
        x_new_half = y_scale / 2 * (ax.get_position().width / ax.get_position().height) # Adjust for aspect ratio
        ax.set_xlim([x_center - x_new_half, x_center + x_new_half])

# --- Single-Step Prediction Plots ---
print("Plotting single-step prediction results...")
# 2D Projections
plot_idx = 0 # Plot the first sample in the batch
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

ax0.plot(X_0_np[plot_idx, 1:, 0], X_0_np[plot_idx, 1:, 1], label="Ground Truth")
ax0.plot(X_hat_single_step_np[plot_idx, :-1, 0], X_hat_single_step_np[plot_idx, :-1, 1], label="Single-Step Pred.", linestyle='--')
ax0.set_xlabel("x [m]")
ax0.set_ylabel("y [m]")
ax0.set_title("XY Projection")
rescale_ax(ax0)
ax0.grid(True)

ax1.plot(X_0_np[plot_idx, 1:, 0], X_0_np[plot_idx, 1:, 2], label="Ground Truth")
ax1.plot(X_hat_single_step_np[plot_idx, :-1, 0], X_hat_single_step_np[plot_idx, :-1, 2], label="Single-Step Pred.", linestyle='--')
ax1.set_xlabel("x [m]")
ax1.set_ylabel("z [m]")
ax1.set_title("XZ Projection")
rescale_ax(ax1)
ax1.grid(True)

ax2.plot(X_0_np[plot_idx, 1:, 1], X_0_np[plot_idx, 1:, 2], label="Ground Truth")
ax2.plot(X_hat_single_step_np[plot_idx, :-1, 1], X_hat_single_step_np[plot_idx, :-1, 2], label="Single-Step Pred.", linestyle='--')
ax2.set_xlabel("y [m]")
ax2.set_ylabel("z [m]")
ax2.set_title("YZ Projection")
rescale_ax(ax2)
ax2.grid(True)

handles, labels = ax0.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Single-Step Prediction: Trajectory Projections")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for titles/legend
plt.savefig(os.path.join(output_dir, f"single_step_projections.pdf"))
plt.close(fig)

# 3D Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.plot(X_0_np[plot_idx, 1:, 0], X_0_np[plot_idx, 1:, 1], X_0_np[plot_idx, 1:, 2], label="Ground Truth")
ax.plot(X_hat_single_step_np[plot_idx, :-1, 0], X_hat_single_step_np[plot_idx, :-1, 1], X_hat_single_step_np[plot_idx, :-1, 2], label="Single-Step Pred.", linestyle='--')
ax.scatter(X_0_np[plot_idx, 1, 0], X_0_np[plot_idx, 1, 1], X_0_np[plot_idx, 1, 2], c='g', marker='o', s=50, label='Start GT', zorder=5)
ax.scatter(X_0_np[plot_idx, -1, 0], X_0_np[plot_idx, -1, 1], X_0_np[plot_idx, -1, 2], c='r', marker='x', s=50, label='End GT', zorder=5)
ax.set_box_aspect([1, 1, 1]) # Equal aspect ratio
ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$y$ [m]")
ax.set_zlabel("$z$ [m]")
ax.set_title("Single-Step Prediction: 3D Trajectory")
ax.legend()
# rescale_ax(ax) # Rescale might make it hard to see differences, disable for now
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "single_step_trajectory_3d.pdf"))
plt.close(fig)

# Ground Truth Trajectory Plot (for reference)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.scatter(X_0_np[plot_idx, 0, 0], X_0_np[plot_idx, 0, 1], X_0_np[plot_idx, 0, 2], c='g', marker='o', s=50, label="Start", zorder=5)
ax.plot(X_0_np[plot_idx, :, 0], X_0_np[plot_idx, :, 1], X_0_np[plot_idx, :, 2], label="Trajectory")
ax.scatter(X_0_np[plot_idx, -1, 0], X_0_np[plot_idx, -1, 1], X_0_np[plot_idx, -1, 2], c='r', marker='x', s=50, label="End", zorder=5)
ax.set_box_aspect([1, 1, 0.5]) # Match original aspect if desired
ax.legend()
ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$y$ [m]")
ax.set_zlabel("$z$ [m]")
ax.set_title("Ground Truth Trajectory Sample")
# rescale_ax(ax) # Disable rescale
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ground_truth_trajectory_3d.pdf"))
plt.close(fig)

# MSE Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True) # Share x-axis

# MSE Position
axs[0, 0].plot(t_pred, (X_0_np[plot_idx, 1:, 0] - X_hat_single_step_np[plot_idx, :-1, 0])**2, label="x")
axs[0, 0].plot(t_pred, (X_0_np[plot_idx, 1:, 1] - X_hat_single_step_np[plot_idx, :-1, 1])**2, label="y")
axs[0, 0].plot(t_pred, (X_0_np[plot_idx, 1:, 2] - X_hat_single_step_np[plot_idx, :-1, 2])**2, label="z")
axs[0, 0].set_title("MSE Position")
axs[0, 0].set_ylabel("MSE $[m^2]$")
axs[0, 0].legend()
axs[0, 0].grid(True)

# MSE Psi
cos_psi = X_0_np[plot_idx, 1:, 3]
sin_psi = X_0_np[plot_idx, 1:, 4]
cos_psi_hat = X_hat_single_step_np[plot_idx, :-1, 3]
sin_psi_hat = X_hat_single_step_np[plot_idx, :-1, 4]
psi = np.arctan2(sin_psi, cos_psi)
psi_hat = np.arctan2(sin_psi_hat, cos_psi_hat)
# Handle angle wrapping for MSE calculation
angle_diff = psi - psi_hat
angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
axs[0, 1].plot(t_pred, angle_diff**2)
axs[0, 1].set_title("MSE Psi (Yaw)")
axs[0, 1].set_ylabel("MSE $[rad^2]$")
axs[0, 1].grid(True)

# MSE Linear Velocity
axs[1, 0].plot(t_pred, (X_0_np[plot_idx, 1:, 5] - X_hat_single_step_np[plot_idx, :-1, 5])**2, label="u (surge)")
axs[1, 0].plot(t_pred, (X_0_np[plot_idx, 1:, 6] - X_hat_single_step_np[plot_idx, :-1, 6])**2, label="v (sway)")
axs[1, 0].plot(t_pred, (X_0_np[plot_idx, 1:, 7] - X_hat_single_step_np[plot_idx, :-1, 7])**2, label="w (heave)")
axs[1, 0].set_title("MSE Linear Velocity")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("MSE $[(m/s)^2]$")
axs[1, 0].legend()
axs[1, 0].grid(True)

# MSE Angular Velocity
axs[1, 1].plot(t_pred, (X_0_np[plot_idx, 1:, 8] - X_hat_single_step_np[plot_idx, :-1, 8])**2, label="r (yaw rate)")
axs[1, 1].set_title("MSE Angular Velocity")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("MSE $[(rad/s)^2]$")
axs[1, 1].legend()
axs[1, 1].grid(True)

fig.suptitle("Single-Step Prediction: Mean Squared Error (MSE)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, "single_step_mse.pdf"))
plt.close(fig)

# --- Rollout Prediction ---
print("Performing rollout prediction...")
# Use the same initial batch X_0, U_0, t_coll_0, time_0 or get a new one if desired
# X_0, U_0, t_coll_0, time_0 = next(iter(dev_dataloader)) # Optional: get new batch

X_0_preds_rollout = torch.zeros_like(X_0).to(device)
X_current_rollout = X_0[:, 0, :].unsqueeze(1).to(device) # Start with initial state

rollout_steps = N_seq - 1 # Rollout for the length of the sequence
print(f"Rollout steps: {rollout_steps}")
for i in range(rollout_steps):
    # Prepare input for this step
    U_step = U_0[:, i, :].unsqueeze(1).to(device)
    time_step = time_0[:, i, :].unsqueeze(1).to(device) # Assuming time diff is needed per step
    
    # Convert input and predict
    Z_step, N_batch_step, N_seq_step, N_x_step = convert_input_data(X_current_rollout, U_step, time_step)
    Z_hat_step = model(Z_step)
    
    # Convert output and store prediction
    X_next_rollout = convert_output_data(Z_hat_step, N_batch_step, N_seq_step, N_x_step)
    X_0_preds_rollout[:, i+1, :] = X_next_rollout.squeeze(1) # Store prediction for the *next* state
    
    # Update current state for next iteration
    X_current_rollout = X_next_rollout

# Detach rollout predictions for plotting
X_0_preds_rollout_np = X_0_preds_rollout.detach().cpu().numpy()

# Extract states for plotting (using the same plot_idx as before)
x_gt = X_0_np[plot_idx, :, 0]
y_gt = X_0_np[plot_idx, :, 1]
z_gt = X_0_np[plot_idx, :, 2]
x_rollout = X_0_preds_rollout_np[plot_idx, :, 0]
y_rollout = X_0_preds_rollout_np[plot_idx, :, 1]
z_rollout = X_0_preds_rollout_np[plot_idx, :, 2]

# --- Rollout Prediction Plots ---
print("Plotting rollout prediction results...")
# 2D Projections
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

ax0.plot(x_gt[1:], y_gt[1:], label="Ground Truth", marker='.', linestyle='-', markersize=4)
ax0.plot(x_rollout[1:], y_rollout[1:], label="Rollout Pred.", marker='x', linestyle='--', markersize=4)
ax0.scatter(x_gt[0], y_gt[0], label="Initial Point", c='g', s=50, zorder=5)
ax0.set_xlabel("x [m]")
ax0.set_ylabel("y [m]")
ax0.set_title("XY Projection")
rescale_ax(ax0)
ax0.grid(True)

ax1.plot(x_gt[1:], z_gt[1:], label="Ground Truth", marker='.', linestyle='-', markersize=4)
ax1.plot(x_rollout[1:], z_rollout[1:], label="Rollout Pred.", marker='x', linestyle='--', markersize=4)
ax1.scatter(x_gt[0], z_gt[0], label="Initial Point", c='g', s=50, zorder=5)
ax1.set_xlabel("x [m]")
ax1.set_ylabel("z [m]")
ax1.set_title("XZ Projection")
rescale_ax(ax1)
ax1.grid(True)

ax2.plot(y_gt[1:], z_gt[1:], label="Ground Truth", marker='.', linestyle='-', markersize=4)
ax2.plot(y_rollout[1:], z_rollout[1:], label="Rollout Pred.", marker='x', linestyle='--', markersize=4)
ax2.scatter(y_gt[0], z_gt[0], label="Initial Point", c='g', s=50, zorder=5)
ax2.set_xlabel("y [m]")
ax2.set_ylabel("z [m]")
ax2.set_title("YZ Projection")
rescale_ax(ax2)
ax2.grid(True)

handles, labels = ax0.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Rollout Prediction: Trajectory Projections")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, "rollout_projections.pdf"))
plt.close(fig)

# 3D Plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.plot(x_gt[1:], y_gt[1:], z_gt[1:], label="Ground Truth", marker='.', linestyle='-', markersize=4)
ax.plot(x_rollout[1:], y_rollout[1:], z_rollout[1:], label="Rollout Pred.", marker='x', linestyle='--', markersize=4)
ax.scatter(x_gt[0], y_gt[0], z_gt[0], label="Initial Point $x_0$", c='g', s=50, zorder=5)
ax.legend()
# plt.rc('text', usetex=True) # Disable LaTeX rendering if dvipng is not installed
# plt.rc('font', family='serif') # Also comment out font setting if it depends on TeX
ax.set_xlabel("$x$ [m]")
ax.set_ylabel("$y$ [m]")
ax.set_zlabel("$z$ [m]")
ax.set_title("Rollout Prediction: 3D Trajectory")
ax.set_box_aspect([1, 1, 1]) # Equal aspect ratio
# rescale_ax(ax) # Disable rescale
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rollout_trajectory_3d.pdf"))
plt.close(fig)

# Individual State Plots (Psi, Velocities)
t_rollout_plot = t[1:] # Time vector for plotting rollout results (N_seq-1 points)

# Extract angles and velocities
cos_psi_gt = X_0_np[plot_idx, :, 3]
sin_psi_gt = X_0_np[plot_idx, :, 4]
cos_psi_rollout = X_0_preds_rollout_np[plot_idx, :, 3]
sin_psi_rollout = X_0_preds_rollout_np[plot_idx, :, 4]
psi_gt = np.arctan2(sin_psi_gt, cos_psi_gt)
psi_rollout = np.arctan2(sin_psi_rollout, cos_psi_rollout)

u_gt = X_0_np[plot_idx, :, 5]
v_gt = X_0_np[plot_idx, :, 6]
w_gt = X_0_np[plot_idx, :, 7]
r_gt = X_0_np[plot_idx, :, 8]
u_rollout = X_0_preds_rollout_np[plot_idx, :, 5]
v_rollout = X_0_preds_rollout_np[plot_idx, :, 6]
w_rollout = X_0_preds_rollout_np[plot_idx, :, 7]
r_rollout = X_0_preds_rollout_np[plot_idx, :, 8]

# Plot Psi (Yaw)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_rollout_plot, psi_gt[1:], label="Ground Truth")
ax.plot(t_rollout_plot, psi_rollout[1:], label="Rollout Pred.", linestyle='--')
ax.scatter(t_rollout_plot[0], psi_gt[1], label="Initial Point", c='g', s=50, zorder=5) # Initial point of the plotted sequence
ax.legend()
ax.set_xlabel("Time (s)")
ax.set_ylabel("Psi (rad)")
ax.set_title("Rollout Prediction: Yaw Angle (Psi)")
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rollout_psi.pdf"))
plt.close(fig)

# Plot Velocities
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# u (surge)
axs[0, 0].plot(t_rollout_plot, u_gt[1:], label="Ground Truth")
axs[0, 0].plot(t_rollout_plot, u_rollout[1:], label="Rollout Pred.", linestyle='--')
axs[0, 0].scatter(t_rollout_plot[0], u_gt[1], label="Initial Point", c='g', s=50, zorder=5)
axs[0, 0].set_title("Surge Velocity (u)")
axs[0, 0].set_ylabel("Velocity (m/s)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# v (sway)
axs[0, 1].plot(t_rollout_plot, v_gt[1:], label="Ground Truth")
axs[0, 1].plot(t_rollout_plot, v_rollout[1:], label="Rollout Pred.", linestyle='--')
axs[0, 1].scatter(t_rollout_plot[0], v_gt[1], label="Initial Point", c='g', s=50, zorder=5)
axs[0, 1].set_title("Sway Velocity (v)")
axs[0, 1].set_ylabel("Velocity (m/s)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# w (heave)
axs[1, 0].plot(t_rollout_plot, w_gt[1:], label="Ground Truth")
axs[1, 0].plot(t_rollout_plot, w_rollout[1:], label="Rollout Pred.", linestyle='--')
axs[1, 0].scatter(t_rollout_plot[0], w_gt[1], label="Initial Point", c='g', s=50, zorder=5)
axs[1, 0].set_title("Heave Velocity (w)")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Velocity (m/s)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# r (yaw rate)
axs[1, 1].plot(t_rollout_plot, r_gt[1:], label="Ground Truth")
axs[1, 1].plot(t_rollout_plot, r_rollout[1:], label="Rollout Pred.", linestyle='--')
axs[1, 1].scatter(t_rollout_plot[0], r_gt[1], label="Initial Point", c='g', s=50, zorder=5)
axs[1, 1].set_title("Yaw Rate (r)")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Ang. Vel. (rad/s)")
axs[1, 1].legend()
axs[1, 1].grid(True)

fig.suptitle("Rollout Prediction: Velocities")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, "rollout_velocities.pdf"))
plt.close(fig)

# Position Error Norm Plot
e_pos = np.sqrt((x_gt[1:] - x_rollout[1:])**2 + (y_gt[1:] - y_rollout[1:])**2 + (z_gt[1:] - z_rollout[1:])**2)
plt.figure(figsize=(8, 4))
plt.plot(t_rollout_plot, e_pos)
plt.xlabel("Time (s)")
plt.ylabel("$||\mathbf{e}_{pos}||_2$ [m]")
plt.title("Rollout Prediction: Position Error Norm")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rollout_pos_error_norm.pdf"))
plt.close()

# --- IVP Calculation ---
print("Calculating Interval of Valid Prediction (IVP)...")
ivp_sum = 0.0
e_thr = 0.05
# Use torch tensors for calculation if possible, otherwise numpy
X_0_tensor = X_0.to(device) # Ensure tensors are on the correct device
X_0_preds_rollout_tensor = X_0_preds_rollout.to(device)

for n in range(N_batch):
    error_norm_n = torch.sqrt(
        (X_0_tensor[n, 1:, 0] - X_0_preds_rollout_tensor[n, 1:, 0])**2 +
        (X_0_tensor[n, 1:, 1] - X_0_preds_rollout_tensor[n, 1:, 1])**2 +
        (X_0_tensor[n, 1:, 2] - X_0_preds_rollout_tensor[n, 1:, 2])**2
    )
    valid_indices = torch.nonzero(error_norm_n <= e_thr)
    if len(valid_indices) > 0:
        last_valid_index = valid_indices[-1][0]
        ivp_sum += dt * (last_valid_index + 1) # +1 because index is 0-based
    # Else: contribution is 0

ivp = ivp_sum / N_batch if N_batch > 0 else torch.tensor(0.0) # Avoid division by zero
print(f"Average IVP (e_thr={e_thr}): {ivp.item():.4f} s")

# --- Save Numerical Results (Optional) ---
# Example: Save losses and IVP to a text file
results_summary_path = os.path.join(output_dir, "summary_results.txt")
with open(results_summary_path, 'w') as f:
    f.write(f"Model: {model_path}\n")
    f.write(f"Overall Single-Step MSE: {single_step_mse:.6f}\n")
    f.write(f"Rollout Loss (10 steps, dB): {rollout_loss.item():.2f}\n")
    f.write(f"Physics Loss (dB): {physics_loss.item():.2f}\n")
    f.write(f"Average IVP (e_thr={e_thr}): {ivp.item():.4f} s\n")
print(f"Numerical results saved to {results_summary_path}")

print(f"--- Script finished. Plots saved to {output_dir} ---")
