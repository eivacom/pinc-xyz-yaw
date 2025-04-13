# System Patterns

**Project Name:** pinc-xyz-yaw

**System Architecture:** The system involves generating data, training a Physics-Informed Neural Network (PINN) model, and evaluating/visualizing its performance. Key components seem to be located in `src/`, `data/`, `models/`, `training/`, and `scripts/`.

**Key Technical Decisions:**
- Use of PyTorch for neural network implementation.
- Physics-Informed Neural Network approach (details TBD).
- Separation of concerns into different directories (data, models, training, scripts).

**Design Patterns in Use:** N/A (Further analysis needed)

**Component Relationships:**
- `scripts/pi_dnn.py`: Loads a pre-trained PINN model (`models/pinn_no_rollout_rotated_less_layers_0`). It uses `get_data_sets` (from `models/model_utility.py`) to load development data. It performs two main types of evaluation:
    - **Single-step prediction:** Compares the model's direct output (`X_hat`) against the next ground truth state (`X_0`). Calculates and plots MSE for position (x, y, z), orientation (psi), linear velocities (u, v, w), and angular velocity (r). Visualizes trajectory comparisons in 2D and 3D.
    - **Rollout prediction:** Simulates a longer trajectory (`X_0_preds`) by iteratively feeding the model's predictions back as input. Compares this rollout trajectory against the ground truth. Plots individual state comparisons (psi, u, v, w, r) and trajectory comparisons (2D projections, 3D plot). Calculates and plots the position error norm over time.
    - It also calculates rollout loss and physics loss using helper functions (`rollout_loss_fn`, `physics_loss_fn`).
    - It saves some plots directly to the CWD (e.g., `input_development.pdf`, `trajectory.pdf`). It utilizes core simulation/model logic from `src/bluerov_torch.py`.
- Other relationships TBD as more components are analyzed.
