# Training and Evaluation Summary: pinc-xyz-yaw

This document summarizes the training process, model architecture, and evaluation results for the physics-informed neural network (PINN) model trained for the BlueROV system.

## Training Process

- **Script:** `training/train_model.py`
- **Experiment Name:** `noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed`
- **TensorBoard Index (`tb_idx`):** 2
- **Epochs:** 1200
- **Batch Size:** 3
- **Initial Learning Rate:** 8e-3
- **Learning Rate Scheduler:** ReduceLROnPlateau
    - Factor: 0.5
    - Patience: 1200 epochs
    - Threshold: 1e-4
    - Minimum LR: 1e-5
- **Loss Components:**
    - Data Loss (1-step prediction MSE): Enabled
    - Physics Loss (PINN): Enabled
    - Rollout Loss (20 steps): Enabled
    - Initial Condition Loss: Enabled (Implicitly via data/rollout loss starting from t=0)
- **Noise Level (Training):** 0.0
- **Optimizer:** AdamW
- **Gradient Handling:** Gradient scaling and combination (details in `models/model_utility.py::train`)
- **Device:** cuda

## Model Architecture

- **Model Class:** `DNN` (defined in `models/model_utility.py`)
- **Type:** Feedforward Neural Network
- **Input Size:** 14 (9 state variables + 4 control inputs + 1 time variable)
- **Output Size:** 9 (predicted state variables)
- **Hidden Layers:** 4 layers, each with 32 neurons ([32, 32, 32, 32])
- **Activation Function:** `torch.nn.Softplus` (wrapped in custom `AdaptiveSoftplus` with `LayerNorm`)
- **Final Model Saved:** `models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2`

## Evaluation Results (`scripts/evaluate_model.py`)

These metrics were calculated on a sample from the development dataset (`dev_set`).

- **One-step Prediction MSE:** 0.000020
- **Rollout Loss (10 steps, log10):** -3.24
- **Physics Loss (log10):** -2.74

Plots saved in `results/plots/`:
- `input_development.pdf`
- `trajectory_projections.pdf`
- `trajectory_3d.pdf`
- `mse_position.pdf`
- `mse_psi.pdf`
- `mse_linear_velocity.pdf`
- `mse_angular_velocity.pdf`

## Rollout Analysis (`scripts/analyze_rollout.py`)

This analysis evaluates the model's ability to predict multiple steps into the future by feeding its own predictions back as input (rollout).

- **Rollout Steps:** 64 (limited by sequence length)
- **Dataset:** Sample from `dev_set`
- **Average Interval of Validity Prediction (IVP) @ 0.05m error:** 0.0000 seconds

**Interpretation:**

The IVP metric of 0.0 seconds indicates that, on average across the test batch, the model's multi-step rollout prediction error exceeded the 0.05m threshold almost immediately (within the first timestep). This suggests **significant difficulty with long-term prediction accuracy**.

While the one-step prediction MSE is very low (0.000020), indicating the model is good at predicting the very next state, errors accumulate rapidly during rollout. The `results/plots/rollout_pos_error.pdf` plot likely shows a steep increase in position error over the 64 steps. Similarly, the individual state rollout plots (`rollout_*.pdf`) probably show divergence between the predicted trajectory and the ground truth relatively early in the sequence.

**Conclusion on Long-Term Prediction:** The current model performs poorly in long-term rollout predictions. The predictions start failing very early when compared against the ground truth, as evidenced by the near-zero IVP score. Further work would be needed to improve the stability and accuracy of multi-step predictions.

Plots saved in `results/plots/`:
- `rollout_xy.pdf`
- `rollout_xz.pdf`
- `rollout_yz.pdf`
- `rollout_psi.pdf`
- `rollout_u.pdf`
- `rollout_v.pdf`
- `rollout_w.pdf`
- `rollout_r.pdf`
- `rollout_pos_error.pdf`
