# Model Explanation

This document explains the neural network architecture, loss functions used in the model, and the training parameters.

## Neural Network Architecture

The residual deep neural network architecture is used to learn ROV dynamics in a physics-informed manner. The model is trained using the AdamW optimizer with a ReduceLROnPlateau scheduler. See the Training Parameters section for more details.

### Residual Formulation for ODE Integration
Solving an ODE over a time interval $I = [0,T]$ can be written as
$$
    \mathbf{x}(T) = \mathbf{x}(0) + \int_0^T f(\mathbf{x}(\tau), \mathbf{u}(\tau)) d\tau.
$$
When the control is assumed constant over $I$ (zero-order hold), the integral can be approximated with a neural network that learns to “integrate” the dynamics from $\mathbf{x}(0)$ to $\mathbf{x}(T)$:
$$
    \mathbf{x}(T) \approx \mathbf{x}(0) + \mathcal{N}\bigl(\underbrace{\begin{bmatrix}
    \mathbf{x}(0) & \mathbf{u}(0) & T
\end{bmatrix}}_{\mathbf{z}(0)}\bigr).
$$

### Layers, Neurons, and Activations
The architecture includes fully connected $N_L$ hidden layers, each containing $N_H$ neurons. Both adaptive `tanh` and `softplus` activation functions are tested with an adaptable parameter, denoted $\beta$, that is unique for each layer.

### State Re-Parameterization for Yaw
Since yaw angle $\psi$ wraps around at $\pm\pi$, it is replaced with two states: $\cos(\psi)$ and $\sin(\psi)$. This ensures the continuity of the states and avoids angle discontinuities.

### Rotational Structural Information
To reflect the natural geometry of planar motion, the network’s predicted increments in $x$ and $y$ are rotated from the body frame $\mathcal{F}_B$ to the world-fixed $\mathcal{F}_W$. Specifically, if the raw network outputs are $\Delta \hat{x}_b$ and $\Delta \hat{y}_b$, they are transformed as follows:
$$
    \begin{bmatrix}
        \Delta\hat{x}_n\\
        \Delta\hat{y}_n
    \end{bmatrix}=
    \begin{bmatrix}
        \cos(\hat{\psi}) & -\sin(\hat{\psi})\\
        \sin(\hat{\psi}) & \cos(\hat{\psi})
    \end{bmatrix}
    \begin{bmatrix}
        \Delta\hat{x}_b\\
        \Delta\hat{y}_b
    \end{bmatrix}.
$$
Because the model learns increments in the body frame, it can capture the simpler local dynamics. Those increments are then rotated back into $\mathcal{F}_W$.

### Layer Normalization
Layer normalization \cite{ba_layer_2016} is another regularization technique employed in this work, which normalizes activations within each layer using learnable parameters, mitigating internal covariate shifts.

## Loss Functions

The total loss is a weighted sum of the data loss, initial condition loss, physics loss (if `pinn` is True), and rollout loss (if `rollout` is True):

$$
\mathcal{L}_{total} = \mathcal{L}_D + \mathcal{L}_{IC} + \alpha * \mathcal{L}_P + \mathcal{L}_R + \alpha * \mathcal{L}_{PR}
$$

Where \(\alpha = 0.5\) is the weighting factor for the physics losses.

### Data Loss

The one-step-ahead prediction loss ($\mathcal{L}_D$) measures the mean squared error between the predicted state $\hat{\mathbf{x}}_{n,m}(T)$ and the ground-truth next state $\mathbf{x}_{n+1,m}(0)$ for each consecutive pair in all trajectories:

Equation:
$$
    \mathcal{L}_D = \frac{1}{N_B(N_D-1)} \sum^{N_B-1}_{m=0} \sum^{N_D-2}_{n=0} ||\mathbf{x}_{n+1,m}(0)-\hat{\mathbf{x}}_{n,m}(T)||_2^2,
$$
This loss encourages the model to match the known data at discrete intervals $T$.

Where:
*   $N_B$: Number of trajectories in a batch.
*   $N_D$: Number of data points in each trajectory.

PyTorch Implementation (from `models/model_utility.py`):
```python
l_data = mse_loss(X_2_hat, X_1[:, 1:])
```

### Physics Loss

The physics loss ($\mathcal{L}_P$) regularizes the model’s predictions to respect the underlying physics by penalizing deviations from the governing differential equations. The `compute_time_derivatives` function in `models/model_utility.py` uses PyTorch's automatic differentiation (`torch.func.jvp`) to compute the time derivative of the predicted state. This allows the model to learn the underlying physics of the system without explicitly providing the derivatives.

Equation:
$$
    F(\mathbf{x}\dot{,\mathbf{x}},\mathbf{u}) = \dot{\mathbf{x}} - f(\mathbf{x},\mathbf{u}).
$$

The physics loss is then computed as the mean squared error of these residuals across all collocation points, trajectories, and time steps:

$$
    \mathcal{L}_P = \frac{1}{N_B N_D N_P} \sum_{k=0}^{N_P - 1} \sum_{m=0}^{N_B - 1} \sum_{n=0}^{N_D - 1} \left\| F\left(\cdot\right) \right\|_2^2.
$$

Where:
*   $N_B$: Number of trajectories in a batch.
*   $N_D$: Number of data points in each trajectory.
*   $N_P$: Number of collocation points per data point.

PyTorch Implementation (from `models/model_utility.py`):
```python
l_phy = ((dX2_hat_dt_flat - bluerov_compute(0, X_2_hat_coll_flat, U_coll_flat))**2).mean()
```

### Initial Condition Loss

The initial condition loss $\mathcal{L}_{IC}$ ensures internal consistency of the network’s outputs at $t=0$, treating each data point as a new initial condition. This is formulated as follows:
$$
    \mathcal{L}_{IC} = \frac{1}{N_B N_D} \sum^{N_B-1}_{m=0}\sum^{N_D-1}_{n=0}||\mathbf{x}_{n,m}(0)-\hat{\mathbf{x}}_{n,m}(0)||_2^2.
$$
Where:
*   $N_B$: Number of trajectories in a batch.
*   $N_D$: Number of data points in each trajectory.

PyTorch Implementation (from `models/model_utility.py`):
```python
l_ic = mse_loss(X_1_hat, X_1)
```

### Rollout Loss

Inspired by \cite{zhao_research_2024}, an $N$-step-ahead (rollout) loss is used to penalize the accumulation of errors when predicting multiple steps forward. In a trajectory with $N_D$ points, only the first $N_R=N_D-N_{pred}$ points can be rolled out. For each trajectory, rollouts are initiated from the first $N_R$ points, and the predicted sequence is compared to the ground truth trajectory using the rollout loss function:
$$
    \mathcal{L}_R = \frac{1}{{N_B N_R N_P}} \sum^{N_P}_{k=1} \sum^{N_B-1}_{m=0} \sum^{N_R-1}_{n=0} ||\mathbf{x}_{n+k,m}(0) - \hat{\mathbf{x}}_{n,m}(kT)||_2^2.
$$
Where:
*   $N_B$: Number of trajectories in a batch.
*   $N_R$: Number of points where rollouts are initiated. Calculated as $N_{seq} - N_{roll}$ where $N_{seq}$ is the original sequence length and $N_{roll}$ is the number of rollout steps.
*   $N_P$: Number of prediction steps (equivalent to $N_{roll}$).

PyTorch Implementation (from `models/model_utility.py`):
```python
l_roll += mse_loss(X_hat, X_0[:, i+1:i+1+N_seq_slice])
```

### Physics Rollout Loss

When both the rollout loss and physics loss are included in the optimization, a physics rollout loss is also computed to regularize the intermediate states in multistep predictions. This loss is calculated as the average of the physics loss over all rollout steps:

$$
    \mathcal{L}_{PR} = \frac{1}{N_{roll}} \sum_{i=0}^{N_{roll}-1} \mathcal{L}_P
$$

Where:
*   $N_{roll}$: Number of rollout steps.
*   $\mathcal{L}_P$: The physics loss calculated by `physics_loss_fn`.

## Training Parameters

The model is trained with the following parameters (defined in `training/train_model.py`):

*   **Filter Constant (alpha):** 0.6
*   **Batch Size:** 3
*   **Epochs:** 1200
*   **Initial Learning Rate (lr_0):** 8e-3
*   **Learning Rate Decay Factor (lr_factor):** 0.5
*   **Learning Rate Patience (lr_patience):** 1200
*   **Learning Rate Threshold (lr_thr):** 1e-4
*   **Minimum Learning Rate (lr_min):** 1e-5
*   **Physics-Informed Neural Network (pinn):** True
*   **Rollout Training (rollout):** True
*   **Activation Function:** Softplus
*   **Noise Level:** 0.0
*   **Gradient Method:** config
*   **Optimizer:** AdamW
*   **Learning Rate Scheduler:** ReduceLROnPlateau
*   **Hidden Layer Sizes (N_h):** \[32, 32, 32, 32]
*   **Number of Hidden Layers (N_layer):** 4

</final_file_content>

IMPORTANT: For any future changes to this file, use the final_file_content shown above as your reference. This content reflects the current state of the file, including any auto-formatting (e.g., if you used single quotes but the formatter converted them to double quotes). Always base your SEARCH/REPLACE operations on this final version to ensure accuracy.

<environment_details>
# VSCode Visible Files
model_explanation.md

# VSCode Open Tabs
.clinerules
requirements.py
pi_dnn.ipynb
data_utility.py
create_data.py
model_utility.py
data/create_data.py
memory-bank/projectbrief.md
memory-bank/productContext.md
memory-bank/techContext.md
notebooks/pi_dnn.ipynb
src/bluerov_torch.py
src/bluerov.py
scripts/evaluate_model.py
scripts/compare_computation_time.py
scripts/analyze_rollout.py
results/summary.md
results/computation_time_comparison.md
models/model_utility.py
requirements.txt
setup.py
models/__init__.py
.gitignore
training/__init__.py
training/train_model.py
README.md
paper.text
paper.md
model_explanation.md
model_explanation.md
tests/src/test_bluerov.py
tests/src/test_bluerov_torch.py
tests/data/test_data_utility.py
tests/models/test_model_utility.py
scripts/pi_dnn.py
results/paper_results/rollout_projections.pdf
data/data_utility.py
scripts/evaluate_multi_rollouts.py
data/create_data_2.py
memory-bank/activeContext.md
memory-bank/progress.md
memory-bank/systemPatterns.md
results/multi_rollout_results/test_set_figure8/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_0/rollout_pos_error_norm.pdf

# Current Time
4/18/2025, 1:35:10 PM (Europe/Copenhagen, UTC+2:00)

# Context Window Usage
162,022 / 1,048.576K tokens used (15%)

# Current Mode
ACT MODE
</environment_details>
