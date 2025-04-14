# System Patterns

**Project Name:** pinc-xyz-yaw

**System Architecture:** The system involves generating simulation data for a BlueROV, training Physics-Informed Neural Network (PINN) models using PyTorch, and evaluating model performance through single-step and multi-step (rollout) predictions against ground truth data. Key components are organized into `src/` (core dynamics), `data/` (data generation/utilities), `models/` (model definitions/weights), `training/` (training scripts), `scripts/` (evaluation scripts), and `results/` (output plots/metrics).

**Key Technical Decisions:**
- Use of PyTorch for neural network implementation (`DNN` class in `models/model_utility.py`).
- Physics-Informed Neural Network approach (specific loss functions like `physics_loss_fn` in `models/model_utility.py`).
- Separation of concerns into different directories.
- Use of `control` library for simulating ground truth dynamics (`src/bluerov.py`, `data/create_data_2.py`).
- Models are saved as individual weight files (`state_dict`) corresponding to a fixed architecture defined in scripts/model definitions.

**Design Patterns in Use:**
- **Dataset Class:** `TrajectoryDataset` in `data/data_utility.py` provides a standard interface for loading trajectory data.
- **Modular Scripts:** Separate scripts for data generation (`data/create_data_2.py`), single-model evaluation (`scripts/pi_dnn.py`), and multi-model/multi-dataset evaluation (`scripts/evaluate_multi_rollouts.py`).

**Component Relationships:**
- **Data Generation (`data/create_data_2.py`, `data/data_utility.py`, `src/bluerov.py`):**
    - `create_data_2.py` orchestrates data generation.
    - It uses `random_input` from `data_utility.py` to generate control inputs (U) based on specified types ('noise', 'sine', 'line', 'circle', 'figure8').
    - It uses `random_x0` from `data_utility.py` to generate initial states (x_0).
    - It uses the `bluerov` dynamics defined in `src/bluerov.py` (via `control.input_output_response`) to simulate the ground truth state trajectories (X) based on U and x_0.
    - It saves X, U, time (t), and collocation times (t_coll) as `.pt` files in dataset directories (e.g., `training_set`, `test_set_line`).
- **Model Definition (`models/model_utility.py`):**
    - Defines the `DNN` class representing the neural network architecture.
    - Defines activation functions (`Softplus`).
    - Defines helper functions for data conversion (`convert_input_data`, `convert_output_data`) and loss calculation (`rollout_loss_fn`, `physics_loss_fn`).
- **Single Model Evaluation (`scripts/pi_dnn.py`):**
    - Defines a specific `DNN` architecture.
    - Loads a specific pre-trained model's weights (e.g., `models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_993`).
    - Uses `get_data_sets` (from `models/model_utility.py`) to load development data (`dev_set`).
    - Performs **single-step prediction** and **rollout prediction**.
    - Generates various comparison plots (MSE, trajectories, individual states) and calculates losses.
    - Saves results to `results/paper_results/`.
- **Multi-Rollout Evaluation (`scripts/evaluate_multi_rollouts.py`):**
    - Defines the same `DNN` architecture as `pi_dnn.py`.
    - Takes lists of model weight file paths and dataset directory paths as input (via command-line arguments).
    - Iterates through each model and dataset combination.
    - Loads model weights and dataset using `TrajectoryDataset`.
    - Performs **rollout prediction** for the first trajectory in each dataset.
    - Generates comparison plots (3D trajectory, 2D projections, position error norm).
    - Calculates Interval of Valid Prediction (IVP).
    - Saves plots and summary metrics to `results/multi_rollout_results/`, organized by dataset and model name.
