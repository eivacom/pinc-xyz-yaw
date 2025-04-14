# Progress

**Project Name:** pinc-xyz-yaw

**What Works:**
- **Data Generation (`data/create_data_2.py` & `data/data_utility.py`):**
    - Generates ground truth simulation data for the BlueROV model.
    - Supports generating random 'noise' and 'sine' based input trajectories.
    - Supports generating specific 'line', 'circle', and 'figure8' trajectories using defined control inputs.
    - Saves generated datasets (X, U, t, t_coll) as `.pt` files in specified directories.
    - Successfully generated `training_set`, `dev_set`, `test_set_interp`, `test_set_extrap`, `test_set_line`, `test_set_circle`, `test_set_figure8`.
- **Single Model Evaluation (`scripts/pi_dnn.py`):**
    - Loads a specific pre-trained PINN model (originally `pinn_no_rollout_rotated_less_layers_0`, now uses `noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_993`).
    - Loads development data (`dev_set`).
    - Performs single-step predictions and compares them to ground truth via MSE plots and trajectory plots.
    - Performs rollout predictions and compares them via state plots, trajectory plots, and position error norm plot.
    - Calculates rollout and physics losses.
    - Saves results to `results/paper_results/`.
- **Multi-Rollout Evaluation (`scripts/evaluate_multi_rollouts.py`):**
    - Loads multiple specified model weight files (e.g., `_best_dev_l_0`, `_best_dev_l_993`).
    - Loads specified datasets (e.g., `test_set_line`, `test_set_circle`, `test_set_figure8`).
    - Performs rollout predictions for each model on each specified dataset.
    - Generates comparison plots (3D trajectory, 2D projections, position error norm).
    - Calculates Interval of Valid Prediction (IVP) metric.
    - Saves results to `results/multi_rollout_results/`, organized by dataset and model.

**What's Left to Build:**
- Reorganize the overall code structure.
- Update requirements file (`requirements.txt`).
- Update `README.md`.
- Potentially train or refine models.
- Potentially add more evaluation metrics or plots.
- Push changes to GitHub.

**Current Status:** Completed the task of adding multi-step simulation results for different models and new trajectory types (line, circle, figure 8). Generated results are available in `results/multi_rollout_results/`.

**Known Issues:**
- The model file `models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_50` was not found during the multi-rollout evaluation, so it was skipped.
