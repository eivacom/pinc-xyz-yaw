# Progress

**Project Name:** pinc-xyz-yaw

**What Works:**
- `scripts/pi_dnn.py`: Converted from `notebooks/pi_dnn.ipynb`. This script successfully:
    - Loads a specific pre-trained PINN model (`models/pinn_no_rollout_rotated_less_layers_0`).
    - Loads development data using `models/model_utility.py`.
    - Performs single-step predictions and compares them to ground truth via MSE plots (position, orientation, velocities) and trajectory plots (2D, 3D).
    - Performs rollout predictions (multi-step simulation) and compares them to ground truth via individual state plots (psi, u, v, w, r), trajectory plots (2D, 3D), and position error norm plot.
    - Calculates rollout and physics losses.
    - Saves some plots directly to the CWD (e.g., `input_development.pdf`, `trajectory.pdf`).

**What's Left to Build:**
- Reorganize the overall code structure.
- Update requirements file (`requirements.txt`).
- Update `README.md`.
- Potentially train or refine models (current script uses a pre-trained one).
- Define clear data generation/processing pipelines.

**Current Status:** Analyzing existing components and planning reorganization. Converted and analyzed `pi_dnn` notebook/script.

**Known Issues:** N/A
