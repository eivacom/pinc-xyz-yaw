# Physics-Informed Neural Control (PINC) for Learning BlueROV Dynamics

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=PyTorch&logoColor=white)
![Tests Passed](https://img.shields.io/badge/tests-passed-brightgreen)



This repository implements a Physics-Informed Neural network with Control (PINC) approach to model and simulate the dynamics of a BlueROV2 underwater vehicle using PyTorch. It leverages the power of neural networks while incorporating physical principles to achieve data effecient neural dynamic models.


> **📄 Accepted Paper @ [IJCNN 2025](https://2025.ijcnn.org/)**  
> **Title:** *Modelling of Underwater Vehicles using Physics-Informed Neural Networks with Control*


## Key Features

*   **BlueROV Dynamic Simulation**: Core simulation logic for the BlueROV2 vehicle implemented in `src/bluerov.py` and `src/bluerov_torch.py`.
*   **Synthetic Data Generation**: Scripts (`data/create_data_2.py`) to generate customizable trajectory datasets for training and evaluation.
*   **PINC Model Implementation**: Neural network models designed to learn system dynamics, potentially incorporating physical laws directly into the loss function (PINN) or network structure. See [Model Explanation](model_explanation.md) for details on the architecture and loss functions.
*   **Model Training Pipeline**: Robust training script (`training/train_model.py`) using PyTorch, with support for hyperparameter tuning.


## Project Structure

```
.
├── data/             # Scripts for data generation and utility functions
├── memory-bank/      # Project documentation (context, progress, etc.)
├── models/           # Saved model checkpoints and utility functions. Contains trained neural network models. These models are the result of training the PINC architecture on the BlueROV dynamics.
├── notebooks/        # Jupyter notebooks for experimentation (potentially outdated)
├── results/          # Saved evaluation results, plots, summaries
├── runs/             # TensorBoard log files
├── scripts/          # Evaluation, analysis, and visualization scripts
├── src/              # Core source code (BlueROV model, parameters)
├── training/         # Model training scripts
├── .gitignore        # Git ignore rules
├── LICENSE           # Project license file (Should be updated to GPLv3)
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

## Installation

### Prerequisites

*   Python 3.8+
*   Git
*   PyTorch (>=2.0 recommended)
*   Optional: CUDA-compatible GPU for accelerated training

### Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/abdelhakim96/underwater_pinns.git
    cd underwater_pinns
    ```

2.  **Create a Virtual Environment (Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```
    *Core Dependencies:*
    *   `torch`: Neural network framework
    *   `numpy`: Numerical computing
    *   `scipy`: Scientific computing tools
    *   `pandas`: Data manipulation and analysis
    *   `matplotlib`: Plotting and visualization
    *   `tqdm`: Progress bars
    *   `control`: Control system analysis library
    *   `conflictfree`: ConFig method

## Usage

### 1. Generate Data

Run the data generation script. You might need to adjust parameters within the script.

```bash
python data/create_data_2.py
```

### 2. Train the Model

Execute the training script. Hyperparameters can often be adjusted within the script or via command-line arguments (if implemented).

```bash
python training/train_model.py
```

### 3. Monitor Training with TensorBoard (Optional)

TensorBoard provides powerful visualizations to monitor the training process in real-time or analyze it afterward. This helps in understanding model convergence, comparing different runs, and debugging potential issues.

To launch TensorBoard, run the following command in your terminal from the project's root directory:

```bash
tensorboard --logdir runs
```

This command points TensorBoard to the `runs/` directory where the training script saves its log files.

Once TensorBoard is running, open your web browser and navigate to the URL provided in the terminal output (usually `http://localhost:6006/`). You should see visualizations including:

*   Loss curves (training and validation)
*   Accuracy or other relevant metrics over epochs
*   Potentially model graphs and hyperparameter comparisons (depending on logging implementation in `training/train_model.py`)

### 4. Evaluate and Analyze Models

Use the scripts provided to evaluate performance and visualize results:

*   **Evaluate Metrics:**
    ```bash
    python scripts/evaluate_model.py --model_path path/to/your/model.pt
    ```
*   **Analyze Rollouts:**
    ```bash
    python scripts/analyze_rollout.py --model_path path/to/your/model.pt
    ```
*   **Visualize Predictions (Example):**
    ```bash
    python scripts/pi_dnn.py # (May need adjustments to load specific models/data)
    ```
*(Note: You might need to modify these scripts to point to the specific model checkpoints you want to analyze, typically found in `models/`)*

## References / Acknowledgements

*   **This Repository:** [https://github.com/eivacom/pinc-xyz-yaw](https://github.com/eivacom/pinc-xyz-yaw)
*   **ConFIG (Conflict-Free Gradient Combination):** The gradient combination techniques explored in this project are inspired by or utilize concepts from the ConFIG library. [https://github.com/tum-pbs/ConFIG](https://github.com/tum-pbs/ConFIG)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
