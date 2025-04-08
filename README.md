# BlueROV Physics-Informed Neural Network (PINN)

A Physics-Informed Neural network with Control (PINC) for modeling the dynamics of the BlueROV underwater vehicle using PyTorch.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Generation](#data-generation)
  - [Training the Model](#training-the-model)
  - [Monitoring with TensorBoard](#monitoring-with-tensorboard)
- [License](#license)

## Features

- **Physics-Informed Neural Network**: Incorporates governing equations into the loss function by utilizing automatic differentiation.
- **Data Generation Scripts**: Generate synthetic datasets with customizable parameters.
- **Training and Evaluation Pipelines**: Scripts for training the model and evaluating performance.
- **TensorBoard Integration**: Real-time monitoring of training metrics.

## Installation

### Prerequisites

- Python 3.8+
- Git
- PyTorch 2.0+
- Optional: CUDA-compatible GPU for acceleration

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/felsager/xyz-yaw.git
   cd xyz-yaw
2. **Install dependencies using pip**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generate Data

Run the data generation script:

```bash
python data/create_data.py
```

### Train the Model

Train the model using:

```bash
python training/train_model.py
```

Adjust hyperparameters in training/train_model.py as needed.
### Monitor Training with TensorBoard

Start TensorBoard to monitor training progress:

tensorboard --logdir runs

Then open http://localhost:6006/ in your web browser.

License
