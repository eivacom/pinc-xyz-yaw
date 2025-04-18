## Neural Network Architecture

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

### One-step-ahead Prediction Loss
The one-step-ahead prediction loss ($\mathcal{L}_D$) measures the mean squared error between the predicted state $\hat{\mathbf{x}}_{n,m}(T)$ and the ground-truth next state $\mathbf{x}_{n+1,m}(0)$ for each consecutive pair in all trajectories:

$$
    \mathcal{L}_D = \frac{1}{N_B(N_D-1)} \sum^{N_B-1}_{m=0} \sum^{N_D-2}_{n=0} ||\mathbf{x}_{n+1,m}(0)-\hat{\mathbf{x}}_{n,m}(T)||_2^2,
$$
This loss encourages the model to match the known data at discrete intervals $T$.

Where:
*   $N_B$ is the number of trajectories in a batch.
*   $N_D$ is the number of data points in each trajectory.

### Physics Loss
The physics loss ($\mathcal{L}_P$) regularizes the model’s predictions to respect the underlying physics by penalizing deviations from the governing differential equations. Fig.~\ref{fig:pinc_loss_processing} illustrates the processing of the data and physics losses within the PINC framework.

Each point in each trajectory has some corresponding collocation points $N_P$, indexed by $k$. Thus, the total number of collocation points evaluated in a batch is $N_D\times N_B\times N_P$. The $k$th collocation point in the $n$th trajectory, at its $m$th point is denoted $T_{n,m,k}^{coll}$, which is sampled by LHS on the interval $T^{coll}\in[0,T]$. The physics residual at each collocation point is defined as:
$$
    F(\mathbf{x}\dot{,\mathbf{x}},\mathbf{u}) = \dot{\mathbf{x}} - f(\mathbf{x},\mathbf{u}).
$$
where $f(\mathbf{x},\mathbf{u})$ refers to the right-hand side of the ROV dynamics equation. When $F(\mathbf{x},\dot{\mathbf{x}},\mathbf{u}) =0$, the learned dynamics perfectly match the true dynamics. Specifically the physics residual $F\left(\cdot\right) = F\left(\hat{\mathbf{x}}_{n,m}(T_{n,m,k}^{\text{coll}}), \dot{\hat{\mathbf{x}}}_{n,m}(T_{n,m,k}^{\text{coll}}), \mathbf{u}_{n,m}(0)\right)$ is used in the physics loss. The physics loss is then computed as the mean squared error of these residuals across all collocation points, trajectories, and time steps:

$$
    \mathcal{L}_P = \frac{1}{N_B N_D N_P} \sum_{k=0}^{N_P - 1} \sum_{m=0}^{N_B - 1} \sum_{n=0}^{N_D - 1} \left\| F\left(\cdot\right) \right\|_2^2.
$$

Where:
*   $N_B$ is the number of trajectories in a batch.
*   $N_D$ is the number of data points in each trajectory.
*   $N_P$ is the number of collocation points per data point.

### Initial Condition Loss
The initial condition loss $\mathcal{L}_{IC}$ ensures internal consistency of the network’s outputs at $t=0$, treating each data point as a new initial condition. This is formulated as follows:
$$
    \mathcal{L}_{IC} = \frac{1}{N_B N_D} \sum^{N_B-1}_{m=0}\sum^{N_D-1}_{n=0}||\mathbf{x}_{n,m}(0)-\hat{\mathbf{x}}_{n,m}(0)||_2^2.
$$
Where:
*   $N_B$ is the number of trajectories in a batch.
*   $N_D$ is the number of data points in each trajectory.

### Rollout Loss
Inspired by \cite{zhao_research_2024}, an $N$-step-ahead (rollout) loss is used to penalize the accumulation of errors when predicting multiple steps forward. In a trajectory with $N_D$, points, only the first $N_R=N_D-N_{pred}$ points can be rolled out. For each trajectory, rollouts are initiated from the first $N_R$ points, and the predicted sequence is compared to the ground truth trajectory using the rollout loss function:
$$
    \mathcal{L}_R = \frac{1}{{N_B N_R N_P}} \sum^{N_P}_{k=1} \sum^{N_B-1}_{m=0} \sum^{N_R-1}_{n=0} ||\mathbf{x}_{n+k,m}(0) - \hat{\mathbf{x}}_{n,m}(kT)||_2^2.
$$
Where:
*   $N_B$ is the number of trajectories in a batch.
*   $N_R$ is the number of points where rollouts are initiated.
*   $N_P$ is the number of prediction steps.

### Physics Rollout Loss
When both the rollout loss and physics loss are included in the optimization, a physics rollout loss is also computed to regularize the intermediate states in multistep predictions. This loss is calculated as the average of the physics loss over all rollout steps:

$$
    \mathcal{L}_{PR} = \frac{1}{N_{roll}} \sum_{i=0}^{N_{roll}-1} \mathcal{L}_P
$$

Where:
*   $N_{roll}$ is the number of rollout steps.
*   $\mathcal{L}_P$ is the physics loss calculated by `physics_loss_fn`.
## Neural Network Architecture

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

### One-step-ahead Prediction Loss
The one-step-ahead prediction loss ($\mathcal{L}_D$) measures the mean squared error between the predicted state $\hat{\mathbf{x}}_{n,m}(T)$ and the ground-truth next state $\mathbf{x}_{n+1,m}(0)$ for each consecutive pair in all trajectories:

$$
    \mathcal{L}_D = \frac{1}{N_B(N_D-1)} \sum^{N_B-1}_{m=0} \sum^{N_D-2}_{n=0} ||\mathbf{x}_{n+1,m}(0)-\hat{\mathbf{x}}_{n,m}(T)||_2^2,
$$
This loss encourages the model to match the known data at discrete intervals $T$.

Where:
*   $N_B$ is the number of trajectories in a batch.
*   $N_D$ is the number of data points in each trajectory.

### Physics Loss
The physics loss ($\mathcal{L}_P$) regularizes the model’s predictions to respect the underlying physics by penalizing deviations from the governing differential equations. Fig.~\ref{fig:pinc_loss_processing} illustrates the processing of the data and physics losses within the PINC framework.

Each point in each trajectory has some corresponding collocation points $N_P$, indexed by $k$. Thus, the total number of collocation points evaluated in a batch is $N_D\times N_B\times N_P$. The $k$th collocation point in the $n$th trajectory, at its $m$th point is denoted $T_{n,m,k}^{coll}$, which is sampled by LHS on the interval $T^{coll}\in[0,T]$. The physics residual at each collocation point is defined as:
$$
    F(\mathbf{x}\dot{,\mathbf{x}},\mathbf{u}) = \dot{\mathbf{x}} - f(\mathbf{x},\mathbf{u}).
$$
where $f(\mathbf{x},\mathbf{u})$ refers to the right-hand side of the ROV dynamics equation. When $F(\mathbf{x},\dot{\mathbf{x}},\mathbf{u}) =0$, the learned dynamics perfectly match the true dynamics. Specifically the physics residual $F\left(\cdot\right) = F\left(\hat{\mathbf{x}}_{n,m}(T_{n,m,k}^{\text{coll}}), \dot{\hat{\mathbf{x}}}_{n,m}(T_{n,m,k}^{\text{coll}}), \mathbf{u}_{n,m}(0)\right)$ is used in the physics loss. The physics loss is then computed as the mean squared error of these residuals across all collocation points, trajectories, and time steps:

$$
    \mathcal{L}_P = \frac{1}{N_B N_D N_P} \sum_{k=0}^{N_P - 1} \sum_{m=0}^{N_B - 1} \sum_{n=0}^{N_D - 1} \left\| F\left(\cdot\right) \right\|_2^2.
$$

Where:
*   $N_B$ is the number of trajectories in a batch.
*   $N_D$ is the number of data points in each trajectory.
*   $N_P$ is the number of collocation points per data point.

### Initial Condition Loss
The initial condition loss $\mathcal{L}_{IC}$ ensures internal consistency of the network’s outputs at $t=0$, treating each data point as a new initial condition. This is formulated as follows:
$$
    \mathcal{L}_{IC} = \frac{1}{N_B N_D} \sum^{N_B-1}_{m=0}\sum^{N_D-1}_{n=0}||\mathbf{x}_{n,m}(0)-\hat{\mathbf{x}}_{n,m}(0)||_2^2.
$$
Where:
*   $N_B$ is the number of trajectories in a batch.
*   $N_D$ is the number of data points in each trajectory.

### Rollout Loss
Inspired by \cite{zhao_research_2024}, an $N$-step-ahead (rollout) loss is used to penalize the accumulation of errors when predicting multiple steps forward. In a trajectory with $N_D$, points, only the first $N_R=N_D-N_{pred}$ points can be rolled out. For each trajectory, rollouts are initiated from the first $N_R$ points, and the predicted sequence is compared to the ground truth trajectory using the rollout loss function:
$$
    \mathcal{L}_R = \frac{1}{{N_B N_R N_P}} \sum^{N_P}_{k=1} \sum^{N_B-1}_{m=0} \sum^{N_R-1}_{n=0} ||\mathbf{x}_{n+k,m}(0) - \hat{\mathbf{x}}_{n,m}(kT)||_2^2.
$$
Where:
*   $N_B$ is the number of trajectories in a batch.
*   $N_R$ is the number of points where rollouts are initiated.
*   $N_P$ is the number of prediction steps.

### Physics Rollout Loss
When both the rollout loss and physics loss are included in the optimization, a physics rollout loss is also computed to regularize the intermediate states in multistep predictions. This loss is calculated as the average of the physics loss over all rollout steps:

$$
    \mathcal{L}_{PR} = \frac{1}{N_{roll}} \sum_{i=0}^{N_{roll}-1} \mathcal{L}_P
$$

Where:
*   $N_{roll}$ is the number of rollout steps.
*   $\mathcal{L}_P$ is the physics loss calculated by `physics_loss_fn`.
