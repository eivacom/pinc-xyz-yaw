import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.func import jvp
from torch.nn import Linear, Softplus, Sequential, Parameter, Module, LayerNorm
from torch.nn.init import xavier_uniform_, zeros_

from conflictfree.grad_operator import ConFIGOperator
from conflictfree.utils import get_gradient_vector, apply_gradient_vector
from conflictfree.length_model import TrackHarmonicAverage
# we go with this length_model because it works the best with more losses

operator = ConFIGOperator(length_model=TrackHarmonicAverage()) # initialize operator

from data.data_utility import TrajectoryDataset
from src.bluerov_torch import bluerov_compute

def convert_input_data(X: Tensor, U: Tensor, time: Tensor):
    """
    Converts the inputs for the model by concatenating state, control input, and time tensors.

    Args:
        X (Tensor): State tensor of shape [N_batch, N_seq, N_x].
        U (Tensor): Control input tensor of shape [N_batch, N_seq, N_u].
        time (Tensor): Time tensor of shape [N_batch, N_seq, 1].

    Returns:
        Tuple[Tensor, int, int, int]: Flattened input tensor Z, batch size N_batch, sequence length N_seq, number of state variables N_x.
    """
    N_batch, N_seq, N_x = X.shape
    Z = torch.cat((X, U, time), dim=2)
    N_in = Z.shape[2]
    Z = Z.view(-1, N_in)
    Z.requires_grad_()
    return Z, N_batch, N_seq, N_x

def convert_input_collocation(X: Tensor, U: Tensor, t_coll: Tensor):
    """
    Converts inputs for evaluating the model at collocation points.

    Args:
        X (Tensor): State tensor of shape [N_batch, N_seq, N_x]
        U (Tensor): Control input tensor of shape [N_batch, N_seq, N_u]
        t_coll (Tensor): Collocation time tensor of shape [N_batch, N_seq, N_coll]

    Returns:
        Tuple[Tensor, Tensor]: Flattened input tensor Z, expanded control input U_coll
    """
    N_coll = t_coll.shape[2]

    # Expand X and U to match the collocation points
    X_coll = X.unsqueeze(2).expand(-1, -1, N_coll, -1)  # Shape: [N_batch, N_seq, N_coll, N_x]
    U_coll = U.unsqueeze(2).expand(-1, -1, N_coll, -1).contiguous()  # Shape: [N_batch, N_seq, N_coll, N_u]

    # Concatenate inputs
    Z = torch.cat((X_coll, U_coll, t_coll.unsqueeze(3)), dim=3)  # Shape: [N_batch, N_seq, N_coll, N_in]
    N_in = Z.shape[3]

    # Flatten Z
    Z = Z.view(-1, N_in)
    Z.requires_grad_()
    return Z, U_coll

def convert_output_data(X_hat, N_batch, N_seq, N_out):
    """
    Reshapes the model's output back into trajectories.

    Args:
        X_hat (Tensor): Flattened model output tensor of shape [N_batch * N_seq, N_out].
        N_batch (int): Batch size.
        N_seq (int): Sequence length.
        N_out (int): Number of output variables (states).

    Returns:
        Tensor: Reshaped output tensor of shape [N_batch, N_seq, N_out].
    """
    return X_hat.view(N_batch, N_seq, N_out)

def compute_time_derivatives(Z_1, N_in, model):
    """
    Computes time derivatives using forward-mode automatic differentiation (jvp).

    Args:
        Z_1 (Tensor): Input tensor.
        N_in (int): Total input size.
        model (Module): Neural network model.

    Returns:
        Tuple[Tensor, Tensor]: Model output and its time derivative.
    """
    # Direction vector for time input
    v = torch.zeros_like(Z_1)
    v[:, N_in-1] = 1.0  # Set 1.0 for time input derivative
    
    X_2_hat, dX_2_hat_dt = jvp(model, (Z_1,), (v,))
    return X_2_hat, dX_2_hat_dt  # Shape: [N_total, N_x]

def compute_physics_loss(X_2_hat_coll_flat, dX2_hat_dt_flat, U_coll_flat):
    """
    Computes the physics-based loss at collocation points.

    Args:
        X_2_hat_coll_flat (Tensor): Flattened predicted states.
        dX2_hat_dt_flat (Tensor): Flattened time derivatives of predicted states.
        U_coll_flat (Tensor): Flattened control inputs.

    Returns:
        Tensor: Physics loss.
    """  
    # Compute the mean squared residuals
    l_phy = ((dX2_hat_dt_flat - bluerov_compute(0, X_2_hat_coll_flat, U_coll_flat))**2).mean()

    return l_phy

def data_loss_fn(model, X_1, U, time, device, noise_level = 0.0):
        """
        Calculates the 1-step ahead prediction loss.
        """
        Z_1, N_batch, N_seq_1, N_x = convert_input_data(X_1[:, :-1, :] + torch.normal(0, noise_level, X_1[:, :-1, :].shape, device=device), U[:, :-1, :], time[:, :-1, :])
        Z_1 = Z_1.to(device)
        X_2_hat = model(Z_1)
        X_2_hat = convert_output_data(X_2_hat, N_batch, N_seq_1, N_x)
        
        l_data = mse_loss(X_2_hat, X_1[:, 1:]) # output, target
        return l_data
    

def rollout_loss_fn(model, X_0, U, time, N_roll: int, device, t_coll, pinn, noise_level = 0.0):
    """
    Computes the rollout loss over multiple steps.

    Args:
        model (Module): Neural network model.
        X_0 (Tensor): Initial states.
        U (Tensor): Control inputs.
        time (Tensor): Time tensor.
        N_roll (int): Number of rollout steps.
        device (torch.device): Device to run computations on.
        t_coll (Tensor): Collocation times.
        pinn (bool): Flag to include physics-informed loss.
        noise_level (float): Noise level for data augmentation.

    Returns:
        Tuple[Tensor, Tensor]: Rollout loss and physics loss.
    """
    N_seq = X_0.shape[1]
    N_seq_slice = N_seq - N_roll
    X_hat = X_0[:, :N_seq_slice, :]    
    l_roll = 0
    l_phy = 0
    for i in range(N_roll):
        Z_0, N_batch, N_seq, N_x = convert_input_data(X_hat + torch.normal(0, noise_level, X_hat.shape, device=device), U[:, i:i+N_seq_slice, :], time[:, i:i+N_seq_slice, :])
        X_hat = model(Z_0)
        X_hat = convert_output_data(X_hat, N_batch, N_seq, N_x)
        l_roll += mse_loss(X_hat, X_0[:, i+1:i+1+N_seq_slice])
        if pinn:
            l_phy += physics_loss_fn(model, X_hat, U[:, i:i+N_seq_slice, :], t_coll[:, i:i+N_seq_slice, :], device)
    
    l_roll /= N_roll
    l_phy /= N_roll
    
    return l_roll, l_phy
        
def initial_condition_loss(model, X_1, U, time, device):
    """
    Calculates the initial condition prediction loss.
    """
    Z_1, N_batch, N_seq_1, N_x = convert_input_data(X_1, U, torch.zeros_like(time))
    Z_1 = Z_1.to(device)
    
    X_1_hat = model(Z_1)
    X_1_hat = convert_output_data(X_1_hat, N_batch, N_seq_1, N_x)
    
    l_ic = mse_loss(X_1_hat, X_1) # output, target
    
    return l_ic
        
def physics_loss_fn(model, X_1, U, t_coll, device, noise_level=0.0):
    """
    Computes the physics-based loss.

    Args:
        model (Module): Neural network model.
        X_1 (Tensor): States.
        U (Tensor): Control inputs.
        t_coll (Tensor): Collocation times.
        device (torch.device): Device to run computations on.
        noise_level (float): Noise level for data augmentation.

    Returns:
        Tensor: Physics loss.
    """
    N_x = X_1.shape[2]
    N_u = U.shape[-1]
    N_in = N_x + N_u + 1
            
    # Prepare collocation inputs
    Z_1_coll, U_coll = convert_input_collocation(X_1 + torch.normal(0, noise_level, X_1.shape, device=device), U, t_coll)
    Z_1_coll = Z_1_coll.to(device)
    U_coll_flat = (U_coll.to(device)).view(-1, N_u)    
    X_2_hat_coll_flat, dX2_hat_dt_flat = compute_time_derivatives(Z_1_coll, N_in, model)
    l_phy = compute_physics_loss(X_2_hat_coll_flat, dX2_hat_dt_flat, U_coll_flat) 
    
    return l_phy

def get_data_sets(N_batch=32, train_path='training_set', dev_path='dev_set', test_1_path='test_set_interp', test_2_path='test_set_extrap'):
    """
    Loads the training, development, and test datasets.

    Args:
        N_batch (int): Batch size.
        train_path (str): Path to training data.
        dev_path (str): Path to development data.
        test_1_path (str): Path to the first test data.
        test_2_path (str): Path to the second test data.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for training, development, and test sets.
    """
    training_data = TrajectoryDataset(train_path)
    train_dataloader = DataLoader(training_data, batch_size=N_batch, shuffle=True)
    dev_data = TrajectoryDataset(dev_path)
    dev_dataloader = DataLoader(dev_data, batch_size=N_batch, shuffle=True)
    test_1_data = TrajectoryDataset(test_1_path)
    test_1_dataloader = DataLoader(test_1_data, batch_size=N_batch, shuffle=True)
    test_2_data = TrajectoryDataset(test_2_path)
    test_2_dataloader = DataLoader(test_2_data, batch_size=N_batch, shuffle=True)
    return train_dataloader, dev_dataloader, test_1_dataloader, test_2_dataloader

def train(
    model: Module, 
    train_loader: DataLoader, 
    optimizer: AdamW, 
    epoch: int,
    device,
    writer: SummaryWriter,
    pinn: bool,
    rollout: bool,
    noise_level = 0.0,
    gradient_method = 'normalize' # 'normalize', 'direct', or 'config'
):
    model.train()

    epoch_loss_data = 0.0
    epoch_loss_phys = 0.0
    epoch_loss_ic = 0.0
    epoch_loss_roll = 0.0
    epoch_loss_roll_phy = 0.0
    num_batches = 0
 
    for X_1, U, t_coll, time in train_loader:
        num_batches += 1
        grads = []
        X_1 = X_1.to(device); U = U.to(device); time = time.to(device); t_coll = t_coll.to(device)
        
        # --- Calculate all loss components first ---
        l_data = data_loss_fn(model, X_1, U, time, device, noise_level)
        epoch_loss_data += l_data.item() # Use .item() for logging scalars

        l_ic = initial_condition_loss(model, X_1, U, time, device)
        epoch_loss_ic += l_ic.item()

        l_phy = torch.tensor(0.0, device=device)
        if pinn:
            l_phy = physics_loss_fn(model, X_1, U, t_coll, device, noise_level)
            epoch_loss_phys += l_phy.item()

        l_roll = torch.tensor(0.0, device=device)
        l_roll_phy = torch.tensor(0.0, device=device)
        if rollout:
            l_roll, l_roll_phy_calc = rollout_loss_fn(model, X_1, U, time, N_roll=20, device=device, t_coll=t_coll, pinn=pinn, noise_level=noise_level)
            epoch_loss_roll += l_roll.item()
            if pinn:
                l_roll_phy = l_roll_phy_calc
                epoch_loss_roll_phy += l_roll_phy.item()

        # --- Apply gradients based on method ---
        optimizer.zero_grad()

        if gradient_method == 'normalize':
            # --- Normalize Method (Existing Logic) ---
            grads = []
            l_data.backward(retain_graph=True)
            grads.append(get_gradient_vector(model))
            if pinn:
                l_phy.backward(retain_graph=True)
                grads.append(get_gradient_vector(model))
            # Note: l_ic gradient is not explicitly used in the original combination
            if rollout:
                l_roll.backward(retain_graph=True)
                grads.append(get_gradient_vector(model))
                if pinn:
                    l_roll_phy.backward(retain_graph=True)
                    grads.append(get_gradient_vector(model))

            # Scale the physics gradient
            if pinn and rollout:
                g_0_norm = torch.norm(grads[0])
                g_1_norm = torch.norm(grads[1]) # l_phy grad
                g_2_norm = torch.norm(grads[2]) # l_roll grad
                g_3_norm = torch.norm(grads[3]) # l_roll_phy grad
                scaling_factor_1 = g_0_norm / (g_1_norm + 1e-12)
                scaling_factor_2 = g_0_norm / (g_2_norm + 1e-12)
                scaling_factor_3 = g_0_norm / (g_3_norm + 1e-12)
                # Apply scaling: data, phy, roll, roll_phy
                combined_grad = 1.0*grads[0] + 0.5*scaling_factor_1*grads[1] + 1.0*scaling_factor_2*grads[2] + 0.5*scaling_factor_3*grads[3]
                combined_grad = g_0_norm / torch.norm(combined_grad) * combined_grad
            elif rollout: # No pinn
                g_0_norm = torch.norm(grads[0])
                g_1_norm = torch.norm(grads[1]) # l_roll grad
                scaling_factor_1 = g_0_norm / (g_1_norm + 1e-12)
                # Apply scaling: data, roll
                combined_grad = 1.0*grads[0] + 1.0*scaling_factor_1*grads[1]
                combined_grad = g_0_norm / torch.norm(combined_grad) * combined_grad
            elif pinn: # No rollout
                g_0_norm = torch.norm(grads[0])
                g_1_norm = torch.norm(grads[1]) # l_phy grad
                scaling_factor_1 = g_0_norm / (g_1_norm + 1e-12)
                # Apply scaling: data, phy
                combined_grad = 1.0 * grads[0] + 0.5 * scaling_factor_1 * grads[1]
                combined_grad = g_0_norm / torch.norm(combined_grad) * combined_grad
            else: # Only data loss
                combined_grad = grads[0]

            apply_gradient_vector(model, combined_grad)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            writer.add_scalar("Gradient Norm", total_norm, epoch)

        elif gradient_method == 'direct':
            # --- Direct Method (New Logic) ---
            # Combine losses with weights (adjust weights as needed)
            l_total = 1.0 * l_data + 1.0 * l_ic # Always include data and IC loss
            if pinn:
                l_total += 0.5 * l_phy
            if rollout:
                l_total += 1.0 * l_roll
                if pinn:
                    l_total += 0.5 * l_roll_phy

            l_total.backward() # Calculate gradients for the combined loss
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            writer.add_scalar("Gradient Norm", total_norm, epoch)
            writer.add_scalar("Loss_train/total", l_total.item(), epoch) # Log total loss for direct method
            writer.add_scalar("Loss_train/total_log", torch.log10(l_total).item(), epoch)

        elif gradient_method == 'config':
             # --- ConFIG Method (Using conflictfree library) ---
            grads = []
            # Calculate losses and gradients in a specific order expected by the operator
            # Order based on previous 'normalize' logic: data, phy, roll, roll_phy
            l_data.backward(retain_graph=True)
            grads.append(get_gradient_vector(model))
            if pinn:
                l_phy.backward(retain_graph=True)
                grads.append(get_gradient_vector(model))
            # Note: l_ic gradient is not explicitly used here, matching 'normalize'
            if rollout:
                l_roll.backward(retain_graph=True)
                grads.append(get_gradient_vector(model))
                if pinn:
                    l_roll_phy.backward(retain_graph=True)
                    grads.append(get_gradient_vector(model))

            # Use the ConFIG operator
            combined_grad = operator.calculate_gradient(grads) # Use the imported operator

            apply_gradient_vector(model, combined_grad)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            writer.add_scalar("Gradient Norm", total_norm, epoch)

        else:
            raise ValueError(f"Unknown gradient_method: {gradient_method}")

    # --- Logging (remains mostly the same, uses accumulated scalar values) ---
    
    avg_loss_data = epoch_loss_data / num_batches
    writer.add_scalar("Loss_train/data", avg_loss_data, epoch)
    writer.add_scalar("Loss_train/data_log", torch.log10(torch.as_tensor(avg_loss_data)).item(), epoch)
    
    avg_loss_ic = epoch_loss_ic / num_batches
    writer.add_scalar("Loss_train/ic", avg_loss_ic, epoch)
    writer.add_scalar("Loss_train/ic_log", torch.log10(torch.as_tensor(avg_loss_ic)).item(), epoch)
    if rollout:
        avg_loss_roll = epoch_loss_roll / num_batches
        writer.add_scalar("Loss_train/roll", avg_loss_roll, epoch)
        writer.add_scalar("Loss_train/roll_log", torch.log10(torch.as_tensor(avg_loss_roll)).item(), epoch)

    if pinn:
        avg_loss_phys = epoch_loss_phys / num_batches
        writer.add_scalar("Loss_train/phys", avg_loss_phys, epoch)
        writer.add_scalar("Loss_train/phys_log", torch.log10(torch.as_tensor(avg_loss_phys)).item(), epoch)
        if rollout:
            avg_loss_roll_phy = epoch_loss_roll_phy / num_batches
            writer.add_scalar("Loss_train/roll_phy", avg_loss_roll_phy, epoch)
            writer.add_scalar("Loss_train/roll_phy_log", torch.log10(torch.as_tensor(avg_loss_roll_phy)).item(), epoch)
    
    writer.flush()
    return avg_loss_data # return data loss for learning rate scheduler
        
def test_dev_set(model: Module, test_loader: DataLoader, epoch: int, device, writer: SummaryWriter):
    model.eval()
    
    epoch_loss_data = 0.0
    epoch_loss_roll = 0.0
    epoch_loss_roll_phy = 0.0
    epoch_loss_phys_nom = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X_1, U, t_coll, time in test_loader: 
            num_batches += 1
            X_1 = X_1.to(device)
            U = U.to(device)
            t_coll = t_coll.to(device)
            time = time.to(device)
        
            Z, N_batch, N_seq, N_x = convert_input_data(X_1, U, time)
            Z = Z.to(device)
            
            X_2_hat = model(Z)
            X_2_hat = convert_output_data(X_2_hat, N_batch, N_seq, N_x)
        
            l_data = mse_loss(X_2_hat[:, :-1], X_1[:, 1:]) # output, target
            epoch_loss_data += l_data.item()
            if epoch % 25 == 0:
                l_roll, l_roll_phy = rollout_loss_fn(model, X_1, U, time, N_roll=10, device=device, t_coll=t_coll, pinn=1) 
                l_phy_nom = physics_loss_fn(model, X_1, U, t_coll, device)
                epoch_loss_roll += l_roll.item()
                epoch_loss_roll_phy += l_roll_phy
                epoch_loss_phys_nom += l_phy_nom.item()

        avg_loss_data = epoch_loss_data / num_batches
        writer.add_scalar("Loss_dev/data", avg_loss_data, epoch)
        writer.add_scalar("Loss_dev/data_log", torch.log10(torch.as_tensor(avg_loss_data)).item(), epoch)
        if epoch % 25 == 0:
            avg_loss_phys_nom = epoch_loss_phys_nom / num_batches
            avg_loss_roll = epoch_loss_roll / num_batches
            avg_loss_roll_phy = epoch_loss_roll_phy / num_batches
            writer.add_scalar("Loss_dev/phys_nominal", avg_loss_phys_nom, epoch)
            writer.add_scalar("Loss_dev/phys_nominal_log", torch.log10(torch.as_tensor(avg_loss_phys_nom)).item(), epoch)
            writer.add_scalar("Loss_dev/roll", avg_loss_roll, epoch)
            writer.add_scalar("Loss_dev/roll_log", torch.log10(torch.as_tensor(avg_loss_roll)).item(), epoch)
            writer.add_scalar("Loss_dev/roll_phy", avg_loss_roll_phy, epoch)
            writer.add_scalar("Loss_dev/roll_phy_log", torch.log10(torch.as_tensor(avg_loss_roll_phy)).item(), epoch)
            
        writer.flush()
    return avg_loss_data # return data loss for learning rate scheduler

class AdaptiveSoftplus(Module):
    def __init__(self, activation, beta):
        super().__init__()
        self.softplus = activation()
        self.beta = Parameter(torch.tensor(beta, requires_grad=True))  # Per-layer beta parameter

    def forward(self, x):
        return torch.reciprocal(self.beta) * self.softplus(self.beta * x)

class DNN(Module):
    def __init__(self, N_in, N_out, N_h, N_layer, activation=Softplus):
        ''' N_in: input size, N_out: output size, N_h: hidden layer size, N_layer: number of hidden layers, activation: activation function '''
        super().__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.N_h = N_h
        
        activation_adapt = AdaptiveSoftplus
        
        layers = []
        layers.append(Linear(N_in, N_h[0]))
        layers.append(activation_adapt(activation, beta=1.0))
        layers.append(LayerNorm(N_h[0])) 
        
        for i in range(N_layer - 1):
            layers.append(Linear(N_h[i], N_h[i+1]))
            layers.append(activation_adapt(activation, beta=1.0))
            if not (i+1)%2:
                layers.append(LayerNorm(N_h[i+1]))
                
        layers.append(Linear(N_h[-1], N_out))
        
        self.layers = Sequential(*layers)
        self.initialize_weights()
        
    def forward(self, Z):
        X_hat_delta = self.layers(Z)
        cos_psi = X_hat_delta[:, 3] + Z[:, 3]
        sin_psi = X_hat_delta[:, 4] + Z[:, 4]
        x = cos_psi*X_hat_delta[:, 0] - sin_psi*X_hat_delta[:, 1] + Z[:, 0]
        y = sin_psi*X_hat_delta[:, 0] + cos_psi*X_hat_delta[:, 1] + Z[:, 1]
        X_hat = X_hat_delta + Z[:, :self.N_out]
        X_hat[:, 0] = x # overwrite x and y positions with global coordinates
        X_hat[:, 1] = y 
        
        return X_hat
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
