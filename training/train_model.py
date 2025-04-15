import os
import sys # Keep sys import if needed elsewhere, otherwise remove
import time
import torch
import numpy as np
from torch.nn import Softplus
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# Use relative import now that training/ and models/ are packages
from ..models.model_utility import (
    get_data_sets,
    DNN,
    convert_input_data,
    train,
    test_dev_set
)

# Set seed for reproducibility
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
alpha = 0.6 # filter constant
batch_size = 3
tb_idx = 2
exp_name = "noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed" # experiment name

epochs = 1200
lr_0 = 8e-3
lr_factor = 0.5
lr_patience = 1200
lr_thr = 1e-4
lr_min = 1e-5   
pinn = True
rollout = True
activation = Softplus
noise_level = 0.0 # std of noise level
gradient_method = 'config' # 'normalize', 'direct', or 'config'

def main():
    # Load data
    train_dataloader, dev_dataloader, _, _ = get_data_sets(batch_size)
    X_0, U_0, t_coll_0, tau_0 = next(iter(train_dataloader))
    N_coll = t_coll_0.shape[2]

    # Define network parameters
    N_x = N_out = X_0.shape[-1]
    N_u = U_0.shape[-1]
    N_in = N_x + N_u + 1
    N_h = [32, 32, 32, 32]
    N_layer = len(N_h)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f"runs/{exp_name}_{tb_idx}_{timestamp}") 

    # Initialize model, optimizer, and scheduler
    model = DNN(N_in=N_in, N_out=N_out, N_h=N_h, N_layer=N_layer, activation=activation).to(device)
    optimizer = AdamW(model.parameters(), lr=lr_0)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=lr_factor,
        patience=lr_patience,
        threshold=lr_thr,
        min_lr=lr_min
    )

    # Prepare for training
    Z_0, _, _, N_x = convert_input_data(
        X_0.to(device),
        U_0.to(device),
        tau_0.to(device)
    )
    Z_0.requires_grad_()

    # Create directory for saving models
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    # Include gradient method in the base filename
    base_model_name = f"{exp_name}_{gradient_method}_{tb_idx}" 
    model_path = os.path.join(model_dir, base_model_name)

    l_dev_best = np.float32('inf')
    l_dev_smooth = 1.0  # Initial value of smoothed l_dev
    
    try:
        for epoch in trange(epochs):
            # Training step
            l_train = train(
                model,
                train_dataloader,
                optimizer,
                epoch,
                device,
                writer,
                pinn=pinn,
                rollout=rollout,
                noise_level=noise_level,
                gradient_method=gradient_method # Pass gradient method
            )
            # Validation step
            l_dev = test_dev_set(
                model,
                dev_dataloader,
                epoch,
                device,
                writer # Added missing writer argument
            )
            
            # Update smoothed validation loss
            l_dev_smooth = alpha*l_dev_smooth + (1 - alpha)*l_dev
            
            # Save the best model
            if l_dev < l_dev_best:
                # Include gradient method in the best model filename
                model_path_best = os.path.join(
                    model_dir,
                    f"{base_model_name}_best_dev_l_{epoch}" 
                )
                torch.save(model.state_dict(), model_path_best)
                l_dev_best = l_dev

            # Scheduler step
            scheduler.step(l_dev_smooth)
                
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
            writer.flush() 
        torch.save(model, model_path)    
        writer.add_hparams(
            {
                "Number of collocation points": N_coll,
                "PINN": pinn,
                "Rollout": rollout,
                "lr_0": lr_0,
                "lr_decay": lr_factor,
                "lr_patience": lr_patience,
                "lr_thr": lr_thr,
                "N_batch": batch_size,
                "epochs": int(epoch+1),
                "Number of states (x)": N_x,
                "Number of inputs (u)": N_u,
                "Total input size": N_in,
                "Number of hidden layers": N_layer,
                "Hidden layers size": f"{N_h}",
                "Activation function": f"{activation.__name__}",
                "Noise level": noise_level,
                "Gradient Method": gradient_method,
            },
            {
                "final_train_loss": l_train,
                "final_dev_loss": l_dev
            }
        )
        writer.close()
    except Exception as e:
        print(f"An error occured: {e}")
        writer.close()
        raise

if __name__ == "__main__":
    main()
