# import xarray as xr
import numpy as np
import torch
# import torch.nn as nn
import os
import time
# from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau



############################################################   TRAINING   ####################################################################

def inverse_zca_transformation_4d_torch(data_zca, zca_matrix, data_mean):
    """
    Inverse ZCA transformation in PyTorch for 4D data (NCHW format).

    Parameters:
    -----------
    data_zca : torch.Tensor
        ZCA-transformed data of shape (batch, channels, height, width).
    zca_matrix : torch.Tensor
        The whitening matrix used for ZCA, shape (C*H*W, C*H*W).
        Must be on the same device as data_zca.
    data_mean : torch.Tensor
        The mean that was subtracted before ZCA; shape (C*H*W,) or (1, C*H*W).
        Also on the same device as data_zca.

    Returns:
    --------
    data_reconstructed : torch.Tensor
        Data in the original (unwhitened) space, shape (batch, channels, height, width).
    """

    B, C, H, W = data_zca.shape

    # Flatten the data to shape (B, C*H*W)
    data_zca_flat = data_zca.view(B, -1)

    # Possibly invert the matrix once outside of this function, if performance is an issue
    # If zca_matrix is symmetrical or if you already have the inverse, skip this step.
    inverse_zca_matrix = torch.linalg.inv(zca_matrix)

    # Multiply: (B, C*H*W) @ (C*H*W, C*H*W) -> (B, C*H*W)
    data_reconstructed_flat = data_zca_flat @ inverse_zca_matrix

    # Add the mean
    # Ensure data_mean is shape (C*H*W,) or (1, C*H*W) for broadcasting
    data_reconstructed_flat += data_mean

    # Reshape to (B, C, H, W)
    data_reconstructed = data_reconstructed_flat.view(B, C, H, W)

    return data_reconstructed



def train_model(model, train_loader, val_loader,
                zca_matrix, data_mean,
                loss_train, loss_val,
                optimizer, device,
                probabilistic=False, 
                save_path='/home/jovyan/SSH/B_data/updated_dm/test3/model.pth',
                n_epochs=2000,
                patience=50):

    """
    This function trains a deep learning model.
    
    Parameters:
    - model: The neural network model to be trained
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - criterion: Loss function
    - optimizer: Optimization algorithm
    - device: Device to run the model on (CPU or GPU)
    - save_path: Path to save the trained model
    - n_epochs: Number of training epochs
    - patience: Number of epochs to wait for improvement in validation loss
                before early stopping.
    """

    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Variables for early stopping and loss tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Check if a saved model exists and load it if it does
    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Resuming from epoch {start_epoch} with best_val_loss = {best_val_loss:.3e}")
    else:
        start_epoch = 0

    # Main training loop
    for epoch in range(start_epoch, n_epochs):
        
        start_time = time.time()
        model.train()  # Set model to training mode
        train_running_loss = 0.0

        # Training phase
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = loss_train(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item() * batch_x.size(0)
            
        # Calculate average training loss for the epoch
        epoch_loss = train_running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)


        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y_ubm in val_loader:
                batch_x = batch_x.to(device)           # SSH input
                batch_y_ubm = batch_y_ubm.to(device)   # UBM target
                outputs_zca = model(batch_x)           # In ZCA space
                
                # Handle both probabilistic and deterministic cases
                if probabilistic:
                    # For probabilistic case, select the mean (first channel)
                    mu_zca = outputs_zca[:, 0, ...]
                    outputs_zca = mu_zca.unsqueeze(1) 

                # Perform the inverse ZCA transform
                outputs_ubm = inverse_zca_transformation_4d_torch(
                    outputs_zca, zca_matrix, data_mean
                )
        
                # Now compare with the actual UBM in the criterion
                loss = loss_val(outputs_ubm, batch_y_ubm)
                val_running_loss += loss.item() * batch_x.size(0)
                
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Calculate epoch duration
        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.2e}, Val Loss: {val_loss:.2e}, "
              f"Epoch Time: {epoch_duration:.2f}s")


        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save checkpoint only if we have a new best validation loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(checkpoint, save_path)
            print(f"Best model so far saved at epoch {epoch+1} (Val Loss: {best_val_loss:.3e})")
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
        
        # Early stopping check
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    
    print("Training complete")


############################################################   TESTING   ####################################################################

def inverse_zca_transform(data_zca, zca_matrix, data_mean):
    num_images, img_rows, img_cols = data_zca.shape
    data_zca_flat = data_zca.reshape(num_images, img_rows * img_cols)
    inverse_zca_matrix = np.linalg.inv(zca_matrix.T)
    data_reconstructed_flat = np.dot(data_zca_flat, inverse_zca_matrix)
    data_reconstructed = data_reconstructed_flat + data_mean
    data_reconstructed = data_reconstructed.reshape(num_images, img_rows, img_cols)
    return data_reconstructed


def evaluate_model_zca(model, device, test_loader, zca_matrix, data_mean, ssh_test, checkpoint_path, probabilistic=False):
    """
    Evaluate model with flexibility for probabilistic or deterministic predictions.
    
    Args:
        model: Neural network model
        device: Computing device (CPU/GPU)
        test_loader: DataLoader for test data
        zca_matrix: ZCA whitening matrix
        data_mean: Mean used in ZCA whitening
        ssh_test: Sea surface height test data
        checkpoint_path: Path to model checkpoint
        probabilistic: Boolean flag for probabilistic prediction (default: False)
    
    Returns:
        If probabilistic:
            mu_zca_np: Mean predictions in ZCA space
            sigma_zca_np: Standard deviation predictions in ZCA space
            ubm_mu_pre: Mean predictions in original space
            bm_mu_pre: Balanced motion predictions
        If deterministic:
            None, None, ubm_pre, bm_pre: Same format but first two are None
    """
    model = model.to(device)
    model.eval()
    eps = 1e-6

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model parameters from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    predictions = []
    uncertainties = [] if probabilistic else None

    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs_zca = model(batch_x)
            
            if probabilistic:
                # Split into mean and uncertainty
                mu = outputs_zca[:, 0, ...]
                log_sigma = outputs_zca[:, 1, ...]
                sigma = torch.exp(log_sigma).clamp(min=eps)
                
                predictions.append(mu.cpu())
                uncertainties.append(sigma.cpu())
            else:
                # For deterministic case, output is already the prediction
                predictions.append(outputs_zca.cpu())

    # Process predictions
    if probabilistic:
        mu_zca_np = torch.cat(predictions, dim=0).numpy()
        sigma_zca_np = torch.cat(uncertainties, dim=0).numpy()
    else:
        # For deterministic case, ensure correct shape
        predictions_np = torch.cat(predictions, dim=0)
        if predictions_np.dim() == 4:  # If output includes channel dimension
            predictions_np = predictions_np.squeeze(1)
        mu_zca_np = predictions_np.numpy()
        sigma_zca_np = None
    
    # Apply inverse ZCA transformation
    ubm_prediction = xr_da(inverse_zca_transform(mu_zca_np, zca_matrix, data_mean), ssh_test)
    bm_prediction = ssh_test - ubm_prediction
    
    return mu_zca_np, sigma_zca_np, ubm_prediction, bm_prediction

# def evaluate_model(model, device, test_loader, ssh_test, checkpoint_path):

#     """
#     Evaluate a trained model on test data and generate predictions.

#     Parameters:
#     - model: The neural network model to be evaluated
#     - device: Device to run the model on (CPU or GPU)
#     - test_loader: DataLoader containing the test data
#     - ssh_test: Sea Surface Height test data (xarray DataArray)
#     - checkpoint_path: Path to the saved model checkpoint

#     Returns:
#     - bm_prediction: Balanced motion prediction (SSH_test - UBM_prediction)
#     - ubm_prediction: Unbalanced motion prediction
#     """
    
#     # Move the model to the specified device (CPU or GPU)
#     model = model.to(device)
#     # Set the model to evaluation mode
#     model.eval()  
    
#     # Load the trained model parameters from the checkpoint
#     if os.path.isfile(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"Loaded model parameters from {checkpoint_path}")
#     else:
#         raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

#     predictions = []
    
#     with torch.no_grad():
#         for batch_x, _ in test_loader:  
#             batch_x = batch_x.to(device)
#             y_pred = model(batch_x)
#             predictions.append(y_pred.cpu())
            
#     # Combine all batch predictions and convert to numpy array
#     prediction = (torch.cat(predictions, dim=0)).squeeze(1).numpy() 

#     # Convert predictions to xarray DataArray with the same structure as ssh_test
#     ubm_prediction = xr_da(prediction, ssh_test)
    
#     bm_prediction = ssh_test - ubm_prediction
    
#     return bm_prediction, ubm_prediction


# def evaluate_model_zca(model, device, test_loader, zca_matrix, data_mean, ssh_test, checkpoint_path):
#     """
#     Evaluate a trained model on test data and generate predictions, with ZCA whitening.

#     Parameters:
#     - model: The neural network model to be evaluated
#     - device: Device to run the model on (CPU or GPU)
#     - test_loader: DataLoader containing the test data
#     - zca_matrix: ZCA whitening matrix for inverse transformation
#     - data_mean: Mean of the data used for ZCA whitening
#     - ssh_test: Sea Surface Height test data (xarray DataArray)
#     - checkpoint_path: Path to the saved model checkpoint

#     Returns:
#     - bm_prediction: Balanced motion prediction (SSH_test - UBM_prediction)
#     - ubm_prediction: Unbalanced motion prediction
#     """    
#     model = model.to(device)
#     model.eval()  

#     if os.path.isfile(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"Loaded model parameters from {checkpoint_path}")
#     else:
#         raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

#     predictions = []
    
#     with torch.no_grad():
#         for batch_x, _ in test_loader:  
#             batch_x = batch_x.to(device)
#             y_pred = model(batch_x)
#             predictions.append(y_pred.cpu())

#     prediction_np = (torch.cat(predictions, dim=0)).squeeze(1).numpy() 
    
#     # Apply inverse ZCA transformation to the predictions
#     prediction = inverse_zca_transformation(prediction_np, zca_matrix, data_mean)
    
#     # Convert predictions to xarray DataArray with the same structure as ssh_test
#     ubm_prediction = xr_da(prediction, ssh_test)
    
#     bm_prediction = ssh_test - ubm_prediction
    
#     return bm_prediction, ubm_prediction



