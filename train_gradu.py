import numpy as np
import torch
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *


############################################################   TRAINING   ####################################################################


def train_model(model, train_loader, val_loader,
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
                outputs_val = model(batch_x)           
                
                # Handle both probabilistic and deterministic cases
                if probabilistic:
                    # For probabilistic case, select the mean (first channel)
                    mu_val = outputs_val[:, 0, ...]
                    outputs_val = mu_val.unsqueeze(1) 

                # Now compare with the actual UBM in the criterion
                loss = loss_val(outputs_val, batch_y_ubm)
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







# def train_model(model, train_loader, val_loader,
#                 loss_train, loss_val,
#                 optimizer, device,
#                 save_path='/home/jovyan/SSH/B_data/updated_dm/test3/model.pth',
#                 n_epochs=2000,
#                 patience=50):

#     """
#     This function trains a deep learning model.
    
#     Parameters:
#     - model: The neural network model to be trained
#     - train_loader: DataLoader for training data
#     - val_loader: DataLoader for validation data
#     - criterion: Loss function
#     - optimizer: Optimization algorithm
#     - device: Device to run the model on (CPU or GPU)
#     - save_path: Path to save the trained model
#     - n_epochs: Number of training epochs
#     - patience: Number of epochs to wait for improvement in validation loss
#                 before early stopping.
#     """

#     model.to(device)
    
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
#     # Variables for early stopping and loss tracking
#     best_val_loss = float('inf')
#     patience_counter = 0
#     train_losses = []
#     val_losses = []
    
#     # Check if a saved model exists and load it if it does
#     if os.path.isfile(save_path):
#         checkpoint = torch.load(save_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
#         best_val_loss = checkpoint['best_val_loss']
#         patience_counter = checkpoint['patience_counter']
#         train_losses = checkpoint.get('train_losses', [])
#         val_losses = checkpoint.get('val_losses', [])
#         print(f"Resuming from epoch {start_epoch} with best_val_loss = {best_val_loss:.3e}")
#     else:
#         start_epoch = 0

#     # Main training loop
#     for epoch in range(start_epoch, n_epochs):
        
#         start_time = time.time()
#         model.train()  # Set model to training mode
#         train_running_loss = 0.0

#         # Training phase
#         for batch_x, batch_y in train_loader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
#             # Forward pass
#             outputs = model(batch_x)
#             loss = loss_train(outputs, batch_y)
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             train_running_loss += loss.item() * batch_x.size(0)
            
#         # Calculate average training loss for the epoch
#         epoch_loss = train_running_loss / len(train_loader.dataset)
#         train_losses.append(epoch_loss)


#         # Validation phase
#         model.eval()
#         val_running_loss = 0.0
#         with torch.no_grad():
#             for batch_x, batch_y in val_loader:
#                 batch_x = batch_x.to(device)          
#                 batch_y = batch_y.to(device)   
#                 outputs = model(batch_x)           
#                 loss = loss_val(outputs, batch_y)
#                 val_running_loss += loss.item() * batch_x.size(0)
                
#         val_loss = val_running_loss / len(val_loader.dataset)
#         val_losses.append(val_loss)
        
#         # Update learning rate based on validation loss
#         scheduler.step(val_loss)

#         # Calculate epoch duration
#         end_time = time.time()
#         epoch_duration = end_time - start_time

#         print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.2e}, Val Loss: {val_loss:.2e}, "
#               f"Epoch Time: {epoch_duration:.2f}s")


#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             # Save checkpoint only if we have a new best validation loss
#             checkpoint = {
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'best_val_loss': best_val_loss,
#                 'patience_counter': patience_counter,
#                 'train_losses': train_losses,
#                 'val_losses': val_losses
#             }
#             torch.save(checkpoint, save_path)
#             print(f"Best model so far saved at epoch {epoch+1} (Val Loss: {best_val_loss:.3e})")
#         else:
#             patience_counter += 1
#             print(f"Patience counter: {patience_counter}/{patience}")
        
#         # Early stopping check
#         if patience_counter >= patience:
#             print("Early stopping triggered.")
#             break

    
#     print("Training complete")


############################################################   TESTING   ####################################################################



def evaluate_model(model, device, test_loader, ssh_test, checkpoint_path):

    model = model.to(device)
    model.eval()

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model parameters from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    predictions = []

    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu())

    predictions_np = torch.cat(predictions, dim=0)
    if predictions_np.dim() == 4:  
        predictions_np = predictions_np.squeeze(1)

    ubm_prediction = xr_da(predictions_np, ssh_test)
    bm_prediction = ssh_test - ubm_prediction
    
    return ubm_prediction, bm_prediction


