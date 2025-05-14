

def inverse_zca_transform_torch(data_zca, Vt, scale, data_mean):
    """
    Inverse ZCA transformation in PyTorch (GPU-compatible).
    """
    num_images, img_rows, img_cols = data_zca.shape
    D = img_rows * img_cols
    data_zca_flat = data_zca.view(num_images, D)
    
    # Handle NaNs by imputing with 0 (mean in original space)
    mask = ~torch.isnan(data_zca_flat)
    data_zca_flat_clean = torch.where(mask, data_zca_flat, torch.zeros_like(data_zca_flat))
    
    # Step 1: Project to PCA space
    transformed = torch.matmul(data_zca_flat_clean, Vt.T)
    # Step 2: Undo scaling
    rescaled = transformed / scale
    # Step 3: Project back to original space
    original_centered = torch.matmul(rescaled, Vt)
    data_reconstructed_flat = original_centered + data_mean
    
    # Restore NaNs and reshape
    data_reconstructed_flat[~mask] = torch.nan
    return data_reconstructed_flat.view(num_images, img_rows, img_cols)

def inverse_zca_variance_torch(var_zca, Vt, scale):
    """
    Propagate variance through inverse ZCA in PyTorch.
    """
    num_images, img_rows, img_cols = var_zca.shape
    D = img_rows * img_cols
    var_zca_flat = var_zca.view(num_images, D)
    
    # Step 1: Project variance through Vt.T
    var_step1 = torch.matmul(var_zca_flat, (Vt.T ** 2))
    # Step 2: Scale variance
    var_step2 = var_step1 / (scale ** 2)
    # Step 3: Project variance through Vt
    var_data_flat = torch.matmul(var_step2, (Vt ** 2))
    
    return var_data_flat.view(num_images, img_rows, img_cols)

def inverse_zca_transform_sigma_torch(sigma_zca, Vt, scale):
    """
    Propagate standard deviation through inverse ZCA in PyTorch.
    """
    var_zca = sigma_zca ** 2
    var_data = inverse_zca_variance_torch(var_zca, Vt, scale)
    return torch.sqrt(var_data)
    
def evaluate_model_zca(model, device, test_loader, Vt, scale, data_mean, ssh_test, checkpoint_path, probabilistic=False):
    model = model.to(device)
    model.eval()
    eps = 1e-6

    # Load checkpoint
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    # Convert ZCA params to PyTorch tensors on the same device
    Vt_tensor = torch.from_numpy(Vt).float().to(device)
    scale_tensor = torch.from_numpy(scale).float().to(device)
    data_mean_tensor = torch.from_numpy(data_mean).float().to(device)

    # Collect predictions
    mu_zca_list, sigma_zca_list = [], []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            if probabilistic:
                mu = outputs[:, 0, ...]
                log_sigma = outputs[:, 1, ...]
                sigma = torch.exp(log_sigma).clamp(min=eps)
                mu_zca_list.append(mu)
                sigma_zca_list.append(sigma)
            else:
                mu_zca_list.append(outputs)

    # Process outputs
    mu_zca = torch.cat(mu_zca_list, dim=0)
    if probabilistic:
        sigma_zca = torch.cat(sigma_zca_list, dim=0)
        # Propagate uncertainty on GPU
        sigma_ubm = inverse_zca_transform_sigma_torch(sigma_zca, Vt_tensor, scale_tensor)
        sigma_ubm_np = sigma_ubm.cpu().numpy()
        sigma_zca_np = sigma_zca.cpu().numpy()
    else:
        sigma_zca = None
        sigma_ubm_np = None

    # Invert ZCA for mean prediction
    ubm_prediction = inverse_zca_transform_torch(mu_zca, Vt_tensor, scale_tensor, data_mean_tensor)
    ubm_prediction_np = ubm_prediction.cpu().numpy()
    
    # Convert to xarray DataArrays (assuming `xr_da` handles numpy inputs)
    mu_zca_da = xr_da(mu_zca.cpu().numpy(), ssh_test)
    sigma_zca_da = xr_da(sigma_zca_np, ssh_test) if probabilistic else None
    sigma_ubm_da = xr_da(sigma_ubm_np, ssh_test) if probabilistic else None
    ubm_da = xr_da(ubm_prediction_np, ssh_test)
    bm_da = ssh_test - ubm_da

    return mu_zca_da, sigma_zca_da, sigma_ubm_da, ubm_da, bm_da




def evaluate_model(model, device, test_loader, ssh_test, checkpoint_path):
    model = model.to(device)
    model.eval()

    # Load checkpoint
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    # Collect predictions
    predictions_list = []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions_list.append(outputs)

    # Process outputs
    predictions = torch.cat(predictions_list, dim=0)
    predictions_np = predictions.squeeze(1).cpu().numpy()
    
    # Convert to xarray DataArrays (assuming `xr_da` handles numpy inputs)
    ubm_da = xr_da(predictions_np, ssh_test)
    bm_da = ssh_test - ubm_da

    return ubm_da, bm_da




def isotropic_spectra(data):
    """
    Calculate the isotropic power spectrum of input data.
    
    Parameters:
    -----------
    data : xarray.DataArray
        Input data array with dimensions 'i' and 'j'.
    
    Returns:
    --------
    xarray.DataArray
        The isotropic power spectrum calculated using xrft.
    """
    iso_psd = xrft.isotropic_power_spectrum(
        data, 
        dim=['i', 'j'], 
        detrend='constant', 
        window=True,
        nfactor=2
    )
    return iso_psd



def calculate_psd_km(field, dx=dx_km):
    """
    field : 2-D numpy array [i, j]   (e.g., ubm sample)
    dx    : km per grid cell
    returns: PSD with coords in cycles km⁻¹ and values in m²/(cycles km⁻¹)
    """
    i_km = np.arange(field.shape[0]) * dx
    j_km = np.arange(field.shape[1]) * dx
    da = xr.DataArray(
        field,
        dims=['i', 'j'],
        coords={'i': i_km, 'j': j_km},
        name='field'
    )
    return isotropic_spectra(da) 








def find_best_worst_correlation_indices(true_array, pred_array):
    """
    Find indices of samples with highest and lowest correlation between true and predicted values.
    Works with xarray DataArrays.
    
    Parameters:
    true_array: xarray DataArray of true values 
    pred_array: xarray DataArray of predicted values
    
    Returns:
    best_indices: Indices of 30 samples with highest correlation
    worst_indices: Indices of 30 samples with lowest correlation
    correlations: Array of correlation values for all samples
    """
    n_samples = len(true_array.sample)
    correlations = np.zeros(n_samples)
    
    # Calculate correlation for each sample
    for i in range(n_samples):
        # Get numpy arrays from the DataArrays for this sample
        true_values = true_array.isel(sample=i).values
        pred_values = pred_array.isel(sample=i).values
        
        # Flatten the 2D numpy arrays to 1D for correlation calculation
        true_flat = true_values.flatten()
        pred_flat = pred_values.flatten()
        
        # Calculate Pearson correlation coefficient
        # Handle potential warnings or errors with try-except
        try:
            corr, _ = pearsonr(true_flat, pred_flat)
            # Handle NaN values that might occur
            if np.isnan(corr):
                correlations[i] = -1  # Assign a low value to NaN correlations
            else:
                correlations[i] = corr
        except Exception as e:
            print(f"Error calculating correlation for sample {i}: {e}")
            correlations[i] = -1  # Assign a low value if correlation calculation fails
    
    # Find indices of highest correlations (best samples)
    best_indices = np.argsort(correlations)[-30:][::-1]  # Sort in descending order
    
    # Find indices of lowest correlations (worst samples)
    worst_indices = np.argsort(correlations)[:30]
    
    return best_indices, worst_indices, correlations
    