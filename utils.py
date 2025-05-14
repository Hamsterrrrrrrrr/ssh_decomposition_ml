import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xrft
from scipy import linalg
import os
import torch



#------------------------------------------------------------------------------#
#                         ZCA Parameters Calculation                            #
#------------------------------------------------------------------------------#

def calculate_zca_params(data, epsilon=1e-5):
    """
    Calculate the ZCA whitening matrix (zca_matrix) and the mean of the data.
    
    Parameters:
    -----------
    data : np.array
        Input data of shape (n_samples, height, width) or (n_samples, ...).
    epsilon : float, optional
        A small value added to the eigenvalues for numerical stability 
        when computing the inverse sqrt of the covariance matrix.
    
    Returns:
    --------
    zca_matrix : np.array
        The ZCA whitening matrix.
    mean : np.array
        The mean of the data along the flattening dimension.
    """

    # Reshape the data to 2D (n_samples, flattened_features)
    data_flat = data.reshape((data.shape[0], -1))
    
    # Create a mask to track NaN values
    mask = ~np.isnan(data_flat)
    
    # Compute mean while ignoring NaNs
    mean = np.nanmean(data_flat, axis=0)
    
    # Center the data by subtracting the mean
    data_centered = data_flat - mean
    # Replace NaNs with 0 after centering
    data_centered[~mask] = 0
    
    # Compute the covariance matrix using the centered data
    cov_matrix = np.dot(data_centered.T, data_centered) / np.sum(mask, axis=0).clip(min=1)
    
    # Perform Singular Value Decomposition (SVD)
    U, S, _ = linalg.svd(cov_matrix)
    
    # Compute the ZCA whitening matrix
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    
    return zca_matrix, mean




#------------------------------------------------------------------------------#
#                            ZCA Whitening Application                          #
#------------------------------------------------------------------------------#

def apply_zca_whitening(data, zca_matrix, mean):
    """
    Apply ZCA whitening transformation to the input data using the provided
    ZCA matrix and mean.
    
    Parameters:
    -----------
    data : np.array
        Input data of shape (n_samples, height, width) or (n_samples, ...).
    zca_matrix : np.array
        ZCA whitening matrix calculated from `calculate_zca_params`.
    mean : np.array
        Mean vector used for centering.
    
    Returns:
    --------
    np.array
        Whitened data with the same shape as the input.
    """

    # Store the original shape for reshaping later
    original_shape = data.shape
    
    # Reshape data into 2D (n_samples, flattened_features)
    data_flat = data.reshape((data.shape[0], -1))
    
    # Create a mask to identify non-NaN values
    mask = ~np.isnan(data_flat)
    
    # Center the data using the mean
    data_centered = data_flat - mean
    # Replace NaNs with 0 for processing
    data_centered[~mask] = 0
    
    # Apply the ZCA whitening transformation
    data_whitened = np.dot(data_centered, zca_matrix)
    
    # Restore NaN values in the places where the original data had NaNs
    data_whitened[~mask] = np.nan
    
    # Reshape the whitened data back to its original shape
    return data_whitened.reshape(original_shape)




#------------------------------------------------------------------------------#
#                         Inverse ZCA Transformation                            #
#------------------------------------------------------------------------------#

def inverse_zca_transform(data_zca, zca_matrix, data_mean):
    """
    Apply inverse ZCA transformation to reconstruct the original data from the 
    whitened data.
    
    Parameters:
    -----------
    data_zca : np.array
        Whitened data of shape (n_samples, height, width) or (n_samples, ...).
    zca_matrix : np.array
    data_mean : np.array
    
    Returns:
    --------
    np.array
        The reconstructed data with the same shape as `data_zca`.
    """
    
    # Extract dimensions of the data
    num_images, img_rows, img_cols = data_zca.shape
    
    # Flatten the whitened data for matrix multiplication
    data_zca_flat = data_zca.reshape(num_images, img_rows * img_cols)
    
    # Compute the inverse of the ZCA matrix
    inverse_zca_matrix = np.linalg.inv(zca_matrix.T)
    
    # Multiply the flattened ZCA data by the inverse matrix
    data_reconstructed_flat = np.dot(data_zca_flat, inverse_zca_matrix)
    
    # Add back the mean to 'uncenter' the data
    data_reconstructed = data_reconstructed_flat + data_mean
    
    # Reshape the data back to its original (num_images, rows, cols)
    data_reconstructed = data_reconstructed.reshape(num_images, img_rows, img_cols)
    
    return data_reconstructed





#------------------------------------------------------------------------------#
#                         Evaluate ZCA-UNet Performance                         #
#------------------------------------------------------------------------------#





#------------------------------------------------------------------------------#
#                         Isotropic Power Spectrum                              #
#------------------------------------------------------------------------------#



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




#------------------------------------------------------------------------------#
#                      NumPy to xarray.DataArray Conversion                     #
#------------------------------------------------------------------------------#

def xr_da(data_np, data_like):
    """
    Convert NumPy array to xarray DataArray using dimensions and coordinates 
    from a template DataArray.
    
    Parameters:
    -----------
    data_np : numpy.ndarray
        Input NumPy array to be converted.
    data_like : xarray.DataArray
        Template DataArray with the desired dimensions and coordinates.
    
    Returns:
    --------
    xarray.DataArray
        The converted data with dimensions and coordinates from data_like.
    """
    coords = {dim: data_like.coords[dim].values for dim in data_like.dims}
    data_xr = xr.DataArray(data_np, dims=data_like.dims, coords=coords)
    return data_xr






def plot_field(data_xr, title, ax, cmap='jet', vmin=None, vmax=None, add_colorbar=False):

    data = data_xr.values
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, extend='both', orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.text(0.5, 1.05, '(m)', transform=cbar.ax.transAxes, ha='center', va='bottom')


def reconstruct_from_patches(patches, original_shape=(5, 2160, 2160), patch_size=108):
    time_steps, height, width = original_shape
    samples_per_timestep = patches.sizes['sample'] // time_steps
    
    reconstructed = np.zeros(original_shape)
    
    for t in range(time_steps):
        start_idx = t * samples_per_timestep
        end_idx = (t + 1) * samples_per_timestep
        time_patches = patches[start_idx:end_idx]
        
        patch_idx = 0
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                reconstructed[t, i:i+patch_size, j:j+patch_size] = time_patches[patch_idx].values
                patch_idx += 1
    
    return xr.DataArray(reconstructed, dims=['time', 'y', 'x'])