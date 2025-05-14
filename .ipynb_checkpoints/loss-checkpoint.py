import torch
import torch.nn as nn
import torch.nn.functional as F


class GradLoss(nn.Module):
    def __init__(self, alpha=1):
        super(GradLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        
    def nan_mse_loss(self, output, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(float('nan'), device=input.device)
        out = (output[mask] - target[mask]) ** 2
        return out.mean()
        
    def compute_gradient(self, img):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
        
        return grad_x, grad_y

    def forward(self, output, target):
        # MSE loss ignoring NaN values
        mse_loss = self.nan_mse_loss(output, target)
        
        # Gradient loss
        output_grad_x, output_grad_y = self.compute_gradient(output)
        target_grad_x, target_grad_y = self.compute_gradient(target)
        
        # Mask for gradient loss (propagate NaN mask to gradients)
        grad_mask = ~torch.isnan(target)
        grad_mask = grad_mask & F.pad(grad_mask[:, :, 1:, :], (0, 0, 1, 0))  # shift mask for x gradient
        grad_mask = grad_mask & F.pad(grad_mask[:, :, :, 1:], (1, 0, 0, 0))  # shift mask for y gradient
        
        grad_loss_x = self.nan_mse_loss(output_grad_x[grad_mask], target_grad_x[grad_mask])
        grad_loss_y = self.nan_mse_loss(output_grad_y[grad_mask], target_grad_y[grad_mask])
        grad_loss = grad_loss_x + grad_loss_y
        
        # Combine losses
        combined_loss = mse_loss + self.alpha * grad_loss
        
        # Handle case where all losses are NaN
        if torch.isnan(combined_loss):
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        
        return combined_loss



class MseLoss(nn.Module):
    def __init__(self):
        super(MseLoss, self).__init__()

    def forward(self, output, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(float('nan'), device=output.device)
        out = (output[mask] - target[mask]) ** 2
        return out.mean()


class GaussianLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(GaussianLoss, self).__init__()
        self.eps = eps
        
    def nan_gaussian_nll(self, mu, log_sigma, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(float('nan'), device=mu.device)
            
        # Clamp sigma and compute corrected log_sigma
        sigma = torch.exp(log_sigma).clamp(min=self.eps)
        log_sigma_clamped = torch.log(sigma)  # Key correction
        
        # Use log_sigma_clamped instead of original log_sigma
        nll = log_sigma_clamped[mask] + 0.5 * ((target[mask] - mu[mask])**2) / (sigma[mask]**2)
        
        return nll.mean()
    
    def forward(self, outputs, target):
        mu = outputs[:, 0, ...]
        log_sigma = outputs[:, 1, ...]
        target_squeezed = target.squeeze(1)
        
        loss = self.nan_gaussian_nll(mu, log_sigma, target_squeezed)
        
        if torch.isnan(loss):
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        return loss





class ZcaMseLoss(nn.Module):
    """
    Loss function that applies ZCA transformation to model outputs before 
    computing MSE against the target ground truth.
    
    This loss handles NaN values in the target.
    """
    
    def __init__(self):
        super(ZcaMseLoss, self).__init__()
    
    def forward(self, output, target, Vt, scale, mean):
        """
        Forward pass to compute the ZCA-MSE loss.
        
        Parameters:
        -----------
        output : torch.Tensor
            Model output tensor of shape (B, 1, H, W)
        target : torch.Tensor
            Target ground truth tensor of shape (B, 1, H, W)
        Vt : torch.Tensor
            The component matrix (eigenvectors transposed)
        scale : torch.Tensor
            The scale factors for the transformation
        mean : torch.Tensor
            The mean that was subtracted before ZCA
            
        Returns:
        --------
        loss : torch.Tensor
            The computed MSE loss between ZCA-transformed output and target.
        """
        # Apply ZCA whitening to model output
        output_zca = self.apply_zca_whitening_4d_torch(output, Vt, scale, mean)
        
        # Compute MSE between transformed output and target, handling NaNs
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(float('nan'), device=output.device)
        
        out = (output_zca[mask] - target[mask]) ** 2
        return out.mean()
    
    def apply_zca_whitening_4d_torch(self, data, Vt, scale, mean):
        """
        Apply ZCA whitening for 4D PyTorch tensors with shape (B, 1, H, W).
        
        Parameters:
        -----------
        data : torch.Tensor
            Input tensor of shape (B, 1, H, W)
        Vt : torch.Tensor
            The component matrix (eigenvectors transposed)
        scale : torch.Tensor
            The scale factors for the transformation
        mean : torch.Tensor
            The mean that was subtracted before ZCA
            
        Returns:
        --------
        whitened : torch.Tensor
            ZCA-whitened tensor of shape (B, 1, H, W)
        """
        original_shape = data.shape
        B = data.shape[0]
        # Flatten the data: shape (B, features)
        data_flat = data.reshape(B, -1)
        
        # Center data (no need to handle NaNs in output)
        data_centered = data_flat - mean
        
        # Apply transformation in 3 steps
        transformed = data_centered @ Vt.T    # (1) Project to PCA space
        transformed = transformed * scale      # (2) Scale components
        whitened = transformed @ Vt            # (3) Project back to original space
        
        return whitened.reshape(original_shape)

        

# class GaussianLoss(nn.Module):
#     """
#     Probabilistic loss assuming outputs come from a Normal distribution 
#     parameterized by mean and log_std, with NaN handling.
    
#     Model outputs shape = (B, 2, H, W):
#        mu        = output[:, 0, ...]
#        log_sigma = output[:, 1, ...]
#     Target shape = (B, 1, H, W).
#     """
#     def __init__(self, eps=1e-6):
#         """
#         eps: small constant to avoid numerical issues when exponentiating log_sigma.
#         """
#         super(GaussianLoss, self).__init__()
#         self.eps = eps
        
#     def nan_gaussian_nll(self, mu, log_sigma, target):
#         """
#         Compute Gaussian negative log-likelihood while handling NaN values.
        
#         Args:
#             mu: (B, H, W) mean predictions
#             log_sigma: (B, H, W) log-standard deviation predictions
#             target: (B, H, W) target values
            
#         Returns:
#             Scalar mean negative log-likelihood, ignoring NaN values.
#         """
#         # Create mask for valid (non-NaN) target values
#         mask = ~torch.isnan(target)
        
#         # If all values are NaN, return NaN
#         if mask.sum() == 0:
#             return torch.tensor(float('nan'), device=mu.device)
            
#         # Convert log_sigma to sigma, applying eps for numerical stability
#         sigma = torch.exp(log_sigma).clamp(min=self.eps)
        
#         # Compute NLL only for valid values
#         # NLL = log(sigma) + ((y - mu)**2) / (2 * sigma^2)
#         nll = log_sigma[mask] + 0.5 * ((target[mask] - mu[mask])**2) / (sigma[mask]**2)
        
#         return nll.mean()
    
#     def forward(self, outputs, target):
#         """
#         outputs: (B, 2, H, W)
#            - outputs[:, 0, ...] = mu
#            - outputs[:, 1, ...] = log_sigma
#         target: (B, 1, H, W)
#         Returns:
#           Scalar (mean) negative log-likelihood, handling NaN values.
#         """
#         # Separate the channels: mean and log-sigma
#         mu = outputs[:, 0, ...]         # shape (B, H, W)
#         log_sigma = outputs[:, 1, ...]  # shape (B, H, W)
        
#         # Squeeze the single channel from target
#         target_squeezed = target.squeeze(1)  # shape (B, H, W)
        
#         # Compute NLL with NaN handling
#         loss = self.nan_gaussian_nll(mu, log_sigma, target_squeezed)
        
#         # Handle case where all values are NaN
#         if torch.isnan(loss):
#             return torch.tensor(0.0, device=outputs.device, requires_grad=True)
            
#         return loss

# class GaussianNLLLoss(nn.Module):
#     """
#     Probabilistic loss assuming outputs come from a Normal distribution 
#     parameterized by mean and log_std.
    
#     Model outputs shape = (B, 2, H, W):
#        mu        = output[:, 0, ...]
#        log_sigma = output[:, 1, ...]
#     Target shape = (B, 1, H, W).
#     """
#     def __init__(self, eps=1e-6):
#         """
#         eps: small constant to avoid numerical issues when exponentiating log_sigma.
#         """
#         super(GaussianNLLLoss, self).__init__()
#         self.eps = eps

#     def forward(self, outputs, target):
#         """
#         outputs: (B, 2, H, W)
#            - outputs[:, 0, ...] = mu
#            - outputs[:, 1, ...] = log_sigma
#         target: (B, 1, H, W)

#         Returns:
#           Scalar (mean) negative log-likelihood.
#         """

#         # Separate the channels: mean and log-sigma
#         mu        = outputs[:, 0, ...]   # shape (B, H, W)
#         log_sigma = outputs[:, 1, ...]   # shape (B, H, W)

#         # Convert log_sigma -> sigma
#         sigma = torch.exp(log_sigma).clamp(min=self.eps)

#         # Squeeze the single channel from target => (B, H, W)
#         target_squeezed = target.squeeze(1)  # shape (B, H, W)

#         # Negative log-likelihood of a Gaussian:
#         #  NLL = log(sigma) + ((y - mu)**2) / (2 * sigma^2)
#         nll = log_sigma + 0.5 * ((target_squeezed - mu)**2) / (sigma**2)

#         return nll.mean()



