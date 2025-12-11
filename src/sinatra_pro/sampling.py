"""
Gaussian Process Classification with Elliptical Slice Sampling in PyTorch
"""

import torch
import numpy as np
from typing import Optional, Callable
from torch.distributions import MultivariateNormal, Normal


def gaussian_kernel_covariance(
    X: torch.Tensor,
    bandwidth: float = 0.01
) -> torch.Tensor:
    """
    Gaussian kernel covariance.
    
    Args:
        X: Design matrix of shape (n_features, n_samples)
        bandwidth: Bandwidth parameter
    
    Returns:
        Covariance matrix K of shape (n_samples, n_samples)
    """
    n = X.shape[1]
    
    # Compute pairwise squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
    X_norm_sq = (X**2).sum(dim=0, keepdim=True)  # (1, n)
    squared_dists = X_norm_sq.T + X_norm_sq - 2 * (X.T @ X)  # (n, n)
    
    # Average over features (mean squared distance)
    mean_squared_dists = squared_dists / X.shape[0]
    
    # Apply Gaussian kernel
    bandwidth_factor = 1.0 / (2 * bandwidth**2)
    K = torch.exp(-mean_squared_dists * bandwidth_factor)
    
    return K


def probit_log_likelihood(
    latent_variables: torch.Tensor,
    class_labels: torch.Tensor
) -> torch.Tensor:
    """
    Probit link function log-likelihood.
    
    Args:
        latent_variables: Latent function values, shape (n,)
        class_labels: Binary class labels {-1, +1}, shape (n,)
    
    Returns:
        Log-likelihood (scalar)
    
    Formula:
        log p(y|f) = Σ log Φ(y_i * f_i)
        where Φ is the standard normal CDF
    """
    # Standard normal distribution
    normal = Normal(0, 1)
    
    # Compute Φ(y * f)
    # class_labels should be {-1, +1}
    cdf_values = normal.cdf(latent_variables * class_labels)
    
    # Clamp to avoid log(0)
    cdf_values = torch.clamp(cdf_values, min=1e-10, max=1-1e-10)
    
    return torch.sum(torch.log(cdf_values))


def logistic_log_likelihood(
    latent_variables: torch.Tensor,
    class_labels: torch.Tensor
) -> torch.Tensor:
    """
    Logistic link function log-likelihood.
    
    Args:
        latent_variables: Latent function values, shape (n,)
        class_labels: Binary class labels {-1, +1}, shape (n,)
    
    Returns:
        Log-likelihood (scalar)
    
    Formula:
        log p(y|f) = -Σ log(1 + exp(-y_i * f_i))
    """
    # Use log-sum-exp trick for numerical stability
    # log(1 + exp(-y*f)) = log(exp(0) + exp(-y*f))
    #                    = max(0, -y*f) + log(exp(-max) + exp(-y*f - max))
    
    product = latent_variables * class_labels
    
    # More stable: use torch.nn.functional.softplus
    # log(1 + exp(x)) = softplus(x)
    return -torch.sum(torch.nn.functional.softplus(-product))


def elliptical_slice_sampling(
    K: torch.Tensor,
    y: torch.Tensor,
    n_mcmc_steps: int = 100000,
    n_burn_in_steps: int = 1000,
    probit: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Elliptical Slice Sampling for Gaussian Process classification.
    
    This is a MCMC sampling algorithm for models with Gaussian priors.
    Reference: Murray, Adams, and MacKay (2010) "Elliptical slice sampling"
    
    Args:
        K: Covariance matrix of GP prior, shape (n, n)
        y: Binary class labels (should be {-1, +1}), shape (n,)
        n_mcmc_steps: Number of MCMC samples to return (default: 100000)
        n_burn_in_steps: Number of burn-in iterations (default: 1000)
        probit: Use probit link if True, logistic link if False (default: True)
        seed: Random seed for reproducibility (default: None)
        verbose: Print progress information (default: False)
        device: Torch device (default: same as K)
    
    Returns:
        MCMC samples of shape (n_mcmc_steps, n)
    
    Algorithm:
        1. Initialize f from the prior
        2. For each iteration:
           a. Draw ν ~ N(0, K)
           b. Choose threshold: log(u) + log p(y|f)
           c. Draw initial angle θ ~ Uniform(0, 2π)
           d. Propose: f' = f*cos(θ) + ν*sin(θ)
           e. Accept if log p(y|f') > threshold, else shrink bracket
    """
    if device is None:
        device = K.device
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Choose likelihood function
    log_lik = probit_log_likelihood if probit else logistic_log_likelihood
    
    n = K.shape[0]
    N = y.shape[0]
    
    assert n == N, f"Dimension mismatch: K is {n}x{n} but y has {N} elements"
    
    # Ensure y is on the correct device and is float
    y = y.to(device).float()
    K = K.to(device)
    
    if verbose:
        print(f"Running elliptical slice sampling...")
        print(f"  Samples: {n}")
        print(f"  Burn-in: {n_burn_in_steps}")
        print(f"  MCMC iterations: {n_mcmc_steps}")
        print(f"  Link function: {'probit' if probit else 'logistic'}")
    
    # Pre-allocate arrays
    total_iterations = n_burn_in_steps + n_mcmc_steps
    mcmc_samples = torch.zeros((total_iterations, N), dtype=torch.float32, device=device)
    
    # Pre-generate random numbers for efficiency
    # Sample from N(0, K) using Cholesky decomposition
    try:
        L = torch.linalg.cholesky(K)
        standard_normals = torch.randn((total_iterations, N), device=device)
        norm_samples = (L @ standard_normals.T).T  # (total_iterations, N)
    except RuntimeError:
        # If Cholesky fails, add small jitter and try again
        if verbose:
            print("  Warning: Adding jitter to covariance matrix")
        K_jitter = K + torch.eye(n, device=device) * 1e-6
        L = torch.linalg.cholesky(K_jitter)
        standard_normals = torch.randn((total_iterations, N), device=device)
        norm_samples = (L @ standard_normals.T).T
    
    unif_samples = torch.rand(total_iterations, device=device)
    theta = torch.rand(total_iterations, device=device) * 2 * np.pi
    theta_min = theta - 2 * np.pi
    theta_max = theta + 2 * np.pi
    
    # Initialize first sample
    mcmc_samples[0] = norm_samples[0]
    
    # Main sampling loop
    for i in range(1, total_iterations):
        if verbose:
            if i <= n_burn_in_steps:
                if i % 100 == 0:
                    print(f"\r  Burn-in: {i}/{n_burn_in_steps}", end='')
            else:
                if (i - n_burn_in_steps) % 1000 == 0:
                    print(f"\r  Sampling: {i - n_burn_in_steps}/{n_mcmc_steps}", end='')
        
        # Current state
        f = mcmc_samples[i - 1]
        
        # Log-likelihood threshold
        llh_thresh = log_lik(f, y) + torch.log(unif_samples[i])
        
        # Propose new point on ellipse
        f_star = f * torch.cos(theta[i]) + norm_samples[i] * torch.sin(theta[i])
        
        # Shrink bracket until acceptance
        max_iterations = 1000  # Prevent infinite loops
        iteration_count = 0
        
        while log_lik(f_star, y) < llh_thresh and iteration_count < max_iterations:
            # Shrink bracket
            if theta[i] < 0:
                theta_min[i] = theta[i]
            else:
                theta_max[i] = theta[i]
            
            # Draw new angle from shrunken bracket
            theta[i] = theta_min[i] + torch.rand(1, device=device) * (theta_max[i] - theta_min[i])
            
            # New proposal
            f_star = f * torch.cos(theta[i]) + norm_samples[i] * torch.sin(theta[i])
            
            iteration_count += 1
        
        if iteration_count >= max_iterations and verbose:
            print(f"\n  Warning: Max iterations reached at step {i}")
        
        mcmc_samples[i] = f_star
    
    if verbose:
        print("\n  Sampling complete!")
    
    # Return samples after burn-in
    return mcmc_samples[n_burn_in_steps:]
