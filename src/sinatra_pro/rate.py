import torch
import numpy as np
from typing import Optional, Tuple, Dict
from torch.linalg import svd
from sinatra_pro.sampling import gaussian_kernel_covariance, elliptical_slice_sampling


def sherman_morrison(A_inv: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Sherman-Morrison formula: compute A_inv - (A_inv @ u)(v^T @ A_inv) / (1 + v^T @ A_inv @ u)
    
    Args:
        A_inv: Inverse matrix (or precision matrix) of shape (n, n)
        u: Column vector of shape (n,) or (n, 1)
        v: Column vector of shape (n,) or (n, 1)
    
    Returns:
        Updated inverse matrix of shape (n, n)
    """
    # Ensure column vectors
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)
    
    # Compute denominator: 1 + v^T @ A_inv @ u
    denominator = 1 + (v.T @ A_inv @ u).item()
    
    # Compute numerator: (A_inv @ u) @ (v^T @ A_inv)
    numerator = (A_inv @ u) @ (v.T @ A_inv)
    
    return A_inv - numerator / denominator


def calc_kld(mu: torch.Tensor, Lambda: torch.Tensor, V: torch.Tensor, q: int) -> torch.Tensor:
    """
    Calculate Kullback-Leibler divergence for variant q.
    
    This measures how much information variant q provides about all other variants
    in the posterior distribution of effect size analogs.
    
    Args:
        mu: Mean vector of effect size analogs, shape (p,)
        Lambda: Precision matrix (inverse covariance), shape (p, p)
        V: Covariance matrix of effect size analogs, shape (p, p)
        q: Index of variant to test (0 to p-1)
    
    Returns:
        KLD value (scalar tensor)
    
    Formula (from paper equation 9):
        KLD_q = 0.5 * μ_q^2 * α_q
        where α_q = λ_{-q,q}^T Λ_{-q,-q}^{-1} λ_{-q,q}
    """
    # Sherman-Morrison update using q-th column of covariance matrix
    v_q = V[:, q]
    Lambda_updated = sherman_morrison(Lambda, v_q, v_q)
    
    # Create mask to exclude index q
    mask = torch.ones(Lambda_updated.shape[0], dtype=torch.bool, device=Lambda.device)
    mask[q] = False
    
    # Extract submatrices
    # λ_{-q,q}: column q without row q
    lambda_minus_q = Lambda_updated[mask, q]
    
    # Λ_{-q,-q}: precision matrix without row/col q
    Lambda_minus_minus = Lambda_updated[mask][:, mask]
    
    # Compute α_q = λ_{-q,q}^T @ Λ_{−q,−q}^{−1} @ λ_{-q,q}
    # Add a small jitter for numerical stability to prevent singular matrix errors.
    # Lambda_minus_minus += 2e-6 * (torch.rand_like(Lambda_minus_minus) - 0.5)
    # lambda_minus_q += 2e-6 * (torch.rand_like(lambda_minus_q) - 0.5)
    # alpha_q = lambda_minus_q @ torch.linalg.solve(Lambda_minus_minus, lambda_minus_q)
    alpha_q = lambda_minus_q.T @ Lambda_minus_minus @ lambda_minus_q
    
    # Compute KLD = 0.5 * μ_q^2 * α_q
    kld = 0.5 * mu[q]**2 * alpha_q
    
    return kld


def rate(
    X: torch.Tensor,
    f_draws: Optional[torch.Tensor] = None,
    beta_draws: Optional[torch.Tensor] = None,
    prop_var: float = 1.0,
    low_rank: bool = False,
    rank_r: Optional[int] = None,
    nullify: Optional[list] = None,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Variable Prioritization via RelATive cEntrality (RATE) centrality measures.
    
    Computes RATE scores for each variable to prioritize important predictors
    in nonlinear regression models (e.g., Gaussian processes, neural networks).
    
    Args:
        X: Design matrix of shape (n_samples, n_features)
        f_draws: Nonparametric function estimates of shape (n_draws, n_samples)
                 Posterior draws from GP, Bayesian NN, etc.
        beta_draws: Direct effect size analog draws of shape (n_draws, n_features)
                    Provide this OR f_draws (not both)
        prop_var: Proportion of variance to explain in SVD (default: 1.0)
        low_rank: Use low-rank approximation (recommended for p >> n)
        rank_r: Rank for approximation (default: min(n, p))
        nullify: List of variable indices to condition out (default: None)
        verbose: Print progress information (default: False)
    
    Returns:
        Dictionary containing:
            - 'KLD': Kullback-Leibler divergence for each variable
            - 'RATE': Relative centrality measure (normalized KLD)
            - 'Delta': Entropic deviation from uniform distribution
            - 'ESS': Effective sample size (calibrated importance measure)
            - 'mu': Mean effect size analogs
            - 'Lambda': Precision matrix (low-rank or full)
    
    Example:
        >>> X = torch.randn(100, 50)  # 100 samples, 50 features
        >>> f_draws = torch.randn(1000, 100)  # 1000 MCMC draws
        >>> results = rate(X, f_draws=f_draws, low_rank=True)
        >>> print(results['RATE'])  # Importance scores for each feature
    """
    n, p = X.shape
    # device = X.device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move everything to device 
    X = X.to(device)
    if f_draws is not None:
        f_draws = f_draws.to(device)
    if beta_draws is not None:
        beta_draws = beta_draws.to(device)   
    
    if f_draws is None and beta_draws is None:
        raise ValueError("Must provide either f_draws or beta_draws")
    
    if f_draws is not None and beta_draws is not None:
        raise ValueError("Provide only one of f_draws or beta_draws")
    
    if rank_r is None:
        rank_r = min(n, p)
    
    if verbose:
        print(f"Computing RATE for {p} variables...")
        print(f"Design matrix: {n} samples × {p} features")
        print(f"Using {'low-rank' if low_rank else 'full-rank'} approximation")
    
    # Compute effect size analogs if needed
    if beta_draws is None:
        if verbose:
            print("Computing effect size analogs from function draws...")
        X_pinv = torch.linalg.pinv(X)
        beta_draws = (X_pinv @ f_draws.T).T  # Shape: (n_draws, p)
    assert beta_draws is not None, "beta_draws should not be None"
    assert beta_draws.shape[1] == p, f"beta_draws should have shape (n_draws, {p})"

    

    # Compute mean effect size analogs
    mu = beta_draws.mean(dim=0)  # Shape: (p,)
    
    if low_rank:
        if verbose:
            print("Computing low-rank approximation...")
        
        # SVD of design matrix X (n x p)
        U_x, S_x, Vh_x = svd(X, full_matrices=False)
        # U_x: (n, min(n,p)), S_x: (min(n,p),), Vh_x: (min(n,p), p)
        
        # Select components based on variance explained
        S_x_sq = S_x**2
        var_explained = torch.cumsum(S_x_sq / S_x_sq.sum(), dim=0)
        keep_x = (S_x > 1e-10) & (var_explained <= prop_var)
        
        if keep_x.sum() == 0:
            keep_x[0] = True  # Keep at least one component
        
        r_x = keep_x.sum().item()
        
        # u: (r_x, n), v: (p, r_x)
        u = (1 / S_x[keep_x]).unsqueeze(1) * U_x[:, keep_x].T  # (r_x, n)
        v = Vh_x[keep_x, :].T  # (p, r_x)
        
        if verbose:
            print(f"  Selected {r_x} components from design matrix SVD")
        
        # Compute Sigma_fhat: covariance of f_draws (n_draws x n)
        # f_draws is (n_draws, n), so cov should give (n, n)
        if f_draws.shape[0] > 1:
            Sigma_fhat = torch.cov(f_draws.T)  # (n, n)
        else:
            Sigma_fhat = torch.eye(f_draws.shape[1], device=device) * 1e-6
        
        # Compute Sigma_star: (r_x, r_x)
        Sigma_star = u @ Sigma_fhat @ u.T
        
        # SVD of Sigma_star
        U_sig, S_sig, _ = svd(Sigma_star, full_matrices=False)
        keep_sig = S_sig > 1e-10
        r_sig = keep_sig.sum().item()
        
        if verbose:
            print(f"  Selected {r_sig} components from Sigma_star SVD")
        
        # Compute low-rank precision approximation
        # U should be (p, r_sig)
        v_pinv = torch.linalg.pinv(v)  # (r_x, p)
        sqrt_inv_S = 1 / torch.sqrt(S_sig[keep_sig])  # (r_sig,)
        U_sig_kept = U_sig[:, keep_sig]  # (r_x, r_sig)
        
        U = v_pinv.T @ (sqrt_inv_S.unsqueeze(0) * U_sig_kept.T).T  # (p, r_sig)
        
        # Covariance matrix (for Sherman-Morrison): (p, p)
        V = v @ Sigma_star @ v.T
        
        # Update mean: mu should be (p,)
        f_mean = f_draws.mean(dim=0)  # (n,)
        mu = v @ u @ f_mean  # (p, r_x) @ (r_x, n) @ (n,) = (p,)
        
    else:
        if verbose:
            print("Computing full covariance and precision matrices...")

        # Full rank computation with regularization
        V = torch.cov(beta_draws.T)  # Covariance matrix

        # Add small ridge regularization to ensure V is invertible
        ridge = 1e-6
        V_reg = V + ridge * torch.eye(V.shape[0], device=device)

        D = torch.linalg.inv(V_reg)  # Precision matrix (now guaranteed invertible)

        # SVD of precision matrix
        U_d, S_d, _ = svd(D, full_matrices=False)
        r = (S_d > 1e-10).sum().item()

        if verbose:
            print(f"  Keeping {r}/{len(S_d)} singular values")

        # Low-rank factorization: Lambda = U @ U^T
        # Scale columns of U_d[:, :r] by sqrt of singular values
        U = U_d[:, :r] * torch.sqrt(S_d[:r]).unsqueeze(0)

    # Create precision matrix Lambda = U @ U^T
    # Add regularization to ensure non-singularity
    Lambda = U @ U.T + 1e-8 * torch.eye(p, device=device)
    
    if verbose:
        print(f"Precision matrix rank: {U.shape[1]}")
        print("Computing KLD for each variable...")
    
    # Determine which variables to test
    test_indices = list(range(p))
    if nullify is not None:
        test_indices = [i for i in test_indices if i not in nullify]
    
    # Compute KLD for each variable
    KLD = torch.zeros(len(test_indices), device=device)
    
    for idx, j in enumerate(test_indices):
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_indices)} variables...")
        
        KLD[idx] = calc_kld(mu, Lambda, V, j)
    
    # Compute RATE (normalized KLD)
    RATE_scores = KLD / KLD.sum()
    
    # Compute entropic deviation from uniform distribution
    n_active = len(test_indices)
    uniform_rate = 1 / n_active

    # Delta = Σ RATE_i * log(n * RATE_i)
    # Handle 0 * log(0) = 0 case (limit as x->0 of x*log(x) = 0)
    mask = RATE_scores > 0
    Delta = torch.zeros(1, device=device)
    if mask.any():
        Delta = (RATE_scores[mask] * torch.log(n_active * RATE_scores[mask])).sum()

    # Effective sample size (calibration measure)
    ESS = 100 / (1 + Delta)
    
    if verbose:
        print(f"\nResults:")
        print(f"  Delta (entropy deviation): {Delta:.4f}")
        print(f"  ESS (effective sample size): {ESS:.2f}%")
        print(f"  Top 5 variables by RATE:")
        top_indices = torch.argsort(RATE_scores, descending=True)[:5]
        for rank, idx in enumerate(top_indices, 1):
            var_idx = test_indices[idx]
            print(f"    {rank}. Variable {var_idx}: RATE = {RATE_scores[idx]:.4f}")
    
    # Create full-size tensors with zeros for nullified variables
    full_KLD = torch.zeros(p, device=device)
    full_RATE = torch.zeros(p, device=device)
    
    for idx, j in enumerate(test_indices):
        full_KLD[j] = KLD[idx]
        full_RATE[j] = RATE_scores[idx]
    
    return {
        'KLD': full_KLD,
        'RATE': full_RATE,
        'Delta': Delta,
        'ESS': ESS,
        'mu': mu,
        'Lambda': Lambda,
        'test_indices': test_indices
    }


def calc_rate(
    X: torch.Tensor,
    y: torch.Tensor,
    bandwidth: float = 0.01,
    n_mcmc_steps: int = 100000,
    n_burn_in_steps: int = 1000,
    probit: bool = True,
    seed: Optional[int] = None,
    prop_var: float = 1.0,
    low_rank: bool = False,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Calculate RelATive cEntrality (RATE) centrality measures from data.
    
    This is an end-to-end function that:
    1. Computes GP covariance matrix from design matrix
    2. Runs elliptical slice sampling for GP classification
    3. Computes RATE measures from the posterior samples
    
    Args:
        X: Design matrix where COLUMNS are observations, shape (n_features, n_samples)
           Note: This follows GP convention where each column is a data point
        y: Binary class labels for each data point {-1, +1}, shape (n_samples,)
        bandwidth: Bandwidth parameter for Gaussian kernel (default: 0.01)
        n_mcmc_steps: Number of MCMC samples to return (default: 100000)
        n_burn_in_steps: Number of burn-in iterations (default: 1000)
        probit: Use probit link if True, logistic if False (default: True)
        seed: Random seed for reproducibility (default: None)
        prop_var: Proportion of variance for SVD in RATE (default: 1.0)
        low_rank: Use low-rank approximation in RATE (default: False)
        verbose: Print progress information (default: False)
    
    Returns:
        Dictionary containing:
            - 'KLD': Kullback-Leibler divergence for each variable
            - 'RATE': Relative centrality measure (normalized KLD)
            - 'Delta': Entropic deviation from uniform distribution
            - 'ESS': Effective sample size
            - 'samples': MCMC samples from elliptical slice sampling
    
    Example:
        >>> X = torch.randn(5, 100)  # 5 features, 100 samples
        >>> y = torch.sign(torch.randn(100))  # Binary labels
        >>> results = calc_rate(X, y, n_mcmc_steps=1000, n_burn_in_steps=500, verbose=True)
        >>> print(results['RATE'])  # Variable importance scores
    """
    n_features, n_samples = X.shape
    
    if verbose:
        print(f"Starting RATE calculation...")
        print(f"  Design matrix: {n_features} features × {n_samples} samples")
        print(f"  MCMC samples: {n_mcmc_steps} (burn-in: {n_burn_in_steps})")
        print(f"  Link function: {'probit' if probit else 'logistic'}")
        print(f"  Low-rank approximation: {low_rank}")
    
    # Step 1: Compute covariance matrix (GP uses columns as observations)
    if verbose:
        print('\nStep 1: Computing Covariance Matrix...')
    Kn = gaussian_kernel_covariance(X, bandwidth)
    
    # Step 2: Run elliptical slice sampling
    if verbose:
        print('\nStep 2: Running Elliptical Slice Sampling...')
    samples = elliptical_slice_sampling(
        Kn, y,
        n_mcmc_steps=n_mcmc_steps,
        n_burn_in_steps=n_burn_in_steps,
        probit=probit,
        seed=seed,
        verbose=verbose
    )
    
    # Step 3: Compute RATE measures
    # IMPORTANT: rate() expects X to have shape (n_samples, n_features)
    # So we need to transpose X before passing to rate()
    if verbose:
        print('\nStep 3: Computing RATE measures...')
    
    X_transposed = X.T  # Convert from (n_features, n_samples) to (n_samples, n_features)
    
    res_rate = rate(
        X=X_transposed,
        f_draws=samples,
        prop_var=prop_var,
        low_rank=low_rank,
        verbose=verbose
    )
    
    if verbose:
        print('\n' + '='*80)
        print('RATE Calculation Complete!')
        print('='*80)
    
    return {
        'KLD': res_rate['KLD'],
        'RATE': res_rate['RATE'],
        'Delta': res_rate['Delta'],
        'ESS': res_rate['ESS'],
        'samples': samples,
        'mu': res_rate.get('mu'),
        'Lambda': res_rate.get('Lambda'),
    } # type: ignore
