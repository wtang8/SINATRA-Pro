import torch
from typing import Dict
from sinatra_pro.rate import sherman_morrison, calc_kld, rate


def test_sherman():
    A = torch.tensor([[1., 2.], [3., 4.]])
    u = torch.tensor([1., 2.])
    v = torch.tensor([3., 4.])
    Ainv = A.inverse()
    lhs = (A + torch.outer(u, v)).inverse()
    rhs = sherman_morrison(Ainv, u, v)
    assert torch.allclose(lhs, rhs)


def test_calc_kld():
    """Test against known properties"""
    p = 10
    
    # Create test data
    mu = torch.randn(p)
    
    # Create positive definite covariance matrix V
    A = torch.randn(p, p)
    V = A @ A.T + torch.eye(p) * 0.1
    
    # Create precision matrix Lambda
    Lambda = torch.inverse(V)
    
    q = 3
    
    # Calculate KLD
    kld = calc_kld(mu, Lambda, V, q)
    
    # Test properties
    assert kld >= 0, "KLD must be non-negative"
    assert torch.isfinite(kld), "KLD must be finite"
    
    # Test: KLD should be 0 when mu[q] = 0
    mu_zero = mu.clone()
    mu_zero[q] = 0
    kld_zero = calc_kld(mu_zero, Lambda, V, q)
    assert torch.allclose(kld_zero, torch.tensor(0.0), atol=1e-6)
    
    # Test: KLD should scale quadratically with mu[q]
    mu_scaled = mu.clone()
    mu_scaled[q] *= 2
    kld_scaled = calc_kld(mu_scaled, Lambda, V, q)
    assert torch.allclose(kld_scaled, kld * 4, rtol=1e-4)
    
    print("  All tests passed!")


def verify_low_rank_components(
    X: torch.Tensor,
    f_draws: torch.Tensor,
    prop_var: float = 1.0,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verify that low-rank approximation components are computed correctly.
    
    This checks the mathematical properties that should hold according to the
    RATE paper supplementary material.
    
    Args:
        X: Design matrix (n, p)
        f_draws: Function draws (n_draws, n)
        prop_var: Proportion of variance to retain
        verbose: Print detailed verification steps
    
    Returns:
        Dictionary with verification results and intermediate values
    """
    n, p = X.shape
    device = X.device
    
    if verbose:
        print("="*80)
        print("VERIFYING LOW-RANK APPROXIMATION COMPONENTS")
        print("="*80)
    
    # Step 1: Compute beta_draws (effect size analogs)
    X_pinv = torch.linalg.pinv(X)
    beta_draws = (X_pinv @ f_draws.T).T  # (n_draws, p)
    
    if verbose:
        print(f"\n1. Effect size analogs (beta_draws):")
        print(f"   Shape: {beta_draws.shape}")
        print(f"   Mean: {beta_draws.mean():.6f}")
        print(f"   Std:  {beta_draws.std():.6f}")
    
    # Step 2: SVD of design matrix X
    U_x, S_x, Vh_x = torch.linalg.svd(X, full_matrices=False)
    
    if verbose:
        print(f"\n2. SVD of design matrix X:")
        print(f"   U_x shape: {U_x.shape}")
        print(f"   S_x shape: {S_x.shape}")
        print(f"   Vh_x shape: {Vh_x.shape}")
        print(f"   Top 5 singular values: {S_x[:5].tolist()}")
    
    # Step 3: Select components based on variance explained
    S_x_sq = S_x**2
    var_explained = torch.cumsum(S_x_sq / S_x_sq.sum(), dim=0)
    keep_x = (S_x > 1e-10) & (var_explained <= prop_var)
    
    if keep_x.sum() == 0:
        keep_x[0] = True
    
    r_x = keep_x.sum().item()
    
    if verbose:
        print(f"\n3. Component selection:")
        print(f"   Components kept (r_x): {r_x}")
        print(f"   Variance explained by kept components: {var_explained[r_x-1]:.4f}")
    
    # Step 4: Compute u and v matrices
    # u = diag(1/S_x[keep]) @ U_x[:, keep].T  =>  shape (r_x, n)
    u = (1 / S_x[keep_x]).unsqueeze(1) * U_x[:, keep_x].T
    # v = Vh_x[keep, :].T  =>  shape (p, r_x)
    v = Vh_x[keep_x, :].T
    
    if verbose:
        print(f"\n4. Low-rank factors u and v:")
        print(f"   u shape: {u.shape} (should be ({r_x}, {n}))")
        print(f"   v shape: {v.shape} (should be ({p}, {r_x}))")
    
    # Verify: X ≈ U_x @ diag(S_x) @ Vh_x
    X_reconstructed = U_x @ torch.diag(S_x) @ Vh_x
    X_error = torch.norm(X - X_reconstructed) / torch.norm(X)
    
    if verbose:
        print(f"\n5. SVD reconstruction check:")
        print(f"   ||X - U*S*V^T|| / ||X||: {X_error:.2e}")
        assert X_error < 1e-5, "SVD reconstruction should be accurate"
        print(f"   ✓ SVD reconstruction accurate")
    
    # Verify: X ≈ v @ diag(S_x[keep]) @ u (low-rank approximation)
    X_lowrank = v @ torch.diag(S_x[keep_x]) @ u
    X_lr_error = torch.norm(X - X_lowrank) / torch.norm(X)
    
    if verbose:
        print(f"\n6. Low-rank approximation check:")
        print(f"   ||X - v*S*u|| / ||X||: {X_lr_error:.2e}")
        if prop_var < 1.0:
            print(f"   (Some approximation error expected with prop_var={prop_var})")
    
    # Step 5: Compute Sigma_fhat (covariance of f_draws)
    if f_draws.shape[0] > 1:
        Sigma_fhat = torch.cov(f_draws.T)  # (n, n)
    else:
        Sigma_fhat = torch.eye(f_draws.shape[1], device=device) * 1e-6
    
    if verbose:
        print(f"\n7. Covariance of function draws (Sigma_fhat):")
        print(f"   Shape: {Sigma_fhat.shape} (should be ({n}, {n}))")
        print(f"   Trace: {torch.trace(Sigma_fhat):.6f}")
        print(f"   Condition number: {torch.linalg.cond(Sigma_fhat):.2e}")
    
    # Step 6: Compute Sigma_star = u @ Sigma_fhat @ u.T
    Sigma_star = u @ Sigma_fhat @ u.T  # (r_x, r_x)
    
    if verbose:
        print(f"\n8. Sigma_star = u @ Sigma_fhat @ u^T:")
        print(f"   Shape: {Sigma_star.shape} (should be ({r_x}, {r_x}))")
        print(f"   Trace: {torch.trace(Sigma_star):.6f}")
    
    # Step 7: SVD of Sigma_star
    U_sig, S_sig, _ = torch.linalg.svd(Sigma_star, full_matrices=False)
    keep_sig = S_sig > 1e-10
    r_sig = keep_sig.sum().item()
    
    if verbose:
        print(f"\n9. SVD of Sigma_star:")
        print(f"   Singular values kept (r_sig): {r_sig}")
        print(f"   Top 5 singular values: {S_sig[:min(5, r_sig)].tolist()}")
    
    # Step 8: Compute U for precision approximation
    # U = pinv(v)^T @ diag(1/sqrt(S_sig[keep])) @ U_sig[:, keep]
    v_pinv = torch.linalg.pinv(v)  # (r_x, p)
    sqrt_inv_S = 1 / torch.sqrt(S_sig[keep_sig])  # (r_sig,)
    U_sig_kept = U_sig[:, keep_sig]  # (r_x, r_sig)
    
    U = v_pinv.T @ (sqrt_inv_S.unsqueeze(0) * U_sig_kept.T).T  # (p, r_sig)
    
    if verbose:
        print(f"\n10. Precision matrix factor U:")
        print(f"    Shape: {U.shape} (should be ({p}, {r_sig}))")
    
    # Step 9: Compute Lambda = U @ U.T (precision approximation)
    Lambda = U @ U.T  # (p, p)
    
    if verbose:
        print(f"\n11. Precision matrix Lambda = U @ U^T:")
        print(f"    Shape: {Lambda.shape} (should be ({p}, {p}))")
        print(f"    Trace: {torch.trace(Lambda):.6f}")
        print(f"    Rank: {torch.linalg.matrix_rank(Lambda).item()}")
    
    # Step 10: Compute V (covariance for Sherman-Morrison)
    V_cov = v @ Sigma_star @ v.T  # (p, p)
    
    if verbose:
        print(f"\n12. Covariance matrix V = v @ Sigma_star @ v^T:")
        print(f"    Shape: {V_cov.shape} (should be ({p}, {p}))")
        print(f"    Trace: {torch.trace(V_cov):.6f}")
    
    # Step 11: Compute mean
    f_mean = f_draws.mean(dim=0)  # (n,)
    mu_lowrank = v @ u @ f_mean  # (p,)
    
    mu_fullrank = beta_draws.mean(dim=0)  # (p,)
    
    mu_diff = torch.norm(mu_lowrank - mu_fullrank) / torch.norm(mu_fullrank)
    
    if verbose:
        print(f"\n13. Mean effect size analogs (mu):")
        print(f"    Low-rank shape: {mu_lowrank.shape}")
        print(f"    Full-rank shape: {mu_fullrank.shape}")
        print(f"    Relative difference: {mu_diff:.6f}")
    
    # Verification checks
    checks = {
        'svd_reconstruction_error': X_error.item(),
        'lowrank_approximation_error': X_lr_error.item(),
        'mu_relative_difference': mu_diff.item(),
        'components_kept': r_x,
        'precision_rank': r_sig,
        'Lambda_condition': torch.linalg.cond(Lambda).item() if Lambda.shape[0] > 0 else float('inf'),
    }
    
    # Check that Lambda @ V approximates identity (up to low-rank)
    LV = Lambda @ V_cov
    identity_error = torch.norm(LV - torch.eye(p, device=device)) / torch.sqrt(torch.tensor(p, dtype=torch.float32))
    
    if verbose:
        print(f"\n14. Verification: Lambda @ V ≈ I:")
        print(f"    ||Lambda @ V - I||_F / sqrt(p): {identity_error:.6f}")
        if identity_error < 0.5:
            print(f"    ✓ Lambda and V are approximate inverses")
        else:
            print(f"    ⚠ Lambda @ V does not approximate identity well")
            print(f"    This is expected with low-rank approximation")
    
    checks['lambda_v_identity_error'] = identity_error.item()
    
    # Compare with full-rank precision
    V_full = torch.cov(beta_draws.T)
    Lambda_full = torch.linalg.pinv(V_full)
    
    lambda_diff = torch.norm(Lambda - Lambda_full) / torch.norm(Lambda_full)
    
    if verbose:
        print(f"\n15. Comparison with full-rank:")
        print(f"    ||Lambda_lowrank - Lambda_fullrank|| / ||Lambda_full||: {lambda_diff:.6f}")
    
    checks['lambda_relative_difference'] = lambda_diff.item()
    
    if verbose:
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        print(f"✓ SVD reconstruction error: {checks['svd_reconstruction_error']:.2e}")
        print(f"✓ Components kept: {checks['components_kept']}")
        print(f"✓ Precision rank: {checks['precision_rank']}")
        print(f"✓ Mean relative diff: {checks['mu_relative_difference']:.6f}")
        print(f"✓ Lambda condition: {checks['Lambda_condition']:.2e}")
        print("="*80)
    
    return {
        'checks': checks,
        'U_x': U_x,
        'S_x': S_x,
        'Vh_x': Vh_x,
        'u': u,
        'v': v,
        'Sigma_fhat': Sigma_fhat,
        'Sigma_star': Sigma_star,
        'U': U,
        'Lambda': Lambda,
        'V_cov': V_cov,
        'mu_lowrank': mu_lowrank,
        'mu_fullrank': mu_fullrank,
    }


def compare_low_rank_full_rank(
    X: torch.Tensor,
    f_draws: torch.Tensor,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Compare low-rank and full-rank RATE computations.
    
    Args:
        X: Design matrix of shape (n_samples, n_features)
        f_draws: Function draws of shape (n_draws, n_samples)
        verbose: Print detailed comparison (default: True)
    
    Returns:
        Dictionary with comparison metrics
    """
    if verbose:
        print("="*80)
        print("COMPARING LOW-RANK vs FULL-RANK RATE")
        print("="*80)
    
    # Compute full-rank RATE
    if verbose:
        print("\n" + "="*80)
        print("FULL-RANK COMPUTATION")
        print("="*80)
    results_full = rate(X, f_draws=f_draws, low_rank=False, verbose=verbose)
    
    # Compute low-rank RATE
    if verbose:
        print("\n" + "="*80)
        print("LOW-RANK COMPUTATION")
        print("="*80)
    results_lr = rate(X, f_draws=f_draws, low_rank=True, verbose=verbose)
    
    # Compare results
    if verbose:
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
    
    # Compute differences
    rate_diff = torch.abs(results_full['RATE'] - results_lr['RATE'])
    kld_diff = torch.abs(results_full['KLD'] - results_lr['KLD'])
    
    # Correlation between rankings
    rate_corr = torch.corrcoef(torch.stack([results_full['RATE'], results_lr['RATE']]))[0, 1]
    
    # Top-k overlap
    def top_k_overlap(scores1, scores2, k=5):
        top1 = set(torch.topk(scores1, k).indices.tolist())
        top2 = set(torch.topk(scores2, k).indices.tolist())
        return len(top1 & top2) / k
    
    top5_overlap = top_k_overlap(results_full['RATE'], results_lr['RATE'], k=5)
    top10_overlap = top_k_overlap(results_full['RATE'], results_lr['RATE'], k=min(10, X.shape[1]))
    
    if verbose:
        print(f"\nRATE Score Differences:")
        print(f"  Mean absolute difference: {rate_diff.mean():.6f}")
        print(f"  Max absolute difference:  {rate_diff.max():.6f}")
        print(f"  Correlation:              {rate_corr:.6f}")
        
        print(f"\nKLD Differences:")
        print(f"  Mean absolute difference: {kld_diff.mean():.6f}")
        print(f"  Max absolute difference:  {kld_diff.max():.6f}")
        
        print(f"\nRanking Agreement:")
        print(f"  Top-5 overlap:            {top5_overlap*100:.1f}%")
        print(f"  Top-10 overlap:           {top10_overlap*100:.1f}%")
        
        print(f"\nDelta (entropy deviation):")
        print(f"  Full-rank:  {results_full['Delta']:.6f}")
        print(f"  Low-rank:   {results_lr['Delta']:.6f}")
        print(f"  Difference: {abs(results_full['Delta'] - results_lr['Delta']):.6f}")
        
        print(f"\nESS (effective sample size):")
        print(f"  Full-rank:  {results_full['ESS']:.2f}%")
        print(f"  Low-rank:   {results_lr['ESS']:.2f}%")
        
        print(f"\nTop 5 variables by RATE (Full-rank):")
        top_full = torch.argsort(results_full['RATE'], descending=True)[:5]
        for rank, idx in enumerate(top_full, 1):
            print(f"  {rank}. Variable {idx.item():2d}: RATE = {results_full['RATE'][idx]:.4f}")
        
        print(f"\nTop 5 variables by RATE (Low-rank):")
        top_lr = torch.argsort(results_lr['RATE'], descending=True)[:5]
        for rank, idx in enumerate(top_lr, 1):
            in_full_top5 = "✓" if idx in top_full else " "
            print(f"  {rank}. Variable {idx.item():2d}: RATE = {results_lr['RATE'][idx]:.4f} {in_full_top5}")
        
        print("\n" + "="*80)
    
    return {
        'full_rank': results_full,
        'low_rank': results_lr,
        'rate_diff_mean': rate_diff.mean().item(),
        'rate_diff_max': rate_diff.max().item(),
        'rate_correlation': rate_corr.item(),
        'kld_diff_mean': kld_diff.mean().item(),
        'kld_diff_max': kld_diff.max().item(),
        'top5_overlap': top5_overlap,
        'top10_overlap': top10_overlap,
    }


def test_rate():
    """Test the RATE implementation with synthetic data"""
    print("Testing RATE implementation...\n")
    
    # Generate synthetic data
    torch.manual_seed(42)
    n, p = 100, 20
    X = torch.randn(n, p)
    
    # Generate synthetic nonlinear function draws
    n_draws = 500
    f_draws = torch.randn(n_draws, n)
    
    # Test basic functionality
    print("Test 1: Basic RATE computation (Full-rank)")
    results = rate(X, f_draws=f_draws, verbose=True)
    
    assert results['RATE'].shape == (p,)
    assert torch.allclose(results['RATE'].sum(), torch.tensor(1.0))
    assert torch.all(results['RATE'] >= 0)
    assert torch.all(results['RATE'] <= 1)
    print("✓ Basic tests passed\n")
    
    # Test with nullification
    print("\nTest 2: RATE with nullified variables")
    results_null = rate(X, f_draws=f_draws, nullify=[0, 1], verbose=False)
    assert results_null['RATE'][0] == 0
    assert results_null['RATE'][1] == 0
    print("✓ Nullification test passed\n")
    
    # Test low-rank approximation
    print("\nTest 3: Low-rank approximation")
    results_lr = rate(X, f_draws=f_draws, low_rank=True, verbose=True)
    assert results_lr['RATE'].shape == (p,)
    print("✓ Low-rank test passed\n")
    
    # Compare low-rank vs full-rank
    print("\nTest 4: Comparing low-rank vs full-rank")
    comparison = compare_low_rank_full_rank(X, f_draws, verbose=True)
    
    # Check that they're reasonably similar
    assert comparison['rate_correlation'] > 0.8, "Low-rank and full-rank should be highly correlated"
    assert comparison['top5_overlap'] > 0.4, "At least 2 of top 5 should overlap"
    print("\n✓ Comparison test passed")
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)


if __name__ == "__main__":
    test_sherman()
    test_calc_kld()
    test_rate()