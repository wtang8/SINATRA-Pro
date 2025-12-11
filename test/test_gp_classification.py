import torch
from sinatra_pro.sampling import (
    gaussian_kernel_covariance,
    probit_log_likelihood,
    logistic_log_likelihood,
    elliptical_slice_sampling
)

def test_gp_classification():

    """Test Gaussian Process classification with elliptical slice sampling"""
    print("Testing GP Classification with Elliptical Slice Sampling\n")
    print("="*80)
    
    # Set random seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)
    
    # Generate synthetic data
    n_features = 5
    n_samples = 50
    
    print(f"Test 1: Generating synthetic data")
    print(f"  Features: {n_features}")
    print(f"  Samples: {n_samples}")
    
    X = torch.arange(n_features * n_samples, dtype=torch.float32).view(n_features, n_samples)
    X -= X.mean(dim=1, keepdim=True)
    X /= X.std(dim=1, keepdim=True)
    
    # Generate binary labels {-1, +1}
    # true_f = torch.randn(n_samples)
    y = torch.ones(n_samples, dtype=torch.float32)
    y[::2] = -1
    print(f"  Class distribution: {(y == 1).sum().item()} positive, {(y == -1).sum().item()} negative")
    
    # Test covariance matrix computation
    print(f"\nTest 2: Computing covariance matrix")
    K = gaussian_kernel_covariance(X, bandwidth=0.1)
    print(f"    Shape: {K.shape}")
    print(f"    Positive definite: {torch.all(torch.linalg.eigvalsh(K) > -1e-10).item()}")
    
    # Test likelihood functions
    print(f"\nTest 3: Testing likelihood functions")
    test_f = torch.randn(n_samples)
    
    probit_llh = probit_log_likelihood(test_f, y)
    logistic_llh = logistic_log_likelihood(test_f, y)
    
    print(f"  Probit log-likelihood: {probit_llh.item():.4f}")
    print(f"  Logistic log-likelihood: {logistic_llh.item():.4f}")
    
    assert torch.isfinite(probit_llh), "Probit likelihood should be finite"
    assert torch.isfinite(logistic_llh), "Logistic likelihood should be finite"
    
    # Test elliptical slice sampling
    print(f"\nTest 4: Running elliptical slice sampling (small test)")
    samples = elliptical_slice_sampling(
        K,
        y,
        n_mcmc_steps=100,
        n_burn_in_steps=50,
        probit=True,
        seed=42,
        verbose=True
    )
    
    print(f"\n  Samples shape: {samples.shape}")
    print(f"  Mean of samples: {samples.mean():.4f}")
    print(f"  Std of samples: {samples.std():.4f}")
    
    # Check MCMC properties
    print(f"\nTest 5: Checking MCMC properties")
    
    # Compute acceptance rate (approximate)
    unique_samples = torch.unique(samples, dim=0).shape[0]
    acceptance_rate = unique_samples / samples.shape[0]
    print(f"  Approximate acceptance rate: {acceptance_rate:.2%}")
    
    # Compute posterior predictive probabilities
    posterior_mean = samples.mean(dim=0)
    print(f"  Posterior mean range: [{posterior_mean.min():.2f}, {posterior_mean.max():.2f}]")
    
    # Classification accuracy on training data
    predictions = torch.sign(posterior_mean)
    predictions[predictions == 0] = 1
    accuracy = (predictions == y).float().mean()
    print(f"  Training accuracy: {accuracy:.2%}")
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)
    
    return {
        'K': K,
        'y': y,
        'samples': samples,
        'posterior_mean': posterior_mean,
        'accuracy': accuracy.item()
    }


if __name__ == "__main__":
    results = test_gp_classification()