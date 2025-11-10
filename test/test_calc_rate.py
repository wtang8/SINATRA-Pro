import torch
import numpy as np
from sinatra_pro.rate import calc_rate


def test_rate_calculation():
    """Test end-to-end RATE calculation"""
    print("Testing End-to-End RATE Calculation")
    print("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    n_features = 5
    n_samples = 50
    
    print(f"\nTest Setup:")
    print(f"  Features: {n_features}")
    print(f"  Samples: {n_samples}")
    
    # Create design matrix (features × samples) - deterministic and non-singular
    # Vandermonde-like matrix (rows are powers of a sequence)
    t = torch.linspace(-1, 1, n_samples, dtype=torch.float32)
    X = torch.stack([t**i for i in range(n_features)])  # Each row: [t^0, t^1, t^2, ...]

    # Standardize each feature
    X -= X.mean(dim=1, keepdim=True)
    X /= X.std(dim=1, keepdim=True) + 1e-8
    
    print(f"  X shape: {X.shape} (features × samples)")
    
    # Generate binary labels {-1, +1}
    y = torch.ones(n_samples, dtype=torch.float32)
    y[n_samples//2:] = -1
    
    print(f"  y shape: {y.shape}")
    print(f"  Class distribution: {(y == 1).sum().item()} positive, {(y == -1).sum().item()} negative")
    
    # Run RATE calculation
    print("\n" + "="*80)
    res_rate = calc_rate(
        X,
        y,
        bandwidth=0.01,
        n_mcmc=2000,
        burn_in=1000,
        low_rank=False,
        verbose=True
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    kld = res_rate['KLD']
    rate_scores = res_rate['RATE']
    delta = res_rate['Delta']
    ess = res_rate['ESS']
    
    print(f"\nKLD scores:")
    for i, score in enumerate(kld):
        print(f"  Feature {i}: {score:.6f}")
    
    print(f"\nRATE scores (normalized importance):")
    for i, score in enumerate(rate_scores):
        print(f"  Feature {i}: {score:.6f}")
    
    print(f"\nSummary statistics:")
    print(f"  Delta (entropy deviation): {delta:.6f}")
    print(f"  ESS (effective sample size): {ess:.2f}%")
    print(f"  Top feature: {torch.argmax(rate_scores).item()}")
    
    # Validate results
    assert kld.shape == (n_features,), f"KLD shape should be ({n_features},), got {kld.shape}"
    assert rate_scores.shape == (n_features,), f"RATE shape should be ({n_features},), got {rate_scores.shape}"
    assert torch.allclose(rate_scores.sum(), torch.tensor(1.0), atol=1e-5), "RATE scores should sum to 1"
    assert torch.all(rate_scores >= 0), "RATE scores should be non-negative"
    assert torch.all(rate_scores <= 1), "RATE scores should be <= 1"
    
    print("\n" + "="*80)
    print("✓ All validation checks passed!")
    print("="*80)
    
    return res_rate


if __name__ == "__main__":
    results = test_rate_calculation()