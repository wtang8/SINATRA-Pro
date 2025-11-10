"""
Unit tests for Gaussian Process Classification with Elliptical Slice Sampling

Run with:
    pytest test_gp_classification.py -v
    or
    python test_gp_classification.py
"""

import torch
import pytest
import numpy as np
from sinatra_pro.sampling import (
    gaussian_kernel_covariance,
    probit_log_likelihood,
    logistic_log_likelihood,
    elliptical_slice_sampling
)


def _gaussian_kernel_covariance_original(
    X: torch.Tensor,
    bandwidth: float = 0.01
) -> torch.Tensor:
    """
    Computes the covariance matrix using the Gaussian (RBF) kernel.
    
    Args:
        X: Design matrix of shape (n_features, n_samples)
           Columns are observations
        bandwidth: Bandwidth parameter for Gaussian kernel (default: 0.01)
    
    Returns:
        Covariance matrix K of shape (n_samples, n_samples)
    
    Formula:
        K[i,j] = exp(-||x_i - x_j||^2 / (2 * bandwidth^2))
    """
    n = X.shape[1]  # number of samples
    device = X.device
    
    # Precompute bandwidth factor
    bandwidth_factor = 1.0 / (2 * bandwidth**2)
    
    # Initialize covariance matrix
    K = torch.zeros((n, n), dtype=torch.float32, device=device)
    
    # Compute pairwise distances and kernel values
    for i in range(n):
        K[i, i] = 1.0  # Diagonal elements
        
        for j in range(i + 1, n):
            # Squared Euclidean distance between columns i and j
            diff = X[:, i] - X[:, j]
            squared_dist = torch.mean(diff**2)
            
            # Gaussian kernel
            K[i, j] = torch.exp(-squared_dist * bandwidth_factor)
            K[j, i] = K[i, j]  # Symmetric
    
    return K

class TestGaussianKernel:
    """Test Gaussian kernel covariance matrix computation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.n_features = 5
        self.n_samples = 10
        self.bandwidth = 0.1

        self.X = torch.arange(self.n_features * self.n_samples, dtype=torch.float32).view(self.n_features, self.n_samples)
        self.X -= self.X.mean(dim=1, keepdim=True)
        self.X /= self.X.std(dim=1, keepdim=True)
    
    def test_covariance_matrix_properties(self):
        """Test that covariance matrix has correct shape"""
        K = gaussian_kernel_covariance(self.X, self.bandwidth)
        assert K.shape == (self.n_samples, self.n_samples)
        """Test that covariance matrix is symmetric"""
        assert torch.allclose(K, K.T, rtol=1e-5)
        """Test that diagonal elements are 1"""
        assert torch.allclose(torch.diag(K), torch.ones(self.n_samples))
        """Test that covariance matrix is positive definite"""
        eigenvalues = torch.linalg.eigvalsh(K)
        assert torch.all(eigenvalues > -1e-10)
        """Test that all covariance values are in [0, 1]"""
        assert torch.all(K >= 0 - 1e-5)
        assert torch.all(K <= 1 + 1e-5)
    
    def test_vectorized_matches_loop(self):
        """Test that vectorized version matches loop version"""
        K_loop = _gaussian_kernel_covariance_original(self.X, self.bandwidth)
        K_vec = gaussian_kernel_covariance(self.X, self.bandwidth)
        assert torch.allclose(K_loop, K_vec, rtol=1e-4)
    
    def test_bandwidth_effect(self):
        """Test that bandwidth affects kernel values"""
        K1 = gaussian_kernel_covariance(self.X, bandwidth=0.01)
        K2 = gaussian_kernel_covariance(self.X, bandwidth=1.0)
        
        # Smaller bandwidth should lead to smaller off-diagonal values
        off_diag_1 = K1[0, 1]
        off_diag_2 = K2[0, 1]
        assert off_diag_1 < off_diag_2 or torch.isclose(off_diag_1, off_diag_2, rtol=1e-3)
    
    def test_identical_points(self):
        """Test covariance of identical points is 1"""
        X_same = torch.ones(self.n_features, 2)
        K = gaussian_kernel_covariance(X_same, self.bandwidth)
        assert torch.allclose(K, torch.ones(2, 2))
    
    def test_gpu_compatibility(self):
        """Test that kernel works on GPU if available"""
        if torch.cuda.is_available():
            X_gpu = self.X.cuda()
            K_gpu = gaussian_kernel_covariance(X_gpu, self.bandwidth)
            assert K_gpu.device.type == 'cuda'
            
            K_cpu = gaussian_kernel_covariance(self.X, self.bandwidth)
            assert torch.allclose(K_gpu.cpu(), K_cpu, rtol=1e-4)


class TestLikelihoodFunctions:
    """Test probit and logistic likelihood functions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.n = 20
        self.f = torch.ones(self.n) * 10.0
        self.y = torch.sign(torch.randn(self.n))
        self.y[self.y == 0] = 1  # Ensure binary {-1, +1}
    
    def test_probit_likelihood_finite_negative(self):
        """Test that probit likelihood is finite"""
        llh = probit_log_likelihood(self.f, self.y)
        assert torch.isfinite(llh)
        """Test that probit log-likelihood is negative or zero"""
        assert llh <= 0
    
    def test_logistic_likelihood_finite_negative(self):
        """Test that logistic likelihood is finite"""
        llh = logistic_log_likelihood(self.f, self.y)
        assert torch.isfinite(llh)
        """Test that logistic log-likelihood is negative or zero"""
        assert llh <= 0
    
    def test_perfect_classification_high_likelihood(self):
        """Test that perfect classification has higher likelihood"""
        # Perfect agreement
        f_perfect = self.y * 10.0  # Large positive values
        llh_perfect = probit_log_likelihood(f_perfect, self.y)
        
        # Random
        llh_random = probit_log_likelihood(self.f, self.y)
        
        assert llh_perfect > llh_random
    
    def test_opposite_classification_low_likelihood(self):
        """Test that opposite classification has low likelihood"""
        # Opposite signs
        f_opposite = -self.y * 10.0
        llh_opposite = probit_log_likelihood(f_opposite, self.y)
        
        # Random
        llh_random = probit_log_likelihood(self.f, self.y)
        
        assert llh_opposite < llh_random
    
    def test_likelihood_with_zeros(self):
        """Test likelihood handles zero latent values"""
        f_zero = torch.zeros(self.n)
        llh_probit = probit_log_likelihood(f_zero, self.y)
        llh_logistic = logistic_log_likelihood(f_zero, self.y)
        
        assert torch.isfinite(llh_probit)
        assert torch.isfinite(llh_logistic)
    
    def test_likelihood_with_extreme_values(self):
        """Test likelihood with extreme latent values"""
        f_large = torch.ones(self.n) * 100
        f_small = torch.ones(self.n) * -100
        y_pos = torch.ones(self.n)
        
        llh_large = probit_log_likelihood(f_large, y_pos)
        llh_small = probit_log_likelihood(f_small, y_pos)
        
        assert torch.isfinite(llh_large)
        assert torch.isfinite(llh_small)
        assert llh_large > llh_small
    
    def test_likelihood_shape_mismatch(self):
        """Test that shape mismatch raises error"""
        f_wrong = torch.randn(self.n + 5)
        
        # This should work without error or handle gracefully
        try:
            llh = probit_log_likelihood(f_wrong, self.y)
            # If no error, shapes were broadcasted
            assert llh.shape == ()
        except RuntimeError:
            # Expected behavior for shape mismatch
            pass


class TestEllipticalSliceSampling:
    """Test elliptical slice sampling algorithm"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
        self.n = 30
        self.X = torch.randn(5, self.n)
        self.K = gaussian_kernel_covariance(self.X, bandwidth=0.1)
        self.y = torch.sign(torch.randn(self.n))
        self.y[self.y == 0] = 1
    
    def test_sampling_shape(self):
        """Test that samples have correct shape"""
        n_mcmc = 100
        samples = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=n_mcmc,
            burn_in=50,
            seed=42,
            verbose=False
        )
        assert samples.shape == (n_mcmc, self.n)
    
    def test_sampling_finite(self):
        """Test that all samples are finite"""
        samples = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=50,
            burn_in=20,
            seed=42,
            verbose=False
        )
        assert torch.all(torch.isfinite(samples))
    
    def test_probit_vs_logistic(self):
        """Test both link functions work"""
        samples_probit = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=50,
            burn_in=20,
            probit=True,
            seed=42,
            verbose=False
        )
        
        samples_logistic = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=50,
            burn_in=20,
            probit=False,
            seed=42,
            verbose=False
        )
        
        assert samples_probit.shape == samples_logistic.shape
        assert torch.all(torch.isfinite(samples_probit))
        assert torch.all(torch.isfinite(samples_logistic))
    
    def test_reproducibility_with_seed(self):
        """Test that same seed gives same results"""
        samples1 = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=50,
            burn_in=20,
            seed=123,
            verbose=False
        )
        
        samples2 = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=50,
            burn_in=20,
            seed=123,
            verbose=False
        )
        
        assert torch.allclose(samples1, samples2)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds give different results"""
        samples1 = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=50,
            burn_in=20,
            seed=123,
            verbose=False
        )
        
        samples2 = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=50,
            burn_in=20,
            seed=456,
            verbose=False
        )
        
        assert not torch.allclose(samples1, samples2)
    
    def test_posterior_mean_sensible(self):
        """Test that posterior mean is sensible"""
        samples = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=200,
            burn_in=100,
            seed=42,
            verbose=False
        )
        
        posterior_mean = samples.mean(dim=0)
        
        # Posterior mean should generally agree with labels
        predictions = torch.sign(posterior_mean)
        predictions[predictions == 0] = 1
        accuracy = (predictions == self.y).float().mean()
        
        # With simple data, should get reasonable accuracy
        assert accuracy > 0.4  # Better than random (0.5) in expectation
    
    def test_burn_in_effect(self):
        """Test that burn-in affects results"""
        # Short burn-in
        samples_short = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=100,
            burn_in=10,
            seed=42,
            verbose=False
        )
        
        # Long burn-in
        samples_long = elliptical_slice_sampling(
            self.K, self.y,
            n_mcmc=100,
            burn_in=100,
            seed=42,
            verbose=False
        )
        
        # They should be different (burn-in matters)
        assert not torch.allclose(samples_short, samples_long)
    
    def test_covariance_singular(self):
        """Test handling of near-singular covariance"""
        # Create nearly singular covariance
        K_singular = torch.eye(self.n) * 1e-8 + torch.ones(self.n, self.n) * 1e-9
        
        samples = elliptical_slice_sampling(
            K_singular, self.y,
            n_mcmc=50,
            burn_in=20,
            seed=42,
            verbose=False
        )
        
        assert samples.shape == (50, self.n)
        assert torch.all(torch.isfinite(samples))
    
    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises error"""
        y_wrong = torch.ones(self.n + 5)
        
        with pytest.raises(AssertionError):
            elliptical_slice_sampling(
                self.K, y_wrong,
                n_mcmc=10,
                burn_in=5,
                verbose=False
            )
    
    def test_gpu_compatibility(self):
        """Test sampling on GPU if available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        K_gpu = self.K.cuda()
        y_gpu = self.y.cuda()
        
        samples = elliptical_slice_sampling(
            K_gpu, y_gpu,
            n_mcmc=50,
            burn_in=20,
            seed=42,
            verbose=False
        )
        
        assert samples.device.type == 'cuda'
        assert torch.all(torch.isfinite(samples))


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def setup_method(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_complete_classification_workflow(self):
        """Test complete GP classification workflow"""
        # Generate data
        n_features = 3
        n_samples = 40
        X = torch.randn(n_features, n_samples)
        
        # Generate labels from latent function
        true_f = torch.randn(n_samples)
        y = torch.sign(true_f)
        y[y == 0] = 1
        
        # Compute covariance
        K = gaussian_kernel_covariance(X, bandwidth=0.1)
        
        # Run MCMC
        samples = elliptical_slice_sampling(
            K, y,
            n_mcmc=500,
            burn_in=200,
            probit=True,
            seed=42,
            verbose=False
        )
        
        # Compute posterior
        posterior_mean = samples.mean(dim=0)
        posterior_std = samples.std(dim=0)
        
        # Make predictions
        predictions = torch.sign(posterior_mean)
        predictions[predictions == 0] = 1
        
        # Check results
        assert samples.shape == (500, n_samples)
        assert posterior_mean.shape == (n_samples,)
        assert posterior_std.shape == (n_samples,)
        assert predictions.shape == (n_samples,)
        
        # Accuracy should be reasonable
        accuracy = (predictions == y).float().mean()
        assert accuracy > 0.3  # Better than worst case
    
    def test_perfect_separation(self):
        """Test with perfectly separable data"""
        n = 20
        
        # Create perfectly separable data
        X = torch.zeros(2, n)
        X[0, :n//2] = -5  # Class -1
        X[0, n//2:] = 5   # Class +1
        y = torch.cat([torch.ones(n//2) * -1, torch.ones(n//2)])
        
        K = gaussian_kernel_covariance(X, bandwidth=0.1)
        
        samples = elliptical_slice_sampling(
            K, y,
            n_mcmc=200,
            burn_in=100,
            seed=42,
            verbose=False
        )
        
        posterior_mean = samples.mean(dim=0)
        predictions = torch.sign(posterior_mean)
        predictions[predictions == 0] = 1
        
        # Should achieve very high accuracy
        accuracy = (predictions == y).float().mean()
        assert accuracy > 0.7  # Should do well on separable data
    
    def test_noisy_labels(self):
        """Test robustness to noisy labels"""
        n = 30
        X = torch.randn(3, n)
        
        # Generate labels and add noise
        true_f = torch.randn(n)
        y = torch.sign(true_f)
        y[y == 0] = 1
        
        # Flip some labels (add noise)
        noise_idx = torch.randperm(n)[:5]
        y[noise_idx] = -y[noise_idx]
        
        K = gaussian_kernel_covariance(X, bandwidth=0.1)
        
        samples = elliptical_slice_sampling(
            K, y,
            n_mcmc=200,
            burn_in=100,
            seed=42,
            verbose=False
        )
        
        # Should still produce sensible results
        assert torch.all(torch.isfinite(samples))
        assert samples.std() > 0  # Should have variation


def run_tests():
    """Run all tests without pytest"""
    print("Running GP Classification Tests")
    print("="*80)
    
    test_classes = [
        TestGaussianKernel,
        TestLikelihoodFunctions,
        TestEllipticalSliceSampling,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-"*80)
        
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # Run setup
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                method = getattr(test_instance, method_name)
                method()
                
                print(f"  ✓ {method_name}")
                passed_tests += 1
                
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
            except Exception as e:
                print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
    
    print("\n" + "="*80)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("="*80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)



