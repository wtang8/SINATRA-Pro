import pytest
import numpy as np
from pathlib import Path
import sys

# Import the directions module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sinatra_pro.directions import (
    rodrigues,
    generate_equidistributed_points,
    generate_equidistributed_cones,
)


class TestRodrigues:
    """Test Rodrigues rotation formula for generating cone directions"""

    def test_rodrigues_basic(self):
        """Test basic Rodrigues rotation"""
        z = np.array([0.0, 0.0, 1.0])  # Z-axis
        r = 0.1
        j = 8

        directions = rodrigues(z, r, j)

        # Should return j directions
        assert directions.shape == (j, 3)

        # All directions should be unit vectors (approximately)
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_rodrigues_zero_radius(self):
        """Test Rodrigues with zero radius (all directions same as axis)"""
        z = np.array([1.0, 0.0, 0.0])
        r = 0.0
        j = 5

        directions = rodrigues(z, r, j)

        # All directions should be very close to z direction
        for direction in directions:
            # Should be nearly parallel (dot product close to 1)
            dot = np.dot(direction, z / np.linalg.norm(z))
            assert np.isclose(dot, 1.0, atol=1e-6)

    def test_rodrigues_arbitrary_axis(self):
        """Test Rodrigues with arbitrary axis"""
        z = np.array([1.0, 1.0, 1.0])  # Arbitrary direction
        r = 0.2
        j = 12

        directions = rodrigues(z, r, j)

        # Check shape
        assert directions.shape == (j, 3)

        # All directions should be unit vectors
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_rodrigues_orthogonal_axis(self):
        """Test Rodrigues with axis along each coordinate direction"""
        for axis in [np.array([1.0, 0.0, 0.0]),
                     np.array([0.0, 1.0, 0.0]),
                     np.array([0.0, 0.0, 1.0])]:
            directions = rodrigues(axis, r=0.15, j=6)

            # All should be unit vectors
            norms = np.linalg.norm(directions, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_rodrigues_evenly_spaced(self):
        """Test that directions are evenly spaced around cone"""
        z = np.array([0.0, 0.0, 1.0])
        r = 0.3
        j = 4  # 4 directions should be 90 degrees apart

        directions = rodrigues(z, r, j)

        # Project onto xy-plane and check angles
        angles = []
        for direction in directions:
            angle = np.arctan2(direction[1], direction[0])
            angles.append(angle)

        angles = np.array(angles)
        # Sort angles to handle wrapping
        angles_sorted = np.sort(angles)

        # Compute angular differences (should be approximately 2π/j)
        expected_spacing = 2 * np.pi / j

        # Check consecutive differences (with wrapping)
        for i in range(j):
            diff = (angles_sorted[(i + 1) % j] - angles_sorted[i]) % (2 * np.pi)
            # Allow for wrapping: difference should be close to expected_spacing
            # or close to 0 (wrapped around)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            assert np.isclose(diff, expected_spacing, atol=0.1) or np.isclose(diff, 0, atol=0.1)

    def test_rodrigues_single_direction(self):
        """Test Rodrigues with single direction"""
        z = np.array([1.0, 2.0, 3.0])
        r = 0.1
        j = 1

        directions = rodrigues(z, r, j)

        assert directions.shape == (1, 3)
        assert np.isclose(np.linalg.norm(directions[0]), 1.0)


class TestGenerateEquidistributedPoints:
    """Test equidistributed point generation on spheres"""

    def test_sphere_basic(self):
        """Test basic sphere point generation"""
        n = 100
        points = generate_equidistributed_points(n, n, hemisphere=False)

        # Should have at least n points
        assert points.shape[0] >= n
        assert points.shape[1] == 3

        # All points should be unit vectors
        norms = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_hemisphere_basic(self):
        """Test hemisphere point generation"""
        n = 50
        points = generate_equidistributed_points(n, n, hemisphere=True)

        # Should have at least n points
        assert points.shape[0] >= n

        # All points should be unit vectors
        norms = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

        # All z-coordinates should be non-negative (upper hemisphere)
        assert np.all(points[:, 2] >= -1e-10)  # Allow small numerical errors

    def test_sphere_coverage(self):
        """Test that points cover full sphere"""
        n = 200
        points = generate_equidistributed_points(n, n, hemisphere=False)

        # Check that we have points in all octants
        # At least some positive and negative in each dimension
        for dim in range(3):
            assert np.any(points[:, dim] > 0.5)
            assert np.any(points[:, dim] < -0.5)

    def test_hemisphere_coverage(self):
        """Test that hemisphere points are in upper half"""
        n = 100
        points = generate_equidistributed_points(n, n, hemisphere=True)

        # All z should be >= 0
        assert np.all(points[:, 2] >= -1e-10)

        # Should have variety in x and y
        assert np.any(points[:, 0] > 0.5)
        assert np.any(points[:, 0] < -0.5)
        assert np.any(points[:, 1] > 0.5)
        assert np.any(points[:, 1] < -0.5)

    def test_minimum_points_generated(self):
        """Test that at least desired number of points are generated"""
        for desired in [10, 50, 100, 200]:
            points = generate_equidistributed_points(desired, desired)
            assert points.shape[0] >= desired

    def test_recursive_increase(self):
        """Test that algorithm increases N if needed"""
        # Start with N smaller than desired
        desired = 100
        N = 10
        points = generate_equidistributed_points(desired, N)

        # Should still get at least desired number
        assert points.shape[0] >= desired

    def test_small_number(self):
        """Test with small number of points"""
        points = generate_equidistributed_points(5, 5, hemisphere=False)

        assert points.shape[0] >= 5
        norms = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_large_number(self):
        """Test with large number of points"""
        n = 1000
        points = generate_equidistributed_points(n, n, hemisphere=False)

        assert points.shape[0] >= n
        norms = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)


class TestGenerateEquidistributedCones:
    """Test equidistributed cone generation"""

    def test_single_direction_per_cone(self):
        """Test with one direction per cone (just sphere points)"""
        n_cones = 50
        directions = generate_equidistributed_cones(
            n_cones=n_cones,
            cap_radius=0.1,
            n_direction_per_cone=1,
            hemisphere=False
        )

        # Should have exactly n_cones directions
        assert directions.shape[0] == n_cones
        assert directions.shape[1] == 3

        # All should be unit vectors
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_multiple_directions_per_cone(self):
        """Test with multiple directions per cone"""
        n_cones = 20
        n_dir = 5
        directions = generate_equidistributed_cones(
            n_cones=n_cones,
            cap_radius=0.2,
            n_direction_per_cone=n_dir,
            hemisphere=False
        )

        # Should have n_cones * n_dir directions
        assert directions.shape[0] == n_cones * n_dir
        assert directions.shape[1] == 3

        # All should be unit vectors
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_hemisphere_cones(self):
        """Test cone generation on hemisphere"""
        n_cones = 30
        directions = generate_equidistributed_cones(
            n_cones=n_cones,
            cap_radius=0.15,
            n_direction_per_cone=3,
            hemisphere=True
        )

        # Should have n_cones * 3 directions
        assert directions.shape[0] == n_cones * 3

        # Most directions should have positive z (allowing for cone spread)
        # At least the central axes should be in upper hemisphere
        assert np.sum(directions[:, 2] > 0) > n_cones * 0.5

    def test_zero_cap_radius(self):
        """Test with very small cap radius"""
        n_cones = 10
        directions = generate_equidistributed_cones(
            n_cones=n_cones,
            cap_radius=0.01,
            n_direction_per_cone=4,
            hemisphere=False
        )

        assert directions.shape[0] == n_cones * 4

        # All should be unit vectors
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_large_cap_radius(self):
        """Test with large cap radius"""
        n_cones = 10
        directions = generate_equidistributed_cones(
            n_cones=n_cones,
            cap_radius=0.5,
            n_direction_per_cone=6,
            hemisphere=False
        )

        assert directions.shape[0] == n_cones * 6

        # All should be unit vectors
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_default_parameters(self):
        """Test with default parameters"""
        n_cones = 25
        directions = generate_equidistributed_cones(n_cones=n_cones)

        # Default: n_direction_per_cone=1, cap_radius=0.1, hemisphere=False
        assert directions.shape[0] == n_cones

        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_cone_structure(self):
        """Test that directions form proper cone structure"""
        n_cones = 5
        n_dir = 8
        cap_radius = 0.2

        directions = generate_equidistributed_cones(
            n_cones=n_cones,
            cap_radius=cap_radius,
            n_direction_per_cone=n_dir,
            hemisphere=False
        )

        # Check structure: every n_dir directions should form a cone
        for i in range(n_cones):
            start_idx = i * n_dir
            end_idx = (i + 1) * n_dir
            cone_dirs = directions[start_idx:end_idx]

            # First direction is the central axis
            central_axis = cone_dirs[0]

            # Other directions should be around this axis
            if n_dir > 1:
                for j in range(1, n_dir):
                    # Dot product should show they're close in angle
                    dot = np.dot(cone_dirs[j], central_axis)
                    # Should be > cos(some reasonable angle)
                    assert dot > 0.8  # Within ~36 degrees


class TestIntegration:
    """Integration tests for direction generation workflows"""

    def test_complete_workflow_sphere(self):
        """Test complete workflow for sphere direction generation"""
        # Generate 100 cones with 5 directions each on full sphere
        directions = generate_equidistributed_cones(
            n_cones=100,
            cap_radius=0.1,
            n_direction_per_cone=5,
            hemisphere=False
        )

        # Verify total count
        assert directions.shape == (500, 3)

        # Verify all unit vectors
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

        # Verify coverage across sphere
        for dim in range(3):
            assert np.any(directions[:, dim] > 0.7)
            assert np.any(directions[:, dim] < -0.7)

    def test_complete_workflow_hemisphere(self):
        """Test complete workflow for hemisphere direction generation"""
        # Generate 50 cones with 4 directions each on hemisphere
        directions = generate_equidistributed_cones(
            n_cones=50,
            cap_radius=0.15,
            n_direction_per_cone=4,
            hemisphere=True
        )

        # Verify total count
        assert directions.shape == (200, 3)

        # Verify all unit vectors
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

        # Verify hemisphere constraint (most should be positive z)
        assert np.sum(directions[:, 2] > 0) > 150

    def test_reproducibility(self):
        """Test that generation is deterministic (reproducible)"""
        n_cones = 30
        cap_radius = 0.12
        n_dir = 3

        # Generate twice with same parameters
        dirs1 = generate_equidistributed_cones(
            n_cones=n_cones,
            cap_radius=cap_radius,
            n_direction_per_cone=n_dir,
            hemisphere=False
        )

        dirs2 = generate_equidistributed_cones(
            n_cones=n_cones,
            cap_radius=cap_radius,
            n_direction_per_cone=n_dir,
            hemisphere=False
        )

        # Should be identical
        np.testing.assert_array_equal(dirs1, dirs2)

    def test_different_scales(self):
        """Test direction generation at different scales"""
        for n_cones in [10, 50, 100, 200]:
            directions = generate_equidistributed_cones(
                n_cones=n_cones,
                n_direction_per_cone=1,
                hemisphere=False
            )

            # Verify correct count
            assert directions.shape[0] == n_cones

            # Verify unit vectors
            norms = np.linalg.norm(directions, axis=1)
            np.testing.assert_allclose(norms, 1.0, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_point(self):
        """Test with single point"""
        points = generate_equidistributed_points(1, 1, hemisphere=False)
        assert points.shape[0] >= 1
        assert np.isclose(np.linalg.norm(points[0]), 1.0)

    def test_single_cone(self):
        """Test with single cone"""
        directions = generate_equidistributed_cones(
            n_cones=1,
            n_direction_per_cone=1,
            hemisphere=False
        )
        assert directions.shape[0] == 1
        assert np.isclose(np.linalg.norm(directions[0]), 1.0)

    def test_rodrigues_with_normalized_input(self):
        """Test Rodrigues with already normalized input"""
        z = np.array([0.0, 0.0, 1.0])  # Already normalized
        directions = rodrigues(z, 0.1, 5)

        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_rodrigues_with_unnormalized_input(self):
        """Test Rodrigues with unnormalized input"""
        z = np.array([0.0, 0.0, 5.0])  # Not normalized
        directions = rodrigues(z, 0.1, 5)

        # Should still produce unit vectors
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
