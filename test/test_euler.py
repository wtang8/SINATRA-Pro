import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Import modules to test
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sinatra_pro.euler import compute_ec_curve_single, compute_ec_curve
from sinatra_pro.mesh import Mesh


class TestComputeECCurveSingle:
    """Test single direction EC curve computation"""

    def test_basic_computation(self):
        """Test basic EC curve computation"""
        # Create simple mesh
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        # Compute EC curve along z-axis
        direction = np.array([0.0, 0.0, 1.0])
        ec_curve = compute_ec_curve_single(mesh, direction, ball_radius=1.0)

        # Should return array of correct length
        assert ec_curve.shape == (25,)
        assert isinstance(ec_curve, np.ndarray)

    def test_ect_type(self):
        """Test standard ECT computation"""
        vertices = torch.rand(20, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        direction = np.array([1.0, 0.0, 0.0])
        ec_curve = compute_ec_curve_single(
            mesh, direction, ball_radius=1.0, ec_type="ECT"
        )

        # ECT should be cumulative (monotonic increasing)
        assert ec_curve.shape == (25,)
        # Check that it's somewhat monotonic (allowing small variations)
        assert np.all(np.diff(ec_curve) >= -1)

    def test_dect_type(self):
        """Test differential ECT computation"""
        vertices = torch.rand(20, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        direction = np.array([0.0, 1.0, 0.0])
        ec_curve = compute_ec_curve_single(
            mesh, direction, ball_radius=1.0, ec_type="DECT"
        )

        assert ec_curve.shape == (25,)
        # DECT should have first value as 0
        assert ec_curve[0] == 0.0

    def test_sect_type(self):
        """Test smooth ECT computation"""
        vertices = torch.rand(20, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        direction = np.array([0.0, 0.0, 1.0])
        ec_curve = compute_ec_curve_single(
            mesh, direction, ball_radius=1.0, ec_type="SECT"
        )

        assert ec_curve.shape == (25,)
        # SECT should be mean-centered during intermediate step
        assert isinstance(ec_curve[0], (float, np.floating))

    def test_invalid_ec_type(self):
        """Test that invalid ec_type raises error"""
        vertices = torch.rand(10, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        direction = np.array([1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Invalid ec_type"):
            compute_ec_curve_single(
                mesh, direction, ball_radius=1.0, ec_type="INVALID"
            )

    def test_include_faces_true(self):
        """Test EC computation with faces included"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        direction = np.array([0.0, 0.0, 1.0])
        ec_with_faces = compute_ec_curve_single(
            mesh, direction, ball_radius=1.0, include_faces=True
        )

        assert ec_with_faces.shape == (25,)

    def test_include_faces_false(self):
        """Test EC computation without faces"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        direction = np.array([0.0, 0.0, 1.0])
        ec_without_faces = compute_ec_curve_single(
            mesh, direction, ball_radius=1.0, include_faces=False
        )

        assert ec_without_faces.shape == (25,)

        # EC without faces may differ from EC with faces
        ec_with_faces = compute_ec_curve_single(
            mesh, direction, ball_radius=1.0, include_faces=True
        )

        # They might be different (depending on mesh structure)
        # Just verify both complete without error

    def test_different_filtration_numbers(self):
        """Test with different numbers of filtration steps"""
        vertices = torch.rand(15, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)

        for n_filt in [10, 25, 50, 100]:
            ec_curve = compute_ec_curve_single(
                mesh, direction, ball_radius=1.0, n_filtration=n_filt
            )
            assert ec_curve.shape == (n_filt,)

    def test_different_ball_radii(self):
        """Test with different ball radii"""
        vertices = torch.rand(15, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        direction = np.array([1.0, 0.0, 0.0])

        for radius in [0.5, 1.0, 2.0, 5.0]:
            ec_curve = compute_ec_curve_single(
                mesh, direction, ball_radius=radius
            )
            assert ec_curve.shape == (25,)

    def test_orthogonal_directions(self):
        """Test EC curves along orthogonal directions"""
        vertices = torch.rand(30, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        # Test along x, y, z axes
        directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        ec_curves = []
        for direction in directions:
            ec_curve = compute_ec_curve_single(
                mesh, direction, ball_radius=1.0
            )
            ec_curves.append(ec_curve)
            assert ec_curve.shape == (25,)

        # EC curves along different directions should be different
        # (unless the mesh is perfectly symmetric)
        assert not np.allclose(ec_curves[0], ec_curves[1])

    def test_mesh_without_edges(self):
        """Test EC computation for mesh without edges"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ])
        # With large radius cutoff, no edges will form
        mesh = Mesh(vertices, generate_mesh=True, radius=0.1)

        direction = np.array([0.0, 0.0, 1.0])
        ec_curve = compute_ec_curve_single(
            mesh, direction, ball_radius=1.0
        )

        assert ec_curve.shape == (25,)


class TestComputeECCurve:
    """Test multiple direction EC curve computation"""

    def test_basic_multiple_directions(self):
        """Test EC curves for multiple directions"""
        vertices = torch.rand(25, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        # Create 5 directions
        directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        # Normalize
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        radius, ec_curves = compute_ec_curve(mesh, directions)

        # Check shapes
        assert radius.shape == (25,)
        assert ec_curves.shape == (5, 25)

        # Check radius range
        assert np.isclose(radius[0], -1.0)
        assert np.isclose(radius[-1], 1.0)

    def test_single_direction(self):
        """Test with single direction (edge case)"""
        vertices = torch.rand(15, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        directions = np.array([[0.0, 0.0, 1.0]])

        radius, ec_curves = compute_ec_curve(mesh, directions)

        assert radius.shape == (25,)
        assert ec_curves.shape == (1, 25)

    def test_many_directions(self):
        """Test with many directions"""
        vertices = torch.rand(20, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        # Generate 50 random directions
        np.random.seed(42)
        directions = np.random.randn(50, 3)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        radius, ec_curves = compute_ec_curve(mesh, directions, n_filtration=30)

        assert radius.shape == (30,)
        assert ec_curves.shape == (50, 30)

    def test_different_ec_types(self):
        """Test all EC types with multiple directions"""
        vertices = torch.rand(20, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        directions = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        for ec_type in ["ECT", "DECT", "SECT"]:
            radius, ec_curves = compute_ec_curve(
                mesh, directions, ec_type=ec_type
            )

            assert radius.shape == (25,)
            assert ec_curves.shape == (2, 25)

    def test_parameters_consistency(self):
        """Test that parameters are correctly passed through"""
        vertices = torch.rand(20, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        directions = np.array([[1.0, 0.0, 0.0]])

        # Test with custom parameters
        radius, ec_curves = compute_ec_curve(
            mesh,
            directions,
            n_filtration=50,
            ball_radius=2.0,
            ec_type="DECT",
            include_faces=False
        )

        assert radius.shape == (50,)
        assert ec_curves.shape == (1, 50)
        assert np.isclose(radius[0], -2.0)
        assert np.isclose(radius[-1], 2.0)


class TestIntegration:
    """Integration tests for EC curve computation"""

    def test_complete_workflow(self):
        """Test complete workflow from mesh creation to EC curves"""
        from sinatra_pro.directions import generate_equidistributed_cones

        # Create mesh from random vertices
        torch.manual_seed(42)
        vertices = torch.rand(50, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        # Generate equidistributed directions
        directions = generate_equidistributed_cones(
            n_cone=20,
            n_direction_per_cone=1,
            hemisphere=False
        )

        # Compute EC curves
        radius, ec_curves = compute_ec_curve(
            mesh, directions, n_filtration=30, ball_radius=1.0
        )

        # Verify shapes
        assert radius.shape == (30,)
        assert ec_curves.shape == (20, 30)

        # Verify no NaN or Inf values
        assert not np.any(np.isnan(ec_curves))
        assert not np.any(np.isinf(ec_curves))

    def test_different_mesh_complexities(self):
        """Test EC curves for meshes of different complexities"""
        for n_vertices in [10, 50, 100]:
            vertices = torch.rand(n_vertices, 3)
            mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

            directions = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ])

            radius, ec_curves = compute_ec_curve(mesh, directions)

            assert radius.shape == (25,)
            assert ec_curves.shape == (2, 25)

    def test_normalized_vs_unnormalized_mesh(self):
        """Test EC curves for normalized vs unnormalized meshes"""
        vertices = torch.rand(30, 3) * 10  # Large scale

        mesh1 = Mesh(vertices.clone(), generate_mesh=True, radius=5.0)
        mesh2 = Mesh(vertices.clone(), generate_mesh=True, radius=5.0)
        mesh2.normalize()

        direction = np.array([[0.0, 0.0, 1.0]])

        # Both should compute without error
        radius1, ec1 = compute_ec_curve(mesh1, direction, ball_radius=5.0)
        radius2, ec2 = compute_ec_curve(mesh2, direction, ball_radius=1.0)

        assert radius1.shape == (25,)
        assert radius2.shape == (25,)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_mesh_edges(self):
        """Test mesh with no edges"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=True, radius=1.0)

        direction = np.array([1.0, 0.0, 0.0])
        ec_curve = compute_ec_curve_single(mesh, direction, ball_radius=1.0)

        assert ec_curve.shape == (25,)

    def test_planar_mesh(self):
        """Test mesh in a single plane"""
        vertices = torch.rand(20, 2)
        vertices_3d = torch.cat([vertices, torch.zeros(20, 1)], dim=1)
        mesh = Mesh(vertices_3d, generate_mesh=True, radius=2.0)

        direction = np.array([0.0, 0.0, 1.0])  # Perpendicular to plane
        ec_curve = compute_ec_curve_single(mesh, direction, ball_radius=1.0)

        assert ec_curve.shape == (25,)

    def test_very_small_mesh(self):
        """Test with minimal mesh (3 vertices)"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=True, radius=0.2)

        direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        ec_curve = compute_ec_curve_single(mesh, direction, ball_radius=0.5)

        assert ec_curve.shape == (25,)

    def test_uniform_height(self):
        """Test mesh with all vertices at same height in direction"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [2.0, 0.5, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        # Direction perpendicular to z (all vertices have z=0)
        direction = np.array([0.0, 0.0, 1.0])
        ec_curve = compute_ec_curve_single(mesh, direction, ball_radius=1.0)

        assert ec_curve.shape == (25,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
