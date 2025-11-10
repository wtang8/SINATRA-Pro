import pytest
import numpy as np
import torch

from sinatra_pro.mesh import Mesh


class TestMeshBasics:
    """Test basic Mesh properties and initialization"""

    def test_init(self):
        """Test Mesh initialization without auto-generating mesh"""
        vertices = torch.rand(10, 3)
        mesh = Mesh(vertices, generate_mesh=False)
        assert mesh.vertices is not None
        torch.testing.assert_close(mesh.vertices, vertices)

    def test_init_with_mesh_generation(self):
        """Test Mesh initialization with automatic mesh generation"""
        vertices = torch.rand(10, 3)
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)
        assert mesh.vertices is not None
        assert hasattr(mesh, 'edges')
        assert hasattr(mesh, 'faces')

    def test_n_vertices_property(self):
        """Test n_vertices property"""
        vertices = torch.rand(15, 3)
        mesh = Mesh(vertices, generate_mesh=False)
        assert mesh.n_vertices == 15

    def test_calc_radius(self):
        """Test radius calculation"""
        # Create vertices with known maximum radius
        vertices = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        mesh = Mesh(vertices, generate_mesh=False)
        radius = mesh.calc_radius()
        # The furthest vertex is [0, 0, 3] with distance 3.0
        assert np.isclose(radius, 3.0)

    def test_normalize(self):
        """Test vertex normalization"""
        vertices = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
        ])
        mesh = Mesh(vertices, generate_mesh=False)
        mesh.normalize()
        # After normalization, max distance should be 1.0
        max_dist = torch.amax(torch.linalg.norm(mesh.vertices, dim=1))
        assert torch.isclose(max_dist, torch.tensor(1.0))

    def test_normalize_with_radius(self):
        """Test vertex normalization with explicit radius"""
        vertices = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 6.0],
        ])
        mesh = Mesh(vertices, generate_mesh=False)
        mesh.normalize(radius=2.0)
        # After normalization by 2.0, max coord should be 3.0
        max_coord = torch.amax(torch.abs(mesh.vertices))
        assert torch.isclose(max_coord, torch.tensor(3.0))

    def test_centering(self):
        """Test centering to origin"""
        vertices = torch.tensor([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ])
        mesh = Mesh(vertices, generate_mesh=False)
        mesh.centering()
        # After centering, mean should be close to origin
        center = torch.mean(mesh.vertices, dim=0)
        torch.testing.assert_close(center, torch.zeros(3), atol=1e-6, rtol=0)


class TestKDTreeMethods:
    """Test KD-tree build and edge connection functionality"""

    def test_build_kdtree(self):
        """Test KD-tree construction"""
        vertices = torch.rand(10, 3)
        mesh = Mesh(vertices, generate_mesh=False)
        mesh.build_kdtree()
        assert hasattr(mesh, 'tree')
        assert mesh.tree is not None

    def test_connect_edges_simple(self):
        """Test edge connection with simple case"""
        # Create a simple 3-vertex triangle
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=False)
        mesh.build_kdtree()

        # Search with cutoff that captures all edges
        mesh.connect_edges(radius=2.0)

        # Should find 3 edges (all pairs)
        assert len(mesh.edges) == 3

        # Verify edges are within cutoff
        vertices_np = vertices.numpy()
        for edge in mesh.edges:
            dist = np.linalg.norm(vertices_np[edge[0]] - vertices_np[edge[1]])
            assert dist <= 2.0

    def test_connect_edges_cutoff(self):
        """Test that cutoff is properly enforced"""
        # Create 4 vertices in a line
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=False)
        mesh.build_kdtree()

        # With cutoff=1.5, should only connect adjacent vertices
        mesh.connect_edges(radius=1.5)

        # Should find 3 edges: (0,1), (1,2), (2,3)
        assert len(mesh.edges) == 3

        # All edges should have distance ~1.0
        vertices_np = vertices.numpy()
        for edge in mesh.edges:
            dist = np.linalg.norm(vertices_np[edge[0]] - vertices_np[edge[1]])
            assert dist <= 1.5

    def test_connect_edges_no_neighbors(self):
        """Test with cutoff that finds no neighbors"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=False)
        mesh.build_kdtree()

        # Very small cutoff should find no edges
        mesh.connect_edges(radius=1.0)

        assert len(mesh.edges) == 0

    def test_connect_edges_3d(self):
        """Test edge connection in 3D space"""
        # Create a cube
        vertices = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ], dtype=torch.float32)
        mesh = Mesh(vertices, generate_mesh=False)
        mesh.build_kdtree()

        # Cutoff to get edges of cube (distance = 1.0)
        mesh.connect_edges(radius=1.1)

        # Cube has 12 edges
        assert len(mesh.edges) == 12


class TestConstructFaces:
    """Test face construction from edges"""

    def test_construct_faces_triangle(self):
        """Test simple triangle face detection"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=False)

        # Create edges manually for a triangle
        mesh.edges = np.array([[0, 1], [1, 2], [0, 2]])

        mesh.construct_faces()

        # Should find 1 face
        assert len(mesh.faces) == 1
        # Face should contain all 3 vertices
        assert set(mesh.faces[0]) == {0, 1, 2}

    def test_construct_faces_tetrahedron(self):
        """Test tetrahedron face detection"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0],
            [0.5, np.sqrt(3)/6, np.sqrt(2/3)],
        ])
        mesh = Mesh(vertices, generate_mesh=False)
        mesh.build_kdtree()

        # Use edge connection to find edges
        mesh.connect_edges(radius=1.5)

        mesh.construct_faces()

        # Tetrahedron has 4 faces
        assert len(mesh.faces) == 4

    def test_construct_faces_no_faces(self):
        """Test with edges that don't form faces"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        mesh = Mesh(vertices, generate_mesh=False)

        # Linear edges - no triangular faces
        mesh.edges = np.array([[0, 1], [1, 2]])

        mesh.construct_faces()

        # Should find no faces
        assert len(mesh.faces) == 0

    def test_automatic_mesh_generation(self):
        """Test automatic mesh generation in constructor"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        # Use automatic generation
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        # Should have generated edges and faces
        assert len(mesh.edges) == 3
        assert len(mesh.faces) == 1


class TestIntegration:
    """Test complete mesh generation workflows"""

    def test_full_pipeline_simple(self):
        """Test full mesh generation pipeline"""
        # Create simple tetrahedral vertices
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0],
            [0.5, np.sqrt(3)/6, np.sqrt(2/3)],
        ])

        # Generate mesh automatically
        mesh = Mesh(vertices, generate_mesh=True, radius=2.0)

        # Verify mesh has edges and faces
        assert len(mesh.edges) > 0
        assert len(mesh.faces) > 0

    def test_radius_effect_on_edges(self):
        """Test that radius affects edge count"""
        torch.manual_seed(42)
        vertices = torch.rand(20, 3) * 10

        # Smaller radius
        mesh1 = Mesh(vertices.clone(), generate_mesh=True, radius=2.0)

        # Larger radius
        mesh2 = Mesh(vertices.clone(), generate_mesh=True, radius=5.0)

        # Larger radius should produce more edges
        assert len(mesh2.edges) >= len(mesh1.edges)

    def test_manual_pipeline(self):
        """Test manually building mesh step by step"""
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        mesh = Mesh(vertices, generate_mesh=False)

        # Manually build mesh
        mesh.build_kdtree()
        mesh.connect_edges(radius=2.0)
        mesh.construct_faces()

        # Verify results
        assert len(mesh.edges) == 3
        assert len(mesh.faces) == 1


# class TestPerformance:
#     """Test performance characteristics of KD-tree implementation"""

#     def test_large_vertex_count(self):
#         """Test that algorithm handles large vertex counts efficiently"""
#         # Generate 1000 random vertices
#         torch.manual_seed(42)
#         vertices = torch.rand(1000, 3) * 10

#         # This should complete quickly with KD-tree (< 1 second)
#         import time
#         start = time.time()
#         mesh = Mesh(vertices, generate_mesh=True, radius=2.0)
#         elapsed = time.time() - start

#         # Should complete in reasonable time (< 1 second for 1000 vertices)
#         assert elapsed < 1.0
#         assert len(mesh.edges) > 0

#     def test_scaling_behavior(self):
#         """Test that KD-tree scales better than O(N²)"""
#         torch.manual_seed(42)

#         import time

#         # Test with different sizes
#         times = []
#         sizes = [100, 200, 400]

#         for n in sizes:
#             vertices = torch.rand(n, 3) * 10

#             start = time.time()
#             mesh = Mesh(vertices, generate_mesh=True, radius=2.0)
#             elapsed = time.time() - start
#             times.append(elapsed)

#         # Time should scale roughly as O(N log N), not O(N²)
#         # If it were O(N²), doubling N would quadruple time
#         # For O(N log N), doubling N would roughly double time (or slightly more)

#         # Check that the scaling is sub-quadratic
#         # times[1]/times[0] should be < 4 (would be ~4 for O(N²))
#         if times[0] > 0:  # Avoid division by zero
#             ratio = times[1] / times[0]
#             assert ratio < 4.0, f"Scaling appears quadratic: {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
