import torch
import numpy as np
from typing import Optional
from scipy.spatial import cKDTree # type: ignore


class Mesh:

    vertices: torch.Tensor # (n_vertices, 3) coordinates of vertices
    edges: np.ndarray # List of indices of connected edges
    faces: np.ndarray # List of indices of constructed faces

    @property
    def n_vertices(self):
        return self.vertices.shape[0]

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_faces(self):
        return len(self.faces)

    def __init__(self, vertices: torch.Tensor, generate_mesh: bool = True, radius: float = 1.0):
        self.vertices = vertices
        if generate_mesh:
            self.build_kdtree()
            self.connect_edges(radius)
            self.construct_faces()
        pass
    
    def calc_radius(self) -> float:
        """Calculate distance of the vertex furthest away from origin """
        return torch.amax(torch.linalg.norm(self.vertices,axis=1)).item()
    
    def normalize(self, radius: Optional[float] = None):
        """Normalize vertices to the unit sphere """
        if radius is None:
            radius = self.calc_radius()
        assert radius is not None
        self.vertices /= radius
        return
   
    def centering(self):
        """Center protein to origin by center of geometry"""
        self.vertices -= self.vertices.mean(dim=0)
        return 
    
    def build_kdtree(self):
        self.tree = cKDTree(self.vertices.numpy())
        return

    def connect_edges(self, radius: float):
        """
        Get list of edges within distance cutoff using KD-tree.

        Parameters
        ----------
        radius : float
            Distance cutoff for edges

        Returns
        -------
        pairs : np.ndarray
            Array of vertex index pairs within radius
        """
        pairs = self.tree.query_pairs(r=radius, output_type='ndarray')
        self.edges = pairs
        return
   
    def construct_faces(self):
        """
        Convert edge list to face list

        Iterate through list of connected edges to look for any 3 edges that enclose a triangle. 
        Such enclosed triangles are constructed as faces.
        """
        self.connections = [set() for i in range(self.n_vertices)]
        for edge in self.edges:
            self.connections[edge[0]].add(edge[1])
            self.connections[edge[1]].add(edge[0])
        faces = []
        self.checked = set()
        for u in range(self.n_vertices):
            self.checked.add(u)
            for v in self.connections[u] - self.checked:
                for s in self.connections[u] & self.connections[v]:
                    if s > v and v > u:
                        faces.append([u, v, s])
        self.faces = np.array(faces)
        return
    


