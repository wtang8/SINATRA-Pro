"""
Euler Characteristic Transform (ECT) computation for mesh topology analysis.

This module provides functions for computing Euler characteristic curves
along different directions for 3D meshes. The ECT is a topological signature
that captures shape information through directional height functions.
"""

import numpy as np
from fast_histogram import histogram1d
from sinatra_pro.mesh import Mesh


def compute_ec_curve_single(
    mesh: Mesh,
    direction: np.ndarray,
    ball_radius: float,
    n_filtrations: int = 25,
    ec_type: str = "ECT",
    include_faces: bool = True
) -> np.ndarray:
    """
    Compute Euler Characteristic (EC) curve in a given direction.

    Calculates the EC curve for a mesh by projecting vertices onto a direction
    vector and computing Euler characteristics at discrete filtration levels.

    Parameters
    ----------
    mesh : Mesh
        Mesh object containing vertices, edges, and faces.
    direction : np.ndarray
        Unit direction vector for EC curve calculation. Shape: (3,)
    ball_radius : float
        Radius of the bounding ball for filtration range.
        EC curves are computed from -ball_radius to +ball_radius.
    n_filtrations : int, default=25
        Number of filtration levels (discrete steps) for EC computation.
    ec_type : str, default="ECT"
        Type of Euler Characteristic Transform:
        - "ECT": Standard ECT (cumulative sum)
        - "DECT": Differential ECT (normalized by bin width)
        - "SECT": Smooth ECT (double cumulative sum with mean centering)
    include_faces : bool, default=True
        If True, includes faces in EC calculation (V - E + F).
        If False, computes only V - E.

    Returns
    -------
    np.ndarray
        EC curve values at n_filtrations discrete levels. Shape: (n_filtrations,)

    Notes
    -----
    The Euler characteristic is computed using the formula: χ = V - E + F
    where V, E, F are the number of vertices, edges, and faces respectively
    in each sublevel set.

    Examples
    --------
    >>> import numpy as np
    >>> from sinatra_pro.mesh import Mesh
    >>> vertices = np.random.rand(100, 3)
    >>> mesh = Mesh(vertices, generate_mesh=True, radius=2.0)
    >>> direction = np.array([0.0, 0.0, 1.0])
    >>> ec_curve = compute_ec_curve_single(mesh, direction, ball_radius=1.0)
    >>> ec_curve.shape
    (25,)
    """
    # Initialize EC curve
    eulers = np.zeros(n_filtrations, dtype=float)

    # Project vertices onto direction vector (height function)
    vertex_function = np.dot(mesh.vertices, direction)

    # Create filtration range
    radius = np.linspace(-ball_radius, ball_radius, n_filtrations)

    # Filtrate vertices - count vertices in each bin
    V = histogram1d(
        vertex_function,
        range=[-ball_radius, ball_radius],
        bins=(n_filtrations - 1)
    )

    # Filtrate edges - use maximum height of edge endpoints
    if len(mesh.edges) > 0:
        edge_function = np.amax(vertex_function[mesh.edges], axis=1)
        E = histogram1d(
            edge_function,
            range=[-ball_radius, ball_radius],
            bins=(n_filtrations - 1)
        )
    else:
        E = 0

    # Filtrate faces - use maximum height of face vertices
    if include_faces and len(mesh.faces) > 0:
        face_function = np.amax(vertex_function[mesh.faces], axis=1)
        F = histogram1d(
            face_function,
            range=[-ball_radius, ball_radius],
            bins=(n_filtrations - 1)
        )
    else:
        F = 0

    # Compute Euler characteristic: χ = V - E + F
    eulers[1:] = V - E + F

    # Apply transformation based on ec_type
    if ec_type == "ECT":
        # Standard ECT: cumulative sum
        eulers = np.cumsum(eulers)
        return eulers

    elif ec_type == "DECT":
        # Differential ECT: normalize by bin width
        eulers[1:] = eulers[1:] / (radius[1:] - radius[:-1])
        return eulers

    elif ec_type == "SECT":
        # Smooth ECT: double cumulative sum with mean centering
        eulers = np.cumsum(eulers)
        eulers -= np.mean(eulers)
        eulers = np.cumsum(eulers) * ((radius[-1] - radius[0]) / n_filtrations)
        return eulers

    else:
        raise ValueError(
            f"Invalid ec_type '{ec_type}'. "
            "Must be one of: 'ECT', 'DECT', 'SECT'"
        )


def compute_ec_curve(
    mesh: Mesh,
    directions: np.ndarray,
    n_filtrations: int = 25,
    ball_radius: float = 1.0,
    ec_type: str = "ECT",
    include_faces: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Euler Characteristic curves for multiple directions.

    Computes EC curves for a mesh along multiple direction vectors,
    useful for capturing comprehensive shape information.

    Parameters
    ----------
    mesh : Mesh
        Mesh object containing vertices, edges, and faces.
    directions : np.ndarray
        Array of unit direction vectors. Shape: (n_directions, 3)
    n_filtrations : int, default=25
        Number of filtration levels for each EC curve.
    ball_radius : float, default=1.0
        Radius of bounding ball (typically 1.0 for normalized meshes).
    ec_type : str, default="ECT"
        Type of EC Transform: "ECT", "DECT", or "SECT".
    include_faces : bool, default=True
        Whether to include faces in EC calculation.

    Returns
    -------
    radius : np.ndarray
        Filtration radius values. Shape: (n_filtrations,)
    eulers : np.ndarray
        EC curves for all directions. Shape: (n_directions, n_filtrations)

    Examples
    --------
    >>> import numpy as np
    >>> from sinatra_pro.mesh import Mesh
    >>> from sinatra_pro.directions import generate_equidistributed_cones
    >>> vertices = np.random.rand(100, 3)
    >>> mesh = Mesh(vertices, generate_mesh=True, radius=2.0)
    >>> directions = generate_equidistributed_cones(n_cone=50)
    >>> radius, ec_curves = compute_ec_curve(mesh, directions)
    >>> ec_curves.shape
    (50, 25)
    """
    n_directions = directions.shape[0]
    eulers = np.zeros((n_directions, n_filtrations), dtype=float)

    # Compute EC curve for each direction
    for i in range(n_directions):
        eulers[i] = compute_ec_curve_single(
            mesh, directions[i], ball_radius, n_filtrations, ec_type, include_faces
        )

    # Generate filtration radius values
    radius = np.linspace(-ball_radius, ball_radius, n_filtrations)

    return radius, eulers
