"""
Utilities for generating equidistributed directions on spheres and cones.

This module provides functions for:
- Generating equidistributed points on spheres/hemispheres
- Computing directions around cones using Rodrigues' rotation formula
- Creating equidistributed cones for directional analysis
"""

import numpy as np


def rodrigues(z: np.ndarray, r: float, j: int) -> np.ndarray:
    """
    Compute directions around a cone using Rodrigues' rotation formula.

    Parameters
    ----------
    z : np.ndarray
        Vector defining the central axis of the cone (will be normalized).
        Shape: (3,)
    r : float
        Radius of the cone that controls the size of the cone opening.
        Range: [0, 1] where 0 is along the axis and 1 is perpendicular.
    j : int
        Number of directions to generate around the cone.

    Returns
    -------
    np.ndarray
        Array of unit direction vectors around the cone.
        Shape: (j, 3)

    Notes
    -----
    Uses Rodrigues' rotation formula to rotate a vector around the cone axis
    at evenly spaced angles.
    """
    # Normalize the central axis
    z = z / np.linalg.norm(z)

    # Find a vector perpendicular to z
    z0 = np.zeros(3)
    if np.any(z == 0):
        # If any component is zero, use a unit vector along that axis
        z0[z == 0] = 1
    else:
        # Create perpendicular vector using plane equation dot(z, z0) = 0
        z0[0] = 2 / z[0]
        z0[1] = -1 / z[1]
        z0[2] = -1 / z[2]

    # Normalize and scale by radius, then add to central axis
    z0 = z0 / np.linalg.norm(z0)
    z0 = z0 * r
    z0 = z + z0
    z0 = z0 / np.linalg.norm(z0)

    # Compute rotation axis and projection components
    B = np.cross(z, z0)  # Rotation axis
    C = np.dot(z, z0) * z  # Component along z

    # Generate j evenly spaced directions around the cone
    directions = np.zeros((j, 3))
    angles = 2 * np.pi * (np.arange(j) + 1) / j

    for i, angle in enumerate(angles):
        # Rodrigues' rotation formula
        directions[i] = (
            z0 * np.cos(angle) + B * np.sin(angle) + C * (1 - np.cos(angle))
        )

    return directions


def generate_equidistributed_points(
    desired_number: int, N: int, hemisphere: bool = False
) -> np.ndarray:
    """
    Generate equidistributed points on a sphere or hemisphere.

    Uses the Fibonacci sphere algorithm to generate approximately uniform
    point distributions on the unit sphere.

    Parameters
    ----------
    desired_number : int
        Desired number of equidistributed points on the sphere.
    N : int
        Initial number of points to generate. If fewer than desired_number
        points are generated, N is automatically incremented recursively.
    hemisphere : bool, default=False
        If True, generates points on upper hemisphere only (z >= 0).
        If False, generates points on entire sphere.

    Returns
    -------
    np.ndarray
        Array of unit vectors representing points on the sphere.
        Shape: (n_points, 3) where n_points >= desired_number

    Notes
    -----
    The algorithm uses spherical coordinates (theta, phi) with:
    - theta: polar angle from z-axis [0, π] or [0, π/2] for hemisphere
    - phi: azimuthal angle [0, 2π]

    The number of points at each latitude band is proportional to sin(theta)
    to maintain approximately uniform density.
    """
    # Calculate angular spacing
    if hemisphere:
        a = 2 * np.pi / N  # Surface area per point on hemisphere
        d = np.sqrt(a)
        M_theta = int(round(np.pi * 0.5 / d))
        d_theta = np.pi * 0.5 / M_theta
    else:
        a = 4 * np.pi / N  # Surface area per point on sphere
        d = np.sqrt(a)
        M_theta = int(round(np.pi / d))
        d_theta = np.pi / M_theta

    d_phi = a / d_theta

    points = []
    for i in range(M_theta):
        # Polar angle
        if hemisphere:
            theta = np.pi * 0.5 * i / M_theta
        else:
            theta = np.pi * (i + 0.5) / M_theta

        # Number of points at this latitude
        M_phi = int(round(2 * np.pi * np.sin(theta) / d_phi))

        # Azimuthal angles at this latitude
        for j in range(M_phi):
            phi = 2 * np.pi * j / M_phi

            # Convert spherical to Cartesian coordinates
            point = np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )
            point = point / np.linalg.norm(point)
            points.append(point)

    points = np.array(points)

    # Recursively increase N if we don't have enough points
    if points.shape[0] < desired_number:
        return generate_equidistributed_points(desired_number, N + 1, hemisphere)

    return points


def generate_equidistributed_cones(
    n_cones: int,
    cap_radius: float = 0.1,
    n_direction_per_cone: int = 1,
    hemisphere: bool = False,
) -> np.ndarray:
    """
    Generate equidistributed cones on a sphere or hemisphere.

    Creates a set of cones with central axes distributed uniformly across
    the sphere, with optional additional directions within each cone.

    Parameters
    ----------
    n_cones : int
        Number of cones (equidistributed central axes) to generate.
    cap_radius : float, default=0.1
        Radius of each cone, controlling the cone opening angle.
        Range: [0, 1] where 0 is infinitely narrow and 1 is 90 degrees.
    n_direction_per_cone : int, default=1
        Number of directions per cone:
        - If 1, only the central axis direction is used
        - If > 1, additional directions are generated around each cone
          using Rodrigues' rotation formula
    hemisphere : bool, default=False
        If True, generates cones on upper hemisphere only.
        If False, generates cones on entire sphere.

    Returns
    -------
    np.ndarray
        Array of unit direction vectors.
        Shape: (n_cones * n_direction_per_cone, 3)

    Examples
    --------
    >>> # Generate 100 uniformly distributed directions
    >>> directions = generate_equidistributed_cones(n_cones=100)
    >>> directions.shape
    (100, 3)

    >>> # Generate 50 cones with 4 directions each (200 total)
    >>> directions = generate_equidistributed_cones(
    ...     n_cones=50,
    ...     cap_radius=0.2,
    ...     n_direction_per_cone=4
    ... )
    >>> directions.shape
    (200, 3)
    """
    # Generate equidistributed points on sphere (cone central axes)
    sphere = generate_equidistributed_points(n_cones, n_cones, hemisphere)

    directions = []
    for i in range(n_cones):
        # Add the central axis direction (already normalized from generation)
        directions.append(sphere[i])

        # Add additional directions around the cone if requested
        if n_direction_per_cone > 1:
            cone_directions = rodrigues(sphere[i], cap_radius, n_direction_per_cone - 1)
            directions.extend(cone_directions)

    return np.array(directions)
