import numpy as np


def rpy_to_matrix(coords):
    """Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.
    The roll-pitch-yaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.
    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    Parameters
    ----------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    Returns
    -------
    R : (3,3) float
        The corresponding homogenous 3x3 rotation matrix.
    """
    coords = np.asanyarray(coords, dtype=np.float64)
    c3, c2, c1 = np.cos(coords)
    s3, s2, s1 = np.sin(coords)

    return np.array([
        [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
        [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
        [-s2, c2 * s3, c2 * c3]
    ], dtype=np.float64)


def matrix_to_rpy(R, solution=1):
    """Convert a 3x3 transform matrix to roll-pitch-yaw coordinates.
    The roll-pitch-yaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.
    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    There are typically two possible roll-pitch-yaw coordinates that could have
    created a given rotation matrix. Specify ``solution=1`` for the first one
    and ``solution=2`` for the second one.
    Parameters
    ----------
    R : (3,3) float
        A 3x3 homogenous rotation matrix.
    solution : int
        Either 1 or 2, indicating which solution to return.
    Returns
    -------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    """
    R = np.asanyarray(R, dtype=np.float64)
    r = 0.0
    p = 0.0
    y = 0.0

    if np.abs(R[2, 0]) >= 1.0 - 1e-12:
        y = 0.0
        if R[2, 0] < 0:
            p = np.pi / 2
            r = np.arctan2(R[0, 1], R[0, 2])
        else:
            p = -np.pi / 2
            r = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        if solution == 1:
            p = -np.arcsin(R[2, 0])
        else:
            p = np.pi + np.arcsin(R[2, 0])
        r = np.arctan2(R[2, 1] / np.cos(p), R[2, 2] / np.cos(p))
        y = np.arctan2(R[1, 0] / np.cos(p), R[0, 0] / np.cos(p))

    return np.array([r, p, y], dtype=np.float64)


def matrix_to_xyz_rpy(matrix):
    """Convert a 4x4 homogenous matrix to xyzrpy coordinates.
    Parameters
    ----------
    matrix : (4,4) float
        The homogenous transform matrix.
    Returns
    -------
    xyz_rpy : (6,) float
        The xyz_rpy vector.
    """
    xyz = matrix[:3, 3]
    rpy = matrix_to_rpy(matrix[:3, :3])
    return np.hstack((xyz, rpy))


def xyz_rpy_to_matrix(xyz_rpy):
    """Convert xyz_rpy coordinates to a 4x4 homogenous matrix.
    Parameters
    ----------
    xyz_rpy : (6,) float
        The xyz_rpy vector.
    Returns
    -------
    matrix : (4,4) float
        The homogenous transform matrix.
    """
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = xyz_rpy[:3]
    matrix[:3, :3] = rpy_to_matrix(xyz_rpy[3:])
    return matrix


def inverse_transform(matrix):
    """inverse transform matrix.
        Parameters
        ----------
        matrix : (4,4) float
            The homogenous transform matrix.
        Returns
        -------
        matrix : (4,4) float
            The homogenous inverse transform matrix.
    """
    inv = matrix[:3, :3].T
    inv_matrix = np.eye(4, dtype=np.float64)
    inv_matrix[:3, :3] = inv
    inv_matrix[:3, [3]] = -inv @ matrix[:3, [3]]
    return inv_matrix
