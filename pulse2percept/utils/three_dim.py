"""`parse_3d_orient`"""

import numpy as np

def parse_3d_orient(orient, orient_mode):
    """Parse the orient parameter
    Given either a 3D rotation matrix, vector of angles of rotation,
    or direction vector, this function will calculate and return the 
    all three representations.
    
    Parameters
    ----------
    orient : np.ndarray with shape (3) or (3, 3)
        Orientation of the electrode in 3D space.
        orient can be:
            - A length 3 vector specifying the direction that the 
              thread should extend in (if orient_mode == 'direction')
            - A list of 3 angles, (r_x, r_y, r_z), specifying the rotation 
              in degrees about each axis (x rotation performed first). 
              (If orient_mode == 'angle')
            - 3D rotation matrix, specifying the direction that the thread 
              should extend in (i.e. a unit vector in the z direction will
              point in the direction after being rotated by this matrix)
    orient_mode : str
        If 'direction', orient is a vector specifying the direction that the
        electrode should extend in. If 'angle', orient is a vector of 3 angles,
        (r_x, r_y, r_z), specifying the rotation in degrees about each axis
        (starting with x). Does not apply if orient is a 3D rotation matrix.

    Returns
    -------
    rot : np.ndarray with shape (3, 3)
        Rotation matrix
    angles : np.ndarray with shape (3)
        Angles of rotation (degrees) about each axis (x, y, z).
        Note that this mapping is not unique. This function will always
        set the rotation about the x axis to be 0, meaning that the
        returned coordinates will match spherical coordinates (i.e.
        r_y is phi and r_z is theta).
    direction : np.ndarray with shape (3)
        Unit vector specifying the direction of the orientation.
    """

    def construct_rot_matrix(angles):
        """Construct a rotation matrix from angles of rotation"""
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(angles[0]), -np.sin(angles[0])],
                          [0, np.sin(angles[0]), np.cos(angles[0])]])
        rot_y = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                          [0, 1, 0],
                          [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        rot_z = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                          [np.sin(angles[2]), np.cos(angles[2]), 0],
                          [0, 0, 1]])
        return rot_z @ rot_y @ rot_x
    
    def extract_direction(rot):
        """Extract direction vector from rotation matrix"""
        # Do this by rotating the point (0, 0, 1)
        direction = np.matmul(rot, np.array([0, 0, 1]))
        return direction
    
    def extract_angles(direction):
        """Extract angles of rotation from direction vector"""
        rot_x = 0
        rot_y = np.arctan2(direction[0], direction[2]) # i.e. phi
        rot_z = np.arctan2(direction[1], direction[0]) # i.e. theta
        angles = np.array([rot_x, rot_y, rot_z])
        return angles

    if isinstance(orient, list):
        orient = np.array(orient)
    if not isinstance(orient, np.ndarray) or orient.shape not in [(3,), (3, 3)]:
        raise ValueError(f'Incorrect value for orient parameter {orient}, ', 
                         'please pass an array with shape (3) or (3, 3)')
    if orient.ndim == 1:
        if orient_mode == 'direction':
            if not np.allclose(np.linalg.norm(orient), 1):
                # unnormalized
                if np.linalg.norm(orient) == 0:
                    raise ValueError('orient cannot be a zero vector if orient_mode is "direction"')
                orient = orient / np.linalg.norm(orient)

            direction = orient
            angles = extract_angles(direction)
            rot = construct_rot_matrix(angles)
        elif orient_mode == 'angle':
            angles = orient
            rot = construct_rot_matrix(angles)
            direction = extract_direction(rot)
        else:
            raise ValueError('orient_mode must be either "direction" or "angle".')
        
    elif orient.ndim == 2:
        if not np.allclose(np.linalg.inv(orient), orient.T) or orient.shape != (3, 3):
            raise ValueError(f'Invalid rotation matrix {orient}')
        rot = orient
        direction = extract_direction(rot)
        angles = extract_angles(direction)

    return rot, angles, direction