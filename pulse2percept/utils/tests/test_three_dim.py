import numpy.testing as npt
import numpy as np
from pulse2percept.utils import parse_3d_orient


def test_parse_3d_orient():
    # Invalid input, must be (3) or (3, 3)
    with npt.assert_raises(TypeError):
        parse_3d_orient(0)
    with npt.assert_raises(TypeError):
        parse_3d_orient([0, 1])
    with npt.assert_raises(TypeError):
        parse_3d_orient([0, 1, 2, 3])
    with npt.assert_raises(TypeError):
        parse_3d_orient([0, 1, 2, 3, 4])
    with npt.assert_raises(TypeError):
        parse_3d_orient(np.array([[0, 1, 2, 2], 
                                  [0, 1, 2, 2]]))
    # Invalid rotation matrix:
    with npt.assert_raises(ValueError):
        parse_3d_orient(np.array([[0, 1, 2], 
                                  [0, 1, 2],
                                  [0, 1, 2]]))
    # Invalid direction 0 vector:
    with npt.assert_raises(ValueError):
        parse_3d_orient([0, 0, 0], orient_mode='direction')
    
    # Identity input:
    rot, angles, direction = parse_3d_orient([0, 0, 0], orient_mode='angle')
    npt.assert_almost_equal(rot, np.eye(3))
    npt.assert_almost_equal(angles, [0, 0, 0])
    npt.assert_almost_equal(direction, [0, 0, 1])
    rot, angles, direction = parse_3d_orient([0, 0, 1], orient_mode='direction')
    npt.assert_almost_equal(rot, np.eye(3))
    npt.assert_almost_equal(angles, [0, 0, 0])
    npt.assert_almost_equal(direction, [0, 0, 1])
    rot, angles, direction = parse_3d_orient(np.eye(3), orient_mode='rot')
    npt.assert_almost_equal(rot, np.eye(3))
    npt.assert_almost_equal(angles, [0, 0, 0])
    npt.assert_almost_equal(direction, [0, 0, 1])

    # Angles
    # This rotation does nothing
    rot, angles, direction = parse_3d_orient([0, 0, 90], orient_mode='angle') 
    npt.assert_almost_equal(rot, np.array([[0, -1, 0],
                                           [1, 0, 0],
                                           [0, 0, 1]]))
    npt.assert_almost_equal(angles, [0, 0, 90])
    npt.assert_almost_equal(direction, [0, 0, 1])

    # Simple rotation
    rot, angles, direction = parse_3d_orient([0, 90, 0], orient_mode='angle')
    npt.assert_almost_equal(rot, np.array([[0, 0, 1],
                                           [0, 1, 0],
                                           [-1, 0, 0]]))
    npt.assert_almost_equal(angles, [0, 90, 0])
    npt.assert_almost_equal(direction, [1, 0, 0])

    # Simple rotation
    rot, angles, direction = parse_3d_orient([90, 0, 0], orient_mode='angle')
    npt.assert_almost_equal(rot, np.array([[1, 0, 0],
                                           [0, 0, -1],
                                           [0, 1, 0]]))
    npt.assert_almost_equal(angles, [90, 0, 0])
    npt.assert_almost_equal(direction, [0, -1, 0])

    # all the way around
    rot, angles, direction = parse_3d_orient([180, 180, 180], orient_mode='angle')
    npt.assert_almost_equal(rot, np.array([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]]))
    npt.assert_almost_equal(angles, [180, 180, 180])
    npt.assert_almost_equal(direction, [0, 0, 1])

    # Direction
    # test non unit vector
    rot, angles, direction = parse_3d_orient([0, 0, 5], orient_mode='direction')
    npt.assert_almost_equal(rot, np.eye(3))
    npt.assert_almost_equal(angles, [0, 0, 0])
    npt.assert_almost_equal(direction, [0, 0, 1])

    # towards x axis
    rot, angles, direction = parse_3d_orient([1, 0, 0], orient_mode='direction')
    npt.assert_almost_equal(rot, np.array([[0, 0, 1],
                                           [0, 1, 0],
                                           [-1, 0, 0]]))
    npt.assert_almost_equal(angles, [0, 90, 0])
    npt.assert_almost_equal(direction, [1, 0, 0])

    # towards y axis
    rot, angles, direction = parse_3d_orient([0, 1, 0], orient_mode='direction')
    npt.assert_almost_equal(rot, np.array([[0, -1, 0],
                                           [1, 0, 0],
                                           [0, 0, 1]]) @ np.array(
                                                [[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]))
    # the angle decomposition is ambiguous, but 
    # ours will NEVER give an x angle other than 0
    # So the x axis rotation is split into y and z
    npt.assert_almost_equal(angles, [0, 90, 90])
    npt.assert_almost_equal(direction, [0, 1, 0])

    # both
    rot, angles, direction = parse_3d_orient([1, 1, 0], orient_mode='direction')
    npt.assert_almost_equal(rot, np.array([[0.707107, -0.707107, 0],
                                           [0.707107, 0.707107, 0],
                                           [0, 0, 1]]) @ np.array(
                                               [[0, 0, 1],
                                                [0, 1, 0],
                                                [-1, 0, 0]]),
                                                decimal=5)
    npt.assert_almost_equal(angles, [0, 90, 45])
    npt.assert_almost_equal(direction, [1/np.sqrt(2), 1 / np.sqrt(2), 0])



