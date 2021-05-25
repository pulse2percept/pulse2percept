import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants import AlphaIMS, AlphaAMS


@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_AlphaIMS(ztype, x, y, rot):
    # Height `h` can either be a float or a list
    if ztype == 'float':
        alpha = AlphaIMS(x=x, y=y, z=-100, rot=rot)
        for e in alpha.electrode_objects:
            npt.assert_almost_equal(e.z, -100)
    else:
        alpha = AlphaIMS(x=x, y=y, z=np.arange(1500), rot=rot)
        for i, e in enumerate(alpha.electrode_objects):
            npt.assert_almost_equal(e.z, i)

    # Slots:
    npt.assert_equal(hasattr(alpha, '__slots__'), True)
    npt.assert_equal(hasattr(alpha, '__dict__'), False)

    # Coordinates of first electrode
    # 18.5 *spacing - spacing/2 for middle coordinate if (0,0) is upper-left
    # corner
    xy = np.array([-1368, -1368]).T

    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)

    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(alpha['A1'].x, xy[0] + x)
    npt.assert_almost_equal(alpha['A1'].y, xy[1] + y)

    # Make sure array center is still (x,y)
    y_center = alpha['AM15'].y + (alpha['A25'].y - alpha['AM15'].y) / 2
    npt.assert_almost_equal(y_center, y)
    x_center = alpha['A15'].x + (alpha['AM25'].x - alpha['A15'].x) / 2
    npt.assert_almost_equal(x_center, x)

    # Check width of square electrodes
    for e in ['A1', 'B2', 'C3']:
        npt.assert_equal(alpha[e].a, 50)

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        AlphaIMS(x=-100, y=10, z=np.arange(28))

    # Indexing must work for both integers and electrode names
    alpha = AlphaIMS()
    # enumerate returns ((0, alpha.items()[0]), (1, alpha.items()[1]), ...)
    # idx = 0, ... 36. name = A1, ... e37. electrode = DiskElectrode(...)
    for idx, (name, electrode) in enumerate(alpha.electrodes.items()):
        npt.assert_equal(electrode, alpha[idx])
        npt.assert_equal(electrode, alpha[name])
        npt.assert_equal(alpha["unlikely name for an electrode"], None)

    # Right-eye implant:
    xc, yc = 1600, -1600
    alpha_re = AlphaIMS(eye='RE', x=xc, y=yc)
    npt.assert_equal(alpha_re['A37'].x > alpha_re['A1'].x, True)
    npt.assert_almost_equal(alpha_re['A37'].y, alpha_re['A1'].y)

    # Left-eye implant:
    alpha_le = AlphaIMS(eye='LE', x=xc, y=yc)
    npt.assert_equal(alpha_le['A1'].x > alpha_le['AE37'].x, True)
    npt.assert_almost_equal(alpha_le['A37'].y, alpha_le['A1'].y)

    # In both left and right eyes, rotation with positive angle should be
    # counter-clock-wise (CCW): for (x>0,y>0), decreasing x and increasing y
    for eye, el in zip(['LE', 'RE'], ['A1', 'A37']):
        before = AlphaIMS(eye=eye)
        after = AlphaIMS(eye=eye, rot=10)
        npt.assert_equal(after[el].x > before[el].x, True)
        npt.assert_equal(after[el].y > before[el].y, True)

    # Invalid eye string:
    with pytest.raises(TypeError):
        AlphaIMS(eye=[1, 2])
    with pytest.raises(ValueError):
        AlphaIMS(eye='left eye')


@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_AlphaAMS(ztype, x, y, rot):
    # Height `h` can either be a float or a list
    if ztype == 'float':
        alpha = AlphaAMS(x=x, y=y, z=-100, rot=rot)
        for e in alpha.electrode_objects:
            npt.assert_almost_equal(e.z, -100)
    else:
        alpha = AlphaAMS(x=x, y=y, z=np.arange(1600), rot=rot)
        for i, e in enumerate(alpha.electrode_objects):
            npt.assert_almost_equal(e.z, i)

    # Slots:
    npt.assert_equal(hasattr(alpha, '__slots__'), True)
    npt.assert_equal(hasattr(alpha, '__dict__'), False)

    # Rotate coordinates of first electrode:
    rot_rad = np.deg2rad(rot)
    xy = np.array([-1365, -1365]).T
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)
    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(alpha['A1'].x, xy[0] + x)
    npt.assert_almost_equal(alpha['A1'].y, xy[1] + y)

    # Make sure array center is still (x,y)
    y_center = alpha['AN1'].y + (alpha['A40'].y - alpha['AN1'].y) / 2
    npt.assert_almost_equal(y_center, y)
    x_center = alpha['A1'].x + (alpha['AN40'].x - alpha['A1'].x) / 2
    npt.assert_almost_equal(x_center, x)

    # Check radii of electrodes
    for e in ['A1', 'B2', 'C3']:
        npt.assert_equal(alpha[e].r, 15)

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        AlphaAMS(x=-100, y=10, z=np.arange(12))

    # Indexing must work for both integers and electrode names
    alpha = AlphaAMS()
    # enumerate returns ((0, alpha.items()[0]), (1, alpha.items()[1]), ...)
    # idx = 0, ... 36. name = A1, ... e37. electrode = DiskElectrode(...)
    for idx, (name, electrode) in enumerate(alpha.electrodes.items()):
        npt.assert_equal(electrode, alpha[idx])
        npt.assert_equal(electrode, alpha[name])
        npt.assert_equal(alpha["unlikely name for an electrode"], None)

    # Right-eye implant:
    xc, yc = 1600, -1600
    alpha_re = AlphaAMS(eye='RE', x=xc, y=yc)
    npt.assert_equal(alpha_re['A40'].x > alpha_re['A1'].x, True)
    npt.assert_almost_equal(alpha_re['A40'].y, alpha_re['A1'].y)

    # Left-eye implant:
    alpha_le = AlphaAMS(eye='LE', x=xc, y=yc)
    npt.assert_equal(alpha_le['A1'].x > alpha_le['AE40'].x, True)
    npt.assert_almost_equal(alpha_le['A40'].y, alpha_le['A1'].y)

    # In both left and right eyes, rotation with positive angle should be
    # counter-clock-wise (CCW): for (x>0,y>0), decreasing x and increasing y
    for eye, el in zip(['LE', 'RE'], ['A1', 'A40']):
        before = AlphaAMS(eye=eye)
        after = AlphaAMS(eye=eye, rot=10)
        npt.assert_equal(after[el].x > before[el].x, True)
        npt.assert_equal(after[el].y > before[el].y, True)

    # Invalid eye string:
    with pytest.raises(TypeError):
        AlphaIMS(eye=[1, 2])
    with pytest.raises(ValueError):
        AlphaIMS(eye='left eye')
