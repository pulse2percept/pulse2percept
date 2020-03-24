import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept import implants


@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('r', (-45, 60))
def test_ArgusI(ztype, x, y, r):
    # Create an ArgusI and make sure location is correct
    # Height `z` can either be a float or a list
    z = 100 if ztype == 'float' else np.ones(16) * 20
    # Convert rotation angle to rad
    rot = r * np.pi / 180

    argus = implants.ArgusI(x, y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(argus, '__slots__'), True)
    npt.assert_equal(hasattr(argus, '__dict__'), False)

    # Coordinates of first electrode
    xy = np.array([-1200, -1200]).T

    # Rotate
    R = np.array([np.cos(rot), -np.sin(rot),
                  np.sin(rot), np.cos(rot)]).reshape((2, 2))
    xy = np.matmul(R, xy)

    # Then off-set: Make sure first electrode is placed
    # correctly
    print("argus x set to", argus['A1'].x)
    print("artifical array set to", xy[0] + x)
    npt.assert_almost_equal(argus['A1'].x, xy[0] + x)
    npt.assert_almost_equal(argus['A1'].y, xy[1] + y)

    # Make sure array center is still (x,y)
    y_center = argus['D1'].y + (argus['A4'].y - argus['D1'].y) / 2
    npt.assert_almost_equal(y_center, y)
    x_center = argus['A1'].x + (argus['D4'].x - argus['A1'].x) / 2
    npt.assert_almost_equal(x_center, x)

    # Check radii of electrodes
    for e in ['A1', 'A3', 'B2', 'C1', 'D4']:
        npt.assert_almost_equal(argus[e].r, 130)
    for e in ['A2', 'A4', 'B1', 'C2', 'D3']:
        npt.assert_almost_equal(argus[e].r, 260)

    # Check location of the tack
    tack = np.matmul(R, [-2000, 0])
    tack = tuple(tack + [x_center, y_center])

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        implants.ArgusI(x=-100, y=10, z=np.zeros(5))
    with pytest.raises(ValueError):
        implants.ArgusI(x=-100, y=10, z=[1, 2, 3])

    # Indexing must work for both integers and electrode names
    for use_legacy_names in [True, False]:
        argus = implants.ArgusI(use_legacy_names=use_legacy_names)
        for idx, (name, electrode) in enumerate(argus.items()):
            npt.assert_equal(electrode, argus[idx])
            npt.assert_equal(electrode, argus[name])
        npt.assert_equal(argus["unlikely name for an electrode"], None)

    # Right-eye implant:
    xc, yc = 500, -500
    argus_re = implants.ArgusI(eye='RE', x=xc, y=yc)
    npt.assert_equal(argus_re['D1'].x > argus_re['A1'].x, True)
    npt.assert_almost_equal(argus_re['D1'].y, argus_re['A1'].y)

    # need to adjust for reflection about y-axis
    # Left-eye implant:
    argus_le = implants.ArgusI(eye='LE', x=xc, y=yc)
    npt.assert_equal(argus_le['A1'].x > argus_le['D4'].x, True)
    npt.assert_almost_equal(argus_le['D1'].y, argus_le['A1'].y)

    # In both left and right eyes, rotation with positive angle should be
    # counter-clock-wise (CCW): for (x>0,y>0), decreasing x and increasing y
    for eye, el in zip(['LE', 'RE'], ['A1', 'D1']):
        before = implants.ArgusI(eye=eye)
        after = implants.ArgusI(eye=eye, rot=np.deg2rad(10))
        npt.assert_equal(after[el].x > before[el].x, True)
        npt.assert_equal(after[el].y > before[el].y, True)

    # Check naming scheme
    argus = implants.ArgusI(use_legacy_names=False)
    npt.assert_equal(list(argus.keys())[15], 'D4')
    npt.assert_equal(list(argus.keys())[0], 'A1')

    argus = implants.ArgusI(use_legacy_names=True)
    npt.assert_equal(list(argus.keys())[15], 'M1')
    npt.assert_equal(list(argus.keys())[0], 'L6')

    # Set a stimulus via dict:
    argus = implants.ArgusI(stim={'B3': 13})
    npt.assert_equal(argus.stim.shape, (1, 1))
    npt.assert_equal(argus.stim.electrodes, ['B3'])

    # Set a stimulus via array:
    argus = implants.ArgusI(stim=np.ones(16))
    npt.assert_equal(argus.stim.shape, (16, 1))
    npt.assert_almost_equal(argus.stim.data, 1)


@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('r', (-45, 60))
def test_ArgusII(ztype, x, y, r):
    # Create an ArgusII and make sure location is correct
    # Height `h` can either be a float or a list
    z = 100 if ztype == 'float' else np.ones(60) * 20
    # Convert rotation angle to rad
    rot = np.deg2rad(r)
    argus = implants.ArgusII(x=x, y=y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(argus, '__slots__'), True)
    npt.assert_equal(hasattr(argus, '__dict__'), False)

    # Coordinates of first electrode
    xy = np.array([-2362.5, -1312.5]).T

    # Rotate
    R = np.array([np.cos(rot), -np.sin(rot),
                  np.sin(rot), np.cos(rot)]).reshape((2, 2))
    xy = np.matmul(R, xy)

    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(argus['A1'].x, xy[0] + x)
    npt.assert_almost_equal(argus['A1'].y, xy[1] + y)

    # Make sure array center is still (x,y)
    y_center = argus['F1'].y + (argus['A10'].y - argus['F1'].y) / 2
    npt.assert_almost_equal(y_center, y)
    x_center = argus['A1'].x + (argus['F10'].x - argus['A1'].x) / 2
    npt.assert_almost_equal(x_center, x)

    # Make sure radius is correct
    for e in ['A1', 'B3', 'C5', 'D7', 'E9', 'F10']:
        npt.assert_almost_equal(argus[e].r, 100)

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        implants.ArgusII(x=-100, y=10, z=np.zeros(5))
    with pytest.raises(ValueError):
        implants.ArgusII(x=-100, y=100, z=[1, 2, 3])

    # Indexing must work for both integers and electrode names
    argus = implants.ArgusII()
    for idx, (name, electrode) in enumerate(argus.items()):
        npt.assert_equal(electrode, argus[idx])
        npt.assert_equal(electrode, argus[name])
    npt.assert_equal(argus["unlikely name for an electrode"], None)

    # Right-eye implant:
    xc, yc = 500, -500
    argus_re = implants.ArgusII(eye='RE', x=xc, y=yc)
    npt.assert_equal(argus_re['A10'].x > argus_re['A1'].x, True)
    npt.assert_almost_equal(argus_re['A10'].y, argus_re['A1'].y)

    # Left-eye implant:
    argus_le = implants.ArgusII(eye='LE', x=xc, y=yc)
    npt.assert_equal(argus_le['A1'].x > argus_le['A10'].x, True)
    npt.assert_almost_equal(argus_le['A10'].y, argus_le['A1'].y)

    # In both left and right eyes, rotation with positive angle should be
    # counter-clock-wise (CCW): for (x>0,y>0), decreasing x and increasing y
    for eye, el in zip(['LE', 'RE'], ['F2', 'F10']):
        # By default, electrode 'F1' in a left eye has the same coordinates as
        # 'F10' in a right eye (because the columns are reversed). Thus both
        # cases are testing an electrode with x>0, y>0:
        before = implants.ArgusII(eye=eye)
        after = implants.ArgusII(eye=eye, rot=np.deg2rad(20))
        print(after[el].x, before[el].x)
        print(after[el].y, before[el].y)
        npt.assert_equal(after[el].x < before[el].x, True)
        npt.assert_equal(after[el].y > before[el].y, True)

    # Set a stimulus via dict:
    argus = implants.ArgusII(stim={'B7': 13})
    npt.assert_equal(argus.stim.shape, (1, 1))
    npt.assert_equal(argus.stim.electrodes, ['B7'])

    # Set a stimulus via array:
    argus = implants.ArgusII(stim=np.ones(60))
    npt.assert_equal(argus.stim.shape, (60, 1))
    npt.assert_almost_equal(argus.stim.data, 1)
