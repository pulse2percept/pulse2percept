import numpy as np
import numpy.testing as npt
import pytest

from .. import implants


def test_Electrode():
    num_pts = 10
    r = np.linspace(1, 1000, num_pts)
    x = np.linspace(-1000, 1000, num_pts)
    y = np.linspace(-2000, 2000, num_pts)
    h = np.linspace(0, 1000, num_pts)
    t = ['subretinal', 'epiretinal'] * (num_pts // 2)
    n = ["some name"] * num_pts

    for rr, xx, yy, hh, tt, nn in zip(r, x, y, h, t, n):
        e = implants.Electrode(tt, rr, xx, yy, hh, nn)
        npt.assert_equal(e.radius, rr)
        npt.assert_equal(e.x_center, xx)
        npt.assert_equal(e.y_center, yy)
        # npt.assert_equal(e.h, hh)
        npt.assert_equal(e.etype, tt)
        npt.assert_equal(e.name, nn)
        if tt.lower() == 'epiretinal':
            # `height` property should return `h_ofl`
            npt.assert_equal(e.height, e.h_ofl)

            # `h_ofl` should be the same as the user-specified height
            npt.assert_equal(e.height, hh)

            # `h_inl` should be further away from the array
            npt.assert_equal(e.h_inl > hh, True)
        else:
            # `height` property should return `h_inl`
            npt.assert_equal(e.height, e.h_inl)

            # Subretinal arrays have layer thicknesses added to `hh`.
            npt.assert_equal(e.height > hh, True)

    # Invalid type
    with pytest.raises(ValueError):
        implants.Electrode('suprachoroidal', 10, 0, 0, 0)

    # Invalid layer
    e = implants.Electrode('epiretinal', 10, 0, 0, 0)
    with pytest.raises(ValueError):
        e.current_spread(0, 0, 'RGC')

    # Invalid type lookup
    e.etype = 'suprachoroidal'
    with pytest.raises(ValueError):
        e.height


def test_ElectrodeArray():
    implant = implants.ElectrodeArray('subretinal', 10, 0, 0)
    npt.assert_equal(implant.num_electrodes, 1)

    # Make sure ElectrodeArray can accept ints, floats, lists, np.arrays
    arrays = [None] * 4
    arrays[0] = implants.ElectrodeArray('epiretinal', [0], [1], [2],
                                        hs=[3])
    arrays[1] = implants.ElectrodeArray('epiretinal', 0, 1, 2, hs=3)
    arrays[2] = implants.ElectrodeArray('epiretinal', .0, [1], 2.0,
                                        hs=[3])
    arrays[3] = implants.ElectrodeArray('epiretinal', np.array([0]), [1],
                                        [2], hs=[[3]])

    for arr in arrays:
        npt.assert_equal(arr.num_electrodes, 1)
        npt.assert_equal(arr.electrodes[0].radius, 0)
        npt.assert_equal(arr.electrodes[0].x_center, 1)
        npt.assert_equal(arr.electrodes[0].y_center, 2)
        npt.assert_equal(arr.electrodes[0].h_ofl, 3)
        npt.assert_equal(arr.electrodes[0].etype, 'epiretinal')

    # Make sure electrodes can be addressed by index
    vals = range(5)
    implant = implants.ElectrodeArray('subretinal', vals, vals, vals,
                                      hs=vals)
    npt.assert_equal(implant.num_electrodes, len(vals))
    for v in vals:
        el = implant[v]
        npt.assert_equal(el.radius, v)
        npt.assert_equal(el.x_center, v)
        npt.assert_equal(el.y_center, v)
        npt.assert_equal(el.h_inl, v + 23.0 / 2.0)
        npt.assert_equal(el.h_ofl, v + 83.0)

    # Test left/right eye
    for validstr in ['left', 'LEFT', 'l', 'LE']:
        implant = implants.ElectrodeArray('epiretinal', 10, 0, 0,
                                          eye=validstr)
        npt.assert_equal(implant.eye, 'LE')
    for validstr in ['right', 'Right', 'r', 'RE']:
        implant = implants.ElectrodeArray('epiretinal', 10, 0, 0,
                                          eye=validstr)
        npt.assert_equal(implant.eye, 'RE')
    for invalidstr in ['both', 'lefteye', 'invalid']:
        with pytest.raises(ValueError):
            implant = implants.ElectrodeArray('epiretinal', 10, 0, 0,
                                              eye=invalidstr)


def test_ElectrodeArray_add_electrode():
    implant = implants.ElectrodeArray('epiretinal', 10, 0, 0)
    with pytest.raises(TypeError):
        implant.add_electrode(implant)

    with pytest.raises(ValueError):
        implant.add_electrode(implants.Electrode('subretinal', 10, 0, 0))

    # Make sure electrode count is correct
    for j in range(5):
        implant.add_electrode(implants.Electrode('epiretinal', 10, 10, 10))
        npt.assert_equal(implant.num_electrodes, j + 2)


def test_ElectrodeArray_add_electrodes():
    for j in range(5):
        implant = implants.ElectrodeArray('epiretinal', 10, 0, 0)
        implant.add_electrodes(range(1, j + 1), range(j), range(j))
        npt.assert_equal(implant.num_electrodes, j + 1)

    # However, all input arguments must have the same number of elements
    with pytest.raises(AssertionError):
        implant.add_electrodes([0], [1, 2], [3, 4, 5], [6])


def test_ArgusI():
    # Create an ArgusI and make sure location is correct
    for htype in ['float', 'list']:
        for x in [0, -100, 200]:
            for y in [0, -200, 400]:
                for r in [0, -30, 45, 60, -90]:
                    # Height `h` can either be a float or a list
                    if htype == 'float':
                        h = 100
                    else:
                        h = np.ones(16) * 20

                    # Convert rotation angle to rad
                    rot = r * np.pi / 180
                    argus = implants.ArgusI(x, y, h=h, rot=rot)

                    # Coordinates of first electrode
                    xy = np.array([-1200, -1200]).T

                    # Rotate
                    R = np.array([np.cos(rot), -np.sin(rot),
                                  np.sin(rot), np.cos(rot)]).reshape((2, 2))
                    xy = np.matmul(R, xy)

                    # Then off-set: Make sure first electrode is placed
                    # correctly
                    npt.assert_almost_equal(argus['A1'].x_center,
                                            xy[0] + x)
                    npt.assert_almost_equal(argus['A1'].y_center,
                                            xy[1] + y)

                    # Make sure array center is still (x,y)
                    y_center = argus['D1'].y_center + \
                        (argus['A4'].y_center - argus['D1'].y_center) / 2
                    npt.assert_almost_equal(y_center, y)
                    x_center = argus['A1'].x_center + \
                        (argus['D4'].x_center - argus['A1'].x_center) / 2
                    npt.assert_almost_equal(x_center, x)

                    # Check radii of electrodes
                    for e in ['A1', 'A3', 'B2', 'C1', 'D4']:
                        npt.assert_almost_equal(argus[e].radius, 130)
                    for e in ['A2', 'A4', 'B1', 'C2', 'D3']:
                        npt.assert_almost_equal(argus[e].radius, 260)

                    # Check location of the tack
                    tack = np.matmul(R, [-2000, 0])
                    tack = tuple(tack + [x_center, y_center])

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        implants.ArgusI(-100, 10, h=np.zeros(5))
    with pytest.raises(ValueError):
        implants.ArgusI(-100, 10, h=[1, 2, 3])

    for use_legacy_names in [False, True]:
        # Indexing must work for both integers and electrode names
        argus = implants.ArgusI(use_legacy_names=use_legacy_names)
        for idx, electrode in enumerate(argus):
            name = electrode.name
            npt.assert_equal(electrode, argus[idx])
            npt.assert_equal(electrode, argus[name])
        npt.assert_equal(argus[16], None)
        npt.assert_equal(argus["unlikely name for an electrode"], None)

        if use_legacy_names:
            name_idx1 = 'L2'
            name_idx4 = 'L5'
        else:
            name_idx1 = 'B1'
            name_idx4 = 'A2'

        # Indexing must have the right order
        npt.assert_equal(argus.get_index(name_idx1), 1)
        npt.assert_equal(argus[name_idx1], argus[1])
        npt.assert_equal(argus.get_index(name_idx4), 4)
        npt.assert_equal(argus[name_idx4], argus[4])

    # Right-eye implant:
    xc, yc = 500, -500
    argus_re = implants.ArgusI(eye='RE', x_center=xc, y_center=yc)
    npt.assert_equal(argus_re['D1'].x_center > argus_re['A1'].x_center, True)
    npt.assert_almost_equal(argus_re['D1'].y_center, argus_re['A1'].y_center)
    npt.assert_equal(argus_re.tack[0] < argus_re['D1'].x_center, True)
    npt.assert_almost_equal(argus_re.tack[1], yc)

    # Left-eye implant:
    argus_le = implants.ArgusI(eye='LE', x_center=xc, y_center=yc)
    npt.assert_equal(argus_le['A1'].x_center > argus_le['D1'].x_center, True)
    npt.assert_almost_equal(argus_le['D1'].y_center, argus_le['A1'].y_center)
    npt.assert_equal(argus_le.tack[0] > argus_le['A1'].x_center, True)
    npt.assert_almost_equal(argus_le.tack[1], yc)

    # In both left and right eyes, rotation with positive angle should be
    # counter-clock-wise (CCW): for (x>0,y>0), decreasing x and increasing y
    for eye, el in zip(['LE', 'RE'], ['A1', 'D4']):
        before = implants.ArgusI(eye=eye)
        after = implants.ArgusI(eye=eye, rot=np.deg2rad(10))
        npt.assert_equal(after[el].x_center < before[el].x_center, True)
        npt.assert_equal(after[el].y_center > before[el].y_center, True)

    argus = implants.ArgusI()
    # Old to new
    npt.assert_equal(argus.get_new_name('M1'), 'D4')
    npt.assert_equal(argus.get_new_name('M6'), 'C3')
    # New to old
    npt.assert_equal(argus.get_old_name('B2'), 'L1')
    npt.assert_equal(argus.get_old_name('A1'), 'L6')


def test_ArgusII():
    # Create an ArgusII and make sure location is correct
    for htype in ['float', 'list']:
        for x in [0, -100, 200]:
            for y in [0, -200, 400]:
                for r in [0, -30, 45, 60, -90]:
                    # Height `h` can either be a float or a list
                    if htype == 'float':
                        h = 100
                    else:
                        h = np.ones(60) * 20

                    # Convert rotation angle to rad
                    rot = np.deg2rad(r)
                    argus = implants.ArgusII(x, y, h=h, rot=rot)

                    # Coordinates of first electrode
                    xy = np.array([-2362.5, -1312.5]).T

                    # Rotate
                    R = np.array([np.cos(rot), -np.sin(rot),
                                  np.sin(rot), np.cos(rot)]).reshape((2, 2))
                    xy = np.matmul(R, xy)

                    # Then off-set: Make sure first electrode is placed
                    # correctly
                    npt.assert_almost_equal(argus['A1'].x_center,
                                            xy[0] + x)
                    npt.assert_almost_equal(argus['A1'].y_center,
                                            xy[1] + y)

                    # Make sure array center is still (x,y)
                    y_center = argus['F1'].y_center + \
                        (argus['A10'].y_center - argus['F1'].y_center) / 2
                    npt.assert_almost_equal(y_center, y)
                    x_center = argus['A1'].x_center + \
                        (argus['F10'].x_center - argus['A1'].x_center) / 2
                    npt.assert_almost_equal(x_center, x)

                    # Make sure radius is correct
                    for e in ['A1', 'B3', 'C5', 'D7', 'E9', 'F10']:
                        npt.assert_almost_equal(argus[e].radius, 100)

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        implants.ArgusII(-100, 10, h=np.zeros(5))
    with pytest.raises(ValueError):
        implants.ArgusII(-100, 100, h=[1, 2, 3])

    # Indexing must work for both integers and electrode names
    argus = implants.ArgusII()
    for idx, electrode in enumerate(argus):
        name = electrode.name
        npt.assert_equal(electrode, argus[idx])
        npt.assert_equal(electrode, argus[name])
    npt.assert_equal(argus[60], None)
    npt.assert_equal(argus["unlikely name for an electrode"], None)

    # Indexing must have the right order
    npt.assert_equal(argus.get_index('A2'), 1)
    npt.assert_equal(argus['A2'], argus[1])
    npt.assert_equal(argus.get_index('B1'), 10)
    npt.assert_equal(argus['B1'], argus[10])

    # Right-eye implant:
    xc, yc = 500, -500
    argus_re = implants.ArgusII(eye='RE', x_center=xc, y_center=yc)
    npt.assert_equal(argus_re['A10'].x_center > argus_re['A1'].x_center, True)
    npt.assert_almost_equal(argus_re['A10'].y_center, argus_re['A1'].y_center)
    npt.assert_equal(argus_re.tack[0] < argus_re['A1'].x_center, True)
    npt.assert_almost_equal(argus_re.tack[1], yc)

    # Left-eye implant:
    argus_le = implants.ArgusII(eye='LE', x_center=xc, y_center=yc)
    npt.assert_equal(argus_le['A1'].x_center > argus_le['A10'].x_center, True)
    npt.assert_almost_equal(argus_le['A10'].y_center, argus_le['A1'].y_center)
    npt.assert_equal(argus_le.tack[0] > argus_le['A10'].x_center, True)
    npt.assert_almost_equal(argus_le.tack[1], yc)

    # In both left and right eyes, rotation with positive angle should be
    # counter-clock-wise (CCW): for (x>0,y>0), decreasing x and increasing y
    for eye, el in zip(['LE', 'RE'], ['F1', 'F10']):
        before = implants.ArgusII(eye=eye)
        after = implants.ArgusII(eye=eye, rot=np.deg2rad(10))
        npt.assert_equal(after[el].x_center < before[el].x_center, True)
        npt.assert_equal(after[el].y_center > before[el].y_center, True)


def test_Electrode_receptive_field():
    electrode = implants.Electrode('epiretinal', 100, 0, 0, 0)

    with pytest.raises(ValueError):
        electrode.receptive_field(0, 0, rftype='invalid')
