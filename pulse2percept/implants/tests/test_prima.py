import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants import PRIMA

@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('r', (-45, 60))
def test_PRIMA(ztype, x, y, r):
    # Create an Prima and make sure location is correct
    # Height `z` can either be a float or a list
    z = 100 if ztype == 'float' else np.ones(378) * 20
    # Convert rotation angle to rad
    rot = r * np.pi / 180

    prima = PRIMA(x, y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(prima, '__slots__'), True)
    npt.assert_equal(hasattr(prima, '__dict__'), False)

    # Make sure number of electrodes is correct
    num_elec = 378
    npt.assert_equal(len(prima.earray.electrodes), num_elec)

    # Coordinates of the first electrode
    # Since the original first three electrodes are removed
    # x = (np.arange(cols) * np.sqrt(3) * spc / 2.0 - 
    #     (cols / 2.0 - 0.5) * spc)[3]
    # y = (np.arange(rows) * spc - (rows / 2.0 - 0.5) * spc - 
    #     (spc * 0.25)[0]
    xy = np.array([-592.64, 656.25]).T

    # Rotate
    R = np.array([np.cos(rot), -np.sin(rot),
                np.sin(rot), np.cos(rot)]).reshape((2, 2))
    xy = np.matmul(R, xy)

    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(prima['A1'].x, xy[0] + x, decimal=2)
    npt.assert_almost_equal(prima['A1'].y, xy[1] + y, decimal=2)

    # Make sure the radius is correct
    for e in ['A1', 'B3', 'C5', 'D7', 'E9', 'F11', 'G13', 'H14']:
        npt.assert_almost_equal(prima[e].r, 10)

