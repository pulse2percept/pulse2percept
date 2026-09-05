import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants import AlphaIMS, AlphaAMS


@pytest.mark.parametrize('ztype', ('float', 'list'))
def test_AlphaIMS(ztype):
    # Height `h` can either be a float or a list
    if ztype == 'float':
        alpha = AlphaIMS(z=-100)
        for e in alpha.electrode_objects:
            npt.assert_almost_equal(e.z, -100)
    else:
        alpha = AlphaIMS(z=np.arange(1500))
        for i, e in enumerate(alpha.electrode_objects):
            npt.assert_almost_equal(e.z, i)

    # Slots:
    npt.assert_equal(hasattr(alpha, '__slots__'), True)
    npt.assert_equal(hasattr(alpha, '__dict__'), False)

    # Coordinates of first electrode
    # 18.5 *spacing - spacing/2 for middle coordinate if (0,0) is upper-left
    # corner
    xy = np.array([-1368, -1368]).T
    npt.assert_almost_equal(alpha['A1'].x, xy[0])
    npt.assert_almost_equal(alpha['A1'].y, xy[1])

    # The array is centered on the device's own origin
    y_center = alpha['AM15'].y + (alpha['A25'].y - alpha['AM15'].y) / 2
    npt.assert_almost_equal(y_center, 0)
    x_center = alpha['A15'].x + (alpha['AM25'].x - alpha['A15'].x) / 2
    npt.assert_almost_equal(x_center, 0)

    # Check width of square electrodes
    for e in ['A1', 'B2', 'C3']:
        npt.assert_equal(alpha[e].side_length, 50)


# The checks below don't depend on ztype, so they live outside the
# parametrized test above.
def test_AlphaIMS_indexing():
    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        AlphaIMS(z=np.arange(28))

    # Indexing must work for both integers and electrode names
    alpha = AlphaIMS()
    # enumerate returns ((0, alpha.items()[0]), (1, alpha.items()[1]), ...)
    # idx = 0, ... 36. name = A1, ... e37. electrode = DiskElectrode(...)
    for idx, (name, electrode) in enumerate(alpha.electrodes.items()):
        npt.assert_equal(electrode, alpha[idx])
        npt.assert_equal(electrode, alpha[name])
        with pytest.raises(KeyError):
            alpha["unlikely name for an electrode"]


def test_AlphaIMS_eye():
    # Right-eye implant:
    alpha_re = AlphaIMS(eye='RE')
    npt.assert_equal(alpha_re['A37'].x > alpha_re['A1'].x, True)
    npt.assert_almost_equal(alpha_re['A37'].y, alpha_re['A1'].y)

    # Left-eye implant:
    alpha_le = AlphaIMS(eye='LE')
    npt.assert_equal(alpha_le['A1'].x > alpha_le['AE37'].x, True)
    npt.assert_almost_equal(alpha_le['A37'].y, alpha_le['A1'].y)

    # Invalid eye string:
    with pytest.raises(TypeError):
        AlphaIMS(eye=[1, 2])
    with pytest.raises(ValueError):
        AlphaIMS(eye='left eye')


@pytest.mark.parametrize('ztype', ('float', 'list'))
def test_AlphaAMS(ztype):
    # Height `h` can either be a float or a list
    if ztype == 'float':
        alpha = AlphaAMS(z=-100)
        for e in alpha.electrode_objects:
            npt.assert_almost_equal(e.z, -100)
    else:
        alpha = AlphaAMS(z=np.arange(1600))
        for i, e in enumerate(alpha.electrode_objects):
            npt.assert_almost_equal(e.z, i)

    # Slots:
    npt.assert_equal(hasattr(alpha, '__slots__'), True)
    npt.assert_equal(hasattr(alpha, '__dict__'), False)

    # Coordinates of first electrode, in the device's own frame
    xy = np.array([-1365, -1365]).T
    npt.assert_almost_equal(alpha['A1'].x, xy[0])
    npt.assert_almost_equal(alpha['A1'].y, xy[1])

    # The array is centered on the device's own origin
    y_center = alpha['AN1'].y + (alpha['A40'].y - alpha['AN1'].y) / 2
    npt.assert_almost_equal(y_center, 0)
    x_center = alpha['A1'].x + (alpha['AN40'].x - alpha['A1'].x) / 2
    npt.assert_almost_equal(x_center, 0)

    # Check radii of electrodes
    for e in ['A1', 'B2', 'C3']:
        npt.assert_equal(alpha[e].radius, 15)


# As above: independent of ztype, so run once rather than twice.
def test_AlphaAMS_indexing():
    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        AlphaAMS(z=np.arange(12))

    # Indexing must work for both integers and electrode names
    alpha = AlphaAMS()
    # enumerate returns ((0, alpha.items()[0]), (1, alpha.items()[1]), ...)
    # idx = 0, ... 36. name = A1, ... e37. electrode = DiskElectrode(...)
    for idx, (name, electrode) in enumerate(alpha.electrodes.items()):
        npt.assert_equal(electrode, alpha[idx])
        npt.assert_equal(electrode, alpha[name])
        with pytest.raises(KeyError):
            alpha["unlikely name for an electrode"]


def test_AlphaAMS_eye():
    # Right-eye implant:
    alpha_re = AlphaAMS(eye='RE')
    npt.assert_equal(alpha_re['A40'].x > alpha_re['A1'].x, True)
    npt.assert_almost_equal(alpha_re['A40'].y, alpha_re['A1'].y)

    # Left-eye implant:
    alpha_le = AlphaAMS(eye='LE')
    npt.assert_equal(alpha_le['A1'].x > alpha_le['AE40'].x, True)
    npt.assert_almost_equal(alpha_le['A40'].y, alpha_le['A1'].y)

    # Invalid eye string:
    with pytest.raises(TypeError):
        AlphaAMS(eye=[1, 2])
    with pytest.raises(ValueError):
        AlphaAMS(eye='left eye')
