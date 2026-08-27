import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import numpy.testing as npt
from matplotlib.patches import Circle, RegularPolygon

from pulse2percept.implants import (PhotovoltaicPixel, PRIMA, PRIMA75, PRIMA55,
                                    PRIMA40)
from pulse2percept.stimuli import LogoBVL
from pulse2percept.models import ScoreboardModel

def test_PhotovoltaicPixel():
    electrode = PhotovoltaicPixel(0, 1, 2, 3, 4)
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_almost_equal(electrode.r, 3)
    npt.assert_almost_equal(electrode.a, 4)
    # Slots:
    npt.assert_equal(hasattr(electrode, '__slots__'), True)
    npt.assert_equal(hasattr(electrode, '__dict__'), False)
    # Plots:
    ax = electrode.plot()
    npt.assert_equal(len(ax.texts), 0)
    npt.assert_equal(len(ax.patches), 2)
    npt.assert_equal(isinstance(ax.patches[0], RegularPolygon), True)
    npt.assert_equal(isinstance(ax.patches[1], Circle), True)
    PhotovoltaicPixel(0, 1, 2, 3, 4)


@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_PRIMA(ztype, x, y, rot):
    # 85 um pixel with 15 um trenches:
    spacing = 100
    # Roughly a 12x15 grid, but edges are trimmed off:
    n_elec = 378
    # Create an Prima and make sure location is correct
    # Height `z` can either be a float or a list
    z = -100 if ztype == 'float' else -np.ones(378) * 20

    prima = PRIMA(x, y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(prima, '__slots__'), True)
    npt.assert_equal(hasattr(prima, '__dict__'), False)

    # Make sure number of electrodes is correct
    npt.assert_equal(prima.n_electrodes, n_elec)
    npt.assert_equal(len(prima.earray.electrodes), n_elec)

    # Coordinates of A6 when device is not rotated:
    xy = np.array([-476.31, -925.0]).T
    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)
    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(prima['A6'].x, xy[0] + x, decimal=2)
    npt.assert_almost_equal(prima['A6'].y, xy[1] + y, decimal=2)

    # Make sure the radius is correct
    for e in ['A7', 'B3', 'C5', 'D7', 'E9', 'F11', 'G13', 'H14']:
        npt.assert_almost_equal(prima[e].r, 14)

    # Make sure the pitch is correct:
    distF6E6 = np.sqrt((prima['E6'].x - prima['F6'].x) ** 2 +
                       (prima['E6'].y - prima['F6'].y) ** 2)
    npt.assert_almost_equal(distF6E6, spacing)
    distF6E7 = np.sqrt((prima['E7'].x - prima['F6'].x) ** 2 +
                       (prima['E7'].y - prima['F6'].y) ** 2)
    npt.assert_almost_equal(distF6E7, spacing)

    with pytest.raises(ValueError):
        PRIMA(0, 0, z=np.ones(16))


@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_PRIMA75(ztype, x, y, rot):
    # 70 um pixel with 5 um trenches:
    spacing = 75
    # Roughly a 12x15 grid, but edges are trimmed off:
    n_elec = 142
    # Create an Prima and make sure location is correct
    # Height `z` can either be a float or a list
    z = -100 if ztype == 'float' else -np.ones(142) * 20

    prima = PRIMA75(x, y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(prima, '__slots__'), True)
    npt.assert_equal(hasattr(prima, '__dict__'), False)

    # Make sure number of electrodes is correct
    npt.assert_equal(len(prima.earray.electrodes), n_elec)
    npt.assert_equal(prima.n_electrodes, n_elec)

    # Coordinates of A6 when device is not rotated:
    xy = np.array([-129.90, -431.25]).T
    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)
    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(prima['A6'].x, xy[0] + x, decimal=2)
    npt.assert_almost_equal(prima['A6'].y, xy[1] + y, decimal=2)

    # Make sure the radius is correct
    for e in ['A6', 'B4', 'C5', 'D7', 'E9', 'F11', 'G13', 'H14']:
        npt.assert_almost_equal(prima[e].r, 10)

    # Make sure the pitch is correct:
    distF6E6 = np.sqrt((prima['E6'].x - prima['F6'].x) ** 2 +
                       (prima['E6'].y - prima['F6'].y) ** 2)
    npt.assert_almost_equal(distF6E6, spacing)
    distF6E7 = np.sqrt((prima['E7'].x - prima['F6'].x) ** 2 +
                       (prima['E7'].y - prima['F6'].y) ** 2)
    npt.assert_almost_equal(distF6E7, spacing)

    with pytest.raises(ValueError):
        PRIMA75(0, 0, z=np.ones(16))


@pytest.mark.parametrize('implant_type, spacing, n_elec, elec_radius', [
    (PRIMA55, 55, 250, 7),
    (PRIMA40, 40, 502, 5),
])
@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_PRIMA_Ho2019(implant_type, spacing, n_elec, elec_radius, ztype, x, y,
                      rot):
    """The F55/F40 arrays of Ho et al. (2019)

    Pixel bodies tile the lattice with no open gap, so pixel width equals the
    nearest-neighbor center spacing, and the published pixel count fits on the
    1 mm circular substrate.
    """
    # Height `z` can either be a float or a list:
    z = -100 if ztype == 'float' else -np.ones(n_elec) * 20
    prima = implant_type(x, y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(prima, '__slots__'), True)
    npt.assert_equal(hasattr(prima, '__dict__'), False)

    # The published pixel count:
    npt.assert_equal(prima.n_electrodes, n_elec)
    npt.assert_equal(len(prima.earray.electrodes), n_elec)

    xy = prima.earray.coordinates()[:, :2]
    # Nearest-neighbor center spacing, in every direction:
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    npt.assert_almost_equal(dist.min(), spacing)
    npt.assert_almost_equal(dist.min(axis=1), spacing)
    # Row spacing is derived, not independent:
    npt.assert_almost_equal(prima.row_spacing, spacing * np.sqrt(3) / 2)

    for elec in prima.earray.electrode_objects:
        # Pixel bodies are as wide as the lattice, with no open gap:
        npt.assert_almost_equal(elec.width, spacing)
        npt.assert_almost_equal(prima.pixel_width, spacing)
        npt.assert_almost_equal(prima.gap, 0)
        # Active electrode:
        npt.assert_almost_equal(elec.r, elec_radius)
        # Hex bodies turn with the lattice:
        npt.assert_almost_equal(elec.rot, rot)
        npt.assert_equal(elec.orientation, 'vertical')

    # The whole array fits on the 1 mm substrate, pixel corners included:
    corner = np.hypot(*(xy - [x, y]).T) + spacing / np.sqrt(3)
    npt.assert_array_less(corner, 500)

    with pytest.raises(ValueError):
        implant_type(0, 0, z=np.ones(16))


def test_PRIMA40_reshape_stim():
    # Smoke test a high-res hex implant with an ImageStimulus, where the
    # old approach runs out of memory easily. A picture is not a stimulus an
    # implant can deliver, so the sampling is exercised where an encoder
    # reaches it:
    PRIMA40().reshape_stim(LogoBVL())
    

@pytest.mark.parametrize('implant_type, offset', [
    (PRIMA, (0, 0)),
    (PRIMA75, (0, 0)),
    # PRIMA55 and PRIMA40 keep the pixels nearest the center of the substrate,
    # taking whole antipodal pairs, so their footprints are centered on (x, y)
    # and symmetric under a half turn:
    (PRIMA55, (0, 0)),
    (PRIMA40, (0, 0)),
])
def test_PRIMA_device_center(implant_type, offset):
    """Where the trimmed device sits relative to the requested (x, y)

    Each PRIMA is a regular hex grid with edge electrodes removed afterwards,
    so the finished device is centered only if those removals are symmetric --
    the grid's own centering says nothing about it. The per-electrode
    coordinate tests would all still pass if a device drifted sideways.
    """
    x, y, rot = -100, 400, 37
    xy = implant_type(x=x, y=y).earray.coordinates()[:, :2]
    center = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    npt.assert_almost_equal(center, np.add([x, y], offset))
    # `rot` turns the whole footprint about (x, y), so the offset above is a
    # property of the device rather than of the coordinate axes:
    th = np.deg2rad(rot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rotated = implant_type(x=x, y=y, rot=rot).earray.coordinates()[:, :2]
    npt.assert_almost_equal(rotated, (R @ (xy - [x, y]).T).T + [x, y])
