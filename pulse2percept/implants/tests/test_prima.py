import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, RegularPolygon

from pulse2percept.implants import (PhotovoltaicPixel, PRIMA, PRIMA75, PRIMA55,
                                    PRIMA40)
from pulse2percept.stimuli import LogoBVL
from pulse2percept.units import deg, mm
from pulse2percept.utils.constants import ZORDER
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

    # Every pixel body sits on the 1 mm substrate, corners included. Flat-top
    # hexagons put a vertex every 60 deg at a circumradius of `width`/sqrt(3):
    corner = np.radians(np.arange(6) * 60 + rot)
    verts = ((xy - [x, y])[:, np.newaxis, :] + spacing / np.sqrt(3) *
             np.column_stack([np.cos(corner), np.sin(corner)]))
    npt.assert_array_less(np.hypot(verts[..., 0], verts[..., 1]), 500)

    with pytest.raises(ValueError):
        implant_type(0, 0, z=np.ones(16))


@pytest.mark.parametrize('implant_type', (PRIMA55, PRIMA40))
def test_PRIMA_Ho2019_units(implant_type):
    """Unitful placement must trim the array exactly like bare microns

    The trimming works off the array's coordinates, so it has to normalize
    ``x``/``y`` and read the rotation back off the grid rather than trusting
    whatever the caller spelled them as.
    """
    bare = implant_type(x=1000, y=-500, z=-100, rot=30)
    unitful = implant_type(x=1 * mm, y=-0.5 * mm, z=-0.1 * mm, rot=30 * deg)
    npt.assert_equal(list(unitful.earray.electrodes),
                     list(bare.earray.electrodes))
    npt.assert_allclose(unitful.earray.coordinates(),
                        bare.earray.coordinates(), atol=1e-9)


def test_PRIMA55_layout():
    """The 250-pixel F55 mask of [Ho2019]_, in axial hex coordinates

    Pins the published outline itself rather than electrode names, which
    depend on the size of the grid the mask is cut from.
    """
    expected = {-9: (3, 8), -8: (1, 9), -7: (-1, 9), -6: (-3, 9), -5: (-4, 9),
                -4: (-5, 9), -3: (-6, 9), -2: (-6, 8), -1: (-7, 8),
                0: (-7, 7), 1: (-8, 7), 2: (-8, 6), 3: (-9, 6), 4: (-9, 5),
                5: (-9, 4), 6: (-9, 3), 7: (-9, 2), 8: (-9, 1), 9: (-8, -1)}
    npt.assert_equal(sum(hi - lo + 1 for lo, hi in expected.values()), 250)

    prima = PRIMA55()
    s = prima.spacing
    xy = prima.earray.coordinates()[:, :2]
    # Flat-top axial coordinates, read back off the pixel centers:
    q = xy[:, 0] / (s * np.sqrt(3) / 2)
    r = xy[:, 1] / s - q / 2
    npt.assert_allclose(q, np.round(q), atol=1e-9)
    npt.assert_allclose(r, np.round(r), atol=1e-9)
    q, r = np.round(q).astype(int), np.round(r).astype(int)

    npt.assert_equal(sorted(set(q)), sorted(expected))
    for col in expected:
        rows = np.sort(r[q == col])
        # Every column is a solid run of pixels between its two limits:
        npt.assert_equal(rows, np.arange(rows[0], rows[-1] + 1))
        npt.assert_equal((rows[0], rows[-1]), expected[col])


def _substrate(implant, **kwargs):
    """Plot an implant on a fresh axis and return its substrate patch

    Closes the figure before returning: the axes stays readable, and a stray
    open figure would otherwise become the `plt.gca()` that the next test
    draws onto.
    """
    fig, ax = plt.subplots()
    implant.plot(ax=ax, **kwargs)
    patches = [p for p in ax.patches if isinstance(p, (Circle, Polygon))]
    npt.assert_equal(len(patches), 1)
    plt.close(fig)
    return ax, patches[0]


@pytest.mark.parametrize('implant_type, radius', [
    (PRIMA75, 500), (PRIMA55, 500), (PRIMA40, 500),
])
@pytest.mark.parametrize('rot', (0, 30))
def test_PRIMA_round_substrate(implant_type, radius, rot):
    """The round devices sit on a 1 mm circular die centered on (x, y)

    PRIMA40 keeps the lattice sites nearest the substrate center rather than
    a footprint centered on them, so the substrate must come from the
    requested position and not from where the pixels ended up.
    """
    x, y = -100, 400
    ax, patch = _substrate(implant_type(x=x, y=y, rot=rot))
    npt.assert_almost_equal(patch.center, (x, y))
    npt.assert_almost_equal(patch.radius, radius)
    # Behind the pixels, whatever order they were added in:
    npt.assert_array_less(patch.get_zorder(), ZORDER['foreground'])
    # ...and inside the view, so `autoscale` shows the chip and not just the
    # pixels:
    npt.assert_array_less(ax.get_xlim()[0], x - radius)
    npt.assert_array_less(x + radius, ax.get_xlim()[1])
    npt.assert_array_less(ax.get_ylim()[0], y - radius)
    npt.assert_array_less(y + radius, ax.get_ylim()[1])


@pytest.mark.parametrize('rot', (0, 30, -45))
def test_PRIMA_square_substrate(rot):
    """Clinical PRIMA sits on a 2 x 2 mm die that turns with the implant"""
    x, y = -100, 400
    ax, patch = _substrate(PRIMA(x=x, y=y, rot=rot))
    corners = patch.get_xy()[:4]
    npt.assert_almost_equal(corners.mean(axis=0), (x, y))
    edges = np.roll(corners, -1, axis=0) - corners
    npt.assert_almost_equal(np.linalg.norm(edges, axis=1), 2000)
    # Square, and turned by `rot`:
    npt.assert_almost_equal(np.abs(np.sum(edges[0] * edges[1])), 0)
    npt.assert_almost_equal(
        np.mod(np.degrees(np.arctan2(edges[0, 1], edges[0, 0])) - rot, 90), 0)
    npt.assert_array_less(patch.get_zorder(), ZORDER['foreground'])
    # The whole die is in view, including the corner the rotation swings out:
    npt.assert_array_less(ax.get_xlim()[0], corners[:, 0].min())
    npt.assert_array_less(corners[:, 0].max(), ax.get_xlim()[1])
    npt.assert_array_less(ax.get_ylim()[0], corners[:, 1].min())
    npt.assert_array_less(corners[:, 1].max(), ax.get_ylim()[1])


@pytest.mark.parametrize('implant_type', (PRIMA, PRIMA75, PRIMA55, PRIMA40))
def test_PRIMA_substrate_holds_pixels(implant_type):
    """Every pixel body is drawn on the substrate, corners included"""
    implant = implant_type()
    xy = implant.earray.coordinates()[:, :2]
    th = np.radians(np.arange(6) * 60)
    verts = (xy[:, np.newaxis, :] + implant.pixel_width / np.sqrt(3) *
             np.column_stack([np.cos(th), np.sin(th)])).reshape(-1, 2)
    if implant_type is PRIMA:
        npt.assert_array_less(np.abs(verts), 1000)
    elif implant_type is PRIMA75:
        # The corner pixels of this hand-trimmed layout overhang a nominal
        # 500 um radius by ~22 um. [Lorach2015]_ gives the substrate only as
        # ~1 mm, so the overhang is drawn rather than trimmed away.
        npt.assert_array_less(np.hypot(*verts.T), 525)
    else:
        npt.assert_array_less(np.hypot(*verts.T), 500)


@pytest.mark.parametrize('implant_type', (PRIMA, PRIMA75, PRIMA55, PRIMA40))
def test_PRIMA_plot_passthrough(implant_type):
    """The substrate override keeps the rest of `plot` working"""
    implant = implant_type()
    fig, ax = plt.subplots()
    # `stim_cmap` is not exercised here: colouring a PhotovoltaicPixel is
    # broken upstream of this override, since it draws two patches and
    # `ElectrodeArray.plot` colours only single-patch electrodes.
    implant.plot(ax=ax, annotate=True)
    npt.assert_equal(len(ax.texts), implant.n_electrodes)
    npt.assert_equal(len(ax.collections), 1)
    plt.close(fig)
    # A unitful position places the substrate the same way a bare one does:
    _, bare = _substrate(implant_type(x=1000, y=-500))
    _, unitful = _substrate(implant_type(x=1 * mm, y=-0.5 * mm))
    if isinstance(bare, Circle):
        npt.assert_almost_equal(unitful.center, bare.center)
    else:
        npt.assert_almost_equal(unitful.get_xy(), bare.get_xy())


def test_PRIMA40_reshape_stim():
    # Smoke test a high-res hex implant with an ImageStimulus, where the
    # old approach runs out of memory easily. A picture is not a stimulus an
    # implant can deliver, so the sampling is exercised where an encoder
    # reaches it:
    PRIMA40().reshape_stim(LogoBVL())
    

@pytest.mark.parametrize('implant_type, offset', [
    (PRIMA, (0, 0)),
    (PRIMA75, (0, 0)),
    # PRIMA55's reconstructed mask is centered on the substrate, since where
    # it sits on the die is not published. PRIMA40 instead keeps the 502
    # lattice sites nearest the substrate center, and a discrete lattice
    # leaves that footprint a quarter of a spacing off center:
    (PRIMA55, (0, 0)),
    (PRIMA40, (0, -0.25 * 40)),
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
