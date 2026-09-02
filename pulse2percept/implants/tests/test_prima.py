import matplotlib
matplotlib.use('Agg')
import hashlib
from functools import partial

import numpy as np
import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, RegularPolygon
from scipy.spatial import cKDTree

from pulse2percept.implants import (ArgusII, PhotovoltaicPixel, PRIMAPivotal,
                                    Lorach2015Array, Ho2019FlatArray,
                                    Huang2021Array, PointSource,
                                    Implant, PRIMA, PRIMA75,
                                    PRIMA55, PRIMA40)
from pulse2percept.stimuli import (BiphasicPulse, BiphasicPulseTrain,
                                   ImageStimulus, LogoBVL, PRIMAEncoder,
                                   Stimulus)
from pulse2percept.units import DimensionMismatchError, deg, mW, mm, um, xTh
from pulse2percept.utils.constants import ZORDER
from pulse2percept.models import ScoreboardModel

def test_PhotovoltaicPixel():
    electrode = PhotovoltaicPixel(0, 1, 2, 3, 4)
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_almost_equal(electrode.radius, 3)
    npt.assert_almost_equal(electrode.apothem, 4)
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
def test_PRIMAPivotal(ztype, x, y, rot):
    # 100 um pixel on a 100 um grid, so no open gap between pixel bodies:
    spacing = 100
    # Roughly a 12x15 grid, but edges are trimmed off:
    n_elec = 378
    # Create an Prima and make sure location is correct
    # Height `z` can either be a float or a list
    z = -100 if ztype == 'float' else -np.ones(378) * 20

    prima = PRIMAPivotal(x, y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(prima, '__slots__'), True)
    npt.assert_equal(hasattr(prima, '__dict__'), False)

    # Make sure number of electrodes is correct
    npt.assert_equal(prima.n_electrodes, n_elec)
    npt.assert_equal(len(prima.electrode_array.electrodes), n_elec)

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
        npt.assert_almost_equal(prima[e].radius, 14)

    # Make sure the pitch is correct:
    distF6E6 = np.sqrt((prima['E6'].x - prima['F6'].x) ** 2 +
                       (prima['E6'].y - prima['F6'].y) ** 2)
    npt.assert_almost_equal(distF6E6, spacing)
    distF6E7 = np.sqrt((prima['E7'].x - prima['F6'].x) ** 2 +
                       (prima['E7'].y - prima['F6'].y) ** 2)
    npt.assert_almost_equal(distF6E7, spacing)

    with pytest.raises(ValueError):
        PRIMAPivotal(0, 0, z=np.ones(16))


@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_Lorach2015Array(ztype, x, y, rot):
    # 70 um pixel with 5 um trenches:
    spacing = 75
    # Roughly a 12x15 grid, but edges are trimmed off:
    n_elec = 142
    # Create an Prima and make sure location is correct
    # Height `z` can either be a float or a list
    z = -100 if ztype == 'float' else -np.ones(142) * 20

    prima = Lorach2015Array(x, y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(prima, '__slots__'), True)
    npt.assert_equal(hasattr(prima, '__dict__'), False)

    # Make sure number of electrodes is correct
    npt.assert_equal(len(prima.electrode_array.electrodes), n_elec)
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
        npt.assert_almost_equal(prima[e].radius, 10)

    # Make sure the pitch is correct:
    distF6E6 = np.sqrt((prima['E6'].x - prima['F6'].x) ** 2 +
                       (prima['E6'].y - prima['F6'].y) ** 2)
    npt.assert_almost_equal(distF6E6, spacing)
    distF6E7 = np.sqrt((prima['E7'].x - prima['F6'].x) ** 2 +
                       (prima['E7'].y - prima['F6'].y) ** 2)
    npt.assert_almost_equal(distF6E7, spacing)

    with pytest.raises(ValueError):
        Lorach2015Array(0, 0, z=np.ones(16))


#: pixel size (um), exposed pixels, pixels fabricated on the die, active
#: electrode diameter (um), for the four arrays of [Huang2021]_.
HUANG_VARIANTS = [(55, 421, 526, 22), (40, 821, 1027, 16),
                  (30, 1388, 1735, 12), (20, 2806, 3508, 8)]


def _column_profile(implant):
    """Return the number of pixels in each lattice column."""
    x = np.round(implant.electrode_array.coordinates()[:, 0], 6)
    return [int(n) for n in np.unique(x, return_counts=True)[1]]


def _mask_fingerprint(implant):
    """Return a placement-invariant digest of the pixel layout."""
    xy = implant.electrode_array.coordinates()[:, :2]
    xy = np.round((xy - 0.5 * (xy.min(axis=0) + xy.max(axis=0))) /
                  implant.spacing, 3)
    xy = xy[np.lexsort((xy[:, 1], xy[:, 0]))]
    return hashlib.sha256(xy.tobytes()).hexdigest()[:16]


def _nn_spacing(implant):
    """Return each pixel's nearest-neighbor distance."""
    xy = implant.electrode_array.coordinates()[:, :2]
    return cKDTree(xy).query(xy, k=2)[0][:, 1]


@pytest.mark.parametrize('pixel_size, n_elec, elec_radius', [
    (55, 250, 7),
    (40, 502, 5),
])
@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_Ho2019FlatArray(pixel_size, n_elec, elec_radius, ztype, x, y, rot):
    """Check the published Ho et al. array geometry."""
    # Height `z` can either be a float or a list:
    z = -100 if ztype == 'float' else -np.ones(n_elec) * 20
    prima = Ho2019FlatArray(pixel_size, x, y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(prima, '__slots__'), True)
    npt.assert_equal(hasattr(prima, '__dict__'), False)

    # The published pixel count:
    npt.assert_equal(prima.n_electrodes, n_elec)
    npt.assert_equal(len(prima.electrode_array.electrodes), n_elec)

    xy = prima.electrode_array.coordinates()[:, :2]
    # Nearest-neighbor center spacing, in every direction:
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    npt.assert_almost_equal(dist.min(), pixel_size)
    npt.assert_almost_equal(dist.min(axis=1), pixel_size)
    # Row spacing is derived, not independent:
    npt.assert_almost_equal(prima.row_spacing, pixel_size * np.sqrt(3) / 2)

    for elec in prima.electrode_array.electrode_objects:
        # Pixel bodies are as wide as the lattice, with no open gap:
        npt.assert_almost_equal(elec.width, pixel_size)
        npt.assert_almost_equal(prima.pixel_width, pixel_size)
        npt.assert_almost_equal(prima.spacing, pixel_size)
        npt.assert_almost_equal(prima.gap, 0)
        # Active electrode:
        npt.assert_almost_equal(elec.radius, elec_radius)
        # Hex bodies turn with the lattice:
        npt.assert_almost_equal(elec.rot, rot)
        npt.assert_equal(elec.orientation, 'vertical')

    # Every pixel body sits on the 1 mm substrate, corners included. Flat-top
    # hexagons put a vertex every 60 deg at a circumradius of `width`/sqrt(3):
    corner = np.radians(np.arange(6) * 60 + rot)
    verts = ((xy - [x, y])[:, np.newaxis, :] + pixel_size / np.sqrt(3) *
             np.column_stack([np.cos(corner), np.sin(corner)]))
    npt.assert_array_less(np.hypot(verts[..., 0], verts[..., 1]), 500)

    with pytest.raises(ValueError):
        Ho2019FlatArray(pixel_size, 0, 0, z=np.ones(16))


@pytest.mark.parametrize('pixel_size', (55, 40))
def test_Ho2019FlatArray_units(pixel_size):
    """Unitful and bare coordinates produce the same array."""
    bare = Ho2019FlatArray(pixel_size, x=1000, y=-500, z=-100, rot=30)
    unitful = Ho2019FlatArray(pixel_size * um, x=1 * mm, y=-0.5 * mm,
                              z=-0.1 * mm, rot=30 * deg)
    npt.assert_equal(list(unitful.electrode_array.electrodes),
                     list(bare.electrode_array.electrodes))
    npt.assert_allclose(unitful.electrode_array.coordinates(),
                        bare.electrode_array.coordinates(), atol=1e-9)


def test_Ho2019FlatArray_pixel_size():
    """Only the published Ho et al. variants are accepted."""
    for pixel_size in (20, 30, 45, 54.9, 75, 100):
        with pytest.raises(ValueError, match='does not model'):
            Ho2019FlatArray(pixel_size)
    with pytest.raises(TypeError):
        Ho2019FlatArray([40, 55])
    # A size Huang2021Array does model is still not a Ho2019FlatArray size:
    npt.assert_equal(Huang2021Array(30).n_electrodes, 1388)


def test_Ho2019FlatArray_F55_layout():
    """Check the F55 layout reconstructed from Fig. 2(a)."""
    expected = {-9: (3, 8), -8: (1, 9), -7: (-1, 9), -6: (-3, 9), -5: (-4, 9),
                -4: (-5, 9), -3: (-6, 9), -2: (-6, 8), -1: (-7, 8),
                0: (-7, 7), 1: (-8, 7), 2: (-8, 6), 3: (-9, 6), 4: (-9, 5),
                5: (-9, 4), 6: (-9, 3), 7: (-9, 2), 8: (-9, 1), 9: (-8, -1)}
    npt.assert_equal(sum(hi - lo + 1 for lo, hi in expected.values()), 250)

    prima = Ho2019FlatArray(55)
    s = prima.spacing
    xy = prima.electrode_array.coordinates()[:, :2]
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


@pytest.mark.parametrize('old_cls, pixel_size', [(PRIMA55, 55), (PRIMA40, 40)])
def test_PRIMA55_PRIMA40_are_deprecated(old_cls, pixel_size):
    """Deprecated names map to the Ho et al. arrays."""
    with pytest.deprecated_call(match='Ho et al'):
        old = old_cls(x=-100, y=400, rot=30)
    new = Ho2019FlatArray(pixel_size, x=-100, y=400, rot=30)
    npt.assert_equal(list(old.electrode_array.electrodes),
                     list(new.electrode_array.electrodes))
    npt.assert_allclose(old.electrode_array.coordinates(),
                        new.electrode_array.coordinates())
    npt.assert_equal(old.pixel_size, pixel_size)
    # Still frozen, and still take the rest of the old signature:
    npt.assert_equal(hasattr(old, '__dict__'), False)
    with pytest.deprecated_call():
        old_cls(0, 0, -100, 0, 'LE', False, False)


@pytest.mark.parametrize('old_cls, new_cls',
                         [(PRIMA, PRIMAPivotal),
                          (PRIMA75, Lorach2015Array)])
def test_PRIMA_PRIMA75_are_deprecated(old_cls, new_cls):
    """Deprecated names map to the canonical arrays."""
    with pytest.deprecated_call():
        old = old_cls(x=-100, y=400, rot=30)
    new = new_cls(x=-100, y=400, rot=30)
    npt.assert_equal(list(old.electrode_array.electrodes),
                     list(new.electrode_array.electrodes))
    npt.assert_allclose(old.electrode_array.coordinates(),
                        new.electrode_array.coordinates())
    # Still frozen, and still take the whole old signature:
    npt.assert_equal(hasattr(old, '__dict__'), False)
    with pytest.deprecated_call():
        old_cls(0, 0, -100, 0, 'LE', False, False)


def test_implant_metadata():
    """Check photovoltaic implant metadata."""
    for cls in (PRIMAPivotal, Lorach2015Array, Ho2019FlatArray,
                Huang2021Array):
        npt.assert_equal(cls.placement, 'subretinal')
        npt.assert_equal(cls.technology, 'photovoltaic')
    # Only the clinical device is part of a product family:
    npt.assert_equal(PRIMAPivotal.family, 'PRIMA')
    for cls in (Lorach2015Array, Ho2019FlatArray, Huang2021Array):
        npt.assert_equal(cls.family, None)
    # Unclassified implants default to None rather than to a guess:
    generic = Implant(PointSource(0, 0, 0))
    npt.assert_equal((generic.placement, generic.technology, generic.family),
                     (None, None, None))


def test_prima_public_api():
    """Check canonical and deprecated public names."""
    import pulse2percept.implants as implants
    canonical = ['PRIMAPivotal', 'Lorach2015Array', 'Ho2019FlatArray',
                 'Huang2021Array']
    deprecated = ['PRIMA', 'PRIMA75', 'PRIMA55', 'PRIMA40']
    for name in canonical + deprecated:
        npt.assert_equal(name in implants.__all__, True)
        npt.assert_equal(getattr(implants, name).__name__, name)


@pytest.mark.parametrize('pixel_size, n_elec, n_total, elec_diam',
                         HUANG_VARIANTS)
@pytest.mark.parametrize('ztype', ('float', 'list'))
def test_Huang2021Array(pixel_size, n_elec, n_total, elec_diam, ztype):
    """Check the published Huang et al. array geometry."""
    x, y, rot = -100, 400, 30
    # Height `z` can either be a float or a list, one entry per exposed pixel:
    z = -100 if ztype == 'float' else -np.ones(n_elec) * 20
    prima = Huang2021Array(pixel_size, x, y, z=z, rot=rot)

    # Slots:
    npt.assert_equal(hasattr(prima, '__slots__'), True)
    npt.assert_equal(hasattr(prima, '__dict__'), False)

    npt.assert_equal(prima.n_electrodes, n_elec)
    npt.assert_equal(len(prima.electrode_array.electrodes), n_elec)
    npt.assert_equal(prima.n_total_pixels, n_total)

    # Pixel bodies tile the lattice with no open gap:
    npt.assert_almost_equal(prima.pixel_size, pixel_size)
    npt.assert_almost_equal(prima.spacing, pixel_size)
    npt.assert_almost_equal(prima.pixel_width, pixel_size)
    npt.assert_almost_equal(prima.gap, 0)
    npt.assert_almost_equal(prima.row_spacing, pixel_size * np.sqrt(3) / 2)
    npt.assert_almost_equal(_nn_spacing(prima), pixel_size)

    for elec in prima.electrode_array.electrode_objects:
        npt.assert_almost_equal(elec.width, pixel_size)
        # Active electrode is 40% of the pixel size across:
        npt.assert_almost_equal(2 * elec.radius, elec_diam)
        # Flat-top hex bodies that turn with the lattice:
        npt.assert_equal(elec.orientation, 'vertical')
        npt.assert_almost_equal(elec.rot, rot)

    # Every pixel body sits on the 1.5 mm substrate, corners included:
    xy = prima.electrode_array.coordinates()[:, :2]
    corner = np.radians(np.arange(6) * 60 + rot)
    verts = ((xy - [x, y])[:, np.newaxis, :] + pixel_size / np.sqrt(3) *
             np.column_stack([np.cos(corner), np.sin(corner)]))
    npt.assert_array_less(np.hypot(verts[..., 0], verts[..., 1]), 750)

    # A per-electrode `z` is one entry per exposed pixel, not one per pixel on
    # the die:
    with pytest.raises(ValueError):
        Huang2021Array(pixel_size, z=np.ones(n_total))


#: pixel size (um), smallest hex grid the mask is cut from, pixels in the
#: leftmost and rightmost lattice columns, and a digest of the whole exposed
#: -pixel set (see `_mask_fingerprint`).
HUANG_MASKS = [(55, (22, 25), (7, 3), '9470661f9f60eb1c'),
               (40, (29, 35), (6, 11), 'f4411b9506ef49f2'),
               (30, (40, 45), (6, 10), 'ab65d1cbf16bed60'),
               (20, (56, 64), (11, 11), '0a9c5ea6bf82d21a')]


@pytest.mark.parametrize('pixel_size, shape, edge_columns, digest',
                         HUANG_MASKS)
def test_Huang2021Array_layout(pixel_size, shape, edge_columns, digest):
    """Check the reconstructed Huang et al. layouts."""
    implant = Huang2021Array(pixel_size)
    npt.assert_equal(implant.shape, shape)
    profile = _column_profile(implant)
    npt.assert_equal(len(profile), shape[1])
    npt.assert_equal(sum(profile), implant.n_electrodes)
    # The rim of the exposed region, where a crop would show first:
    npt.assert_equal((profile[0], profile[-1]), edge_columns)
    npt.assert_equal(_mask_fingerprint(implant), digest)


def test_Huang2021Array_pixel_size():
    """Only the published Huang et al. variants are accepted."""
    for pixel_size in (10, 25, 50, 39.9, 75, 100):
        with pytest.raises(ValueError, match='does not model'):
            Huang2021Array(pixel_size)
    with pytest.raises(TypeError):
        Huang2021Array([20, 30])
    # A length quantity names the same variant a bare number of microns does:
    for unitful in (40 * um, 0.04 * mm):
        npt.assert_equal(Huang2021Array(unitful).n_electrodes, 821)


@pytest.mark.parametrize('pixel_size', (55, 20))
def test_Huang2021Array_placement(pixel_size):
    """Placement and rotation preserve the pixel layout."""
    origin = Huang2021Array(pixel_size)
    x, y, rot = -100, 400, 37
    moved = Huang2021Array(pixel_size, x=x, y=y, rot=rot)
    npt.assert_equal(_mask_fingerprint(Huang2021Array(pixel_size, x=x, y=y)),
                     _mask_fingerprint(origin))

    th = np.deg2rad(rot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xy = origin.electrode_array.coordinates()[:, :2]
    npt.assert_allclose(moved.electrode_array.coordinates()[:, :2],
                        (R @ xy.T).T + [x, y], atol=1e-9)
    # The footprint is centered on the requested position:
    npt.assert_almost_equal(0.5 * (xy.min(axis=0) + xy.max(axis=0)), (0, 0))

    unitful = Huang2021Array(pixel_size * um, x=-0.1 * mm, y=0.4 * mm,
                             z=-0.1 * mm, rot=rot * deg)
    npt.assert_equal(list(unitful.electrode_array.electrodes),
                     list(moved.electrode_array.electrodes))
    npt.assert_allclose(unitful.electrode_array.coordinates(),
                        moved.electrode_array.coordinates(), atol=1e-9)


def _substrate(implant, **kwargs):
    """Plot an implant and return its substrate patch."""
    fig, ax = plt.subplots()
    implant.plot(ax=ax, **kwargs)
    patches = [p for p in ax.patches if isinstance(p, (Circle, Polygon))]
    npt.assert_equal(len(patches), 1)
    plt.close(fig)
    return ax, patches[0]


#: Every round-substrate device, paired with the radius (um) of its die.
ROUND_DEVICES = ([(Lorach2015Array, 500),
                  (partial(Ho2019FlatArray, 55), 500),
                  (partial(Ho2019FlatArray, 40), 500)] +
                 [(partial(Huang2021Array, s), 750)
                  for s in (55, 40, 30, 20)])


@pytest.mark.parametrize('implant_type, radius', ROUND_DEVICES)
@pytest.mark.parametrize('rot', (0, 30))
def test_PRIMA_round_substrate(implant_type, radius, rot):
    """Check circular substrate size and position."""
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
    """Check pivotal PRIMA substrate size and rotation."""
    x, y = -100, 400
    ax, patch = _substrate(PRIMAPivotal(x=x, y=y, rot=rot))
    corners = patch.get_xy()[:4]
    npt.assert_almost_equal(corners.mean(axis=0), (x, y))
    edges = np.roll(corners, -1, axis=0) - corners
    npt.assert_almost_equal(np.linalg.norm(edges, axis=1), 2000)
    # Square, and turned by `rot`:
    npt.assert_almost_equal(np.abs(np.sum(edges[0] * edges[1])), 0)
    # Folded into (-45, 45] rather than [0, 90): a residual of -1e-14 is a
    # rounding error, not an 89.99999999999997 deg mismatch.
    off = np.degrees(np.arctan2(edges[0, 1], edges[0, 0])) - rot
    npt.assert_almost_equal(np.mod(off + 45, 90) - 45, 0)
    npt.assert_array_less(patch.get_zorder(), ZORDER['foreground'])
    # The whole die is in view, including the corner the rotation swings out:
    npt.assert_array_less(ax.get_xlim()[0], corners[:, 0].min())
    npt.assert_array_less(corners[:, 0].max(), ax.get_xlim()[1])
    npt.assert_array_less(ax.get_ylim()[0], corners[:, 1].min())
    npt.assert_array_less(corners[:, 1].max(), ax.get_ylim()[1])


@pytest.mark.parametrize('implant_type, radius',
                         [(PRIMAPivotal, None)] + ROUND_DEVICES)
def test_PRIMA_substrate_holds_pixels(implant_type, radius):
    """Check that pixel centers lie on the substrate."""
    implant = implant_type()
    xy = implant.electrode_array.coordinates()[:, :2]
    if radius is None:
        npt.assert_array_less(np.abs(xy), 1000)
    else:
        npt.assert_array_less(np.hypot(*xy.T), radius)

    ax, substrate = _substrate(implant)
    clipped = ax.collections[0].get_clip_path()
    npt.assert_equal(clipped is not None, True)
    npt.assert_allclose(
        clipped.get_fully_transformed_path().get_extents().extents,
        substrate.get_path().transformed(
            substrate.get_transform()).get_extents().extents, atol=1e-6)


@pytest.mark.parametrize('implant_type, radius, whole_bodies', [
    (PRIMAPivotal, None, True), (Lorach2015Array, 500, False),
    (partial(Ho2019FlatArray, 55), 500, True),
    (partial(Ho2019FlatArray, 40), 500, True),
    (partial(Huang2021Array, 40), 750, True),
])
def test_PRIMA_pixel_bodies_vs_substrate(implant_type, radius, whole_bodies):
    """Check which arrays contain clipped rim pixels."""
    implant = implant_type()
    xy = implant.electrode_array.coordinates()[:, :2]
    th = np.radians(np.arange(6) * 60)
    verts = (xy[:, np.newaxis, :] + implant.pixel_width / np.sqrt(3) *
             np.column_stack([np.cos(th), np.sin(th)])).reshape(-1, 2)
    outside = (np.abs(verts).max(axis=1) > 1000 if radius is None
               else np.hypot(*verts.T) > radius)
    npt.assert_equal(not outside.any(), whole_bodies)
    if not whole_bodies:
        # Seven rim pixels of the 142 are cut by the edge of the chip:
        npt.assert_equal(len(np.unique(np.where(outside)[0] // 6)), 7)


@pytest.mark.parametrize('implant_type', [
    PRIMAPivotal, Lorach2015Array, partial(Ho2019FlatArray, 55),
    partial(Ho2019FlatArray, 40),
    partial(Huang2021Array, 55), partial(Huang2021Array, 20),
])
def test_PRIMA_plot_passthrough(implant_type):
    """Check the implant plot override."""
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
    Ho2019FlatArray(40).reshape_stim(LogoBVL())


@pytest.mark.parametrize('implant_type, offset', [
    (PRIMAPivotal, (0, 0)),
    (Lorach2015Array, (0, 0)),
    # The reconstructed masks are centered on the substrate, since where they
    # sit on the die is not published. Ho2019FlatArray(40) instead keeps the
    # 502 lattice sites nearest the substrate center, and a discrete lattice
    # leaves that footprint a quarter of a spacing off center:
    (partial(Ho2019FlatArray, 55), (0, 0)),
    (partial(Ho2019FlatArray, 40), (0, -0.25 * 40)),
    (partial(Huang2021Array, 55), (0, 0)),
    (partial(Huang2021Array, 20), (0, 0)),
])
def test_PRIMA_device_center(implant_type, offset):
    """Where the trimmed device sits relative to the requested (x, y)

    Each PRIMA is a regular hex grid with edge electrodes removed afterwards,
    so the finished device is centered only if those removals are symmetric --
    the grid's own centering says nothing about it. The per-electrode
    coordinate tests would all still pass if a device drifted sideways.
    """
    x, y, rot = -100, 400, 37
    xy = implant_type(x=x, y=y).electrode_array.coordinates()[:, :2]
    center = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    npt.assert_almost_equal(center, np.add([x, y], offset))
    # `rot` turns the whole footprint about (x, y), so the offset above is a
    # property of the device rather than of the coordinate axes:
    th = np.deg2rad(rot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rotated = implant_type(x=x, y=y, rot=rot).electrode_array.coordinates()[:,
                                                                            :2]
    npt.assert_almost_equal(rotated, (R @ (xy - [x, y]).T).T + [x, y])


class LooseEncoder(PRIMAEncoder):
    """PRIMAEncoder variant used to exercise ``safe_mode`` limits."""
    __slots__ = ()

    pulse_step = 0.35
    max_pulse_dur = 21.0
    max_irradiance = 100.0


def test_PRIMAPivotal_is_stimulated_optically():
    implant = PRIMAPivotal()
    npt.assert_equal(implant.stimulus_unit, mW / mm ** 2)
    npt.assert_equal(isinstance(implant.encoder, PRIMAEncoder), True)
    # All photovoltaic pixels may be illuminated simultaneously.
    npt.assert_equal(implant.raster, None)
    # Encoder state is per implant instance.
    implant.encoder.threshold = 0.9
    npt.assert_almost_equal(PRIMAPivotal().encoder.threshold, 0.5)

    stim = implant.prepare_stim(LogoBVL())
    npt.assert_equal(stim.unit, mW / mm ** 2)
    npt.assert_equal(stim.shape[0], 378)
    npt.assert_almost_equal(stim.data.max(), 3.5)
    npt.assert_almost_equal(stim.duration, 500)

    # Disabling the encoder rejects image input.
    with pytest.raises(DimensionMismatchError):
        PRIMAPivotal(encoder=None).prepare_stim(LogoBVL())
    with pytest.raises(TypeError):
        PRIMAPivotal(encoder='binary')


def test_PRIMAPivotal_rejects_threshold_relative_stimuli():
    # Threshold-relative current is invalid for an optical implant.
    train = Stimulus({'A5': BiphasicPulseTrain(20, 2 * xTh, 0.45,
                                               stim_dur=50)})
    with pytest.raises(DimensionMismatchError) as excinfo:
        PRIMAPivotal().prepare_stim(train)
    npt.assert_equal('irradiance' in str(excinfo.value), True)
    # Current-driven implants still accept xTh before calibration.
    npt.assert_equal(ArgusII(encoder=None).prepare_stim(train).unit, xTh)


@pytest.mark.parametrize('implant_type', [Lorach2015Array,
                                          partial(Ho2019FlatArray, 55),
                                          partial(Huang2021Array, 55)])
def test_other_photovoltaic_arrays_have_no_encoder(implant_type):
    # Other photovoltaic arrays do not assume the pivotal PRIMA protocol.
    npt.assert_equal(implant_type().encoder, None)


def test_PRIMA_deprecated_alias_keeps_the_encoder():
    with pytest.deprecated_call():
        implant = PRIMA()
    npt.assert_equal(isinstance(implant.encoder, PRIMAEncoder), True)
    npt.assert_equal(implant.prepare_stim(LogoBVL()).unit, mW / mm ** 2)


def test_PRIMAPivotal_safe_mode_accepts_the_full_device():
    # Full-array illumination at the documented maximum is valid.
    implant = PRIMAPivotal(safe_mode=True)
    stim = implant.prepare_stim(ImageStimulus(np.ones((32, 32))))
    npt.assert_equal(np.count_nonzero(stim.data.max(axis=1)), 378)
    npt.assert_almost_equal(stim.duty_cycle.max(), 0.294)
    # All documented pulse-duration levels are valid.
    for pulse_dur in np.arange(1, 15) * 0.7:
        implant.encoder = PRIMAEncoder(pulse_dur=pulse_dur, grayscale=True)
        implant.prepare_stim(LogoBVL())


@pytest.mark.parametrize('encoder, msg', [
    (LooseEncoder(irradiance=5.0), 'exceeds the 3.5 mW/mm^2'),
    (LooseEncoder(pulse_dur=14.0), 'longest documented ON duration'),
    (LooseEncoder(pulse_dur=1.05), 'whole multiples of 0.7 ms'),
    # Combined frequency/pulse-duration violations are caught by duty cycle.
    (PRIMAEncoder(freq=60), 'duty cycle'),
    # Frame rate is checked independently.
    (PRIMAEncoder(freq=60, pulse_dur=4.9), 'runs the projector at 60 Hz'),
])
def test_PRIMAPivotal_safe_mode_rejects(encoder, msg):
    implant = PRIMAPivotal(safe_mode=True, encoder=encoder)
    with pytest.raises(ValueError) as excinfo:
        implant.prepare_stim(LogoBVL())
    npt.assert_equal(msg in str(excinfo.value), True)
    # The operating-envelope check is disabled when safe_mode=False.
    npt.assert_equal(
        PRIMAPivotal(encoder=encoder).prepare_stim(LogoBVL()).unit,
        mW / mm ** 2)


def test_PRIMAPivotal_safe_mode_reads_the_schedule_not_the_metadata():
    # Envelope checks use schedule state, not mutable metadata.
    implant = PRIMAPivotal(safe_mode=True)
    stim = PRIMAPivotal(encoder=PRIMAEncoder(freq=60)).prepare_stim(LogoBVL())
    stim.metadata['encoder'] = {'frame_time': np.zeros(1), 'frame_dur': 500.0,
                                'optical': {'wavelength': 880.0,
                                            'irradiance': 3.5, 'freq': 30.0,
                                            'pulse_dur': np.zeros((378, 1)),
                                            'grayscale': False}}
    with pytest.raises(ValueError) as excinfo:
        implant.check_stim(stim)
    npt.assert_equal('duty cycle' in str(excinfo.value), True)


def test_PRIMAPivotal_refuses_light_that_is_not_light():
    # Negative or nonfinite irradiance is invalid regardless of safe_mode.
    stim = PRIMAPivotal().prepare_stim(LogoBVL())
    # Negative scaling is rejected by the schedule.
    with pytest.raises(ValueError):
        stim * -1
    # Zero scaling turns illumination off.
    npt.assert_almost_equal((stim * 0).irradiance, 0)
    PRIMAPivotal().check_stim(stim * 0)
    # Nonfinite waveform samples are rejected.
    for factor in (np.nan, np.inf):
        with pytest.raises(ValueError) as excinfo:
            PRIMAPivotal().check_stim(stim * factor)
        npt.assert_equal('non-finite irradiance' in str(excinfo.value), True)
    # Hand-built negative irradiance is also rejected.
    negative = Stimulus(stim)
    negative.metadata = {'user': None}
    negative._stim = {'data': -np.abs(negative.data),
                      'electrodes': negative.electrodes,
                      'time': negative.time}
    with pytest.raises(ValueError) as excinfo:
        PRIMAPivotal().check_stim(negative)
    npt.assert_equal('cannot be negative' in str(excinfo.value), True)


def test_PRIMAPivotal_safe_mode_needs_the_projector_settings():
    implant = PRIMAPivotal(safe_mode=True)
    encoded = PRIMAPivotal().prepare_stim(LogoBVL())
    # Duty cycle cannot be checked after the projector schedule is lost.
    handmade = Stimulus(encoded)
    handmade.metadata = {'user': None}
    with pytest.raises(ValueError) as excinfo:
        implant.check_stim(handmade)
    npt.assert_equal('duty cycle cannot be verified' in str(excinfo.value),
                     True)
    # Without safe_mode, the incomplete envelope check is skipped.
    PRIMAPivotal().check_stim(handmade)
    # Optical safe_mode does not accept electrical stimulation.
    with pytest.raises(DimensionMismatchError):
        implant.check_stim(Stimulus({'A5': BiphasicPulse(10, 0.45)}))
    # max_current is not defined for this optical device.
    implant = PRIMAPivotal()
    implant.max_current = 100
    with pytest.raises(DimensionMismatchError):
        implant.prepare_stim(LogoBVL())
