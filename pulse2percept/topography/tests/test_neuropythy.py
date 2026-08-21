"""Tests for :py:class:`~pulse2percept.topography.NeuropythyMap`

Almost everything this map owns -- the bookkeeping around a lookup, the
hemisphere split, the nearest-neighbor interpolation, the download cache --
works the same whatever subject is loaded, and is tested here against the toy
cortex below. The handful of tests marked ``slow`` are the ones that check p2p
against real Neuropythy and the Benson & Winawer (2018) data; they need
``--runslow``, and they share a single ``fsaverage`` map.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree
from types import SimpleNamespace

from pulse2percept.implants import EnsembleImplant
from pulse2percept.implants.cortex import Neuralink
from pulse2percept.models import ScoreboardModel as BeyelerScoreboard
from pulse2percept.models.cortex import ScoreboardModel
from pulse2percept.topography import CorticalMap, NeuropythyMap
from pulse2percept.units import DimensionMismatchError, dva, mm, um


#: Where each surface of the toy cortex sits along z, in millimeters.
TOY_SURFACE_MM = {'white': -1.0, 'midgray': 0.0, 'pial': 1.0}

#: How far out the toy visual field mesh reaches, in dva. A point beyond it
#: has no face to be addressed to, the way a point outside V1 has none.
TOY_MAX_ECC = 5.0


def toy_cortex_mm(x, y, surface):
    """Where the toy cortex puts the visual field point ``(x, y)``

    The three surfaces are scaled copies of each other, so the direction from
    one to the next depends on where the point is -- which is what
    :py:meth:`~pulse2percept.implants.cortex.Neuralink.from_neuropythy` turns
    into a thread orientation.
    """
    scale = 1 + TOY_SURFACE_MM[surface] / 4
    return np.stack([scale * np.asarray(x, dtype=float),
                     -scale * np.asarray(y, dtype=float),
                     np.full(np.shape(x), TOY_SURFACE_MM[surface])])


class ToyMesh:
    """Stands in for one hemisphere's visual field mesh of one region

    Addressing a point on a real mesh gives the face containing it and the
    point's barycentric coordinates within that face; a point the mesh does
    not contain addresses to NaN. The toy keeps the visual field coordinates
    themselves as the barycentric ones, so the surface below can unaddress
    them without a triangulation, and records what it was asked for.
    """

    def __init__(self, coordinates):
        # `cortex_to_dva` looks a vertex's dva coordinates up here:
        self.coordinates = np.asarray(coordinates, dtype=float)
        self.addressed = []

    def address(self, coords):
        x, y = (np.atleast_1d(np.asarray(c, dtype=float)) for c in coords)
        self.addressed.append((x.copy(), y.copy()))
        bc = np.stack([x, y])
        bc[:, np.hypot(x, y) > TOY_MAX_ECC] = np.nan
        return {'faces': np.zeros((3, x.size), dtype=int), 'coordinates': bc}


class ToySurface:
    """Stands in for one FreeSurfer surface of one hemisphere (in mm)"""

    def __init__(self, name):
        self.name = name

    def unaddress(self, addr):
        x, y = np.asarray(addr['coordinates'], dtype=float)
        return toy_cortex_mm(x, y, self.name)


class ToyHemisphere:
    """Stands in for one hemisphere, recording which surfaces were asked for"""

    def __init__(self):
        self.surfaces_asked = []

    def surface(self, name):
        self.surfaces_asked.append(name)
        return ToySurface(name)


class ToySubject:
    """Stands in for a FreeSurfer subject: two hemispheres of surfaces"""

    def __init__(self):
        self.hemis = {'lh': ToyHemisphere(), 'rh': ToyHemisphere()}


class ToyNeuropythyMap(NeuropythyMap):
    """A NeuropythyMap with a toy cortex in place of a FreeSurfer subject

    Going backward, the cortical mesh is a row of ``n`` vertices 1 mm apart --
    exactly ``cort_nn_thresh`` -- with vertex ``i`` sitting at ``(i, 0, 0)`` mm
    and looking at ``(i, -i)`` dva.

    Going forward, addressing a point is neuropythy's job even here, so the toy
    only supplies the mesh and the surfaces it addresses onto: a visual field
    point lands where :py:func:`toy_cortex_mm` puts it, and a point beyond
    ``TOY_MAX_ECC`` comes back NaN.

    The two directions are separate stand-ins rather than inverses of each
    other. Nothing in ``NeuropythyMap`` composes them, and keeping them apart
    keeps both sets of numbers hand-checkable.
    """

    def __init__(self, n=6, regions=('v1', 'v2', 'v3'), cache_dir=None,
                 **params):
        # NeuropythyMap.__init__ needs neuropythy and a subject, so go
        # straight to the parameter defaults it would have set:
        CorticalMap.__init__(self, regions=list(regions), **params)
        self.cache_dir = cache_dir
        self.cortex_tree = cKDTree(np.stack([np.arange(n, dtype=float),
                                             np.zeros(n), np.zeros(n)],
                                            axis=-1))
        self.addr_idxs = {'addr': np.arange(n),
                          'region': np.array([self.regions[0]] * n),
                          'hemi': np.zeros(n, dtype=int)}
        coords = np.stack([np.arange(n, dtype=float),
                           -np.arange(n, dtype=float)])
        self.region_meshes = {r: (ToyMesh(coords), ToyMesh(coords))
                              for r in self.regions}
        self.subject = ToySubject()


@pytest.fixture(scope='module')
def neuropythy():
    """The neuropythy package, or skip

    Addressing a point is neuropythy's job even when the mesh is a toy, so the
    forward direction needs the package imported -- but no subject, and no
    Benson & Winawer download.
    """
    return pytest.importorskip('neuropythy',
                               reason="requires `pip install neuropythy`")


@pytest.fixture(scope='module')
def fsaverage(neuropythy):
    """One real 'fsaverage' map, shared by every test that needs real data

    Building it predicts retinotopy and meshes three regions, and may download
    the Benson & Winawer (2018) dataset first, which is why every test using it
    is marked slow. A missing neuropythy skips those tests; anything else that
    goes wrong is a failure, not a skip.
    """
    return NeuropythyMap('fsaverage', regions=['v1', 'v2', 'v3'])


# -----------------------------------------------------------------------------
# cortex -> dva
# -----------------------------------------------------------------------------

def test_cortex_to_dva_shape_and_nans():
    """Every input point must get its own output slot (Issue #774)."""
    nmap = ToyNeuropythyMap()
    # Vertex i is at i mm == i * 1000 um, and maps to (i, -i) dva:
    xc = np.array([0., 1000., 2000.])
    xdva, ydva = nmap.cortex_to_dva(xc, np.zeros(3), np.zeros(3))
    npt.assert_almost_equal(xdva, [0, 1, 2])
    npt.assert_almost_equal(ydva, [0, -1, -2])

    # A NaN input must not shift the points after it into its slot:
    xc = np.array([0., np.nan, 2000.])
    xdva, ydva = nmap.cortex_to_dva(xc, np.zeros(3), np.zeros(3))
    npt.assert_equal(xdva.shape, (3,))
    npt.assert_almost_equal(xdva, [0, np.nan, 2])
    npt.assert_almost_equal(ydva, [0, np.nan, -2])
    # A NaN in any one of the three coordinates is enough:
    for coords in ([np.zeros(2), np.array([np.nan, 0.]), np.zeros(2)],
                   [np.zeros(2), np.zeros(2), np.array([np.nan, 0.])]):
        xdva, ydva = nmap.cortex_to_dva(*coords)
        npt.assert_almost_equal(xdva, [np.nan, 0])
        npt.assert_almost_equal(ydva, [np.nan, 0])

    # The output has the shape of the input, whatever that shape is:
    for shape in [(), (1,), (4,), (2, 3), (2, 3, 4)]:
        zeros = np.zeros(shape)
        xdva, ydva = nmap.cortex_to_dva(zeros, zeros, zeros)
        npt.assert_equal(xdva.shape, shape)
        npt.assert_equal(ydva.shape, shape)
        # ... including when every point is NaN, which must still return two
        # arrays rather than a single stacked one:
        nans = np.full(shape, np.nan)
        xdva, ydva = nmap.cortex_to_dva(nans, nans, nans)
        npt.assert_equal(xdva.shape, shape)
        npt.assert_equal(np.all(np.isnan(xdva)), True)
        npt.assert_equal(np.all(np.isnan(ydva)), True)

    with pytest.raises(ValueError):
        nmap.cortex_to_dva(np.zeros(3), np.zeros(2), np.zeros(3))


def test_cortex_to_dva_interpolation():
    """Nearby vertices are averaged; distant ones do not count at all."""
    nmap = ToyNeuropythyMap()
    # A point sitting exactly on a vertex is at zero distance from it, which
    # must not divide by zero (Issue #774). It maps to that vertex, not to a
    # blend with its neighbors and not to NaN:
    verts = nmap.cortex_tree.data * 1000  # mm -> um
    with np.errstate(divide='raise', invalid='raise'):
        xdva, ydva = nmap.cortex_to_dva(verts[:, 0], verts[:, 1], verts[:, 2])
    npt.assert_almost_equal(xdva, np.arange(len(verts)))
    npt.assert_almost_equal(ydva, -np.arange(len(verts)))

    # Halfway between two vertices, both weigh the same:
    xdva, ydva = nmap.cortex_to_dva(np.array([2500.]), np.zeros(1), np.zeros(1))
    npt.assert_almost_equal(xdva, [2.5])
    npt.assert_almost_equal(ydva, [-2.5])

    # `cort_nn_thresh` is how far a vertex reaches, not a rounding: a point
    # exactly that far away still maps, one micron further does not:
    npt.assert_almost_equal(nmap.cort_nn_thresh, 1000)
    xdva, ydva = nmap.cortex_to_dva(np.array([-1000.]), np.zeros(1),
                                    np.zeros(1))
    npt.assert_almost_equal(xdva, [0])
    npt.assert_almost_equal(ydva, [0])
    xdva, ydva = nmap.cortex_to_dva(np.array([-1001.]), np.zeros(1),
                                    np.zeros(1))
    npt.assert_almost_equal(xdva, [np.nan])
    npt.assert_almost_equal(ydva, [np.nan])


def test_cortex_to_dva_region_dispatch():
    """v1/v2/v3 all read the same cortical mesh; only the name differs."""
    nmap = ToyNeuropythyMap()
    coords = (np.array([500.]), np.zeros(1), np.zeros(1))
    expected = nmap.cortex_to_dva(*coords)
    for to_dva in (nmap.v1_to_dva, nmap.v2_to_dva, nmap.v3_to_dva):
        npt.assert_almost_equal(to_dva(*coords), expected)
    npt.assert_equal(sorted(nmap.to_dva().keys()), ['v1', 'v2', 'v3'])


# -----------------------------------------------------------------------------
# dva -> cortex
# -----------------------------------------------------------------------------

def test_dva_to_cortex_regions(neuropythy):
    """A region the map was not built with has nothing to look a point up in."""
    nmap = ToyNeuropythyMap(regions=['v1'])
    npt.assert_equal(sorted(nmap.from_dva().keys()), ['v1'])
    nmap.dva_to_v1(1, 1)
    for dva_to in (nmap.dva_to_v2, nmap.dva_to_v3):
        with pytest.raises(ValueError):
            dva_to(1, 1)

    # Each of the three dispatches to its own region's mesh:
    nmap = ToyNeuropythyMap()
    for region, dva_to in [('v1', nmap.dva_to_v1), ('v2', nmap.dva_to_v2),
                           ('v3', nmap.dva_to_v3)]:
        dva_to(1, 1)
        npt.assert_equal([len(m.addressed)
                          for m in nmap.region_meshes[region]], [1, 0])

    # The hemisphere is not something `dva_to_cortex` can guess:
    with pytest.raises(ValueError):
        nmap.dva_to_cortex(np.ones(1), np.ones(1), region='v1')
    with pytest.raises(ValueError):
        nmap.dva_to_cortex(np.ones(1), np.ones(1), region='v1', hemi='both')


def test_dva_to_cortex_hemispheres_and_shapes(neuropythy):
    """Points are split at x=0, looked up, and put back where they came from."""
    nmap = ToyNeuropythyMap()
    x, y = np.array([-1., 0., 2.]), np.array([1., 1., -1.])
    xc, yc, zc = nmap.dva_to_v1(x, y)
    lh, rh = nmap.region_meshes['v1']
    npt.assert_almost_equal(rh.addressed[-1][0], [-1])
    npt.assert_almost_equal(lh.addressed[-1][0], [0, 2])
    # ... and the answers come back in the caller's order, in microns:
    npt.assert_almost_equal(xc, [-1000, 0, 2000])
    npt.assert_almost_equal(yc, [-1000, -1000, 1000])
    npt.assert_almost_equal(zc, [0, 0, 0])

    # The output has the shape of the input, down to a scalar:
    for shape in [(), (1,), (6,), (2, 3)]:
        zeros = np.zeros(shape)
        npt.assert_equal([c.shape for c in nmap.dva_to_v1(zeros, zeros)],
                         [shape] * 3)
    # ... including an empty one, which never reaches the mesh at all:
    addressed = len(lh.addressed)
    npt.assert_equal([c.shape
                      for c in nmap.dva_to_v1(np.array([]), np.array([]))],
                     [(0,)] * 3)
    npt.assert_equal(len(lh.addressed), addressed)

    for dva_to in (nmap.dva_to_v1, nmap.dva_to_v2, nmap.dva_to_v3):
        with pytest.raises(ValueError):
            dva_to(np.zeros(3), np.zeros(2))

    # A point the mesh cannot address keeps its slot and comes back NaN:
    xc, yc, zc = nmap.dva_to_v1([1., TOY_MAX_ECC + 1], [0., 0.])
    npt.assert_almost_equal(xc, [1000, np.nan])
    npt.assert_almost_equal(yc, [0, np.nan])
    npt.assert_almost_equal(zc, [0, np.nan])


@pytest.mark.parametrize('surface', ['white', 'midgray', 'pial'])
def test_dva_to_cortex_surface(surface, neuropythy):
    """`surface` picks which surface of the subject the point lands on."""
    nmap = ToyNeuropythyMap()
    npt.assert_almost_equal(nmap.dva_to_v1(2., 2., surface=surface),
                            toy_cortex_mm(2., 2., surface) * 1000, decimal=3)
    npt.assert_equal(nmap.subject.hemis['lh'].surfaces_asked, [surface])
    npt.assert_equal(nmap.subject.hemis['rh'].surfaces_asked, [])
    # The address itself is what a caller asking for no surface wants:
    addr = nmap.dva_to_cortex(np.ones(1), np.ones(1), hemi='lh', surface=None)
    npt.assert_equal(sorted(addr.keys()), ['coordinates', 'faces'])


@pytest.mark.parametrize('region', ['v1', 'v2', 'v3'])
def test_dva_to_cortex_jitter_boundary(region, neuropythy):
    """Jittering moves points off the meridians, which have no cortex of their own

    V1 spans the horizontal meridian, so only the vertical one is a
    discontinuity there; V2 and V3 are bounded by both.
    """
    x, y = [0., 1.], [0., 1.]
    nmap = ToyNeuropythyMap(jitter_boundary=False)
    nmap.from_dva()[region](x, y)
    addressed_x, addressed_y = nmap.region_meshes[region][0].addressed[-1]
    npt.assert_almost_equal(addressed_x, [0, 1])
    npt.assert_almost_equal(addressed_y, [0, 1])

    nmap = ToyNeuropythyMap(jitter_boundary=True)
    nmap.from_dva()[region](x, y)
    addressed_x, addressed_y = nmap.region_meshes[region][0].addressed[-1]
    npt.assert_almost_equal(addressed_x, [nmap.jitter_thresh, 1], decimal=6)
    expected_y = [0, 1] if region == 'v1' else [nmap.jitter_thresh, 1]
    npt.assert_almost_equal(addressed_y, expected_y, decimal=6)


def test_NeuropythyMap_units(neuropythy):
    """The FreeSurfer map converts between the same two sides as any other"""
    vfmap = ToyNeuropythyMap()
    npt.assert_equal(vfmap.visual_unit, dva)
    npt.assert_equal(vfmap.tissue_unit, um)
    x, y = np.array([1.0, 3.0]), np.array([1.0, -2.0])
    bare = vfmap.dva_to_v1(x, y)
    npt.assert_allclose(vfmap.dva_to_v1(x * dva, y * dva), bare, rtol=1e-12)
    # `surface=` is not a coordinate and travels through untouched:
    npt.assert_allclose(vfmap.dva_to_v1(x * dva, y * dva, surface='pial'),
                        vfmap.dva_to_v1(x, y, surface='pial'), rtol=1e-12)
    # Back again, with the three coordinates spelled differently:
    xc, yc, zc = np.array([1000.0, 2500.0]), np.zeros(2), np.zeros(2)
    npt.assert_allclose(
        vfmap.v1_to_dva((xc / 1000) * mm, yc * um, (zc / 1000) * mm),
        vfmap.v1_to_dva(xc, yc, zc), rtol=1e-6)
    with pytest.raises(DimensionMismatchError):
        vfmap.dva_to_v1(x * um, y)
    with pytest.raises(DimensionMismatchError):
        vfmap.v1_to_dva(xc * dva, yc, zc)


def test_ndim_mixup():
    """A 3D cortical map cannot drive a model that only knows 2D grids."""
    model = BeyelerScoreboard(vfmap=ToyNeuropythyMap())
    npt.assert_equal(2 in model.ndim, True)
    npt.assert_equal(3 in model.ndim, False)
    with pytest.raises(ValueError):
        model.build()


# -----------------------------------------------------------------------------
# parse_subject
# -----------------------------------------------------------------------------

def fake_config():
    """A stand-in for ``ny.config``, as unconfigured as a fresh install"""
    return {'benson_winawer_2018_path': None, 'freesurfer_subject_paths': []}


class DownloadOnAccess:
    """A stand-in for ``ny.data['benson_winawer_2018'].subjects``

    Looking a subject up here is what downloads it, and is the only reason p2p
    reaches for the dataset at all.
    """

    def __init__(self, downloaded):
        self.downloaded = downloaded

    def __getitem__(self, name):
        self.downloaded.append(name)
        return name


def test_parse_subject_passes_through_loaded_subject(neuropythy, tmp_path,
                                                     monkeypatch):
    """A subject the caller loaded themselves is used as it is."""
    class LoadedSubject:
        pass

    def no_lookup(subject):
        raise AssertionError("should not have gone looking for a subject")

    monkeypatch.setattr(neuropythy.mri.core, 'Subject', LoadedSubject)
    monkeypatch.setattr(neuropythy, 'config', fake_config())
    monkeypatch.setattr(neuropythy, 'freesurfer_subject', no_lookup)
    nmap = ToyNeuropythyMap(cache_dir=str(tmp_path))
    subject = LoadedSubject()
    npt.assert_equal(nmap.parse_subject(subject) is subject, True)


def test_parse_subject_configures_cache(neuropythy, tmp_path, monkeypatch):
    """The cache is created and pointed at before any subject is looked up."""
    config = fake_config()
    monkeypatch.setattr(neuropythy, 'config', config)
    monkeypatch.setattr(neuropythy, 'freesurfer_subject',
                        lambda subject: f'loaded {subject}')
    cache_dir = tmp_path / 'cache'
    nmap = ToyNeuropythyMap(cache_dir=str(cache_dir))
    npt.assert_equal(nmap.parse_subject('fsaverage'), 'loaded fsaverage')

    dataset = cache_dir / 'benson_winawer_2018'
    npt.assert_equal(dataset.is_dir(), True)
    npt.assert_equal(config['benson_winawer_2018_path'], str(dataset))
    npt.assert_equal(config['freesurfer_subject_paths'],
                     [str(dataset / 'freesurfer_subjects')])
    # A second subject must not point neuropythy anywhere else, nor add the
    # same search path twice:
    nmap.parse_subject('S1201')
    npt.assert_equal(config['benson_winawer_2018_path'], str(dataset))
    npt.assert_equal(len(config['freesurfer_subject_paths']), 1)


@pytest.mark.parametrize('subject', ['S1201', 's1201'])
def test_parse_subject_downloads_benson_winawer(subject, neuropythy, tmp_path,
                                                monkeypatch):
    """Neuropythy only downloads the dataset for 'fsaverage' on its own

    Every other Benson & Winawer subject comes back as a ValueError until the
    dataset is on disk, so p2p asks for it and tries again.
    """
    downloaded = []
    attempts = []

    def freesurfer_subject(name):
        attempts.append(name)
        if not downloaded:
            raise ValueError(f"no such subject: {name}")
        return f'loaded {name}'

    dataset = SimpleNamespace(subjects=DownloadOnAccess(downloaded))
    monkeypatch.setattr(neuropythy, 'config', fake_config())
    monkeypatch.setattr(neuropythy, 'freesurfer_subject', freesurfer_subject)
    monkeypatch.setattr(neuropythy, 'data', {'benson_winawer_2018': dataset})

    nmap = ToyNeuropythyMap(cache_dir=str(tmp_path))
    npt.assert_equal(nmap.parse_subject(subject), f'loaded {subject}')
    npt.assert_equal(downloaded, [subject])
    npt.assert_equal(attempts, [subject, subject])


def test_parse_subject_reraises_unknown_subject(neuropythy, tmp_path,
                                                monkeypatch):
    """A subject that is not ours to download stays neuropythy's error."""
    def freesurfer_subject(name):
        raise ValueError(f"no such subject: {name}")

    monkeypatch.setattr(neuropythy, 'config', fake_config())
    monkeypatch.setattr(neuropythy, 'freesurfer_subject', freesurfer_subject)
    nmap = ToyNeuropythyMap(cache_dir=str(tmp_path))
    with pytest.raises(ValueError):
        nmap.parse_subject('invalid_subject')


# -----------------------------------------------------------------------------
# Neuralink.from_neuropythy
# -----------------------------------------------------------------------------

def test_Neuralink_from_neuropythy(neuropythy):
    """One thread per visual field location, sitting on the pial surface

    ``from_neuropythy`` is tested on its own -- grids, thread names, regions,
    insertion angles -- against a stub map in
    ``implants/cortex/tests/test_neuralink.py``. What is left to check here is
    that what a real ``NeuropythyMap`` lookup returns, hemisphere split and
    NaNs and all, is what that factory expects.
    """
    nmap = ToyNeuropythyMap()
    locs = np.array([[0, 0], [3, 3], [-2, -2], [TOY_MAX_ECC, TOY_MAX_ECC]])
    nlink = Neuralink.from_neuropythy(nmap, locs=locs)
    # The last location is off the mesh, so there is nowhere to put a thread:
    npt.assert_equal(list(nlink.implants.keys()), ['A', 'B', 'C'])
    for name, (x, y) in zip(['A', 'B', 'C'], locs[:3]):
        implant = nlink.implants[name]
        pial = np.array(nmap.dva_to_v1(x, y, surface='pial'))
        npt.assert_almost_equal([implant.x, implant.y, implant.z], pial,
                                decimal=3)
        # A thread points from where it enters the cortex to where it ends up:
        orient = np.array(nmap.dva_to_v1(x, y, surface='midgray')) - pial
        npt.assert_almost_equal(implant.direction,
                                orient / np.linalg.norm(orient), decimal=4)


# -----------------------------------------------------------------------------
# The real thing: Neuropythy and the Benson & Winawer (2018) data
# -----------------------------------------------------------------------------

#: Visual field points (dva) looked up on the real 'fsaverage' subject. The
#: two on the vertical meridian have no cortex of their own; for V2 and V3 the
#: horizontal meridian is a boundary too.
FSAVERAGE_POINTS = {
    'v1': ([1, 1, 0, 0, -1, -1], [1, -1, 1, -1, 1, -1]),
    'v2': ([1, 1, 0, 0, -1, -1], [1, -1, 1, -1, 0, -1]),
    'v3': ([1, 1, 0, 0, -1, -1], [1, -1, 1, -1, 0, -1]),
}

#: Where `FSAVERAGE_POINTS` land on the cortex (um), by region and by
#: `jitter_boundary`. These numbers are a regression baseline: they say the
#: mapping has not moved, not that it is anatomically right.
FSAVERAGE_CORTEX = {
    ('v1', False): ([-10035.355, -13315.073, np.nan, np.nan, 12075.739,
                     13630.971],
                    [-96637.12, -102852.29, np.nan, np.nan, -95358.4,
                     -101546.41],
                    [-10769.129, -3861.491, np.nan, np.nan, -7168.826,
                     924.938]),
    ('v1', True): ([-10035.355, -13315.073, -11266.07, -16252.549, 12075.739,
                    13630.971],
                   [-96637.12, -102852.29, -96669.43, -102938.95, -95358.4,
                    -101546.41],
                   [-10769.129, -3861.491, -12831.113, -1908.735, -7168.826,
                    924.938]),
    ('v2', False): ([-11731.504, -20458.03, np.nan, np.nan, np.nan,
                     22283.799],
                    [-93461.92, -100803.35, np.nan, np.nan, np.nan,
                     -99334.945],
                    [-11246.644, 1673.845, np.nan, np.nan, np.nan, 7011.859]),
    ('v2', True): ([-11731.504, -20458.03, np.nan, -18807.701, 26066.922,
                    22283.799],
                   [-93461.92, -100803.35, np.nan, -101528.13, -96025.48,
                    -99334.945],
                   [-11246.644, 1673.845, np.nan, -313.502, 4501.598,
                    7011.859]),
    ('v3', False): ([-23812.113, -23514.828, np.nan, np.nan, np.nan,
                     28547.275],
                    [-84409.51, -93015.07, np.nan, np.nan, np.nan, -93238.63],
                    [-15261.302, 4050.124, np.nan, np.nan, np.nan, 8467.487]),
    ('v3', True): ([-23812.113, -23514.828, -29542.21, -25206.152, 27090.357,
                    28547.275],
                   [-84409.51, -93015.07, -83442.17, -89647.35, -94726.14,
                    -93238.63],
                   [-15261.302, 4050.124, -16078.909, 3062.166, 4468.217,
                    8467.487]),
}


@pytest.mark.slow
def test_fsaverage_meshes(fsaverage):
    """The subject arrives with a predicted retinotopy and a mesh per region."""
    npt.assert_equal(fsaverage.predicted_retinotopy is not None, True)
    npt.assert_equal(list(fsaverage.region_meshes.keys()), ['v1', 'v2', 'v3'])
    for meshes in fsaverage.region_meshes.values():
        # One mesh per hemisphere, each with vertices in it:
        npt.assert_equal(len(meshes), 2)
        for mesh in meshes:
            npt.assert_equal(mesh.coordinates.shape[0], 2)
            npt.assert_equal(mesh.coordinates.shape[1] > 0, True)
    npt.assert_equal(fsaverage.cortex_tree.n > 0, True)


@pytest.mark.slow
@pytest.mark.parametrize('jitter_boundary', [False, True])
@pytest.mark.parametrize('region', ['v1', 'v2', 'v3'])
def test_fsaverage_dva_to_cortex(region, jitter_boundary, fsaverage,
                                 monkeypatch):
    """Known visual field points still land where they used to on fsaverage."""
    monkeypatch.setattr(fsaverage, 'jitter_boundary', jitter_boundary)
    x, y = FSAVERAGE_POINTS[region]
    expected = FSAVERAGE_CORTEX[(region, jitter_boundary)]
    coords = fsaverage.from_dva()[region](x, y)
    npt.assert_equal([c.shape for c in coords], [(6,)] * 3)
    npt.assert_almost_equal(coords[0], expected[0], decimal=3)
    npt.assert_almost_equal(coords[1], expected[1], decimal=2)
    npt.assert_almost_equal(coords[2], expected[2], decimal=3)


@pytest.mark.slow
@pytest.mark.parametrize('region', ['v1', 'v2', 'v3'])
def test_fsaverage_cortex_to_dva(region, fsaverage):
    """The inverse lands back on the visual field point it started from."""
    to_dva = fsaverage.to_dva()[region]
    # The forward regression, run backwards:
    xc, yc, zc = (np.array(c) for c in FSAVERAGE_CORTEX[(region, False)])
    keep = ~np.isnan(xc)
    xdva, ydva = to_dva(xc[keep], yc[keep], zc[keep])
    npt.assert_allclose(xdva, np.array(FSAVERAGE_POINTS[region][0])[keep],
                        rtol=.05, atol=0.1)
    npt.assert_allclose(ydva, np.array(FSAVERAGE_POINTS[region][1])[keep],
                        rtol=.05, atol=0.1)

    # ... and a whole diagonal of the lower left quadrant, which is far enough
    # from every boundary that the round trip has to close:
    x = y = np.arange(-10, -1, .1)
    xdva, ydva = to_dva(*fsaverage.from_dva()[region](x, y))
    npt.assert_allclose(xdva, x, rtol=.05, atol=0.1)
    npt.assert_allclose(ydva, y, rtol=.05, atol=0.1)


@pytest.mark.slow
def test_fsaverage_scoreboard(fsaverage):
    """A real map still drives a real implant through a real model."""
    model = ScoreboardModel(rho=800, step=.25, vfmap=fsaverage).build()
    implants = [Neuralink.from_neuropythy(fsaverage, xrange=(-3, 3),
                                          yrange=(-3, 3), region=region)
                for region in ['v1', 'v2', 'v3']]
    implant = EnsembleImplant(implants)
    implant.stim = {e: 1 for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    npt.assert_almost_equal(np.sum(percept.data), 20245.445, decimal=1)
    npt.assert_almost_equal(np.max(percept.data), 86.4913, decimal=1)
