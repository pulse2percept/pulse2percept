"""Tests for :class:`~pulse2percept.topography.NeuropythyMap`.

Most behavior is tested against a deterministic toy cortex. Slow tests use one
shared ``fsaverage`` map to exercise the real Neuropythy pipeline.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree
from types import SimpleNamespace

from pulse2percept.implants import ArgusII, EnsembleImplant
from pulse2percept.implants.cortex import Neuralink
from pulse2percept.models import ScoreboardModel as BeyelerScoreboard
from pulse2percept.models.cortex import ScoreboardModel
from pulse2percept.topography import CorticalMap, NeuropythyMap
from pulse2percept.units import DimensionMismatchError, dva, mm, um


TOY_SURFACE_MM = {'white': -1.0, 'midgray': 0.0, 'pial': 1.0}
TOY_MAX_ECC = 5.0


def toy_cortex_mm(x, y, surface):
    """Map toy visual-field coordinates onto a cortical surface."""
    scale = 1 + TOY_SURFACE_MM[surface] / 4
    return np.stack([scale * np.asarray(x, dtype=float),
                     -scale * np.asarray(y, dtype=float),
                     np.full(np.shape(x), TOY_SURFACE_MM[surface])])


class ToyMesh:
    """Minimal visual-field mesh used by the toy map."""

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
    """Minimal FreeSurfer-like surface."""

    def __init__(self, name):
        self.name = name

    def unaddress(self, addr):
        x, y = np.asarray(addr['coordinates'], dtype=float)
        return toy_cortex_mm(x, y, self.name)


class ToyHemisphere:
    """Hemisphere that records requested surfaces."""

    def __init__(self):
        self.surfaces_asked = []

    def surface(self, name):
        self.surfaces_asked.append(name)
        return ToySurface(name)


class ToySubject:
    """Minimal two-hemisphere subject."""

    def __init__(self):
        self.hemis = {'lh': ToyHemisphere(), 'rh': ToyHemisphere()}


class ToyNeuropythyMap(NeuropythyMap):
    """NeuropythyMap backed by deterministic toy meshes and surfaces."""

    def __init__(self, n=6, regions=('v1', 'v2', 'v3'), cache_dir=None,
                 **params):
        # Skip NeuropythyMap.__init__, which requires a real subject.
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
    """Import neuropythy, skipping only when the package is absent."""
    try:
        import neuropythy as ny
    except ModuleNotFoundError as err:
        if err.name == 'neuropythy':
            pytest.skip("requires `pip install neuropythy`")
        raise
    return ny


@pytest.fixture(scope='module')
def fsaverage(neuropythy):
    """Shared real fsaverage map for slow integration tests."""
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
    # Exact vertex hits must not divide by zero (Issue #774).
    verts = nmap.cortex_tree.data * 1000  # mm -> um
    with np.errstate(divide='raise', invalid='raise'):
        xdva, ydva = nmap.cortex_to_dva(verts[:, 0], verts[:, 1], verts[:, 2])
    npt.assert_almost_equal(xdva, np.arange(len(verts)))
    npt.assert_almost_equal(ydva, -np.arange(len(verts)))

    # Halfway between two vertices, both weigh the same:
    xdva, ydva = nmap.cortex_to_dva(np.array([2500.]), np.zeros(1), np.zeros(1))
    npt.assert_almost_equal(xdva, [2.5])
    npt.assert_almost_equal(ydva, [-2.5])

    # The threshold is inclusive.
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

    # Each region dispatches to its own mesh.
    nmap = ToyNeuropythyMap()
    for region, dva_to in [('v1', nmap.dva_to_v1), ('v2', nmap.dva_to_v2),
                           ('v3', nmap.dva_to_v3)]:
        dva_to(1, 1)
        npt.assert_equal([len(m.addressed)
                          for m in nmap.region_meshes[region]], [1, 0])

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
    # Preserve input order and convert mm -> um:
    npt.assert_almost_equal(xc, [-1000, 0, 2000])
    npt.assert_almost_equal(yc, [-1000, -1000, 1000])
    npt.assert_almost_equal(zc, [0, 0, 0])

    # Preserve input shape:
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

    # `cort_nn_thresh` is a distance between mesh vertices, so it is stored in
    # microns however it was handed over:
    npt.assert_equal(ToyNeuropythyMap(cort_nn_thresh=1 * mm).cort_nn_thresh,
                     1000)
    npt.assert_equal(ToyNeuropythyMap(cort_nn_thresh=500 * um).cort_nn_thresh,
                     500)
    with pytest.raises(DimensionMismatchError):
        ToyNeuropythyMap(cort_nn_thresh=1 * dva)


def test_ndim_mixup():
    """A 3D cortical map cannot drive a model that only knows 2D grids."""
    model = BeyelerScoreboard(ArgusII(), vfmap=ToyNeuropythyMap())
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
    """Fake Benson-Winawer subject collection that records downloads."""
    subject_ids = ('fsaverage', 'S1201', 'S1202', 'S1203', 'S1204', 'S1205',
                   'S1206', 'S1207', 'S1208')

    def __init__(self, downloaded):
        self.downloaded = downloaded

    def __getitem__(self, name):
        if name not in self.subject_ids:
            raise KeyError(name)
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


@pytest.mark.parametrize(
    ('subject', 'subject_id'),
    [('S1201', 'S1201'),
     ('s1201', 'S1201')],
)
def test_parse_subject_downloads_benson_winawer(neuropythy, tmp_path,
                                                monkeypatch, subject,
                                                subject_id):
    """Benson-Winawer subjects are downloaded using their canonical ID."""
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
    monkeypatch.setattr(neuropythy, 'data',
                        {'benson_winawer_2018': dataset})

    nmap = ToyNeuropythyMap(cache_dir=str(tmp_path))
    npt.assert_equal(nmap.parse_subject(subject), f'loaded {subject_id}')
    npt.assert_equal(downloaded, [subject_id])
    npt.assert_equal(attempts, [subject, subject_id])


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
    """One thread per valid visual-field location."""
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

# Regression points and expected fsaverage cortical coordinates.
FSAVERAGE_POINTS = {
    'v1': ([1, 1, 0, 0, -1, -1], [1, -1, 1, -1, 1, -1]),
    'v2': ([1, 1, 0, 0, -1, -1], [1, -1, 1, -1, 0, -1]),
    'v3': ([1, 1, 0, 0, -1, -1], [1, -1, 1, -1, 0, -1]),
}
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
    """Build real V1-V3 meshes."""
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
    """Check real fsaverage forward mapping."""
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
    """Check real fsaverage inverse mapping."""
    to_dva = fsaverage.to_dva()[region]
    # The forward regression, run backwards:
    xc, yc, zc = (np.array(c) for c in FSAVERAGE_CORTEX[(region, False)])
    keep = ~np.isnan(xc)
    xdva, ydva = to_dva(xc[keep], yc[keep], zc[keep])
    npt.assert_allclose(xdva, np.array(FSAVERAGE_POINTS[region][0])[keep],
                        rtol=.05, atol=0.1)
    npt.assert_allclose(ydva, np.array(FSAVERAGE_POINTS[region][1])[keep],
                        rtol=.05, atol=0.1)

    # ... and a whole diagonal of the lower left quadrant:
    x = y = np.arange(-10, -1, .1)
    xdva, ydva = to_dva(*fsaverage.from_dva()[region](x, y))
    npt.assert_allclose(xdva, x, rtol=.05, atol=0.1)
    npt.assert_allclose(ydva, y, rtol=.05, atol=0.1)


@pytest.mark.slow
def test_fsaverage_scoreboard(fsaverage):
    """Run one end-to-end real-map model integration."""
    # `meridian_blend=0`: the sums below pin the neuropythy map, not the
    # default postprocessing. The blend is covered by
    # `test_CortexSpatial_meridian_blend`.
    implants = [Neuralink.from_neuropythy(fsaverage, xrange=(-3, 3),
                                          yrange=(-3, 3), region=region)
                for region in ['v1', 'v2', 'v3']]
    implant = EnsembleImplant(implants)
    model = ScoreboardModel(implant=implant, rho=800, step=.25,
                            vfmap=fsaverage, meridian_blend=0)
    percept = model.predict_percept({e: 1 for e in implant.electrode_names})
    npt.assert_almost_equal(np.sum(percept.data), 20245.445, decimal=1)
    npt.assert_almost_equal(np.max(percept.data), 86.4913, decimal=1)
