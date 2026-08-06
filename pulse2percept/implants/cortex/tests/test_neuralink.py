from string import ascii_uppercase

import numpy.testing as npt
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pulse2percept.implants import ProsthesisSystem
from pulse2percept.implants.cortex import (EllipsoidElectrode, LinearEdgeThread,
                                           NeuralinkThread, Neuralink, Cortivis)
from pulse2percept.topography import Grid2D, NeuropythyMap, Polimeni2006Map
from pulse2percept.topography.cortex import CorticalMap


class StubNeuropythyMap(NeuropythyMap):
    """A NeuropythyMap whose dva -> cortex mapping is a simple known formula.

    ``NeuropythyMap.__init__`` needs the optional ``neuropythy`` package plus a
    FreeSurfer subject download, so the real thing is only exercised by the
    ``slow`` tests in ``topography/tests/test_neuropythy.py``. Since
    ``Neuralink.from_neuropythy`` only ever calls ``from_dva()[region]``, a stub
    that implements that one mapping runs the whole factory method for free.

    1 dva maps to 1 mm of cortex, and the cortical surface normal at ``(x, y)``
    points along ``(x, y, -10)``, so that every thread gets its own insertion
    direction. Locations at or beyond ``max_ecc`` are off the map and come back
    as NaN, the way neuropythy reports points it cannot address.
    """
    # Eccentricity (dva) at which the stub map ends:
    max_ecc = 4.0
    # Distance (um) between the pial surface and the midgray surface:
    thickness = 1000.0
    # Each region is a copy of the same map, shifted along x:
    region_offsets = {'v1': 0., 'v2': 50000., 'v3': 100000.}

    def __init__(self, **params):
        # Skip NeuropythyMap.__init__ (needs neuropythy), keep everything else:
        CorticalMap.__init__(self, **params)

    @staticmethod
    def normal(x, y):
        """Unit surface normal (pointing into the cortex) at ``(x, y)``"""
        normal = np.array([x, y, -10.0 * np.ones_like(x)], dtype=float)
        return normal / np.linalg.norm(normal, axis=0)

    def dva_to_cortex(self, x, y, region='v1', hemi=None, surface='midgray'):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        pial = np.array([1000 * x + self.region_offsets[region], 1000 * y,
                         np.zeros_like(x)])
        depth = {'pial': 0., 'midgray': 1., 'white': 2.}[surface]
        points = pial + depth * self.thickness * self.normal(x, y)
        points[:, np.hypot(x, y) >= self.max_ecc] = np.nan
        return points[0], points[1], points[2]

    def dva_to_v1(self, x, y, surface='midgray'):
        return self.dva_to_cortex(x, y, region='v1', surface=surface)

    def dva_to_v2(self, x, y, surface='midgray'):
        return self.dva_to_cortex(x, y, region='v2', surface=surface)

    def dva_to_v3(self, x, y, surface='midgray'):
        return self.dva_to_cortex(x, y, region='v3', surface=surface)


def stub_map_expected(locs, region='v1'):
    """Locations and insertion directions the stub map implies for ``locs``

    Returns the (x, y, z) insertion points and unit insertion directions of the
    threads that ``Neuralink.from_neuropythy`` should build, with the off-map
    locations already dropped.
    """
    locs = np.asarray(locs, dtype=float)
    locs = locs[np.hypot(locs[:, 0], locs[:, 1]) < StubNeuropythyMap.max_ecc]
    x, y = locs[:, 0], locs[:, 1]
    offset = StubNeuropythyMap.region_offsets[region]
    points = np.array([1000 * x + offset, 1000 * y, np.zeros_like(x)]).T
    directions = StubNeuropythyMap.normal(x, y).T
    return points, directions


def excel_names(n):
    """The first ``n`` names in the A, B, ..., Z, AA, AB, ... sequence"""
    names = list(ascii_uppercase)
    for first in ascii_uppercase:
        names += [first + second for second in ascii_uppercase]
    return names[:n]


def angle_between(u, v):
    """Angle (degrees) between two unit vectors"""
    return np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1)))


def test_EllipsoidElectrode():
    electrode = EllipsoidElectrode(0, 1, 2, 3, 4, 5, name='A001')
    npt.assert_almost_equal(electrode.x, 0)
    npt.assert_almost_equal(electrode.y, 1)
    npt.assert_almost_equal(electrode.z, 2)
    npt.assert_almost_equal(electrode.rx, 3)
    npt.assert_almost_equal(electrode.ry, 4)
    npt.assert_almost_equal(electrode.rz, 5)
    npt.assert_equal(electrode.name, 'A001')
    with pytest.raises(TypeError):
        EllipsoidElectrode([0], 1, 2)
    with pytest.raises(TypeError):
        EllipsoidElectrode(0, np.array([1, 2]), 2)
    with pytest.raises(TypeError):
        EllipsoidElectrode(0, 1, [2, 3])
    # Slots:
    npt.assert_equal(hasattr(electrode, '__slots__'), True)
    npt.assert_equal(hasattr(electrode, '__dict__'), False)


def test_EllipsoidElectrode_defaults():
    electrode = EllipsoidElectrode()
    npt.assert_almost_equal((electrode.x, electrode.y, electrode.z), (0, 0, 0))
    npt.assert_almost_equal((electrode.rx, electrode.ry, electrode.rz),
                            (7, 7, 12))
    npt.assert_equal(electrode.activated, True)
    # Defaults to pointing along +z, i.e. no rotation at all:
    npt.assert_almost_equal(electrode.direction, [0, 0, 1])
    npt.assert_almost_equal(electrode.angles, [0, 0, 0])
    npt.assert_almost_equal(electrode.rot, np.eye(3))


@pytest.mark.parametrize('orient, orient_mode, direction', [
    ([1, 0, 0], 'direction', [1, 0, 0]),
    ([0, 3, 0], 'direction', [0, 1, 0]),      # normalized for us
    ([0, 90, 0], 'angle', [1, 0, 0]),         # 90deg about y takes +z to +x
    ([90, 0, 0], 'angle', [0, -1, 0]),        # 90deg about x takes +z to -y
])
def test_EllipsoidElectrode_orient(orient, orient_mode, direction):
    electrode = EllipsoidElectrode(orient=orient, orient_mode=orient_mode)
    npt.assert_almost_equal(electrode.direction, direction)
    # A rotation matrix takes +z to the same place:
    rotated = EllipsoidElectrode(orient=electrode.rot, orient_mode='rot')
    npt.assert_almost_equal(rotated.direction, direction)
    npt.assert_almost_equal(rotated.rot, electrode.rot)


def test_EllipsoidElectrode_invalid_orient():
    with pytest.raises(TypeError):
        EllipsoidElectrode(orient=[0, 0])
    with pytest.raises(ValueError):
        # Cannot normalize the zero vector:
        EllipsoidElectrode(orient=[0, 0, 0], orient_mode='direction')
    with pytest.raises(ValueError):
        EllipsoidElectrode(orient=[0, 0, 1], orient_mode='invalid')
    with pytest.raises(ValueError):
        # A rotation matrix needs orient_mode='rot':
        EllipsoidElectrode(orient=np.eye(3), orient_mode='direction')
    with pytest.raises(ValueError):
        EllipsoidElectrode(orient=np.ones((3, 3)), orient_mode='rot')


def test_EllipsoidElectrode_plot_kwargs():
    electrode = EllipsoidElectrode(rx=3, ry=4, rz=5)
    npt.assert_equal(electrode.plot_patch, Ellipse)
    # The 2D patch is the ellipse's cross-section:
    for kwargs in (electrode.plot_kwargs, electrode.plot_deactivated_kwargs):
        npt.assert_almost_equal(kwargs['width'], 3)
        npt.assert_almost_equal(kwargs['height'], 4)
    # Deactivated electrodes are drawn lighter and more transparent:
    npt.assert_equal(electrode.plot_deactivated_kwargs['ec'][0] >
                     electrode.plot_kwargs['ec'][0], True)
    npt.assert_equal(electrode.plot_deactivated_kwargs['fc'][-1] <
                     electrode.plot_kwargs['fc'][-1], True)
    npt.assert_equal(electrode.plot_3d_kwargs['color'], 'yellow')


def test_EllipsoidElectrode_electric_potential():
    with pytest.raises(NotImplementedError):
        EllipsoidElectrode().electric_potential(0, 0, 0, 1)


def test_EllipsoidElectrode_pprint():
    electrode = EllipsoidElectrode(0, 1, 2, 3, 4, 5, name='A001')
    params = electrode._pprint_params()
    npt.assert_equal(params['rx'], 3)
    npt.assert_equal(params['ry'], 4)
    npt.assert_equal(params['rz'], 5)
    npt.assert_almost_equal(params['angles'], electrode.angles)
    # Inherited from Electrode:
    npt.assert_equal(params['name'], 'A001')
    npt.assert_equal('EllipsoidElectrode' in str(electrode), True)


def test_LinearEdgeThread():
    thread = LinearEdgeThread()
    npt.assert_almost_equal(thread.x, 0)
    npt.assert_almost_equal(thread.y, 0)
    npt.assert_almost_equal(thread.z, 0)

    # elecs arent actually at this spot, but are on the edge, a few microns off
    zs = []
    for e in thread.electrode_objects:
        npt.assert_almost_equal(e.x, thread.r + 7 // 2)
        npt.assert_almost_equal(e.y, 0)
        npt.assert_almost_equal(e.rot, thread.rot)
        zs.append(e.z)
    npt.assert_equal(np.allclose(np.diff(zs), thread.spacing), True)


    thread = LinearEdgeThread(orient=[1, 0, 0])
    xs = []
    for e in thread.electrode_objects:
        npt.assert_almost_equal(e.z, -thread.r - 7 // 2)
        npt.assert_almost_equal(e.y, 0)
        xs.append(e.x)
    npt.assert_equal(np.allclose(np.diff(xs), thread.spacing), True)

    thread = LinearEdgeThread(orient=[1, 1, 1], spacing=3*np.sqrt(3))
    locs = []
    for i, e in enumerate(thread.electrode_objects):
        npt.assert_almost_equal(e.x, 3*i + 4.618802, decimal=5)
        npt.assert_almost_equal(e.y, 3*i + 4.618802, decimal=5)
        npt.assert_almost_equal(e.z, 3*i - 4.618802, decimal=5)
        locs.append([e.x, e.y, e.z])
    npt.assert_equal(np.allclose(np.diff(locs, axis=0), 3), True)


def test_LinearEdgeThread_defaults():
    thread = LinearEdgeThread()
    npt.assert_equal(isinstance(thread, NeuralinkThread), True)
    npt.assert_almost_equal(thread.loc, [0, 0, 0])
    npt.assert_almost_equal(thread.r, 5)
    npt.assert_equal(thread.n_elecs, 32)
    npt.assert_almost_equal(thread.spacing, 50)
    npt.assert_almost_equal(thread.insertion_depth, 0)
    npt.assert_equal(thread.electrode, EllipsoidElectrode)
    npt.assert_equal(thread.stim, None)
    npt.assert_equal(thread.safe_mode, False)
    npt.assert_equal(thread.preprocess, False)
    # The thread also sticks out of the cortex, for visualization:
    npt.assert_almost_equal(thread.extracortical_depth, 1000)
    npt.assert_almost_equal(thread.thread_length,
                            32 * 50 + 1000 + 0)
    # One electrode per contact, named by index:
    npt.assert_equal(thread.n_electrodes, 32)
    npt.assert_equal(thread.electrode_names, [str(i) for i in range(32)])
    npt.assert_equal([isinstance(e, EllipsoidElectrode)
                      for e in thread.electrode_objects], [True] * 32)


def test_LinearEdgeThread_geometry():
    thread = LinearEdgeThread(10, 20, 30, n_elecs=4, spacing=25,
                              insertion_depth=100, r=8)
    npt.assert_almost_equal(thread.thread_length, 4 * 25 + 1000 + 100)
    npt.assert_equal(thread.n_electrodes, 4)
    # Default orientation is +z, so electrodes start `insertion_depth` below
    # the insertion point and are offset onto the edge of the thread:
    edge_offset = 8 + 7 // 2
    for i, e in enumerate(thread.electrode_objects):
        npt.assert_almost_equal(e.x, 10 + edge_offset)
        npt.assert_almost_equal(e.y, 20)
        npt.assert_almost_equal(e.z, 30 + 100 + i * 25)


def test_LinearEdgeThread_custom_electrode():
    class BigEllipsoid(EllipsoidElectrode):
        def __init__(self, x=0, y=0, z=0, **kwargs):
            super().__init__(x, y, z, rx=20, ry=21, rz=22, **kwargs)

    thread = LinearEdgeThread(n_elecs=3, electrode=BigEllipsoid)
    npt.assert_equal(thread.electrode, BigEllipsoid)
    for e in thread.electrode_objects:
        npt.assert_equal(isinstance(e, BigEllipsoid), True)
        npt.assert_almost_equal((e.rx, e.ry, e.rz), (20, 21, 22))
        # The thread still hands the electrodes its own orientation:
        npt.assert_almost_equal(e.rot, thread.rot)


def test_LinearEdgeThread_stim():
    thread = LinearEdgeThread(n_elecs=3, stim={'0': 1, '2': 3})
    npt.assert_equal(thread.stim.electrodes, ['0', '2'])
    npt.assert_almost_equal(thread.stim.data.ravel(), [1, 3])
    # safe_mode rejects stimuli that are not charge-balanced:
    with pytest.raises(ValueError):
        LinearEdgeThread(n_elecs=3, safe_mode=True, stim={'0': 1})


def test_LinearEdgeThread_pprint():
    thread = LinearEdgeThread(1, 2, 3, n_elecs=4, spacing=25, r=8)
    params = thread._pprint_params()
    npt.assert_equal(params['location'], (1, 2, 3))
    npt.assert_almost_equal(params['angles'], thread.angles)
    npt.assert_equal(params['r'], 8)
    npt.assert_equal(params['n_elecs'], 4)
    npt.assert_equal(params['spacing'], 25)
    # Inherited from ProsthesisSystem:
    npt.assert_equal(params['stim'], None)
    npt.assert_equal('LinearEdgeThread' in str(thread), True)


def test_Neuralink():
    t1 = LinearEdgeThread(orient=[1, 0, 0])
    t2 = LinearEdgeThread(500, 500, orient=[0, 1, 0])
    nlink = Neuralink([t1, t2])

    # check that positions are the same
    npt.assert_equal(nlink['0-1'].x, t1['1'].x)
    npt.assert_equal(nlink['0-1'].y, t1['1'].y)
    npt.assert_equal(nlink['1-1'].x, t2['1'].x)
    npt.assert_equal(nlink['1-1'].y, t2['1'].y)


def test_Neuralink_from_dict():
    t1 = LinearEdgeThread(n_elecs=2)
    t2 = LinearEdgeThread(500, 500, n_elecs=2)
    nlink = Neuralink({'A': t1, 'B': t2})
    npt.assert_equal(list(nlink.implants.keys()), ['A', 'B'])
    npt.assert_equal(nlink.electrode_names, ['A-0', 'A-1', 'B-0', 'B-1'])
    npt.assert_equal(nlink['B-0'].x, t2['0'].x)


def test_Neuralink_requires_threads():
    thread = LinearEdgeThread(n_elecs=2)
    # Neither a list nor a dict may hold anything but NeuralinkThreads:
    with pytest.raises(TypeError):
        Neuralink([thread, Cortivis()])
    with pytest.raises(TypeError):
        Neuralink({'A': thread, 'B': Cortivis()})
    with pytest.raises(TypeError):
        Neuralink([thread, 'not a thread'])
    with pytest.raises(TypeError):
        Neuralink({'A': None})


def test_Neuralink_stim():
    nlink = Neuralink({'A': LinearEdgeThread(n_elecs=2),
                       'B': LinearEdgeThread(500, 500, n_elecs=2)},
                      stim={'A-0': 1, 'B-1': 2})
    npt.assert_equal(nlink.stim.electrodes, ['A-0', 'B-1'])
    npt.assert_almost_equal(nlink.stim.data.ravel(), [1, 2])
    with pytest.raises(ValueError):
        Neuralink([LinearEdgeThread(n_elecs=2)], safe_mode=True,
                  stim={'0-0': 1})


def _ax3d():
    fig = plt.figure()
    return fig.add_subplot(111, projection='3d')


@pytest.mark.parametrize('make_obj', [
    pytest.param(lambda: EllipsoidElectrode(0, 1, 2, 3, 4, 5, name='A001'),
                 id='EllipsoidElectrode'),
    pytest.param(lambda: LinearEdgeThread(0, 0, 0), id='LinearEdgeThread'),
    pytest.param(lambda: Neuralink([LinearEdgeThread(0, 0, 0),
                                    LinearEdgeThread(100, 0, 0)]),
                 id='Neuralink'),
])
def test_plot3D(make_obj):
    obj = make_obj()

    # Plots onto a given 3D axis:
    plt.close('all')
    ax = _ax3d()
    npt.assert_equal(obj.plot3D(ax=ax) is not None, True)

    # Creates its own 3D axis when none is given:
    plt.close('all')
    npt.assert_equal(obj.plot3D() is not None, True)

    # ... and honors `figsize` when it does:
    plt.close('all')
    ax = obj.plot3D(figsize=(8, 6))
    npt.assert_almost_equal(ax.figure.get_size_inches(), (8, 6))

    # A 2D axis is rejected:
    plt.close('all')
    _, ax2d = plt.subplots()
    with pytest.raises(ValueError):
        obj.plot3D(ax=ax2d)
    plt.close('all')


@pytest.mark.parametrize('make_obj', [
    pytest.param(lambda: EllipsoidElectrode(0, 1, 2, 3, 4, 5, name='A001'),
                 id='EllipsoidElectrode'),
    pytest.param(lambda: LinearEdgeThread(0, 0, 0, n_elecs=2),
                 id='LinearEdgeThread'),
    pytest.param(lambda: Neuralink([LinearEdgeThread(0, 0, 0, n_elecs=2)]),
                 id='Neuralink'),
])
def test_plot3D_reuses_existing_3d_axis(make_obj):
    # An existing 3D axis is drawn onto rather than replaced:
    plt.close('all')
    ax = _ax3d()
    npt.assert_equal(make_obj().plot3D() is ax, True)
    plt.close('all')


def test_plot3D_surfaces():
    # The thread draws its own shaft plus one surface per electrode:
    plt.close('all')
    ax = LinearEdgeThread(n_elecs=3).plot3D()
    npt.assert_equal(len(ax.collections), 4)

    # ... and the implant draws every thread:
    plt.close('all')
    ax = Neuralink([LinearEdgeThread(n_elecs=3),
                    LinearEdgeThread(500, 0, 0, n_elecs=3)]).plot3D()
    npt.assert_equal(len(ax.collections), 8)
    plt.close('all')


def test_Neuralink_from_neuropythy_requires_neuropythy_map():
    # The vfmap must be a NeuropythyMap; this guard runs before any dataset
    # is touched, so it is testable without neuropythy installed:
    from pulse2percept.topography import Watson2014Map
    with pytest.raises(TypeError):
        Neuralink.from_neuropythy(Watson2014Map())
    with pytest.raises(TypeError):
        Neuralink.from_neuropythy(Polimeni2006Map())


def test_Neuralink_from_neuropythy_locs():
    # The last location is off the map and must be dropped:
    locs = np.array([[1., 2.], [-2., 1.], [0., 0.], [3.5, 3.5]])
    nlink = Neuralink.from_neuropythy(StubNeuropythyMap(), locs=locs)

    points, directions = stub_map_expected(locs)
    npt.assert_equal(len(nlink.implants), 3)
    npt.assert_equal(list(nlink.implants.keys()), ['A', 'B', 'C'])
    for thread, point, direction in zip(nlink.implants.values(), points,
                                        directions):
        npt.assert_equal(isinstance(thread, LinearEdgeThread), True)
        npt.assert_almost_equal((thread.x, thread.y, thread.z), point)
        # Threads are inserted perpendicular to the cortical surface:
        npt.assert_almost_equal(thread.direction, direction)
    # The electrodes of all threads end up in one array:
    npt.assert_equal(nlink.n_electrodes, 3 * 32)


def test_Neuralink_from_neuropythy_default_grid():
    # Without locs, threads are placed on a Grid2D covering (-3, 3) dva at a
    # 1 dva spacing, minus the corners that fall off the stub map:
    nlink = Neuralink.from_neuropythy(StubNeuropythyMap())

    grid = Grid2D((-3, 3), (-3, 3), 1)
    locs = np.stack([grid.x.flatten(), grid.y.flatten()], axis=1)
    points, directions = stub_map_expected(locs)

    npt.assert_equal(len(nlink.implants), len(points))
    # More than 26 threads, so the names wrap around to AA, AB, ...:
    npt.assert_equal(len(points) > 26, True)
    npt.assert_equal(list(nlink.implants.keys()), excel_names(len(points)))
    for thread, point, direction in zip(nlink.implants.values(), points,
                                        directions):
        npt.assert_almost_equal((thread.x, thread.y, thread.z), point)
        npt.assert_almost_equal(thread.direction, direction)


def test_Neuralink_from_neuropythy_grid_args():
    nlink = Neuralink.from_neuropythy(StubNeuropythyMap(), xrange=(-1, 1),
                                      yrange=(0, 2), xystep=1)
    grid = Grid2D((-1, 1), (0, 2), 1)
    locs = np.stack([grid.x.flatten(), grid.y.flatten()], axis=1)
    points, _ = stub_map_expected(locs)
    npt.assert_equal(len(nlink.implants), len(points))
    npt.assert_almost_equal([[t.x, t.y, t.z] for t in nlink.implants.values()],
                            points)


@pytest.mark.parametrize('region', ['v1', 'v2', 'v3'])
def test_Neuralink_from_neuropythy_region(region):
    # Each region is a differently offset copy of the same map, so the threads
    # land somewhere else depending on which one is asked for:
    locs = np.array([[1., 1.], [-1., 2.]])
    nlink = Neuralink.from_neuropythy(
        StubNeuropythyMap(regions=['v1', 'v2', 'v3']), locs=locs, region=region)
    points, _ = stub_map_expected(locs, region=region)
    npt.assert_almost_equal([[t.x, t.y, t.z] for t in nlink.implants.values()],
                            points)


def test_Neuralink_from_neuropythy_unmapped_region():
    # A region the map was not built for is not silently ignored:
    locs = np.array([[1., 1.], [-1., 2.]])
    with pytest.raises(KeyError):
        Neuralink.from_neuropythy(StubNeuropythyMap(), locs=locs, region='v2')


def test_Neuralink_from_neuropythy_custom_thread():
    class ShortThread(LinearEdgeThread):
        def __init__(self, x=0, y=0, z=0, **kwargs):
            super().__init__(x, y, z, n_elecs=4, **kwargs)

    nlink = Neuralink.from_neuropythy(StubNeuropythyMap(),
                                      locs=np.array([[1., 1.], [-1., -1.]]),
                                      Thread=ShortThread)
    for thread in nlink.implants.values():
        npt.assert_equal(isinstance(thread, ShortThread), True)
        npt.assert_equal(thread.n_electrodes, 4)


def test_Neuralink_from_neuropythy_rand_insertion_angle():
    locs = np.array([[1., 2.], [-2., 1.], [0., 1.], [2., -2.]])
    _, perpendicular = stub_map_expected(locs)

    np.random.seed(0)
    nlink = Neuralink.from_neuropythy(StubNeuropythyMap(), locs=locs,
                                      rand_insertion_angle=20)
    offsets = [angle_between(t.direction, d)
               for t, d in zip(nlink.implants.values(), perpendicular)]
    # Every thread is tilted, but never by more than the requested angle:
    npt.assert_equal(np.all(np.less_equal(offsets, 20)), True)
    npt.assert_equal(np.all(np.greater(offsets, 0)), True)
    # Insertion points are unaffected by the tilt:
    points, _ = stub_map_expected(locs)
    npt.assert_almost_equal([[t.x, t.y, t.z] for t in nlink.implants.values()],
                            points)

    # An angle of 0 leaves the threads perpendicular:
    nlink = Neuralink.from_neuropythy(StubNeuropythyMap(), locs=locs,
                                      rand_insertion_angle=0)
    npt.assert_almost_equal([t.direction for t in nlink.implants.values()],
                            perpendicular)


def test_Neuralink_from_neuropythy_surface_mismatch():
    class HoleyStubMap(StubNeuropythyMap):
        """A map where a location is on the pial but not the midgray surface"""
        def dva_to_v1(self, x, y, surface='midgray'):
            xc, yc, zc = super().dva_to_v1(x, y, surface=surface)
            if surface == 'midgray':
                xc[0] = np.nan
            return xc, yc, zc

    with pytest.raises(ValueError):
        Neuralink.from_neuropythy(HoleyStubMap(),
                                  locs=np.array([[1., 1.], [-1., -1.]]))


def test_Neuralink_from_cortical_map_requires_thread():
    # Only NeuralinkThreads can go into a Neuralink:
    for implant_type in (Cortivis, ProsthesisSystem):
        with pytest.raises(TypeError):
            Neuralink.from_cortical_map(implant_type, Polimeni2006Map())


def test_Neuralink_from_cortical_map_non_neuropythy():
    # A plain CorticalMap falls through to EnsembleImplant.from_cortical_map,
    # which just centers a thread on each cortical location:
    vfmap = Polimeni2006Map()
    nlink = Neuralink.from_cortical_map(LinearEdgeThread, vfmap,
                                        xrange=(-1, 1), yrange=(0, 0),
                                        xystep=1)
    npt.assert_equal(isinstance(nlink, Neuralink), True)
    xc, yc = vfmap.dva_to_v1(np.array([-1., 0., 1.]), np.array([0., 0., 0.]))
    npt.assert_equal(len(nlink.implants), 3)
    for thread, x, y in zip(nlink.implants.values(), xc, yc):
        npt.assert_almost_equal(thread.x, x, decimal=3)
        npt.assert_almost_equal(thread.y, y, decimal=3)
        # No 3D map, so the threads stay at the default depth/orientation:
        npt.assert_almost_equal(thread.z, 0)
        npt.assert_almost_equal(thread.direction, [0, 0, 1])


def test_Neuralink_from_cortical_map_neuropythy():
    # A NeuropythyMap is instead routed to from_neuropythy, which knows about
    # the third dimension and the insertion angle:
    locs = np.array([[1., 2.], [-2., 1.]])
    nlink = Neuralink.from_cortical_map(LinearEdgeThread, StubNeuropythyMap(),
                                        locs=locs)
    points, directions = stub_map_expected(locs)
    npt.assert_equal(list(nlink.implants.keys()), ['A', 'B'])
    npt.assert_almost_equal([[t.x, t.y, t.z] for t in nlink.implants.values()],
                            points)
    npt.assert_almost_equal([t.direction for t in nlink.implants.values()],
                            directions)
