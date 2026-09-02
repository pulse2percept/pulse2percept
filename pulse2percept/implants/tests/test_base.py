import numpy as np
import collections as coll
import pytest
import numpy.testing as npt
from pulse2percept import implants
from pulse2percept.implants import cortex
from pulse2percept.units import (DimensionMismatchError, Quantity, deg,
                                 dimensionless, dva, mA, mm, ms, nA, rad, uA,
                                 um, xTh)
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

from pulse2percept.implants import (PointSource, ElectrodeArray, ElectrodeGrid,
                                    GridImplant, Implant,
                                    RectangleImplant, PhotovoltaicPixel)
from pulse2percept.stimuli import (Stimulus, ImageStimulus, VideoStimulus,
                                   BostonTrain, LogoBVL)
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BiphasicPulseTrain, FrequencyEncoder,
                                   MonophasicPulse)
from pulse2percept.implants import (ArgusII, DiskElectrode)
from pulse2percept.models import ScoreboardModel


class PhotovoltaicArray(Implant):
    def __init__(self, x=0, y=0, z=-100, r=5, spacing=40, rot=0,
                 preprocess=False, safe_mode=False):
        # 35 um pixels with 5 um trenches, 16 um active electrode:
        self.spacing = spacing  # um
        self.trench = 5  # um
        elec_radius = 8  # um
        self.shape = (int(r * 600 / spacing), int(r * 700 / spacing))
        self.eye = 'RE'
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        dva2ret = 280.0

        self.earray = ElectrodeGrid(self.shape, spacing, x=x, y=y, z=z,
                                    rot=rot, grid_type='hex',
                                    orientation='vertical',
                                    electrode_type=PhotovoltaicPixel,
                                    radius=elec_radius,
                                    apothem=(self.spacing - self.trench) / 2)

        rm_names = []
        for name, electrode in self.earray.electrodes.items():
            if (electrode.x - x) ** 2 + (electrode.y - y) ** 2 > (r * dva2ret) ** 2:
                rm_names.append(name)
        for e in rm_names:
            self.earray.remove_electrode(e)


def test_Implant():
    # Invalid instantiations:
    with pytest.raises(ValueError):
        Implant(ElectrodeArray(PointSource(0, 0, 0)), eye='both')
    with pytest.raises(TypeError):
        Implant(Stimulus)

    # Iterating over the electrode array:
    implant = Implant(PointSource(0, 0, 0))
    npt.assert_equal(implant.n_electrodes, 1)
    npt.assert_equal(implant[0], implant.earray[0])
    npt.assert_equal(implant.electrode_names, implant.earray.electrode_names)
    for i, e in zip(implant, implant.earray):
        npt.assert_equal(i, e)

    # Prepare a stimulus:
    stim = implant.prepare_stim(3)
    npt.assert_equal(isinstance(stim, Stimulus), True)
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.time, None)
    npt.assert_equal(stim.electrodes, [0])

    plt.cla()
    ax = implant.plot()
    npt.assert_equal(len(ax.texts), 0)
    npt.assert_equal(len(ax.collections), 1)

    with pytest.raises(ValueError):
        # Wrong number of stimuli
        implant.prepare_stim([1, 2])
    with pytest.raises(TypeError):
        # Invalid stim type:
        implant.prepare_stim("stim")
    # Invalid electrode names:
    with pytest.raises(ValueError):
        implant.prepare_stim({'A1': 1})
    with pytest.raises(ValueError):
        implant.prepare_stim(Stimulus({'A1': 1}))
    # Safe mode requires charge-balanced pulses:
    with pytest.raises(ValueError):
        implant = Implant(PointSource(0, 0, 0), safe_mode=True)
        implant.prepare_stim(1)

    # Slots:
    npt.assert_equal(hasattr(implant, '__slots__'), True)
    npt.assert_equal(hasattr(implant, '__dict__'), False)


def test_Implant_prepare_stim():
    implant = Implant(ElectrodeGrid((13, 13), 20))
    with pytest.raises(ValueError):
        implant.prepare_stim(Stimulus(np.ones((13 * 13 + 1, 5))))

    # make sure an empty source prepares to None
    npt.assert_equal(implant.prepare_stim(None), None)
    npt.assert_equal(implant.prepare_stim([]), None)
    npt.assert_equal(implant.prepare_stim({}), None)
    npt.assert_equal(implant.prepare_stim(np.array([])), None)

    # color mapping
    source = np.zeros((13*13, 5))
    source[84, 0] = 1
    source[98, 2] = 2
    plt.cla()
    ax = implant.plot(stim=source, stim_cmap='hsv')
    plt.colorbar()
    npt.assert_equal(len(ax.collections), 1)
    npt.assert_equal(ax.collections[0].colorbar.vmax, 2)
    npt.assert_equal(ax.collections[0].cmap(ax.collections[0].norm(1)),
                     (0.0, 1.0, 0.9647031631761764, 1))
    # `stim_cmap` has nothing to color without a stimulus to prepare:
    with pytest.raises(ValueError):
        implant.plot(stim_cmap='hsv')
    # make sure default behaviour unchanged
    plt.cla()
    ax = implant.plot()
    plt.colorbar()
    npt.assert_equal(len(ax.collections), 1)
    npt.assert_equal(ax.collections[0].colorbar.vmax, 1)
    npt.assert_equal(ax.collections[0].cmap(ax.collections[0].norm(1)),
                     (0.993248, 0.906157, 0.143936, 1))

    # Deactivated electrodes cannot receive stimuli:
    implant.deactivate('H4')
    npt.assert_equal(implant['H4'].activated, False)
    npt.assert_equal('H4' in implant.prepare_stim({'H4': 1}).electrodes, False)

    implant.deactivate('all')
    npt.assert_equal(implant.prepare_stim(source).data.size == 0, True)
    implant.activate('all')
    npt.assert_equal('H4' in implant.prepare_stim({'H4': 1}).electrodes, True)


def test_Implant_prepare_stim_is_stateless():
    """Preparing leaves neither the implant nor the caller's source changed"""
    implant = ArgusII(preprocess=False)
    source = Stimulus({'A1': 10, 'B2': 20})
    first = implant.prepare_stim(source)
    # The result is a copy, so the caller can keep using their own object:
    npt.assert_equal(first is source, False)
    npt.assert_almost_equal(first.data, source.data)

    # Two calls with different input do not leak into one another:
    second = implant.prepare_stim({'C3': 30})
    npt.assert_equal(sorted(str(e) for e in first.electrodes), ['A1', 'B2'])
    npt.assert_equal([str(e) for e in second.electrodes], ['C3'])
    npt.assert_equal(hasattr(implant, 'stim'), False)
    npt.assert_equal(hasattr(implant, '_stim'), False)

    # And the same source prepares to the same thing every time:
    again = implant.prepare_stim(source)
    npt.assert_almost_equal(again.data, first.data)


@pytest.mark.parametrize('rot', (0, 30, 92))
@pytest.mark.parametrize('gtype', ('hex', 'rect'))
@pytest.mark.parametrize('n_frames', (1, 3, 4))
def test_Implant_reshape_stim(rot, gtype, n_frames):
    implant = Implant(ElectrodeGrid((10, 10), 30, rot=rot, grid_type=gtype))
    # Smoke test the reshaping. It runs inside `prepare_stim`, but
    # a picture is not a stimulus an implant can deliver, so it is exercised
    # directly here (which is also how an encoder reaches it):
    n_px = 21
    reshaped = implant.reshape_stim(
        ImageStimulus(np.ones((n_px, n_px, n_frames)).squeeze()))
    npt.assert_equal(reshaped.data.shape, (implant.n_electrodes, 1))
    npt.assert_equal(reshaped.time, None)
    reshaped = implant.reshape_stim(
        VideoStimulus(np.ones((n_px, n_px, 3 * n_frames)),
                      time=2 * np.arange(3 * n_frames)))
    npt.assert_equal(reshaped.data.shape,
                     (implant.n_electrodes, 3 * n_frames))
    npt.assert_equal(reshaped.time, 2 * np.arange(3 * n_frames))

    # Verify that a horizontal stimulus will always appear horizontally, even if
    # the device is rotated. What is under test is where `reshape_stim` puts
    # the pixels, so the sampled gray levels are handed to the model as an
    # ordinary electrical stimulus rather than encoded -- a model reads
    # current, and the one-uA-per-gray-level reading has to be written down:
    data = np.zeros((50, 50))
    data[20:-20, 10:-10] = 1
    sampled = implant.reshape_stim(ImageStimulus(data))
    stim = Stimulus(sampled.data, electrodes=sampled.electrodes)
    model = ScoreboardModel(implant=implant, xrange=(-1, 1), yrange=(-1, 1),
                            rho=30, step=0.02)
    model.build()
    percept = label(model.predict_percept(stim).data.squeeze().T > 0.2)
    npt.assert_almost_equal(regionprops(percept)[0].orientation, 0, decimal=1)

    # Smoke test a large hex grid (old code results in MemoryError):
    implant = PhotovoltaicArray(r=2, spacing=40, rot=rot)
    implant.reshape_stim(LogoBVL())


def test_Implant_deactivate():
    implant = Implant(ElectrodeGrid((10, 10), 30))
    source = np.ones(implant.n_electrodes)
    electrode = 'A3'
    npt.assert_equal(electrode in implant.prepare_stim(source).electrodes, True)
    implant.deactivate(electrode)
    npt.assert_equal(implant[electrode].activated, False)
    # Deactivating affects the next preparation, not one already handed out:
    npt.assert_equal(electrode in implant.prepare_stim(source).electrodes,
                     False)

@pytest.mark.parametrize('ztype', ('float', 'list'))
@pytest.mark.parametrize('x', (-100, 200))
@pytest.mark.parametrize('y', (-200, 400))
@pytest.mark.parametrize('rot', (-45, 60))
def test_rectangle_implant(ztype, x, y, rot):
    # Create an argus like implant and make sure location is correct
    z = 100 if ztype == 'float' else np.ones(60) * 20
    implant = RectangleImplant(x=x, y=y, z=z, rot=rot, shape=(6, 10), r=112.5, spacing=575.0)

    # Slots:
    npt.assert_equal(hasattr(implant, '__slots__'), True)

    # Coordinates of first electrode
    xy = np.array([-2587.5, -1437.5]).T

    # Rotate
    rot_rad = np.deg2rad(rot)
    R = np.array([np.cos(rot_rad), -np.sin(rot_rad),
                  np.sin(rot_rad), np.cos(rot_rad)]).reshape((2, 2))
    xy = np.matmul(R, xy)

    # Then off-set: Make sure first electrode is placed
    # correctly
    npt.assert_almost_equal(implant['A1'].x, xy[0] + x)
    npt.assert_almost_equal(implant['A1'].y, xy[1] + y)

    # Make sure array center is still (x,y)
    y_center = implant['F1'].y + (implant['A10'].y - implant['F1'].y) / 2
    npt.assert_almost_equal(y_center, y)
    x_center = implant['A1'].x + (implant['F10'].x - implant['A1'].x) / 2
    npt.assert_almost_equal(x_center, x)

    # Make sure radius is correct
    for e in ['A1', 'B3', 'C5', 'D7', 'E9', 'F10']:
        npt.assert_almost_equal(implant[e].radius, 112.5)

    # Indexing must work for both integers and electrode names
    for idx, (name, electrode) in enumerate(implant.electrodes.items()):
        npt.assert_equal(electrode, implant[idx])
        npt.assert_equal(electrode, implant[name])
    npt.assert_equal(implant["unlikely name for an electrode"], None)

    # Right-eye implant:
    xc, yc = 500, -500
    implant = RectangleImplant(eye='RE', x=xc, y=yc)
    npt.assert_equal(implant['A10'].x > implant['A1'].x, True)
    npt.assert_almost_equal(implant['A10'].y, implant['A1'].y)

    # Left-eye implant:
    implant = RectangleImplant(eye='LE', x=xc, y=yc)
    npt.assert_equal(implant['A1'].x > implant['A10'].x, True)
    npt.assert_almost_equal(implant['A10'].y, implant['A1'].y)

    # In both left and right eyes, rotation with positive angle should be
    # counter-clock-wise (CCW): for (x>0,y>0), decreasing x and increasing y
    for eye, el in zip(['LE', 'RE'], ['O1', 'O15']):
        # By default, electrode 'F1' in a left eye has the same coordinates as
        # 'F10' in a right eye (because the columns are reversed). Thus both
        # cases are testing an electrode with x>0, y>0:
        before = RectangleImplant(eye=eye)
        after = RectangleImplant(eye=eye, rot=20)
        npt.assert_equal(after[el].x < before[el].x, True)
        npt.assert_equal(after[el].y > before[el].y, True)

    # Prepare a stimulus via dict:
    stim = RectangleImplant().prepare_stim({'B7': 13})
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.electrodes, ['B7'])

    # Prepare a stimulus via array:
    stim = RectangleImplant().prepare_stim(np.ones(225))
    npt.assert_equal(stim.shape, (225, 1))
    npt.assert_almost_equal(stim.data, 1)

    # test different shapes
    for shape in [(6, 10), (5, 12), (15, 15)]:
        implant = RectangleImplant(shape=shape)
        npt.assert_equal(implant.earray.shape, shape)


def test_RectangleImplant_is_deprecated():
    """Deprecated in favor of GridImplant, but otherwise unchanged"""
    with pytest.deprecated_call(match='drop-in replacement'):
        implant = RectangleImplant(shape=(3, 4), spacing=100)
    # The legacy defaults and geometry survive the deprecation:
    npt.assert_equal(implant.preprocess, True)
    npt.assert_equal(implant.earray.shape, (3, 4))
    npt.assert_equal(isinstance(implant['A1'], DiskElectrode), True)
    npt.assert_almost_equal(implant['A1'].radius, 75.)
    # Including the left-eye column reversal that GridImplant does not do
    # (see test_GridImplant_does_not_relabel_the_left_eye):
    with pytest.deprecated_call():
        le = RectangleImplant(shape=(3, 4), spacing=100, eye='LE')
    npt.assert_equal(le['A1'].x > le['A4'].x, True)
    npt.assert_almost_equal(le['A1'].x, implant['A4'].x)


def test_ProsthesisSystem_is_a_deprecated_alias():
    """Renamed to Implant in 0.11.0; the old name is the same class"""
    for module in (implants, implants.base):
        with pytest.deprecated_call(match='Use ``Implant``'):
            alias = module.ProsthesisSystem
        npt.assert_equal(alias is implants.Implant, True)
    # An alias, not a subclass, so existing type checks still hold:
    npt.assert_equal(isinstance(ArgusII(), alias), True)
    npt.assert_equal(issubclass(GridImplant, alias), True)
    npt.assert_equal(alias(PointSource(0, 0, 0)).n_electrodes, 1)
    with pytest.raises(AttributeError):
        implants.NotAnImplant


def test_GridImplant_is_a_grid_in_an_implant():
    implant = GridImplant((3, 4), 100)
    npt.assert_equal(isinstance(implant, Implant), True)
    npt.assert_equal(isinstance(implant.earray, ElectrodeGrid), True)
    npt.assert_equal(implant.n_electrodes, 12)
    npt.assert_equal(implant.earray.shape, (3, 4))
    npt.assert_equal(implant.electrode_names,
                     ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4',
                      'C1', 'C2', 'C3', 'C4'])
    npt.assert_equal(isinstance(implant['A1'], PointSource), True)
    # Centered on the origin, 100 um apart:
    npt.assert_almost_equal(implant['A1'].x, -150)
    npt.assert_almost_equal(implant['A1'].y, -100)
    npt.assert_almost_equal(implant['C4'].x, 150)
    npt.assert_almost_equal(implant['C4'].y, 100)
    # `shape`/`spacing` are required; there is no default geometry:
    with pytest.raises(TypeError):
        GridImplant()
    with pytest.raises(TypeError):
        GridImplant((3, 4))


def test_GridImplant_hex():
    implant = GridImplant((3, 4), 100, grid_type='hex')
    npt.assert_equal(implant.earray.grid_type, 'hex')
    npt.assert_equal(implant.n_electrodes, 12)
    # `grid_type` really produces a triangular lattice, not just some other set
    # coordinates: every nearest neighbor is exactly one spacing away, which
    # on a rect grid is only true of the orthogonal ones.
    xy = implant.earray.coordinates()[:, :2]
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    npt.assert_almost_equal(dist.min(axis=1), 100)
    # The array is still centered on (x, y), odd row count and all:
    npt.assert_almost_equal((xy[:, 0].min() + xy[:, 0].max()) / 2, 0)
    npt.assert_almost_equal((xy[:, 1].min() + xy[:, 1].max()) / 2, 0)


def test_GridImplant_electrode_kwargs():
    implant = GridImplant((2, 3), 100, electrode_type=DiskElectrode,
                          radius=20)
    npt.assert_equal(implant.n_electrodes, 6)
    for e in implant.electrode_objects:
        npt.assert_equal(isinstance(e, DiskElectrode), True)
        npt.assert_almost_equal(e.radius, 20)


def test_GridImplant_geometry_passthrough():
    implant = GridImplant((2, 3), 100, x=200, y=-300, z=50, rot=90)
    npt.assert_almost_equal(implant.earray.coordinates().mean(axis=0),
                            [200, -300, 50])
    # 90 deg CCW about the grid center: (dx, dy) -> (-dy, dx)
    unrot = GridImplant((2, 3), 100, x=200, y=-300, z=50)
    dx, dy = unrot['A1'].x - 200, unrot['A1'].y + 300
    npt.assert_almost_equal(implant['A1'].x, 200 - dy)
    npt.assert_almost_equal(implant['A1'].y, -300 + dx)
    # Unitful geometry normalizes to plain microns, as everywhere else:
    unitful = GridImplant((2, 3), 0.1 * mm, x=0.2 * mm, y=-300 * um,
                          z=50 * um, rot=90 * deg)
    npt.assert_allclose(unitful.earray.coordinates(),
                        implant.earray.coordinates(), rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        GridImplant((2, 3), 2 * dva)


def test_GridImplant_device_arguments_reach_Implant():
    """Everything that is not geometry is handed to Implant as given

    What those arguments then do is Implant's business and is tested
    there; all a GridImplant owes them is not to drop or reinterpret one.
    """
    encoder = AmplitudeEncoder(amp_range=(0, 20))
    raster = implants.SequentialRaster(2)
    implant = GridImplant((2, 3), 100, eye='LE',
                          preprocess=True, safe_mode=True, encoder=encoder,
                          raster=raster, max_current=100)
    npt.assert_equal(implant.eye, 'LE')
    npt.assert_equal(
        implant.prepare_stim({'A1': BiphasicPulse(10, 1)}).electrodes, ['A1'])
    npt.assert_equal(implant.preprocess, True)
    npt.assert_equal(implant.safe_mode, True)
    npt.assert_equal(implant.encoder, encoder)
    npt.assert_equal(implant.raster, raster)
    npt.assert_almost_equal(implant.max_current, 100)


def test_GridImplant_does_not_relabel_the_left_eye():
    """Unlike RectangleImplant, a generic grid is the same in either eye"""
    re = GridImplant((3, 4), 100, eye='RE')
    le = GridImplant((3, 4), 100, eye='LE')
    npt.assert_equal(le.electrode_names, re.electrode_names)
    npt.assert_almost_equal(le.earray.coordinates(), re.earray.coordinates())


def test_Implant_reshape_stim_frames_independent():
    """Downsampling a video must treat each frame on its own.

    ``reshape_stim`` builds one interpolator for the whole video rather than
    one per frame, so this checks that a frame lands on the electrodes the
    same way whether it arrives alone or inside a sequence.
    """
    rng = np.random.default_rng(3)
    n_frames = 5
    vid = rng.random((24, 31, n_frames)).astype(np.float32)
    implant = Implant(ElectrodeGrid((6, 8), 200))

    joint = implant.reshape_stim(VideoStimulus(vid,
                                               time=np.arange(n_frames))).data
    npt.assert_equal(joint.shape, (implant.n_electrodes, n_frames))

    for f in range(n_frames):
        single = implant.reshape_stim(ImageStimulus(vid[..., f]))
        npt.assert_allclose(single.data[:, 0], joint[:, f], rtol=1e-5,
                            atol=1e-7)

    # Pixels outside the electrode footprint are filled with zero, not
    # extrapolated, so an all-zero frame stays all zero:
    vid[..., 2] = 0
    sampled = implant.reshape_stim(VideoStimulus(vid,
                                                 time=np.arange(n_frames)))
    npt.assert_equal(np.all(sampled.data[:, 2] == 0), True)


def test_Implant_rgb_video_stim():
    """An RGB video can be presented to an implant directly (Issue #802)"""
    n_frames = 4
    vid = VideoStimulus(np.random.default_rng(0).random((6, 10, 3, n_frames)),
                        metadata={'fps': 20})
    implant = ArgusII(encoder=AmplitudeEncoder(amp_range=(0, 20)))
    stim = implant.prepare_stim(vid)
    npt.assert_equal(len(stim.electrodes), implant.n_electrodes)
    npt.assert_equal(stim._spatial_view().shape,
                     (implant.n_electrodes, n_frames))


def test_implant_geometry_units():
    """Every implant places itself the same way, however its x/y/z is spelled

    Some device constructors inspect or adjust the geometry themselves before
    handing it to an ElectrodeGrid (Orion checks `z`, PRIMA writes a
    per-electrode `z` list onto the electrodes afterwards), so it is not enough
    to normalize inside the grid.
    """
    cases = [
        (implants.ArgusI, {'x': 1 * mm, 'y': -0.5 * mm, 'z': 100 * um},
         {'x': 1000, 'y': -500, 'z': 100}),
        (implants.ArgusII, {'x': 1 * mm, 'y': -0.5 * mm, 'z': 100 * um},
         {'x': 1000, 'y': -500, 'z': 100}),
        (implants.AlphaIMS, {'x': 0.2 * mm, 'z': -0.1 * mm},
         {'x': 200, 'z': -100}),
        (implants.AlphaAMS, {'x': 0.2 * mm, 'z': -0.1 * mm},
         {'x': 200, 'z': -100}),
        (implants.PRIMAPivotal, {'z': -0.1 * mm}, {'z': -100}),
        (implants.Lorach2015Array, {'z': -0.1 * mm}, {'z': -100}),
        (implants.Ho2019FlatArray, {'pixel_size': 55 * um, 'z': -0.1 * mm},
         {'pixel_size': 55, 'z': -100}),
        (implants.Ho2019FlatArray, {'pixel_size': 40 * um, 'z': -0.1 * mm},
         {'pixel_size': 40, 'z': -100}),
        (implants.Huang2021Array, {'pixel_size': 0.03 * mm, 'z': -0.1 * mm},
         {'pixel_size': 30, 'z': -100}),
        (implants.BVT24, {'x': 1 * mm, 'y': -0.5 * mm, 'z': 50 * um},
         {'x': 1000, 'y': -500, 'z': 50}),
        (implants.BVT44, {'x': 1 * mm, 'y': -0.5 * mm, 'z': 50 * um},
         {'x': 1000, 'y': -500, 'z': 50}),
        (implants.IMIE, {'x': 1 * mm, 'z': 100 * um}, {'x': 1000, 'z': 100}),
        (implants.RectangleImplant,
         {'x': 1 * mm, 'spacing': 0.4 * mm, 'r': 75 * um},
         {'x': 1000, 'spacing': 400., 'r': 75.}),
        (cortex.Orion, {'x': 15 * mm}, {'x': 15000}),
        (cortex.Cortivis, {'x': 20 * mm, 'y': -5 * mm}, {'x': 20000,
                                                         'y': -5000}),
        (cortex.ICVP, {'x': 15 * mm}, {'x': 15000}),
    ]
    for cls, unitful, bare in cases:
        coords = cls(**unitful).earray.coordinates()
        npt.assert_allclose(coords, cls(**bare).earray.coordinates(),
                            rtol=1e-12, err_msg=cls.__name__)
        # Plain numbers all the way down, whatever went in:
        npt.assert_equal(coords.dtype, np.float64)
    # Orion is the documented default: 15 mm to the right of the fovea:
    npt.assert_allclose(cortex.Orion(x=15 * mm).earray.coordinates(),
                        cortex.Orion().earray.coordinates(), rtol=1e-12)
    # A conversion that does not land on a round number is no different:
    npt.assert_allclose(
        implants.ArgusII(x=0.8625 * mm, z=0.0417 * mm).earray.coordinates(),
        implants.ArgusII(x=862.5, z=41.7).earray.coordinates(), rtol=1e-12)


def test_implant_rot_units():
    """`rot` is an ordinary angle; the grid does the conversion for everyone"""
    bare = implants.ArgusII(rot=45).earray.coordinates()
    for rot in (45 * deg, np.pi / 4 * rad):
        npt.assert_allclose(implants.ArgusII(rot=rot).earray.coordinates(),
                            bare, rtol=1e-12)


def test_implant_per_electrode_z_units():
    """A per-electrode list of heights never reaches ElectrodeGrid"""
    for cls, n in [(implants.PRIMAPivotal, 378),
                   (implants.Lorach2015Array, 142),
                   (implants.AlphaIMS, 1500)]:
        heights = np.linspace(-150, -50, n)
        unitful = cls(z=[h * um for h in heights])
        npt.assert_allclose(unitful.earray.coordinates(),
                            cls(z=list(heights)).earray.coordinates(),
                            rtol=1e-12, err_msg=cls.__name__)
        npt.assert_allclose(unitful.earray.coordinates()[:, 2], heights,
                            rtol=1e-12)


def test_implant_dimension_errors():
    for cls in (implants.ArgusII, implants.PRIMAPivotal, implants.BVT24,
                cortex.Orion, cortex.Cortivis, cortex.ICVP):
        with pytest.raises(DimensionMismatchError):
            cls(x=5 * ms)
        with pytest.raises(DimensionMismatchError):
            cls(z=10 * uA)
        with pytest.raises(DimensionMismatchError):
            cls(rot=5 * dva)
    with pytest.raises(DimensionMismatchError):
        implants.RectangleImplant(spacing=2 * dva)
    with pytest.raises(DimensionMismatchError):
        implants.RectangleImplant(r=10 * uA)


def test_Implant_max_current_units():
    """`max_current` is a current, stored as a plain number of microamps"""
    earray = ElectrodeArray(DiskElectrode(0, 0, 0, 100))
    for value in (100, 100 * uA, 0.1 * mA, 100000 * nA):
        implant = Implant(earray, max_current=value)
        npt.assert_allclose(implant.max_current, 100, rtol=1e-12)
        npt.assert_equal(isinstance(implant.max_current, Quantity), False)
    # An awkward conversion is no different:
    npt.assert_allclose(
        Implant(earray, max_current=0.0417 * mA).max_current, 41.7,
        rtol=1e-12)
    # None means no limit, and is left alone:
    npt.assert_equal(Implant(earray).max_current, None)
    # Assigning later goes through the same setter:
    implant = Implant(earray)
    implant.max_current = 0.1 * mA
    npt.assert_allclose(implant.max_current, 100, rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        Implant(earray, max_current=5 * ms)
    with pytest.raises(DimensionMismatchError):
        implant.max_current = 5 * dva
    with pytest.raises(ValueError):
        Implant(earray, max_current=-1 * uA)


def test_Implant_safety_checks_are_electrical():
    """Electrical safety may only be asked about an electrical stimulus

    In the ordinary flow ``prepare_stim`` has already refused anything that is
    not a current (see ``test_Implant_requires_an_electrical_stimulus``),
    so these guards are reached by calling ``check_stim`` directly -- which is
    public, and which a subclass may call on a stimulus of its own making.
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    sampled = ArgusII().reshape_stim(img)

    # `safe_mode` is a claim about electricity, and cannot be made about a
    # picture -- it must not integrate gray levels and pronounce them safe:
    implant = ArgusII(preprocess=False, safe_mode=True)
    with pytest.raises(DimensionMismatchError) as excinfo:
        implant.check_stim(sampled)
    npt.assert_equal("Safety check 'safe_mode'" in str(excinfo.value), True)
    npt.assert_equal('dimensionless' in str(excinfo.value), True)

    # ... and so is `max_current`:
    implant = ArgusII(preprocess=False, safe_mode=False)
    implant.max_current = 100 * uA
    with pytest.raises(DimensionMismatchError) as excinfo:
        implant.check_stim(sampled)
    npt.assert_equal("Safety check 'max_current'" in str(excinfo.value), True)
    # Including an empty one: the guard sits before the empty-data fast path,
    # since an empty picture is just as much the wrong kind of thing.
    empty = Stimulus(np.zeros((60, 0)),
                     electrodes=ArgusII().electrode_names)._inherit_units(img)
    with pytest.raises(DimensionMismatchError):
        implant.check_stim(empty)
    # An empty *electrical* stimulus is fine, as before:
    implant.check_stim(Stimulus(np.zeros((60, 0)),
                                electrodes=ArgusII().electrode_names))


def test_Implant_requires_an_electrical_stimulus():
    """An implant delivers current, so a picture it cannot encode is refused

    Not when a model eventually reads it: the preparation is the line that was
    wrong, and without an encoder there is no principled default mapping from a
    gray level onto an amplitude or a frequency for the implant to apply on the
    user's behalf.
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    npt.assert_equal(Implant.stimulus_unit, uA)

    for source in (img, VideoStimulus(np.ones((6, 10, 3)) * 0.5,
                                      time=[0, 20, 40]), BostonTrain()):
        implant = ArgusII(encoder=None)
        with pytest.raises(DimensionMismatchError) as excinfo:
            implant.prepare_stim(source)
        npt.assert_equal('encoder' in str(excinfo.value), True)
        npt.assert_equal('dimensionless' in str(excinfo.value), True)
        # A generic system has no encoder either:
        with pytest.raises(DimensionMismatchError):
            Implant(ArgusII().earray).prepare_stim(source)

    # Encoded, the very same picture goes through:
    implant = ArgusII(encoder=None)
    encoded = AmplitudeEncoder(amp_range=(0, 50)).encode(img, implant=implant)
    npt.assert_equal(implant.prepare_stim(encoded).unit, uA)

    # ... and so does everything that was electrical all along:
    for source in ({'A1': 20}, np.ones(60),
                   {'A1': BiphasicPulse(0.02 * mA, 0.45, stim_dur=50)}):
        npt.assert_equal(ArgusII().prepare_stim(source).unit, uA)
    for source in (20, BiphasicPulse(20, 0.45, stim_dur=50)):
        single = Implant(DiskElectrode(0, 0, 0, 100))
        npt.assert_equal(single.prepare_stim(source).unit, uA)

    # A subclass may declare that it delivers something else, in which case a
    # picture is already what it delivers and the encoder stays out of it:
    class Projector(ArgusII):
        stimulus_unit = dimensionless

    npt.assert_equal(Projector().prepare_stim(img).unit, dimensionless)


def test_Implant_encoder():
    """A picture prepared by an implant with an encoder is encoded on the way
    through
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    implant = Implant(ArgusII().earray)
    npt.assert_equal(implant.encoder, None)
    with pytest.raises(TypeError):
        implant.encoder = 'amplitude'
    with pytest.raises(TypeError):
        Implant(ArgusII().earray, encoder=ArgusII())

    # Giving it one is all it takes:
    implant.encoder = AmplitudeEncoder(amp_range=(0, 50), freq=20)
    npt.assert_equal('encoder' in str(implant), True)
    stim = implant.prepare_stim(img)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.shape[0], implant.n_electrodes)
    npt.assert_almost_equal(np.abs(stim.data).max(), 50)
    # What comes back is exactly what encoding it by hand gives:
    by_hand = implant.encoder.encode(img, implant=implant)
    npt.assert_almost_equal(stim.data, by_hand.data)
    npt.assert_almost_equal(stim.time, by_hand.time)

    # A custom encoder is honored, and its parameters reach the stimulus:
    implant.encoder = AmplitudeEncoder(amp_range=(10, 30), freq=50,
                                       frame_dur=100)
    stim = implant.prepare_stim(img)
    npt.assert_almost_equal(np.abs(stim.data).max(), 30)
    npt.assert_almost_equal(stim.time[-1], 100)
    implant.encoder = FrequencyEncoder(freq_range=(0, 100), amp=42,
                                       frame_dur=100)
    npt.assert_almost_equal(np.abs(implant.prepare_stim(img).data).max(), 42)

    # An electrical stimulus bypasses the encoder entirely, whatever is
    # installed -- there is nothing left to encode:
    implant.encoder = AmplitudeEncoder(amp_range=(0, 50))
    for source in ({'A1': 20}, np.ones(60),
                   BiphasicPulse(20, 0.45, stim_dur=50)):
        stim = implant.prepare_stim(source)
        npt.assert_equal(stim.unit, uA)
        npt.assert_equal('encoder' in stim.metadata, False)

    # The encoder sees the implant it is installed on, so it samples at that
    # implant's electrodes and schedules against that implant's raster. An
    # amplitude range that starts above zero keeps every electrode active, so
    # the schedules are the raster groups and nothing else:
    implant.encoder = AmplitudeEncoder(amp_range=(10, 50))
    implant.raster = implants.SequentialRaster(6)
    stim = implant.prepare_stim(img)
    delays = [stim.time[np.argmax(stim.data[e] < 0)]
              for e in (0, 10, 20, 30, 40, 50)]
    npt.assert_almost_equal(delays, np.arange(6) * 50 / 6, decimal=2)
    npt.assert_equal(len(np.unique(np.abs(stim.data) > 0, axis=0)), 6)


def test_Implant_encoded_stim_is_one_object():
    """An encoded stimulus knows both what it delivers and what it was asked
    for, so preparation returns one of it
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    implant = ArgusII()
    stim = implant.prepare_stim(img)
    # What comes back is the delivered pulse train -- that invariant is what
    # makes the safety checks and the temporal models meaningful:
    npt.assert_equal(stim.time.size > 1, True)
    # ... and the same object says what the encoder asked each electrode for:
    # one column, no waveform, no raster.
    spatial = stim._spatial_view()
    npt.assert_equal(spatial.shape, (60, 1))
    npt.assert_equal(spatial.time, None)
    npt.assert_equal(spatial.unit, uA)
    npt.assert_almost_equal(np.abs(stim.data).max(axis=1),
                            spatial.data.ravel(), decimal=4)

    # A video keeps one column per video frame:
    vid = VideoStimulus(np.random.default_rng(0).random((6, 10, 4)),
                        metadata={'fps': 20})
    with pytest.warns(UserWarning, match='deliver no pulse'):
        # 6 Hz against 20 fps; irrelevant here, but it is not the modulation
        # that goes short of frames, only the train delivering it:
        stim = implant.prepare_stim(vid)
    npt.assert_equal(stim._spatial_view().shape, (60, 4))
    npt.assert_almost_equal(stim._spatial_view().time, np.arange(4) * 50.0)

    # An encoded stimulus carries that description wherever it came from, so
    # encoding by hand and preparing the result is the same thing as letting
    # the implant do it:
    by_hand = ArgusII(encoder=None).prepare_stim(
        AmplitudeEncoder(amp_range=(0, 50)).encode(img, implant=ArgusII()))
    npt.assert_almost_equal(by_hand._spatial_view().data,
                            ArgusII().prepare_stim(img)._spatial_view().data)
    # A stimulus given as current has only the one description of itself:
    for source in ({'A1': 20}, np.ones(60)):
        stim = implant.prepare_stim(source)
        npt.assert_equal(stim._spatial_view() is stim, True)
        npt.assert_equal(stim._has_spatial_view, False)

    # Switching an electrode off reaches both descriptions, or a model reading
    # one of them would go on stimulating through a dead electrode -- and it
    # does not cost the schedule:
    implant = ArgusII()
    implant.deactivate(['A1', 'B2'])
    stim = implant.prepare_stim(img)
    npt.assert_equal(stim._has_spatial_view, True)
    npt.assert_equal(stim.shape[0], 58)
    npt.assert_equal(stim._spatial_view().shape[0], 58)
    npt.assert_equal('A1' in stim._spatial_view().electrodes, False)


def test_Implant_preprocess_crosses_the_boundary():
    """Preprocessing may turn a picture into current before the encoder sees it
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    encoder = AmplitudeEncoder(amp_range=(0, 20), freq=20)
    bare = ArgusII(encoder=None, raster=None)
    implant = ArgusII(safe_mode=True, encoder=None, raster=None,
                      preprocess=lambda x: encoder.encode(x, implant=bare))
    implant.max_current = 100 * mA
    stim = implant.prepare_stim(img)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.is_charge_balanced, True)
    # Preprocessing that already crossed the boundary leaves the encoder with
    # nothing to do, so an installed one does not encode twice:
    encoded = encoder.encode(img, implant=bare)
    twice = ArgusII(raster=None,
                    preprocess=lambda x: encoder.encode(x, implant=bare))
    npt.assert_almost_equal(twice.prepare_stim(img).data, encoded.data)
    # The same chain, presented already-encoded, and this time with a limit
    # tight enough to matter:
    implant = ArgusII(preprocess=False, safe_mode=True)
    implant.max_current = 100 * uA
    with pytest.raises(ValueError) as excinfo:
        implant.prepare_stim(encoded)
    npt.assert_equal('exceeds max_current' in str(excinfo.value), True)
    implant.max_current = 2 * mA
    npt.assert_equal(implant.prepare_stim(encoded).unit, uA)


def test_Implant_historical_stimuli_unchanged():
    """A bare stimulus is electrical by contract, and is checked as before"""
    implant = ArgusII(preprocess=False, safe_mode=True)
    npt.assert_equal(implant.prepare_stim({'A1': BiphasicPulse(50, 0.45)}).unit,
                     uA)
    with pytest.raises(ValueError) as excinfo:
        implant.prepare_stim({'A1': MonophasicPulse(50, 0.45)})
    npt.assert_equal('charge-balanced' in str(excinfo.value), True)
    # A plain number is microamps, and the limit is read the same way:
    implant = ArgusII(preprocess=False)
    implant.max_current = 60
    source = {name: 2 for name in ArgusII().electrode_names}
    with pytest.raises(ValueError) as excinfo:
        implant.prepare_stim(source)
    npt.assert_equal('draws 120.0 uA at once' in str(excinfo.value), True)
    implant.max_current = 0.2 * mA
    npt.assert_equal(implant.prepare_stim(source).unit, uA)


def test_Implant_deactivated_electrodes_do_not_mutate_the_source():
    # Filtering out deactivated electrodes rewrites the stimulus, so it
    # happens on a copy. A stimulus defined by its pulse parameters cannot
    # lose an electrode and remain one, so what comes back for it is an
    # ordinary Stimulus -- presenting a perfectly good pulse must not fail
    # merely because the electrode it names happens to be switched off.
    pulse = BiphasicPulse(20, 0.45, electrode='A1')
    implant = ArgusII()
    implant.deactivate('A1')
    stim = implant.prepare_stim(pulse)
    npt.assert_equal(type(stim), Stimulus)
    npt.assert_equal(stim.shape[0], 0)
    # The caller still holds their pulse, unchanged:
    npt.assert_equal(type(pulse), BiphasicPulse)
    npt.assert_equal(pulse.shape[0], 1)
    npt.assert_almost_equal(pulse.amp, 20)

    # With every electrode on, the pulse keeps its own kind:
    npt.assert_equal(type(ArgusII().prepare_stim(pulse)), BiphasicPulse)

    # An ordinary stimulus is not mutated by either path, which it used to be:
    for deactivate_first in (True, False):
        source = Stimulus({'A1': 10, 'B2': 20})
        implant = ArgusII()
        if deactivate_first:
            implant.deactivate('A1')
            stim = implant.prepare_stim(source)
        else:
            implant.prepare_stim(source)
            implant.deactivate('A1')
            stim = implant.prepare_stim(source)
        npt.assert_equal(sorted(str(e) for e in source.electrodes),
                         ['A1', 'B2'])
        npt.assert_equal([str(e) for e in stim.electrodes], ['B2'])

    # Nothing deactivated means nothing is copied or removed:
    source = Stimulus({'A1': 10, 'B2': 20})
    stim = ArgusII().prepare_stim(source)
    npt.assert_equal(sorted(str(e) for e in stim.electrodes), ['A1', 'B2'])


def test_Implant_deactivated_electrode_does_not_render_the_others():
    # An implant drops a deactivated electrode from a dict of pulse trains by
    # forgetting the entry that drives it, so the trains on the electrodes
    # that are still on never get sampled.
    def unrendered(stim):
        return sum(c._Stimulus__stim['data'] is None
                   for c, _ in stim._components)

    trains = {name: BiphasicPulseTrain(20 + i, 10, 0.45, stim_dur=200)
              for i, name in enumerate(['A1', 'A2', 'A3'])}
    implant = ArgusII()
    npt.assert_equal(unrendered(implant.prepare_stim(trains)), 3)
    implant.deactivate('A2')
    stim = implant.prepare_stim(trains)
    npt.assert_equal([str(e) for e in stim.electrodes], ['A1', 'A3'])
    npt.assert_equal(stim._components is None, False)
    npt.assert_equal(unrendered(stim), 2)
    # The waveform is still the one the trains describe:
    npt.assert_equal(stim.data.shape[0], 2)
    npt.assert_almost_equal(stim.time[-1], 200)


def test_Implant_thresholds():
    implant = ArgusII()
    npt.assert_equal(implant.thresholds, {})
    implant.thresholds = 100 * uA
    npt.assert_equal(len(implant.thresholds), implant.n_electrodes)
    npt.assert_almost_equal(implant.thresholds['A1'], 100)
    implant.thresholds = {'A1': 83 * uA, 'A2': 107 * uA}
    npt.assert_equal(sorted(implant.thresholds), ['A1', 'A2'])
    npt.assert_almost_equal(implant.thresholds['A2'], 107)
    # The getter hands out a copy, not the dict the implant works from:
    implant.thresholds['A1'] = 999
    npt.assert_almost_equal(implant.thresholds['A1'], 83)
    implant.thresholds = None
    npt.assert_equal(implant.thresholds, {})


def test_Implant_thresholds_at_construction():
    npt.assert_equal(ArgusII().thresholds, {})
    for scalar in (80, 80 * uA, 0.08 * mA):
        implant = ArgusII(thresholds=scalar)
        npt.assert_equal(len(implant.thresholds), implant.n_electrodes)
        npt.assert_almost_equal(implant.thresholds['A1'], 80)
    implant = ArgusII(thresholds={'A1': 80, 'A2': 107 * uA})
    npt.assert_equal(sorted(implant.thresholds), ['A1', 'A2'])
    npt.assert_almost_equal(implant.thresholds['A2'], 107)
    # Threshold keys use the final left-eye electrode names:
    implant = ArgusII(eye='LE', thresholds={'A10': 80})
    npt.assert_equal(sorted(implant.thresholds), ['A10'])
    npt.assert_almost_equal(implant.thresholds['A10'], 80)
    stim = ArgusII(thresholds=80).prepare_stim(
        {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45)})
    npt.assert_almost_equal(np.abs(stim.data).max(), 160)
    npt.assert_equal(stim.unit, uA)


def test_Implant_thresholds_at_construction_are_validated():
    for bad in (0, -5, np.nan, np.inf):
        with pytest.raises(ValueError):
            ArgusII(thresholds=bad)
        with pytest.raises(ValueError):
            ArgusII(thresholds={'A1': bad})
    with pytest.raises(ValueError):
        ArgusII(thresholds={'ZZ9': 80 * uA})
    with pytest.raises(DimensionMismatchError):
        ArgusII(thresholds=5 * ms)
    implant = Implant(DiskElectrode(0, 0, 0, 100), thresholds=80)
    npt.assert_almost_equal(implant.thresholds[0], 80)


def test_Implant_thresholds_are_validated():
    implant = ArgusII()
    with pytest.raises(ValueError):
        implant.thresholds = {'ZZ9': 80 * uA}
    for bad in (0, -5, np.nan, np.inf):
        with pytest.raises(ValueError):
            implant.thresholds = {'A1': bad}
        with pytest.raises(ValueError):
            implant.thresholds = bad
    for bad in (5 * ms, 5 * mm):
        with pytest.raises(DimensionMismatchError):
            implant.thresholds = {'A1': bad}
        with pytest.raises(DimensionMismatchError):
            implant.thresholds = bad
    # A rejected assignment leaves the implant as it was:
    npt.assert_equal(implant.thresholds, {})
    # None is normalized away rather than stored:
    implant.thresholds = {'A1': 80 * uA, 'A2': None}
    npt.assert_equal(sorted(implant.thresholds), ['A1'])


def test_Implant_thresholds_calibrate_pulse_trains():
    implant = ArgusII()
    source = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45),
              'A2': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    # Uncalibrated, the stimulus is not a current at all:
    npt.assert_equal(implant.prepare_stim(source).unit, xTh)
    implant.thresholds = {'A1': 80 * uA, 'A2': 120 * uA}
    stim = implant.prepare_stim(source)
    npt.assert_equal(stim.unit, uA)
    for _, src in stim._structured_sources():
        npt.assert_almost_equal(src.amp_factor, 2)
    npt.assert_almost_equal([src.amp for _, src in stim._structured_sources()],
                            [160, 240])
    npt.assert_almost_equal(np.abs(stim['A1']).max(), 160, decimal=3)


def test_Implant_thresholds_hold_current_stimuli_fixed():
    implant = ArgusII()
    train = {'A1': BiphasicPulseTrain(20, 160 * uA, 0.45)}
    source = implant.prepare_stim(train)._structured_sources()[0][1]
    npt.assert_equal(source.amp_factor, None)
    implant.thresholds = 80 * uA
    source = implant.prepare_stim(train)._structured_sources()[0][1]
    npt.assert_almost_equal(source.amp, 160)
    npt.assert_almost_equal(source.amp_factor, 2)


@pytest.mark.parametrize('amp, cleared_amp',
                         [(2 * xTh, 2), (160 * uA, 160)])
def test_Implant_clearing_thresholds_restores_the_train(amp, cleared_amp):
    implant = ArgusII()
    train = {'A1': BiphasicPulseTrain(20, amp, 0.45)}
    implant.thresholds = 40 * uA
    implant.prepare_stim(train)
    implant.thresholds = None
    source = implant.prepare_stim(train)._structured_sources()[0][1]
    npt.assert_almost_equal(source.amp, cleared_amp)
    npt.assert_equal(source.amp_factor, None if cleared_amp == 160 else 2)


def test_Implant_thresholds_beat_the_pulse_trains_own():
    implant = ArgusII()
    train = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45,
                                      threshold_amp=50 * uA)}
    implant.thresholds = 100 * uA
    stim = implant.prepare_stim(train)
    npt.assert_almost_equal(stim._structured_sources()[0][1].amp, 200)
    # Clearing falls back to the train's own threshold, not the reference:
    implant.thresholds = None
    source = implant.prepare_stim(train)._structured_sources()[0][1]
    npt.assert_almost_equal(source.amp, 100)
    npt.assert_almost_equal(source.threshold_amp, 50)


def test_Implant_thresholds_leave_raw_waveforms_alone():
    implant = ArgusII()
    before = implant.prepare_stim({'A1': 30}).data.copy()
    implant.thresholds = 80 * uA
    stim = implant.prepare_stim({'A1': 30})
    npt.assert_array_equal(stim.data, before)
    npt.assert_equal(stim._structured_sources(), None)


def test_Implant_thresholds_are_checked_when_the_stimulus_is():
    """A threshold that puts a stimulus over the limit is caught on preparation

    The implant holds no stimulus to recheck, so the pairing of thresholds and
    delivery limits is decided the next time one is prepared.
    """
    implant = ArgusII()
    train = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    implant.max_current = 250
    implant.thresholds = {'A1': 90 * uA}
    stim = implant.prepare_stim(train)
    npt.assert_almost_equal(stim._structured_sources()[0][1].amp, 180)
    # 2 * 200 uA is over the limit, so preparing against it raises:
    implant.thresholds = {'A1': 200 * uA}
    with pytest.raises(ValueError):
        implant.prepare_stim(train)
    # The stimulus already handed out is the caller's, and is untouched:
    npt.assert_almost_equal(stim._structured_sources()[0][1].amp, 180)


def test_Implant_thresholds_do_not_render_the_stimulus():
    implant = ArgusII()
    implant.thresholds = 80 * uA
    stim = implant.prepare_stim({'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45)})
    source = stim._structured_sources()[0][1]
    # `data is None` is what says no waveform has been generated:
    npt.assert_equal(source._Stimulus__stim['data'], None)


def test_Implant_thresholds_do_not_revive_deactivated_electrodes():
    implant = ArgusII()
    source = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45),
              'A2': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    implant.deactivate('A1')
    npt.assert_equal(list(implant.prepare_stim(source).electrodes), ['A2'])
    implant.thresholds = 80 * uA
    stim = implant.prepare_stim(source)
    npt.assert_equal(list(stim.electrodes), ['A2'])
    npt.assert_almost_equal(stim._structured_sources()[0][1].amp, 160)
    implant.thresholds = None
    npt.assert_equal(list(implant.prepare_stim(source).electrodes), ['A2'])


def test_Implant_thresholds_preserve_metadata():
    implant = ArgusII()
    source = Stimulus({'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45,
                                                metadata='train'),
                       'A2': BiphasicPulseTrain(20, 2 * xTh, 0.45)})
    source.metadata['user'] = 'collection'
    implant.thresholds = 80 * uA
    stim = implant.prepare_stim(source)
    npt.assert_equal(stim.metadata['user'], 'collection')
    npt.assert_equal(stim._structured_sources()[0][1].metadata['user'],
                     'train')


def test_Implant_thresholds_calibrate_from_the_original_source():
    """Each preparation starts from the caller's source, not the last result

    Calibrating twice would compound the factors; calibrating from the source
    every time preserves the original 2xTh basis.
    """
    implant = ArgusII()
    train = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    implant.thresholds = 80 * uA
    npt.assert_almost_equal(
        implant.prepare_stim(train)._structured_sources()[0][1].amp, 160)
    implant.thresholds = 50 * uA
    source = implant.prepare_stim(train)._structured_sources()[0][1]
    npt.assert_almost_equal(source.amp, 100)
    npt.assert_almost_equal(source.amp_factor, 2)


def test_Implant_uncalibrated_xTh_is_not_yet_a_current():
    implant = ArgusII()
    train = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    stim = implant.prepare_stim(train)
    npt.assert_equal(stim.unit, xTh)
    npt.assert_almost_equal(np.abs(stim.data).max(), 2, decimal=3)
    implant.max_current = 250
    with pytest.raises(DimensionMismatchError):
        implant.check_stim(stim)
    implant.safe_mode = True
    with pytest.raises(DimensionMismatchError):
        implant.check_stim(stim)
    implant.thresholds = 80 * uA
    stim = implant.prepare_stim(train)
    implant.check_stim(stim)
    npt.assert_almost_equal(np.abs(stim.data).max(), 160, decimal=3)


def test_Implant_partial_calibration_of_xTh_is_refused():
    implant = ArgusII()
    xth_source = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45),
                  'A2': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    implant.thresholds = {'A1': 80 * uA}
    with pytest.raises(DimensionMismatchError) as err:
        implant.prepare_stim(xth_source)
    npt.assert_equal('A2' in str(err.value), True)
    # A current-valued train is already a current, threshold or no threshold:
    uA_source = {'A1': BiphasicPulseTrain(20, 160 * uA, 0.45),
                 'A2': BiphasicPulseTrain(20, 160 * uA, 0.45)}
    stim = implant.prepare_stim(uA_source)
    factors = [src.amp_factor for _, src in stim._structured_sources()]
    npt.assert_equal(factors, [2, None])


@pytest.mark.parametrize('cls,expected', [
    (implants.ArgusII, 'epiretinal'),
    (implants.IMIE, 'epiretinal'),
    (implants.AlphaAMS, 'subretinal'),
    (implants.Lorach2015Array, 'subretinal'),
    (implants.BVT24, 'suprachoroidal'),
    (cortex.Orion, 'epicortical'),
    (cortex.Cortivis, 'intracortical'),
    (cortex.ICVP, 'intracortical'),
])
def test_named_devices_say_where_they_sit(cls, expected):
    npt.assert_equal(cls.placement, expected)
    npt.assert_equal(cls().placement, expected)


def test_a_generic_array_says_nothing_about_placement():
    # `placement` records what the literature is unambiguous about; a bare
    # grid of electrodes is a shape, not a device, and models read `None` as
    # "no claim" rather than as a placement of its own.
    npt.assert_equal(implants.GridImplant(shape=(2, 2), spacing=500).placement, None)
    npt.assert_equal(
        implants.Implant(implants.PointSource(0, 0, 0)).placement,
        None)
