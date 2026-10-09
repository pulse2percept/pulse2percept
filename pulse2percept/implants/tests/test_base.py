import numpy as np
import collections as coll
from copy import deepcopy
from functools import partial
from inspect import signature
import pytest
import numpy.testing as npt
import torch
from scipy.interpolate import RegularGridInterpolator
from pulse2percept import implants
from pulse2percept.implants import cortex, retina
from pulse2percept.implants.base import _bilinear_tensor
from pulse2percept.units import (DimensionMismatchError, Quantity, deg,
                                 dimensionless, dva, mA, mm, ms, nA, rad, uA,
                                 um, xTh)
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

from pulse2percept.implants import (PointSource, ElectrodeArray,
                                    ElectrodeGrid, GridImplant, Implant)
from pulse2percept.implants.retina import PhotovoltaicPixel
from pulse2percept.stimuli import (Stimulus, ImageStimulus, VideoStimulus,
                                   samples)
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BiphasicPulseTrain, FrequencyEncoder,
                                   MonophasicPulse, TraceEncoder)
from pulse2percept.implants import DiskElectrode
from pulse2percept.implants.retina import ArgusII
from pulse2percept.models.retina import ScoreboardModel, ScoreboardSpatial


class PhotovoltaicArray(Implant):
    def __init__(self, x=0, y=0, z=-100, r=5, spacing=40, rot=0,
                 preprocess=False, safe_mode=False):
        # 35 um pixels with 5 um trenches, 16 um active electrode:
        self.spacing = spacing  # um
        self.trench = 5  # um
        elec_radius = 8  # um
        self.shape = (int(r * 600 / spacing), int(r * 700 / spacing))
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        dva2ret = 280.0

        self.electrode_array = ElectrodeGrid(
            self.shape, spacing, x=x, y=y, z=z, rot=rot, grid_type='hex',
            orientation='vertical', electrode_type=PhotovoltaicPixel,
            radius=elec_radius, apothem=(self.spacing - self.trench) / 2)

        rm_names = []
        for name, electrode in self.electrode_array.electrodes.items():
            if (electrode.x - x) ** 2 + (electrode.y - y) ** 2 > (r * dva2ret) ** 2:
                rm_names.append(name)
        for e in rm_names:
            self.electrode_array.remove_electrode(e)


def test_Implant():
    # Invalid instantiations:
    with pytest.raises(TypeError):
        Implant(Stimulus)

    # Iterating over the electrode array:
    implant = Implant(PointSource(0, 0, 0))
    npt.assert_equal(implant.n_electrodes, 1)
    npt.assert_equal(implant[0], implant.electrode_array[0])
    npt.assert_equal(implant.electrode_names,
                     implant.electrode_array.electrode_names)
    for i, e in zip(implant, implant.electrode_array):
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
    """prepare_stim changes neither the implant nor the source"""
    implant = ArgusII(preprocess=False)
    source = Stimulus({'A1': 10, 'B2': 20})
    first = implant.prepare_stim(source)
    # The result is a copy of the source:
    npt.assert_equal(first is source, False)
    npt.assert_almost_equal(first.data, source.data)

    # Consecutive calls are independent:
    second = implant.prepare_stim({'C3': 30})
    npt.assert_equal(sorted(str(e) for e in first.electrodes), ['A1', 'B2'])
    npt.assert_equal([str(e) for e in second.electrodes], ['C3'])
    npt.assert_equal(hasattr(implant, 'stim'), False)
    npt.assert_equal(hasattr(implant, '_stim'), False)

    # Same source, same result:
    again = implant.prepare_stim(source)
    npt.assert_almost_equal(again.data, first.data)


@pytest.mark.parametrize('rot', (0, 30, 92))
@pytest.mark.parametrize('gtype', ('hex', 'rect'))
@pytest.mark.parametrize('n_frames', (1, 3, 4))
def test_Implant_reshape_stim(rot, gtype, n_frames):
    implant = Implant(ElectrodeGrid((10, 10), 30, rot=rot, grid_type=gtype))
    # Smoke test the reshaping directly (encoders call it too, since a picture
    # is not a deliverable stimulus):
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
    # the device is rotated. Sampled gray levels are passed to the model as a
    # plain Stimulus (1 uA per gray level) to test `reshape_stim` only:
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
    implant.reshape_stim(samples.logo_bvl())


@pytest.mark.parametrize('rot', (0, 30))
def test_Implant__sample_image_tensor(rot):
    # A rotated hex grid puts electrodes between pixel centers:
    implant = Implant(ElectrodeGrid((5, 7), 300, rot=rot, grid_type='hex'))
    # Out-of-range gray levels are clipped later, by the encoder:
    img = np.random.default_rng(3).uniform(-0.5, 1.5, (9, 13))
    expected = implant.reshape_stim(ImageStimulus(img))
    sampled = implant._sample_image_tensor(
        torch.tensor(img, dtype=torch.float32))
    assert sampled.dtype == torch.float32
    assert sampled.shape == (implant.n_electrodes,)
    npt.assert_allclose(sampled.numpy(), expected.data.ravel(), atol=1e-6)


@pytest.mark.parametrize('rot', (0, 30))
def test_Implant__sample_image_tensor_video(rot):
    implant = Implant(ElectrodeGrid((5, 7), 300, rot=rot, grid_type='hex'))
    # Frames on the last axis, each with different content; H, W, F all
    # differ so an (F, H, W) reading would fail:
    vid = np.random.default_rng(4).uniform(-0.5, 1.5, (9, 13, 4))
    vid[..., 1] = vid[..., 1][::-1]
    vid[..., 2] = 0
    expected = implant.reshape_stim(VideoStimulus(vid))
    sampled = implant._sample_image_tensor(
        torch.tensor(vid, dtype=torch.float32))
    assert sampled.shape == (implant.n_electrodes, 4)
    npt.assert_allclose(sampled.numpy(), expected.data, atol=1e-6)
    # Frames are sampled independently, never across time:
    for f in range(4):
        npt.assert_allclose(sampled[:, f].numpy(),
                            implant._sample_image_tensor(
                                torch.tensor(vid[..., f])).numpy(),
                            atol=1e-6)


def test_Implant__sample_image_tensor_orientation():
    # Row 0 lies at the smallest y (no image flip), column 0 at the smallest x.
    # E sits 3/4 of the way along x and 1/4 of the way along y:
    pos = [(-600, -600), (600, -600), (-600, 600), (600, 600), (300, -300)]
    implant = Implant(ElectrodeArray({n: DiskElectrode(x, y, 0, 100)
                                      for n, (x, y) in zip('ABCDE', pos)}))
    img = np.array([[1.0, 2.0], [3.0, 4.0]])
    sampled = implant._sample_image_tensor(torch.tensor(img))
    npt.assert_allclose(sampled.numpy(), [1, 2, 3, 4, 2.25])
    npt.assert_allclose(sampled.numpy(),
                        implant.reshape_stim(ImageStimulus(img)).data.ravel())
    with pytest.raises(TypeError, match='torch.Tensor'):
        implant._sample_image_tensor(img)
    with pytest.raises(TypeError, match='float32 or float64'):
        implant._sample_image_tensor(torch.ones((2, 2), dtype=torch.int64))
    # RGB images and videos are unsupported:
    for shape in ((4,), (2, 2, 3, 5)):
        with pytest.raises(ValueError, match='gray image'):
            implant._sample_image_tensor(torch.ones(shape))


def test__bilinear_tensor():
    img_y, img_x = np.arange(4.0), 10 * np.arange(5.0)
    img = np.arange(1.0, 21.0).reshape((4, 5))
    image = torch.tensor(img, requires_grad=True)
    # One point between rows 1-2 and columns 2-3 depends on those four pixels
    # only, with the bilinear weights as gradient:
    out = _bilinear_tensor(image, img_y, img_x, np.array([1.25]),
                           np.array([26.0]))
    out.sum().backward()
    weights = np.zeros_like(img)
    weights[1:3, 2:4] = [[0.75 * 0.4, 0.75 * 0.6], [0.25 * 0.4, 0.25 * 0.6]]
    npt.assert_allclose(image.grad.numpy(), weights)
    npt.assert_allclose(out.item(), np.sum(weights * img))
    # Grid edges are inside; anything beyond them is 0:
    y = np.array([0, 3, 1.5, -0.1, 3.1, 1, 1])
    x = np.array([0, 40, 13, 10, 10, -1, 40.5])
    out = _bilinear_tensor(image, img_y, img_x, y, x)
    expected = RegularGridInterpolator((img_y, img_x), img,
                                       bounds_error=False, fill_value=0)
    npt.assert_allclose(out.detach().numpy(), expected(np.vstack((y, x)).T))
    npt.assert_equal(out.detach().numpy()[[0, 1, 3, 4, 5, 6]],
                     [1, 20, 0, 0, 0, 0])
    with pytest.raises(ValueError, match='strictly ascending'):
        _bilinear_tensor(image, np.zeros(4), img_x, y, x)


def test_Implant_deactivate():
    implant = Implant(ElectrodeGrid((10, 10), 30))
    source = np.ones(implant.n_electrodes)
    electrode = 'A3'
    npt.assert_equal(electrode in implant.prepare_stim(source).electrodes, True)
    implant.deactivate(electrode)
    npt.assert_equal(implant[electrode].activated, False)
    # Deactivation applies to the next prepare_stim call:
    npt.assert_equal(electrode in implant.prepare_stim(source).electrodes,
                     False)


def test_GridImplant_is_a_grid_in_an_implant():
    implant = GridImplant((3, 4), 100)
    npt.assert_equal(isinstance(implant, Implant), True)
    npt.assert_equal(isinstance(implant.electrode_array, ElectrodeGrid), True)
    npt.assert_equal(implant.n_electrodes, 12)
    npt.assert_equal(implant.electrode_array.shape, (3, 4))
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
    npt.assert_equal(implant.electrode_array.grid_type, 'hex')
    npt.assert_equal(implant.n_electrodes, 12)
    # Hex grid is a triangular lattice: every nearest neighbor is exactly one
    # spacing away (on a rect grid, only the orthogonal ones are):
    xy = implant.electrode_array.coordinates()[:, :2]
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    npt.assert_almost_equal(dist.min(axis=1), 100)
    # Still centered on (x, y), even with an odd row count:
    npt.assert_almost_equal((xy[:, 0].min() + xy[:, 0].max()) / 2, 0)
    npt.assert_almost_equal((xy[:, 1].min() + xy[:, 1].max()) / 2, 0)


def test_GridImplant_electrode_kwargs():
    implant = GridImplant((2, 3), 100, electrode_type=DiskElectrode,
                          radius=20)
    npt.assert_equal(implant.n_electrodes, 6)
    for e in implant.electrode_objects:
        npt.assert_equal(isinstance(e, DiskElectrode), True)
        npt.assert_almost_equal(e.radius, 20)


def test_GridImplant_rejects_the_old_spellings():
    npt.assert_equal(GridImplant((2, 3), 100, grid_type='hex').n_electrodes, 6)
    for kwargs in ({'type': 'hex'}, {'etype': DiskElectrode, 'r': 20},
                   {'electrode_type': DiskElectrode, 'r': 20}):
        with pytest.raises(TypeError):
            GridImplant((2, 3), 100, **kwargs)


def test_GridImplant_geometry_passthrough():
    implant = GridImplant((2, 3), 100, x=200, y=-300, z=50, rot=90)
    npt.assert_almost_equal(implant.electrode_array.coordinates().mean(axis=0),
                            [200, -300, 50])
    # 90 deg CCW about the grid center: (dx, dy) -> (-dy, dx)
    unrot = GridImplant((2, 3), 100, x=200, y=-300, z=50)
    dx, dy = unrot['A1'].x - 200, unrot['A1'].y + 300
    npt.assert_almost_equal(implant['A1'].x, 200 - dy)
    npt.assert_almost_equal(implant['A1'].y, -300 + dx)
    # Unitful geometry normalizes to plain microns, as everywhere else:
    unitful = GridImplant((2, 3), 0.1 * mm, x=0.2 * mm, y=-300 * um,
                          z=50 * um, rot=90 * deg)
    npt.assert_allclose(unitful.electrode_array.coordinates(),
                        implant.electrode_array.coordinates(), rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        GridImplant((2, 3), 2 * dva)


def test_GridImplant_device_arguments_reach_Implant():
    """GridImplant passes all non-geometry arguments to Implant unchanged"""
    encoder = AmplitudeEncoder(amp_range=(0, 20))
    raster = implants.SequentialRaster(2)
    implant = GridImplant((2, 3), 100,
                          preprocess=True, safe_mode=True, encoder=encoder,
                          raster=raster, max_current=100)
    npt.assert_equal(
        implant.prepare_stim({'A1': BiphasicPulse(10, 1)}).electrodes, ['A1'])
    npt.assert_equal(implant.preprocess, True)
    npt.assert_equal(implant.safe_mode, True)
    npt.assert_equal(implant.encoder is encoder, True)
    npt.assert_equal(encoder.implant is implant, True)
    npt.assert_equal(implant.raster, raster)
    npt.assert_almost_equal(implant.max_current, 100)


def test_Implant_reshape_stim_frames_independent():
    """reshape_stim samples each video frame independently

    ``reshape_stim`` uses one interpolator for the whole video, so a frame must
    map to the electrodes the same way alone or inside a sequence.
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

    # Pixels outside the electrode footprint are zero-filled, not extrapolated:
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
    """Unitful and plain x/y/z give the same coordinates for every implant

    Some constructors adjust geometry before building the ElectrodeGrid
    (Orion checks `z`, PRIMA writes per-electrode `z` afterwards), so
    normalizing inside the grid is not enough.
    """
    cases = [
        (retina.ArgusI, {'z': 100 * um}, {'z': 100}),
        (retina.ArgusII, {'z': 100 * um}, {'z': 100}),
        (retina.AlphaIMS, {'z': -0.1 * mm}, {'z': -100}),
        (retina.AlphaAMS, {'z': -0.1 * mm}, {'z': -100}),
        (retina.PRIMAPivotal, {'z': -0.1 * mm}, {'z': -100}),
        (retina.Lorach2015Array, {'z': -0.1 * mm}, {'z': -100}),
        (retina.Ho2019FlatArray, {'pixel_size': 55 * um, 'z': -0.1 * mm},
         {'pixel_size': 55, 'z': -100}),
        (retina.Ho2019FlatArray, {'pixel_size': 40 * um, 'z': -0.1 * mm},
         {'pixel_size': 40, 'z': -100}),
        (retina.Huang2021Array, {'pixel_size': 0.03 * mm, 'z': -0.1 * mm},
         {'pixel_size': 30, 'z': -100}),
        (retina.Suprachoroidal24, {'z': 50 * um}, {'z': 50}),
        (retina.Suprachoroidal44, {'z': 50 * um}, {'z': 50}),
        (retina.IMIE, {'z': 100 * um}, {'z': 100}),
    ]
    for cls, unitful, bare in cases:
        coords = cls(**unitful).electrode_array.coordinates()
        npt.assert_allclose(coords, cls(**bare).electrode_array.coordinates(),
                            rtol=1e-12, err_msg=cls.__name__)
        # Coordinates are plain floats:
        npt.assert_equal(coords.dtype, np.float64)
    # Non-round conversions work too:
    npt.assert_allclose(
        retina.ArgusII(z=0.0417 * mm).electrode_array.coordinates(),
        retina.ArgusII(z=41.7).electrode_array.coordinates(), rtol=1e-12)


def test_implant_rot_units():
    """`rot` accepts angle units"""
    bare = GridImplant((6, 10), 575.0, rot=45).electrode_array.coordinates()
    for rot in (45 * deg, np.pi / 4 * rad):
        npt.assert_allclose(
            GridImplant((6, 10), 575.0, rot=rot).electrode_array.coordinates(),
            bare, rtol=1e-12)


def test_implant_per_electrode_z_units():
    """A per-electrode list of unitful `z` matches the same list in um"""
    for cls, n in [(retina.PRIMAPivotal, 378),
                   (retina.Lorach2015Array, 142),
                   (retina.AlphaIMS, 1500)]:
        heights = np.linspace(-150, -50, n)
        unitful = cls(z=[h * um for h in heights])
        npt.assert_allclose(unitful.electrode_array.coordinates(),
                            cls(z=list(heights)).electrode_array.coordinates(),
                            rtol=1e-12, err_msg=cls.__name__)
        npt.assert_allclose(unitful.electrode_array.coordinates()[:, 2],
                            heights,
                            rtol=1e-12)


def test_implant_dimension_errors():
    for cls in (retina.ArgusII, retina.PRIMAPivotal,
                retina.Suprachoroidal24):
        with pytest.raises(DimensionMismatchError):
            cls(z=10 * uA)
    with pytest.raises(DimensionMismatchError):
        GridImplant((2, 2), 400, rot=5 * dva)
    with pytest.raises(DimensionMismatchError):
        GridImplant((2, 2), 0.4 * dva)


def test_Implant_max_current_units():
    """`max_current` accepts current units and is stored in uA"""
    electrode_array = ElectrodeArray(DiskElectrode(0, 0, 0, 100))
    for value in (100, 100 * uA, 0.1 * mA, 100000 * nA):
        implant = Implant(electrode_array, max_current=value)
        npt.assert_allclose(implant.max_current, 100, rtol=1e-12)
        npt.assert_equal(isinstance(implant.max_current, Quantity), False)
    # Non-round conversions work too:
    npt.assert_allclose(
        Implant(electrode_array, max_current=0.0417 * mA).max_current, 41.7,
        rtol=1e-12)
    # None means no limit:
    npt.assert_equal(Implant(electrode_array).max_current, None)
    # The setter converts too:
    implant = Implant(electrode_array)
    implant.max_current = 0.1 * mA
    npt.assert_allclose(implant.max_current, 100, rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        Implant(electrode_array, max_current=5 * ms)
    with pytest.raises(DimensionMismatchError):
        implant.max_current = 5 * dva
    with pytest.raises(ValueError):
        Implant(electrode_array, max_current=-1 * uA)


def test_Implant_safety_checks_are_electrical():
    """check_stim rejects safety checks on a non-electrical stimulus

    ``prepare_stim`` already rejects non-current input, so this calls the
    public ``check_stim`` directly (as a subclass might).
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    sampled = ArgusII().reshape_stim(img)

    # `safe_mode` cannot integrate gray levels:
    implant = ArgusII(preprocess=False, safe_mode=True)
    with pytest.raises(DimensionMismatchError) as excinfo:
        implant.check_stim(sampled)
    npt.assert_equal("Safety check 'safe_mode'" in str(excinfo.value), True)
    npt.assert_equal('dimensionless' in str(excinfo.value), True)

    # Same for `max_current`:
    implant = ArgusII(preprocess=False, safe_mode=False)
    implant.max_current = 100 * uA
    with pytest.raises(DimensionMismatchError) as excinfo:
        implant.check_stim(sampled)
    npt.assert_equal("Safety check 'max_current'" in str(excinfo.value), True)
    # Also for empty data (unit check comes before the empty-data shortcut):
    empty = Stimulus(np.zeros((60, 0)),
                     electrodes=ArgusII().electrode_names)._inherit_units(img)
    with pytest.raises(DimensionMismatchError):
        implant.check_stim(empty)
    # An empty electrical stimulus passes:
    implant.check_stim(Stimulus(np.zeros((60, 0)),
                                electrodes=ArgusII().electrode_names))


def test_Implant_requires_an_electrical_stimulus():
    """prepare_stim rejects a picture when the implant has no encoder

    There is no default mapping from gray level to amplitude or frequency.
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    npt.assert_equal(Implant.stimulus_unit, uA)

    for source in (img, VideoStimulus(np.ones((6, 10, 3)) * 0.5,
                                      time=[0, 20, 40])):
        implant = ArgusII(encoder=None)
        with pytest.raises(DimensionMismatchError) as excinfo:
            implant.prepare_stim(source)
        npt.assert_equal('encoder' in str(excinfo.value), True)
        npt.assert_equal('dimensionless' in str(excinfo.value), True)
        # Same for a generic Implant:
        with pytest.raises(DimensionMismatchError):
            Implant(ArgusII().electrode_array).prepare_stim(source)

    # An encoded picture passes:
    implant = ArgusII(encoder=None)
    encoded = AmplitudeEncoder(implant, amp_range=(0, 50)).encode(img)
    npt.assert_equal(implant.prepare_stim(encoded).unit, uA)

    # So do electrical stimuli:
    for source in ({'A1': 20}, np.ones(60),
                   {'A1': BiphasicPulse(0.02 * mA, 0.45, stim_dur=50)}):
        npt.assert_equal(ArgusII().prepare_stim(source).unit, uA)
    for source in (20, BiphasicPulse(20, 0.45, stim_dur=50)):
        single = Implant(DiskElectrode(0, 0, 0, 100))
        npt.assert_equal(single.prepare_stim(source).unit, uA)

    # A subclass that delivers dimensionless stimuli takes the picture as is:
    class Projector(ArgusII):
        stimulus_unit = dimensionless

    npt.assert_equal(Projector().prepare_stim(img).unit, dimensionless)


def test_Implant_encoder():
    """prepare_stim encodes a picture with the implant's encoder"""
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    implant = Implant(ArgusII().electrode_array)
    npt.assert_equal(implant.encoder, None)
    with pytest.raises(TypeError):
        implant.encoder = 'amplitude'
    with pytest.raises(TypeError):
        Implant(ArgusII().electrode_array, encoder=ArgusII())

    # Assigning an encoder:
    unbound = AmplitudeEncoder(amp_range=(0, 50), freq=20)
    implant.encoder = unbound
    npt.assert_equal('encoder' in str(implant), True)
    stim = implant.prepare_stim(img)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.shape[0], implant.n_electrodes)
    npt.assert_almost_equal(np.abs(stim.data).max(), 50)
    # Result matches encoding by hand:
    by_hand = AmplitudeEncoder(implant, amp_range=(0, 50), freq=20).encode(img)
    npt.assert_almost_equal(stim.data, by_hand.data)
    npt.assert_almost_equal(stim.time, by_hand.time)
    npt.assert_equal(list(stim.electrodes), list(by_hand.electrodes))

    # The stored encoder is bound to the implant:
    npt.assert_equal(implant.encoder is unbound, True)
    npt.assert_equal(unbound.implant is implant, True)
    # An encoder already bound to this implant is stored as is:
    bound = AmplitudeEncoder(implant)
    implant.encoder = bound
    npt.assert_equal(implant.encoder is bound, True)
    # An encoder bound to another implant is rejected:
    other = Implant(ArgusII().electrode_array)
    with pytest.raises(ValueError, match='already bound'):
        other.encoder = bound
    with pytest.raises(ValueError, match='already bound'):
        Implant(ArgusII().electrode_array, encoder=unbound)
    npt.assert_equal(other.encoder, None)
    npt.assert_equal(bound.implant is implant, True)
    # TraceEncoder requires a model, so it is rejected:
    trace = TraceEncoder(ScoreboardSpatial(implant))
    with pytest.raises(TypeError, match='ImplantEncoder'):
        implant.encoder = trace
    npt.assert_equal(implant.encoder is bound, True)
    # A deep copy binds the copied encoder to the copied implant:
    clone = deepcopy(implant)
    npt.assert_equal(clone.encoder.implant is clone, True)

    # Encoder parameters reach the stimulus:
    implant.encoder = AmplitudeEncoder(amp_range=(10, 30), freq=50,
                                       frame_dur=100)
    stim = implant.prepare_stim(img)
    npt.assert_almost_equal(np.abs(stim.data).max(), 30)
    npt.assert_almost_equal(stim.time[-1], 100)
    implant.encoder = FrequencyEncoder(freq_range=(0, 100), amp=42,
                                       frame_dur=100)
    npt.assert_almost_equal(np.abs(implant.prepare_stim(img).data).max(), 42)

    # Electrical stimuli bypass the encoder:
    implant.encoder = AmplitudeEncoder(amp_range=(0, 50))
    for source in ({'A1': 20}, np.ones(60),
                   BiphasicPulse(20, 0.45, stim_dur=50)):
        stim = implant.prepare_stim(source)
        npt.assert_equal(stim.unit, uA)
        npt.assert_equal('encoder' in stim.metadata, False)

    # The encoder samples at this implant's electrodes and uses its raster.
    # amp_range > 0 keeps every electrode active, so schedules are the raster
    # groups:
    implant.encoder = AmplitudeEncoder(amp_range=(10, 50))
    implant.raster = implants.SequentialRaster(6)
    stim = implant.prepare_stim(img)
    delays = [stim.time[np.argmax(stim.data[e] < 0)]
              for e in (0, 10, 20, 30, 40, 50)]
    npt.assert_almost_equal(delays, np.arange(6) * 50 / 6, decimal=2)
    npt.assert_equal(len(np.unique(np.abs(stim.data) > 0, axis=0)), 6)


def test_Implant_encoded_stim_is_one_object():
    """An encoded stimulus stores the pulse train and the requested values"""
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    implant = ArgusII()
    stim = implant.prepare_stim(img)
    # The result is the delivered pulse train (used by safety checks and
    # temporal models):
    npt.assert_equal(stim.time.size > 1, True)
    # The spatial view stores the requested amplitude per electrode: one
    # column, no waveform, no raster.
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
        # 6 Hz pulses at 20 fps leave some frames without a pulse:
        stim = implant.prepare_stim(vid)
    npt.assert_equal(stim._spatial_view().shape, (60, 4))
    npt.assert_almost_equal(stim._spatial_view().time, np.arange(4) * 50.0)

    # Encoding by hand, then preparing, gives the same spatial view:
    by_hand = ArgusII(encoder=None).prepare_stim(
        AmplitudeEncoder(ArgusII(), amp_range=(0, 50)).encode(img))
    npt.assert_almost_equal(by_hand._spatial_view().data,
                            ArgusII().prepare_stim(img)._spatial_view().data)
    # A current stimulus is its own spatial view:
    for source in ({'A1': 20}, np.ones(60)):
        stim = implant.prepare_stim(source)
        npt.assert_equal(stim._spatial_view() is stim, True)
        npt.assert_equal(stim._has_spatial_view, False)

    # Deactivation applies to both the pulse train and the spatial view:
    implant = ArgusII()
    implant.deactivate(['A1', 'B2'])
    stim = implant.prepare_stim(img)
    npt.assert_equal(stim._has_spatial_view, True)
    npt.assert_equal(stim.shape[0], 58)
    npt.assert_equal(stim._spatial_view().shape[0], 58)
    npt.assert_equal('A1' in stim._spatial_view().electrodes, False)


def test_Implant_preprocess_crosses_the_boundary():
    """Preprocessing may convert a picture into current before encoding"""
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    bare = ArgusII(encoder=None, raster=None)
    encoder = AmplitudeEncoder(bare, amp_range=(0, 20), freq=20)
    implant = ArgusII(safe_mode=True, encoder=None, raster=None,
                      preprocess=encoder.encode)
    implant.max_current = 100 * mA
    stim = implant.prepare_stim(img)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.is_charge_balanced, True)
    # Already-encoded output from preprocessing is not encoded again:
    encoded = encoder.encode(img)
    twice = ArgusII(raster=None, preprocess=encoder.encode)
    npt.assert_almost_equal(twice.prepare_stim(img).data, encoded.data)
    # Same encoded stimulus, with a max_current low enough to fail:
    implant = ArgusII(preprocess=False, safe_mode=True)
    implant.max_current = 100 * uA
    with pytest.raises(ValueError) as excinfo:
        implant.prepare_stim(encoded)
    npt.assert_equal('exceeds max_current' in str(excinfo.value), True)
    implant.max_current = 2 * mA
    npt.assert_equal(implant.prepare_stim(encoded).unit, uA)


def test_Implant_historical_stimuli_unchanged():
    """A plain stimulus is treated as current and checked for safety"""
    implant = ArgusII(preprocess=False, safe_mode=True)
    npt.assert_equal(implant.prepare_stim({'A1': BiphasicPulse(50, 0.45)}).unit,
                     uA)
    with pytest.raises(ValueError) as excinfo:
        implant.prepare_stim({'A1': MonophasicPulse(50, 0.45)})
    npt.assert_equal('charge-balanced' in str(excinfo.value), True)
    # A plain number is in uA, and so is max_current:
    implant = ArgusII(preprocess=False)
    implant.max_current = 60
    source = {name: 2 for name in ArgusII().electrode_names}
    with pytest.raises(ValueError) as excinfo:
        implant.prepare_stim(source)
    npt.assert_equal('draws 120.0 uA at once' in str(excinfo.value), True)
    implant.max_current = 0.2 * mA
    npt.assert_equal(implant.prepare_stim(source).unit, uA)


def test_Implant_deactivated_electrodes_do_not_mutate_the_source():
    # Removing deactivated electrodes works on a copy. A pulse stimulus that
    # loses its electrode becomes a plain Stimulus (and does not fail):
    pulse = BiphasicPulse(20, 0.45, electrode='A1')
    implant = ArgusII()
    implant.deactivate('A1')
    stim = implant.prepare_stim(pulse)
    npt.assert_equal(type(stim), Stimulus)
    npt.assert_equal(stim.shape[0], 0)
    # The caller's pulse is unchanged:
    npt.assert_equal(type(pulse), BiphasicPulse)
    npt.assert_equal(pulse.shape[0], 1)
    npt.assert_almost_equal(pulse.amp, 20)

    # With all electrodes active, the pulse keeps its type:
    npt.assert_equal(type(ArgusII().prepare_stim(pulse)), BiphasicPulse)

    # A plain Stimulus source is not mutated either way:
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

    # Nothing deactivated, nothing removed:
    source = Stimulus({'A1': 10, 'B2': 20})
    stim = ArgusII().prepare_stim(source)
    npt.assert_equal(sorted(str(e) for e in stim.electrodes), ['A1', 'B2'])


def test_Implant_deactivated_electrode_does_not_render_the_others():
    # Deactivating an electrode drops its pulse train entry, so the remaining
    # trains stay unrendered:
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
    # The waveform still matches the trains:
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
    # The getter returns a copy:
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
    implant = ArgusII(eye='left', thresholds={'A10': 80})
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
    # None entries are dropped:
    implant.thresholds = {'A1': 80 * uA, 'A2': None}
    npt.assert_equal(sorted(implant.thresholds), ['A1'])


def test_Implant_thresholds_calibrate_pulse_trains():
    implant = ArgusII()
    source = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45),
              'A2': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    # Without thresholds, the unit is xTh, not a current:
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
    # Clearing falls back to the train's own threshold_amp:
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
    """Thresholds are checked against max_current at the next prepare_stim"""
    implant = ArgusII()
    train = {'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45)}
    implant.max_current = 250
    implant.thresholds = {'A1': 90 * uA}
    stim = implant.prepare_stim(train)
    npt.assert_almost_equal(stim._structured_sources()[0][1].amp, 180)
    # 2 * 200 uA exceeds max_current:
    implant.thresholds = {'A1': 200 * uA}
    with pytest.raises(ValueError):
        implant.prepare_stim(train)
    # The previously returned stimulus is unchanged:
    npt.assert_almost_equal(stim._structured_sources()[0][1].amp, 180)


def test_Implant_thresholds_do_not_render_the_stimulus():
    implant = ArgusII()
    implant.thresholds = 80 * uA
    stim = implant.prepare_stim({'A1': BiphasicPulseTrain(20, 2 * xTh, 0.45)})
    source = stim._structured_sources()[0][1]
    # `data is None` means no waveform has been generated:
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
    """Each prepare_stim calibrates from the source, not the previous result

    Otherwise the 2 xTh factor would compound.
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
    # A current-valued train stays in uA, with or without thresholds:
    uA_source = {'A1': BiphasicPulseTrain(20, 160 * uA, 0.45),
                 'A2': BiphasicPulseTrain(20, 160 * uA, 0.45)}
    stim = implant.prepare_stim(uA_source)
    factors = [src.amp_factor for _, src in stim._structured_sources()]
    npt.assert_equal(factors, [2, None])


@pytest.mark.parametrize('cls,expected', [
    (retina.ArgusII, 'epiretinal'),
    (retina.IMIE, 'epiretinal'),
    (retina.AlphaAMS, 'subretinal'),
    (retina.Lorach2015Array, 'subretinal'),
    (retina.Suprachoroidal24, 'suprachoroidal'),
    (cortex.Orion, 'epicortical'),
    (cortex.NeuroPortArray, 'intracortical'),
    (cortex.ICVP, 'intracortical'),
])
def test_named_devices_say_where_they_sit(cls, expected):
    npt.assert_equal(cls.placement, expected)
    npt.assert_equal(cls().placement, expected)


@pytest.mark.parametrize('cls,expected', [
    # Camera is head-mounted, so it does not move with the eye:
    (retina.ArgusI, 'head'),
    (retina.ArgusII, 'head'),
    (retina.Suprachoroidal24, 'head'),
    (retina.Suprachoroidal44, 'head'),
    (retina.IMIE, 'head'),
    # Photodiode arrays receive light through the eye's optics (PRIMA projects
    # its camera image through the eye):
    (retina.AlphaIMS, 'eye'),
    (retina.AlphaAMS, 'eye'),
    (retina.PRIMAPivotal, 'eye'),
    (retina.Lorach2015Array, 'eye'),
    (retina.Ho2019FlatArray, 'eye'),
    (retina.Huang2021Array, 'eye'),
])
def test_named_devices_say_how_gaze_reaches_them(cls, expected):
    npt.assert_equal(cls._default_scene_input_frame, expected)


def test_a_generic_array_moves_its_input_with_the_eye():
    # Only a head-fixed camera decouples input from gaze; a bare grid has none:
    npt.assert_equal(implants.GridImplant._default_scene_input_frame, 'eye')


def test_one_system_can_override_how_gaze_reaches_it():
    # Eye-tracked Argus II:
    tracked = retina.ArgusII()
    tracked.scene_input_frame = 'eye'
    npt.assert_equal(tracked.scene_input_frame, 'eye')
    npt.assert_equal(retina.ArgusII().scene_input_frame, 'head')
    npt.assert_equal('scene_input_frame' in str(tracked), True)
    # None restores the class default:
    tracked.scene_input_frame = None
    npt.assert_equal(tracked.scene_input_frame, 'head')
    npt.assert_equal('scene_input_frame' in str(tracked), False)
    with pytest.raises(ValueError):
        tracked.scene_input_frame = 'retinal'
    # A generic array takes it at construction:
    grid = implants.GridImplant(shape=(2, 2), spacing=500,
                                scene_input_frame='head')
    npt.assert_equal(grid.scene_input_frame, 'head')


def test_a_generic_array_says_nothing_about_placement():
    # `placement` is set only for named devices; models read `None` as
    # unspecified:
    npt.assert_equal(implants.GridImplant(shape=(2, 2), spacing=500).placement, None)
    npt.assert_equal(
        implants.Implant(implants.PointSource(0, 0, 0)).placement,
        None)


def test_Implant_electrode_array_is_the_canonical_name():
    """The array is `electrode_array`; `earray` was removed"""
    array = ElectrodeArray(DiskElectrode(0, 0, 0, 100))
    implant = Implant(electrode_array=array)
    npt.assert_equal(implant.electrode_array is array, True)
    npt.assert_equal('electrode_array=' in str(implant), True)
    npt.assert_equal(hasattr(implant, 'earray'), False)
    npt.assert_equal(hasattr(ArgusII(), 'earray'), False)
    with pytest.raises(TypeError):
        Implant(earray=array)


def test_Implant_is_a_container():
    """Implant supports len() and indexing like its electrode array"""
    implant = ArgusII()
    npt.assert_equal(len(implant), 60)
    npt.assert_equal(len(implant), len(implant.electrode_array))
    npt.assert_equal(implant['A1'] is implant[0], True)
    npt.assert_equal(implant['F10'] is implant[-1], True)
    with pytest.raises(KeyError):
        implant['Z99']
    with pytest.raises(IndexError):
        implant[60]
    with pytest.raises(TypeError):
        implant[1.2]


#: Every named device; each describes its geometry about its own origin.
NAMED_IMPLANTS = [
    retina.ArgusI, retina.ArgusII, retina.AlphaIMS, retina.AlphaAMS,
    retina.Suprachoroidal24, retina.Suprachoroidal44, retina.IMIE,
    retina.PRIMAPivotal,
    retina.Lorach2015Array, partial(retina.Ho2019FlatArray, 55),
    partial(retina.Huang2021Array, 55), cortex.NeuroPortArray, cortex.ICVP,
    cortex.Orion,
]


def _name_of(implant_type):
    return getattr(implant_type, '__name__', repr(implant_type))


@pytest.mark.parametrize('implant_type', NAMED_IMPLANTS)
def test_a_named_implant_has_no_whole_device_placement(implant_type):
    """Named implants have no x/y arguments; placement is set on the model"""
    params = signature(implant_type).parameters
    for name in ('x', 'y'):
        npt.assert_equal(name in params, False,
                         err_msg=f'{_name_of(implant_type)}.{name}')


@pytest.mark.parametrize('implant_type', NAMED_IMPLANTS)
def test_a_named_implant_is_built_around_its_own_origin(implant_type):
    """Named implant footprints include the device-local origin."""
    xy = implant_type().electrode_array.coordinates()[:, :2]
    # Suprachoroidal return electrodes sit far to one side, so check that the
    # origin lies inside the footprint instead of checking the bounding box:
    npt.assert_array_less(xy.min(axis=0), 1e-9,
                          err_msg=_name_of(implant_type))
    npt.assert_array_less(-1e-9, xy.max(axis=0),
                          err_msg=_name_of(implant_type))


def test_model_side_placement_reproduces_an_old_absolute_position():
    """Model-side placement reproduces the former Argus II pose."""
    implant = retina.ArgusII()
    rot = -28.4
    model = ScoreboardSpatial(implant, implant_position=(-1331, -850) * um,
                              implant_rotation=rot, implant_depth=100 * um,
                              xrange=(-5, 5), yrange=(-5, 5), step=1)
    stim = implant.prepare_stim({'A1': 1})
    placed = np.column_stack(
        model._electrode_coords(implant.electrode_array, stim,
                                electrodes=implant.electrode_names))
    local = implant.electrode_array.coordinates()
    th = np.deg2rad(rot)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    npt.assert_almost_equal(placed[:, :2],
                            (R @ local[:, :2].T).T + [-1331, -850], decimal=3)
    npt.assert_almost_equal(placed[:, 2], local[:, 2] + 100, decimal=3)
    # The implant geometry is unchanged:
    npt.assert_array_equal(implant.electrode_array.coordinates(), local)


def test_one_implant_serves_two_models_at_different_depths():
    """Placement depth preserves local z differences."""
    array = ElectrodeArray([PointSource(0, 0, 0), PointSource(200, 0, 40),
                            PointSource(400, 0, -25)])
    implant = Implant(array)
    stim = implant.prepare_stim({name: 1 for name in implant.electrode_names})
    local = implant.electrode_array.coordinates()

    def z_of(**params):
        model = ScoreboardSpatial(implant, xrange=(-2, 2), yrange=(-2, 2),
                                  step=1, **params)
        return np.asarray(model._electrode_coords(implant.electrode_array,
                                                  stim)[2], dtype=float)

    shallow, deep = z_of(implant_depth=0), z_of(implant_depth=150 * um)
    npt.assert_almost_equal(shallow, local[:, 2], decimal=3)
    npt.assert_almost_equal(deep - shallow, 150, decimal=3)
    # Placement does not flatten a non-planar array:
    npt.assert_almost_equal(np.diff(deep), np.diff(local[:, 2]), decimal=3)
    npt.assert_array_equal(implant.electrode_array.coordinates(), local)


def test_a_flat_named_array_is_flat_in_its_own_frame():
    """Flat named retinal arrays use z=0 in their local frame."""
    for implant_type in (retina.AlphaIMS, retina.AlphaAMS,
                         retina.ArgusI, retina.ArgusII,
                         retina.Suprachoroidal24, retina.Suprachoroidal44,
                         retina.IMIE,
                         retina.PRIMAPivotal, retina.Lorach2015Array,
                         partial(retina.Ho2019FlatArray, 55),
                         partial(retina.Huang2021Array, 55)):
        z = implant_type().electrode_array.coordinates()[:, 2]
        npt.assert_almost_equal(z, 0, decimal=9,
                                err_msg=_name_of(implant_type))
    # Fixed shank depths are kept:
    for implant_type, depths in [(cortex.NeuroPortArray, {-1500.0}),
                                 (cortex.ICVP, {-650.0, -850.0})]:
        z = implant_type().electrode_array.coordinates()[:, 2]
        npt.assert_equal(set(np.round(z, 6)), depths)
