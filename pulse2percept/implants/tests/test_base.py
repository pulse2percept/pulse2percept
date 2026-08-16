import numpy as np
import collections as coll
import pytest
import numpy.testing as npt
from pulse2percept import implants
from pulse2percept.implants import cortex
from pulse2percept.units import (DimensionMismatchError, Quantity,
                                 dimensionless, dva, mA, mm, ms, nA, uA, um)
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

from pulse2percept.implants import (PointSource, ElectrodeArray, ElectrodeGrid,
                                    ProsthesisSystem, RectangleImplant,
                                    PhotovoltaicPixel)
from pulse2percept.stimuli import Stimulus, ImageStimulus, VideoStimulus, LogoBVL
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   MonophasicPulse)
from pulse2percept.implants import (ArgusII, DiskElectrode)
from pulse2percept.models import ScoreboardModel


class PhotovoltaicArray(ProsthesisSystem):
    def __init__(self, x=0, y=0, z=-100, r=5, spacing=40, rot=0,
                 stim=None, preprocess=False, safe_mode=False):
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
                                    rot=rot, type='hex',
                                    orientation='vertical',
                                    etype=PhotovoltaicPixel, r=elec_radius,
                                    a=(self.spacing - self.trench) / 2)

        rm_names = []
        for name, electrode in self.earray.electrodes.items():
            if (electrode.x - x) ** 2 + (electrode.y - y) ** 2 > (r * dva2ret) ** 2:
                rm_names.append(name)
        for e in rm_names:
            self.earray.remove_electrode(e)

        # Beware of race condition: Stim must be set last, because it requires
        # indexing into self.electrodes:
        self.stim = stim


def test_ProsthesisSystem():
    # Invalid instantiations:
    with pytest.raises(ValueError):
        ProsthesisSystem(ElectrodeArray(PointSource(0, 0, 0)),
                         eye='both')
    with pytest.raises(TypeError):
        ProsthesisSystem(Stimulus)

    # Iterating over the electrode array:
    implant = ProsthesisSystem(PointSource(0, 0, 0))
    npt.assert_equal(implant.n_electrodes, 1)
    npt.assert_equal(implant[0], implant.earray[0])
    npt.assert_equal(implant.electrode_names, implant.earray.electrode_names)
    for i, e in zip(implant, implant.earray):
        npt.assert_equal(i, e)

    # Set a stimulus after the constructor:
    npt.assert_equal(implant.stim, None)
    implant.stim = 3
    npt.assert_equal(isinstance(implant.stim, Stimulus), True)
    npt.assert_equal(implant.stim.shape, (1, 1))
    npt.assert_equal(implant.stim.time, None)
    npt.assert_equal(implant.stim.electrodes, [0])

    plt.cla()
    ax = implant.plot()
    npt.assert_equal(len(ax.texts), 0)
    npt.assert_equal(len(ax.collections), 1)

    with pytest.raises(ValueError):
        # Wrong number of stimuli
        implant.stim = [1, 2]
    with pytest.raises(TypeError):
        # Invalid stim type:
        implant.stim = "stim"
    # Invalid electrode names:
    with pytest.raises(ValueError):
        implant.stim = {'A1': 1}
    with pytest.raises(ValueError):
        implant.stim = Stimulus({'A1': 1})
    # Safe mode requires charge-balanced pulses:
    with pytest.raises(ValueError):
        implant = ProsthesisSystem(PointSource(0, 0, 0), safe_mode=True)
        implant.stim = 1

    # Slots:
    npt.assert_equal(hasattr(implant, '__slots__'), True)
    npt.assert_equal(hasattr(implant, '__dict__'), False)


def test_ProsthesisSystem_stim():
    implant = ProsthesisSystem(ElectrodeGrid((13, 13), 20))
    stim = Stimulus(np.ones((13 * 13 + 1, 5)))
    with pytest.raises(ValueError):
        implant.stim = stim

    # make sure empty stimulus causes None stim
    implant.stim = []
    npt.assert_equal(implant.stim, None)
    implant.stim = {}
    npt.assert_equal(implant.stim, None)
    implant.stim = np.array([])
    npt.assert_equal(implant.stim, None)

    # color mapping
    stim = np.zeros((13*13, 5))
    stim[84, 0] = 1
    stim[98, 2] = 2
    implant.stim = stim
    plt.cla()
    ax = implant.plot(stim_cmap='hsv')
    plt.colorbar()
    npt.assert_equal(len(ax.collections), 1)
    npt.assert_equal(ax.collections[0].colorbar.vmax, 2)
    npt.assert_equal(ax.collections[0].cmap(ax.collections[0].norm(1)),
                     (0.0, 1.0, 0.9647031631761764, 1))
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
    implant.stim = {'H4': 1}
    npt.assert_equal('H4' in implant.stim.electrodes, False)

    implant.deactivate('all')
    npt.assert_equal(implant.stim.data.size == 0, True)
    implant.activate('all')
    implant.stim = {'H4': 1}
    npt.assert_equal('H4' in implant.stim.electrodes, True)


@pytest.mark.parametrize('rot', (0, 30, 92))
@pytest.mark.parametrize('gtype', ('hex', 'rect'))
@pytest.mark.parametrize('n_frames', (1, 3, 4))
def test_ProsthesisSystem_reshape_stim(rot, gtype, n_frames):
    implant = ProsthesisSystem(ElectrodeGrid((10, 10), 30, rot=rot, type=gtype))
    # Smoke test the automatic reshaping:
    n_px = 21
    implant.stim = ImageStimulus(np.ones((n_px, n_px, n_frames)).squeeze())
    npt.assert_equal(implant.stim.data.shape, (implant.n_electrodes, 1))
    npt.assert_equal(implant.stim.time, None)
    implant.stim = VideoStimulus(np.ones((n_px, n_px, 3 * n_frames)),
                                 time=2 * np.arange(3 * n_frames))
    npt.assert_equal(implant.stim.data.shape,
                     (implant.n_electrodes, 3 * n_frames))
    npt.assert_equal(implant.stim.time, 2 * np.arange(3 * n_frames))

    # Verify that a horizontal stimulus will always appear horizontally, even if
    # the device is rotated:
    data = np.zeros((50, 50))
    data[20:-20, 10:-10] = 1
    implant.stim = ImageStimulus(data)
    model = ScoreboardModel(xrange=(-1, 1), yrange=(-1, 1), rho=30, xystep=0.02)
    model.build()
    percept = label(model.predict_percept(implant).data.squeeze().T > 0.2)
    npt.assert_almost_equal(regionprops(percept)[0].orientation, 0, decimal=1)

    # Smoke test a large hex grid (old code results in MemoryError):
    implant = PhotovoltaicArray(r=2, spacing=40, rot=rot)
    implant.stim = LogoBVL()


def test_ProsthesisSystem_deactivate():
    implant = ProsthesisSystem(ElectrodeGrid((10, 10), 30))
    implant.stim = np.ones(implant.n_electrodes)
    electrode = 'A3'
    npt.assert_equal(electrode in implant.stim.electrodes, True)
    implant.deactivate(electrode)
    npt.assert_equal(implant[electrode].activated, False)
    npt.assert_equal(electrode in implant.stim.electrodes, False)

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
        npt.assert_almost_equal(implant[e].r, 112.5)

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

    # Set a stimulus via dict:
    implant = RectangleImplant(stim={'B7': 13})
    npt.assert_equal(implant.stim.shape, (1, 1))
    npt.assert_equal(implant.stim.electrodes, ['B7'])

    # Set a stimulus via array:
    implant = RectangleImplant(stim=np.ones(225))
    npt.assert_equal(implant.stim.shape, (225, 1))
    npt.assert_almost_equal(implant.stim.data, 1)

    # test different shapes
    for shape in [(6, 10), (5, 12), (15, 15)]:
        implant = RectangleImplant(shape=shape)
        npt.assert_equal(implant.earray.shape, shape)


def test_ProsthesisSystem_reshape_stim_frames_independent():
    """Downsampling a video must treat each frame on its own.

    ``reshape_stim`` builds one interpolator for the whole video rather than
    one per frame, so this checks that a frame lands on the electrodes the
    same way whether it arrives alone or inside a sequence.
    """
    rng = np.random.default_rng(3)
    n_frames = 5
    vid = rng.random((24, 31, n_frames)).astype(np.float32)
    implant = ProsthesisSystem(ElectrodeGrid((6, 8), 200))

    implant.stim = VideoStimulus(vid, time=np.arange(n_frames))
    joint = implant.stim.data
    npt.assert_equal(joint.shape, (implant.n_electrodes, n_frames))

    for f in range(n_frames):
        implant.stim = ImageStimulus(vid[..., f])
        npt.assert_allclose(implant.stim.data[:, 0], joint[:, f], rtol=1e-5,
                            atol=1e-7)

    # Pixels outside the electrode footprint are filled with zero, not
    # extrapolated, so an all-zero frame stays all zero:
    vid[..., 2] = 0
    implant.stim = VideoStimulus(vid, time=np.arange(n_frames))
    npt.assert_equal(np.all(implant.stim.data[:, 2] == 0), True)


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
        (implants.PRIMA, {'z': -0.1 * mm}, {'z': -100}),
        (implants.PRIMA75, {'z': -0.1 * mm}, {'z': -100}),
        (implants.PRIMA55, {'z': -0.1 * mm}, {'z': -100}),
        (implants.PRIMA40, {'z': -0.1 * mm}, {'z': -100}),
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


def test_implant_per_electrode_z_units():
    """A per-electrode list of heights never reaches ElectrodeGrid"""
    for cls, n in [(implants.PRIMA, 378), (implants.PRIMA75, 142),
                   (implants.AlphaIMS, 1500)]:
        heights = np.linspace(-150, -50, n)
        unitful = cls(z=[h * um for h in heights])
        npt.assert_allclose(unitful.earray.coordinates(),
                            cls(z=list(heights)).earray.coordinates(),
                            rtol=1e-12, err_msg=cls.__name__)
        npt.assert_allclose(unitful.earray.coordinates()[:, 2], heights,
                            rtol=1e-12)


def test_implant_dimension_errors():
    for cls in (implants.ArgusII, implants.PRIMA, implants.BVT24,
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


def test_ProsthesisSystem_max_current_units():
    """`max_current` is a current, stored as a plain number of microamps"""
    earray = ElectrodeArray(DiskElectrode(0, 0, 0, 100))
    for value in (100, 100 * uA, 0.1 * mA, 100000 * nA):
        implant = ProsthesisSystem(earray, max_current=value)
        npt.assert_allclose(implant.max_current, 100, rtol=1e-12)
        npt.assert_equal(isinstance(implant.max_current, Quantity), False)
    # An awkward conversion is no different:
    npt.assert_allclose(
        ProsthesisSystem(earray, max_current=0.0417 * mA).max_current, 41.7,
        rtol=1e-12)
    # None means no limit, and is left alone:
    npt.assert_equal(ProsthesisSystem(earray).max_current, None)
    # Assigning later goes through the same setter:
    implant = ProsthesisSystem(earray)
    implant.max_current = 0.1 * mA
    npt.assert_allclose(implant.max_current, 100, rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        ProsthesisSystem(earray, max_current=5 * ms)
    with pytest.raises(DimensionMismatchError):
        implant.max_current = 5 * dva
    with pytest.raises(ValueError):
        ProsthesisSystem(earray, max_current=-1 * uA)


def test_ProsthesisSystem_safety_checks_are_electrical():
    """Electrical safety may only be asked about an electrical stimulus"""
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    vid = VideoStimulus(np.ones((6, 10, 3)) * 0.5, time=[0, 20, 40])

    # No electrical policy requested, so no electrical question is asked and a
    # picture may still be assigned. This is what keeps preprocessing
    # workflows, which turn images into current, working:
    implant = ArgusII(preprocess=False, safe_mode=False)
    implant.stim = img
    npt.assert_equal(implant.stim.unit, dimensionless)
    implant.stim = vid
    npt.assert_equal(implant.stim.unit, dimensionless)

    # `safe_mode` is a claim about electricity, and cannot be made about a
    # picture -- it must not integrate gray levels and pronounce them safe:
    implant = ArgusII(preprocess=False, safe_mode=True)
    with pytest.raises(DimensionMismatchError) as excinfo:
        implant.stim = img
    npt.assert_equal("Safety check 'safe_mode'" in str(excinfo.value), True)
    npt.assert_equal('dimensionless' in str(excinfo.value), True)

    # ... and so is `max_current`:
    implant = ArgusII(preprocess=False, safe_mode=False)
    implant.max_current = 100 * uA
    with pytest.raises(DimensionMismatchError) as excinfo:
        implant.stim = img
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


def test_ProsthesisSystem_preprocess_crosses_the_boundary():
    """Preprocessing may turn a picture into current before safety sees it"""
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    encoder = AmplitudeEncoder(ArgusII(), amp_range=(0, 20), freq=20)
    implant = ArgusII(safe_mode=True, preprocess=lambda x: encoder.encode(x))
    implant.max_current = 100 * mA
    implant.stim = img
    npt.assert_equal(implant.stim.unit, uA)
    npt.assert_equal(implant.stim.is_charge_balanced, True)
    # The same chain, assigned already-encoded, and this time with a limit
    # tight enough to matter:
    implant = ArgusII(preprocess=False, safe_mode=True)
    implant.max_current = 100 * uA
    with pytest.raises(ValueError) as excinfo:
        implant.stim = encoder.encode(img)
    npt.assert_equal('exceeds max_current' in str(excinfo.value), True)
    implant.max_current = 2 * mA
    implant.stim = encoder.encode(img)
    npt.assert_equal(implant.stim.unit, uA)


def test_ProsthesisSystem_historical_stimuli_unchanged():
    """A bare stimulus is electrical by contract, and is checked as before"""
    implant = ArgusII(preprocess=False, safe_mode=True)
    implant.stim = {'A1': BiphasicPulse(50, 0.45)}
    npt.assert_equal(implant.stim.unit, uA)
    with pytest.raises(ValueError) as excinfo:
        implant.stim = {'A1': MonophasicPulse(50, 0.45)}
    npt.assert_equal('charge-balanced' in str(excinfo.value), True)
    # A plain number is microamps, and the limit is read the same way:
    implant = ArgusII(preprocess=False)
    implant.max_current = 60
    with pytest.raises(ValueError) as excinfo:
        implant.stim = {name: 2 for name in ArgusII().electrode_names}
    npt.assert_equal('draws 120.0 uA at once' in str(excinfo.value), True)
    implant.max_current = 0.2 * mA
    implant.stim = {name: 2 for name in ArgusII().electrode_names}
    npt.assert_equal(implant.stim.unit, uA)
