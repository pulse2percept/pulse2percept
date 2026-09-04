import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept import implants
from pulse2percept.implants import SequentialRaster
from pulse2percept.models import AxonMapModel
from pulse2percept.stimuli import AmplitudeEncoder, BostonTrain, LogoBVL
from pulse2percept.units import DimensionMismatchError, uA


@pytest.mark.parametrize('ztype', ('float', 'list'))
def test_ArgusI(ztype):
    # Create an ArgusI and make sure location is correct
    # Height `z` can either be a float or a list
    z = 100 if ztype == 'float' else np.ones(16) * 20

    argus = implants.ArgusI(z=z)

    # Slots:
    npt.assert_equal(hasattr(argus, '__slots__'), True)
    npt.assert_equal(hasattr(argus, '__dict__'), False)

    # Coordinates of first electrode, in the device's own frame
    xy = np.array([-1200, -1200]).T
    npt.assert_almost_equal(argus['A1'].x, xy[0])
    npt.assert_almost_equal(argus['A1'].y, xy[1])

    # The array is centered on the device's own origin
    y_center = argus['D1'].y + (argus['A4'].y - argus['D1'].y) / 2
    npt.assert_almost_equal(y_center, 0)
    x_center = argus['A1'].x + (argus['D4'].x - argus['A1'].x) / 2
    npt.assert_almost_equal(x_center, 0)

    # Check radii of electrodes
    for e in ['A1', 'A3', 'B2', 'C1', 'D4']:
        npt.assert_almost_equal(argus[e].radius, 125)
    for e in ['A2', 'A4', 'B1', 'C2', 'D3']:
        npt.assert_almost_equal(argus[e].radius, 250)

    # Check location of the tack
    tack = tuple(np.array([-2000, 0]) + [x_center, y_center])

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        implants.ArgusI(z=np.zeros(5))
    with pytest.raises(ValueError):
        implants.ArgusI(z=[1, 2, 3])

    # Indexing must work for both integers and electrode names
    for use_legacy_names in [True, False]:
        argus = implants.ArgusI(use_legacy_names=use_legacy_names)
        for idx, (name, electrode) in enumerate(argus.electrodes.items()):
            npt.assert_equal(electrode, argus[idx])
            npt.assert_equal(electrode, argus[name])
        with pytest.raises(KeyError):
            argus["unlikely name for an electrode"]

    # Right-eye implant:
    argus_re = implants.ArgusI(eye='RE')
    npt.assert_equal(argus_re['D1'].x > argus_re['A1'].x, True)
    npt.assert_almost_equal(argus_re['D1'].y, argus_re['A1'].y)

    # need to adjust for reflection about y-axis
    # Left-eye implant:
    argus_le = implants.ArgusI(eye='LE')
    npt.assert_equal(argus_le['A1'].x > argus_le['D4'].x, True)
    npt.assert_almost_equal(argus_le['D1'].y, argus_le['A1'].y)

    # Check naming scheme
    argus = implants.ArgusI(use_legacy_names=False)
    npt.assert_equal(argus.electrode_names[15], 'D4')
    npt.assert_equal(argus.electrode_names[0], 'A1')

    argus = implants.ArgusI(use_legacy_names=True)
    npt.assert_equal(argus.electrode_names[15], 'M1')
    npt.assert_equal(argus.electrode_names[0], 'L6')

    # Prepare a stimulus via dict:
    stim = implants.ArgusI().prepare_stim({'B3': 13})
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.electrodes, ['B3'])

    # Prepare a stimulus via array:
    stim = implants.ArgusI().prepare_stim(np.ones(16))
    npt.assert_equal(stim.shape, (16, 1))
    npt.assert_almost_equal(stim.data, 1)


@pytest.mark.parametrize('ztype', ('float', 'list'))
def test_ArgusII(ztype):
    # Create an ArgusII and make sure location is correct
    # Height `h` can either be a float or a list
    z = 100 if ztype == 'float' else np.ones(60) * 20
    argus = implants.ArgusII(z=z)

    # Slots:
    npt.assert_equal(hasattr(argus, '__slots__'), True)
    npt.assert_equal(hasattr(argus, '__dict__'), False)

    # Coordinates of first electrode, in the device's own frame
    xy = np.array([-2587.5, -1437.5]).T
    npt.assert_almost_equal(argus['A1'].x, xy[0])
    npt.assert_almost_equal(argus['A1'].y, xy[1])

    # The array is centered on the device's own origin
    y_center = argus['F1'].y + (argus['A10'].y - argus['F1'].y) / 2
    npt.assert_almost_equal(y_center, 0)
    x_center = argus['A1'].x + (argus['F10'].x - argus['A1'].x) / 2
    npt.assert_almost_equal(x_center, 0)

    # Make sure radius is correct
    for e in ['A1', 'B3', 'C5', 'D7', 'E9', 'F10']:
        npt.assert_almost_equal(argus[e].radius, 112.5)

    # `h` must have the right dimensions
    with pytest.raises(ValueError):
        implants.ArgusII(z=np.zeros(5))
    with pytest.raises(ValueError):
        implants.ArgusII(z=[1, 2, 3])

    # Indexing must work for both integers and electrode names
    argus = implants.ArgusII()
    for idx, (name, electrode) in enumerate(argus.electrodes.items()):
        npt.assert_equal(electrode, argus[idx])
        npt.assert_equal(electrode, argus[name])
    with pytest.raises(KeyError):
        argus["unlikely name for an electrode"]

    # Right-eye implant:
    argus_re = implants.ArgusII(eye='RE')
    npt.assert_equal(argus_re['A10'].x > argus_re['A1'].x, True)
    npt.assert_almost_equal(argus_re['A10'].y, argus_re['A1'].y)

    # Left-eye implant:
    argus_le = implants.ArgusII(eye='LE')
    npt.assert_equal(argus_le['A1'].x > argus_le['A10'].x, True)
    npt.assert_almost_equal(argus_le['A10'].y, argus_le['A1'].y)

    # Prepare a stimulus via dict:
    stim = implants.ArgusII().prepare_stim({'B7': 13})
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.electrodes, ['B7'])

    # Prepare a stimulus via array:
    stim = implants.ArgusII().prepare_stim(np.ones(60))
    npt.assert_equal(stim.shape, (60, 1))
    npt.assert_almost_equal(stim.data, 1)


def test_ArgusII_defaults():
    """Argus II brings its own encoder and raster, and each instance a fresh one
    """
    argus = implants.ArgusII()
    # 6 Hz amplitude modulation, which is the rate the device runs video at:
    npt.assert_equal(isinstance(argus.encoder, AmplitudeEncoder), True)
    npt.assert_almost_equal(argus.encoder.freq, 6)
    # Six sequential groups (one row of ten electrodes each), 2 ms apart:
    npt.assert_equal(isinstance(argus.raster, SequentialRaster), True)
    npt.assert_equal(argus.raster.n_groups, 6)
    npt.assert_almost_equal(argus.raster.group_dur, 2)
    npt.assert_equal(argus.raster.groups(argus.electrode_names),
                     np.repeat(np.arange(6), 10))
    # The raster is bound to the implant that owns it, so it plots itself:
    npt.assert_equal(argus.raster.implant is argus, True)

    # Each instance gets its own, so tweaking one implant's does not reach
    # every other Argus II in the session:
    other = implants.ArgusII()
    npt.assert_equal(other.encoder is argus.encoder, False)
    npt.assert_equal(other.raster is argus.raster, False)
    other.encoder.freq = 20
    npt.assert_almost_equal(argus.encoder.freq, 6)

    # An explicit None switches each feature off, and is told apart from the
    # argument simply not being given:
    npt.assert_equal(implants.ArgusII(encoder=None).encoder, None)
    npt.assert_equal(implants.ArgusII(raster=None).raster, None)
    npt.assert_equal(implants.ArgusII(raster=None).encoder is None, False)
    # ... and switching the raster off really does stop the multiplexing: every
    # electrode then fires on the same schedule, at the same instant.
    unrastered = implants.ArgusII(raster=None).prepare_stim(LogoBVL())
    npt.assert_equal(unrastered.metadata['encoder']['cycle'], None)
    # There is an instant at which every electrode is at its own peak, so the
    # stimulator has to source the whole array at once:
    npt.assert_almost_equal(np.abs(unrastered.data).sum(axis=0).max(),
                            np.abs(unrastered.data).max(axis=1).sum(),
                            decimal=3)
    rastered = implants.ArgusII().prepare_stim(LogoBVL())
    npt.assert_array_less(np.abs(rastered.data).sum(axis=0).max(),
                          np.abs(unrastered.data).sum(axis=0).max())
    # ... and either can be replaced outright:
    custom = implants.ArgusII(encoder=AmplitudeEncoder(freq=20),
                              raster=SequentialRaster(3))
    npt.assert_almost_equal(custom.encoder.freq, 20)
    npt.assert_equal(custom.raster.n_groups, 3)
    with pytest.raises(TypeError):
        implants.ArgusII(encoder='amplitude')
    with pytest.raises(TypeError):
        implants.ArgusII(raster='line')


def test_ArgusII_encodes_pictures_on_preparation():
    """The device's own defaults are what make `prepare_stim(picture)` work"""
    argus = implants.ArgusII()
    stim = argus.prepare_stim(LogoBVL())
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.shape[0], argus.n_electrodes)
    npt.assert_equal(list(stim.electrodes), list(argus.electrode_names))
    # 6 Hz over the 500 ms an image is treated as lasting is three pulses:
    npt.assert_almost_equal(stim.time[-1], 500)
    npt.assert_almost_equal(np.abs(stim.data).max(), 50, decimal=4)
    # The raster is in there too: six groups, each 2 ms behind the one before:
    npt.assert_almost_equal(stim.metadata['encoder']['cycle'], 12)
    # ... which is what a raster is for: at no instant is more than one group
    # of electrodes drawing current.
    groups = argus.raster.groups(stim.electrodes)
    for column in stim.data.T:
        npt.assert_equal(np.unique(groups[column != 0]).size <= 1, True)

    # A video keeps its own frame clock, which is what a model reports at:
    with pytest.warns(UserWarning, match='deliver no pulse'):
        # 6 Hz against 29.97 fps: most frames carry no pulse of their own
        stim = argus.prepare_stim(BostonTrain())
    npt.assert_equal(stim.unit, uA)
    meta = stim.metadata['encoder']
    npt.assert_equal(meta['frame_time'].size, 94)
    npt.assert_almost_equal(meta['frame_dur'], 1000 / 29.97, decimal=3)

    # Without an encoder the very same picture is refused, since there is no
    # default mapping from a gray level onto an amplitude:
    with pytest.raises(DimensionMismatchError):
        implants.ArgusII(encoder=None).prepare_stim(LogoBVL())

    # And the whole point of it: a picture goes straight into a model, with no
    # encoding step for the caller to spell out.
    model = AxonMapModel(implant=argus, xrange=(-4, 4), yrange=(-3, 3), step=1,
                         rho=200, lam=100).build()
    percept = model.predict_percept(LogoBVL())
    npt.assert_equal(percept.data.shape[:2], model.spatial.grid.x.shape)
    npt.assert_equal(np.any(percept.data > 0), True)
