import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept.implants import retina, SequentialRaster
from pulse2percept.models.retina import AxonMapModel
from pulse2percept.stimuli import AmplitudeEncoder, samples
from pulse2percept.units import DimensionMismatchError, uA


@pytest.mark.parametrize('ztype', ('float', 'list'))
def test_ArgusI(ztype):
    # Create an ArgusI and make sure location is correct
    # Height `z` can either be a float or a list
    z = 100 if ztype == 'float' else np.ones(16) * 20

    argus = retina.ArgusI(z=z)

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
        retina.ArgusI(z=np.zeros(5))
    with pytest.raises(ValueError):
        retina.ArgusI(z=[1, 2, 3])

    # Indexing must work for both integers and electrode names
    for use_legacy_names in [True, False]:
        argus = retina.ArgusI(use_legacy_names=use_legacy_names)
        for idx, (name, electrode) in enumerate(argus.electrodes.items()):
            npt.assert_equal(electrode, argus[idx])
            npt.assert_equal(electrode, argus[name])
        with pytest.raises(KeyError):
            argus["unlikely name for an electrode"]

    # Right-eye implant:
    argus_re = retina.ArgusI(eye='right')
    npt.assert_equal(argus_re['D1'].x > argus_re['A1'].x, True)
    npt.assert_almost_equal(argus_re['D1'].y, argus_re['A1'].y)

    # need to adjust for reflection about y-axis
    # Left-eye implant:
    argus_le = retina.ArgusI(eye='left')
    npt.assert_equal(argus_le['A1'].x > argus_le['D4'].x, True)
    npt.assert_almost_equal(argus_le['D1'].y, argus_le['A1'].y)

    # Check naming scheme
    argus = retina.ArgusI(use_legacy_names=False)
    npt.assert_equal(argus.electrode_names[15], 'D4')
    npt.assert_equal(argus.electrode_names[0], 'A1')

    argus = retina.ArgusI(use_legacy_names=True)
    npt.assert_equal(argus.electrode_names[15], 'M1')
    npt.assert_equal(argus.electrode_names[0], 'L6')

    # Prepare a stimulus via dict:
    stim = retina.ArgusI().prepare_stim({'B3': 13})
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.electrodes, ['B3'])

    # Prepare a stimulus via array:
    stim = retina.ArgusI().prepare_stim(np.ones(16))
    npt.assert_equal(stim.shape, (16, 1))
    npt.assert_almost_equal(stim.data, 1)


@pytest.mark.parametrize('ztype', ('float', 'list'))
def test_ArgusII(ztype):
    # Create an ArgusII and make sure location is correct
    # Height `h` can either be a float or a list
    z = 100 if ztype == 'float' else np.ones(60) * 20
    argus = retina.ArgusII(z=z)

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
        retina.ArgusII(z=np.zeros(5))
    with pytest.raises(ValueError):
        retina.ArgusII(z=[1, 2, 3])

    # Indexing must work for both integers and electrode names
    argus = retina.ArgusII()
    for idx, (name, electrode) in enumerate(argus.electrodes.items()):
        npt.assert_equal(electrode, argus[idx])
        npt.assert_equal(electrode, argus[name])
    with pytest.raises(KeyError):
        argus["unlikely name for an electrode"]

    # Right-eye implant:
    argus_re = retina.ArgusII(eye='right')
    npt.assert_equal(argus_re['A10'].x > argus_re['A1'].x, True)
    npt.assert_almost_equal(argus_re['A10'].y, argus_re['A1'].y)

    # Left-eye implant:
    argus_le = retina.ArgusII(eye='left')
    npt.assert_equal(argus_le['A1'].x > argus_le['A10'].x, True)
    npt.assert_almost_equal(argus_le['A10'].y, argus_le['A1'].y)

    # Prepare a stimulus via dict:
    stim = retina.ArgusII().prepare_stim({'B7': 13})
    npt.assert_equal(stim.shape, (1, 1))
    npt.assert_equal(stim.electrodes, ['B7'])

    # Prepare a stimulus via array:
    stim = retina.ArgusII().prepare_stim(np.ones(60))
    npt.assert_equal(stim.shape, (60, 1))
    npt.assert_almost_equal(stim.data, 1)


def test_ArgusII_defaults():
    """Argus II has a default encoder and raster, new for each instance"""
    argus = retina.ArgusII()
    # 6 Hz amplitude modulation (the device video rate):
    npt.assert_equal(isinstance(argus.encoder, AmplitudeEncoder), True)
    npt.assert_almost_equal(argus.encoder.freq, 6)
    # Six sequential groups (one row of ten electrodes each), 2 ms apart:
    npt.assert_equal(isinstance(argus.raster, SequentialRaster), True)
    npt.assert_equal(argus.raster.n_groups, 6)
    npt.assert_almost_equal(argus.raster.group_dur, 2)
    npt.assert_equal(argus.raster.groups(argus.electrode_names),
                     np.repeat(np.arange(6), 10))
    # The raster is bound to the implant, so `raster.plot()` needs no argument:
    npt.assert_equal(argus.raster.implant is argus, True)

    # Each instance has its own encoder and raster:
    other = retina.ArgusII()
    npt.assert_equal(other.encoder is argus.encoder, False)
    npt.assert_equal(other.raster is argus.raster, False)
    other.encoder.freq = 20
    npt.assert_almost_equal(argus.encoder.freq, 6)

    # An explicit None disables each one (distinct from omitting the argument):
    npt.assert_equal(retina.ArgusII(encoder=None).encoder, None)
    npt.assert_equal(retina.ArgusII(raster=None).raster, None)
    npt.assert_equal(retina.ArgusII(raster=None).encoder is None, False)
    # Without a raster, all electrodes fire on the same schedule:
    unrastered = retina.ArgusII(raster=None).prepare_stim(samples.logo_bvl())
    npt.assert_equal(unrastered.metadata['encoder']['cycle'], None)
    # At some instant every electrode is at its peak, so the whole array draws
    # current at once:
    npt.assert_almost_equal(np.abs(unrastered.data).sum(axis=0).max(),
                            np.abs(unrastered.data).max(axis=1).sum(),
                            decimal=3)
    rastered = retina.ArgusII().prepare_stim(samples.logo_bvl())
    npt.assert_array_less(np.abs(rastered.data).sum(axis=0).max(),
                          np.abs(unrastered.data).sum(axis=0).max())
    # Either can be replaced:
    custom = retina.ArgusII(encoder=AmplitudeEncoder(freq=20),
                              raster=SequentialRaster(3))
    npt.assert_almost_equal(custom.encoder.freq, 20)
    npt.assert_equal(custom.raster.n_groups, 3)
    with pytest.raises(TypeError):
        retina.ArgusII(encoder='amplitude')
    with pytest.raises(TypeError):
        retina.ArgusII(raster='line')


def test_ArgusII_encodes_pictures_on_preparation(camera_video):
    """The default encoder and raster let `prepare_stim` accept a picture"""
    argus = retina.ArgusII()
    stim = argus.prepare_stim(samples.logo_bvl())
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.shape[0], argus.n_electrodes)
    npt.assert_equal(list(stim.electrodes), list(argus.electrode_names))
    # An image lasts 500 ms, so 6 Hz gives three pulses:
    npt.assert_almost_equal(stim.time[-1], 500)
    npt.assert_almost_equal(np.abs(stim.data).max(), 50, decimal=4)
    # Raster: six groups, each 2 ms after the previous one:
    npt.assert_almost_equal(stim.metadata['encoder']['cycle'], 12)
    # At no instant does more than one group draw current:
    groups = argus.raster.groups(stim.electrodes)
    for column in stim.data.T:
        npt.assert_equal(np.unique(groups[column != 0]).size <= 1, True)

    # A video keeps its own frame clock (the model's output times):
    with pytest.warns(UserWarning, match='deliver no pulse'):
        # 6 Hz at 29.97 fps: most frames have no pulse
        stim = argus.prepare_stim(camera_video)
    npt.assert_equal(stim.unit, uA)
    meta = stim.metadata['encoder']
    npt.assert_equal(meta['frame_time'].size, 94)
    npt.assert_almost_equal(meta['frame_dur'], 1000 / 29.97, decimal=3)

    # Without an encoder, the picture is rejected (no default gray level to
    # amplitude mapping):
    with pytest.raises(DimensionMismatchError):
        retina.ArgusII(encoder=None).prepare_stim(samples.logo_bvl())

    # A picture can be passed directly to a model:
    model = AxonMapModel(implant=argus, xrange=(-4, 4), yrange=(-3, 3), step=1,
                         rho=200, lam=100).build()
    percept = model.predict_percept(samples.logo_bvl())
    npt.assert_equal(percept.data.shape[:2], model.spatial.grid.x.shape)
    npt.assert_equal(np.any(percept.data > 0), True)
