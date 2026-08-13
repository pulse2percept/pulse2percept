"""End-to-end tests you can check by hand

The encoder tests in ``pulse2percept/stimuli/tests`` drive Argus II with
``BostonTrain``, which exercises the plumbing but tells you nothing you could
have predicted with a pencil. These tests run the whole pipeline -- image ->
encoder -> implant -> model -> percept -- on a deliberately tiny setup where
every number has a closed form:

*  Four electrodes on the corners of a 1200 um square, far enough apart that
   each one produces its own phosphene.
*  A 2x2 image, so that ``reshape_stim`` samples exactly one pixel per
   electrode (it maps the image grid linearly onto the electrode bounding box,
   and the electrodes sit on its corners).
*  Gray levels chosen to be distinct and evenly separated.
*  One electrode per raster group, so the stimulator drives exactly one
   electrode at a time.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import trapezoid

from pulse2percept.implants import (CustomRaster, DiskElectrode, ElectrodeArray,
                                    ProsthesisSystem)
from pulse2percept.models import FadingTemporal, ScoreboardSpatial
from pulse2percept.stimuli import (AmplitudeEncoder, FrequencyEncoder,
                                   ImageStimulus)
from pulse2percept.utils.constants import DT

# Electrode names in the order `ElectrodeArray` keeps them, and their positions
# (um). The 2x2 image below is sampled at exactly these four points:
#     A = top-left pixel, B = top-right, C = bottom-left, D = bottom-right
NAMES = ['A', 'B', 'C', 'D']
POS = [(-600.0, -600.0), (600.0, -600.0), (-600.0, 600.0), (600.0, 600.0)]


def make_implant():
    """Four well-separated electrodes, one per raster group"""
    earray = ElectrodeArray({n: DiskElectrode(x, y, 0, 100)
                             for n, (x, y) in zip(NAMES, POS)})
    return ProsthesisSystem(earray)


def one_per_group():
    """A raster that drives exactly one electrode at a time"""
    return CustomRaster({n: i for i, n in enumerate(NAMES)})


def onsets(stim, electrode):
    """The time (ms) at which each of one electrode's pulses begins"""
    neg = stim.data[electrode] < 0
    started = neg & ~np.concatenate(([False], neg[:-1]))
    # A pulse ramps up over DT, so the first sample at full amplitude sits one
    # tick past the onset:
    return stim.time[started] - DT


def at_electrodes(model, implant):
    """The grid index nearest each electrode, and one in the middle"""
    gx = np.asarray(model.grid.ret.x)
    gy = np.asarray(model.grid.ret.y)
    here = {}
    for n in NAMES:
        e = implant[n]
        flat = int(np.argmin((gx - e.x) ** 2 + (gy - e.y) ** 2))
        here[n] = np.unravel_index(flat, gx.shape)
    middle = np.unravel_index(int(np.argmin(gx ** 2 + gy ** 2)), gx.shape)
    return here, middle


def test_endtoend_amplitude_modulation():
    # Four gray levels, evenly spaced, one per electrode:
    implant = make_implant()
    img = ImageStimulus(np.array([[0.25, 0.50], [0.75, 1.00]]))
    stim = AmplitudeEncoder(implant, amp_range=(0, 50), freq=20,
                            raster=one_per_group(),
                            frame_dur=200).encode(img)
    implant.stim = stim
    npt.assert_equal(list(stim.electrodes), NAMES)

    # --- what the encoder produced --------------------------------------
    # Gray level maps onto amplitude absolutely: 50 uA * gray.
    npt.assert_almost_equal(np.abs(stim.data).max(axis=1),
                            [12.5, 25.0, 37.5, 50.0], decimal=4)
    # Every electrode pulses at the requested 20 Hz -- rastering costs no
    # frequency, it only decides where in each 50 ms period an electrode goes.
    # Four groups split that period into four 12.5 ms slots:
    for e in range(4):
        npt.assert_almost_equal(onsets(stim, e)[0], e * 12.5, decimal=3)
        npt.assert_almost_equal(np.diff(onsets(stim, e)), 50.0, decimal=3)
    npt.assert_almost_equal(stim.metadata['encoder']['cycle'], 50.0)
    # The point of the raster: the stimulator sources one electrode's worth of
    # current at a time. All four at once would be 12.5+25+37.5+50 = 125 uA.
    npt.assert_almost_equal(np.abs(stim.data).sum(axis=0).max(), 50.0)
    net = trapezoid(stim.data.astype(np.float64),
                    x=stim.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=4)

    # --- what the model made of it --------------------------------------
    model = ScoreboardSpatial(xrange=(-4, 4), yrange=(-4, 4), xystep=0.2,
                              rho=200).build()
    percept = model.predict_percept(implant)
    here, middle = at_electrodes(model, implant)
    # Brightest over time, since no two electrodes are ever on together:
    env = percept.data.max(axis=-1)
    bright = np.array([env[here[n]] for n in NAMES])

    # Four separate phosphenes, one per electrode, brightness following the
    # gray level that produced it:
    npt.assert_equal(np.all(np.diff(bright) > 0), True)
    npt.assert_allclose(bright / bright[-1], [0.25, 0.5, 0.75, 1.0], rtol=1e-3)
    # ... and nothing in between them, so they really are separate blobs:
    npt.assert_equal(env[middle] < 0.01 * bright[0], True)
    npt.assert_equal(np.unravel_index(int(np.argmax(env)), env.shape),
                     here['D'])

    # Size is set by `rho`, not by amplitude: a brighter phosphene is brighter,
    # not bigger, so all four cover the same area relative to their own peak.
    gx, gy = np.asarray(model.grid.ret.x), np.asarray(model.grid.ret.y)
    areas = []
    for n in NAMES:
        e = implant[n]
        quadrant = ((np.sign(gx) == np.sign(e.x)) &
                    (np.sign(gy) == np.sign(e.y)))
        blob = np.where(quadrant, env, 0.0)
        areas.append(int((blob >= blob.max() / 2).sum()))
    npt.assert_equal(areas, [areas[0]] * 4)
    npt.assert_equal(areas[0] > 4, True)

    # The raster is visible in the percept itself: at any one instant at most
    # one electrode is lit. The threshold sits well above the 0.006 of
    # cross-talk an unlit electrode picks up and well below the 12.5 of the
    # dimmest lit one:
    lit = np.array([[percept.data[here[n]][t] > 1.0 for n in NAMES]
                    for t in range(percept.data.shape[-1])])
    npt.assert_equal(lit.sum(axis=1).max(), 1)
    # ... and over the whole stimulus each of them gets a turn:
    npt.assert_equal(lit.any(axis=0), [True] * 4)


def test_endtoend_frequency_modulation():
    # Gray levels chosen so that the requested rates are 50, 66.7, 100 and
    # 200 Hz -- whole multiples of the 5 ms raster cycle that the fastest
    # electrode sets, so nothing has to be quantized away:
    implant = make_implant()
    img = ImageStimulus(np.array([[0.25, 1 / 3], [0.5, 1.0]]))
    stim = FrequencyEncoder(implant, freq_range=(0, 200), amp=50,
                            raster=one_per_group(),
                            frame_dur=200).encode(img)
    implant.stim = stim

    # --- what the encoder produced --------------------------------------
    # One amplitude for everyone; the gray level sets the rate instead:
    npt.assert_almost_equal(np.abs(stim.data).max(axis=1), 50.0, decimal=4)
    npt.assert_almost_equal(stim.metadata['encoder']['cycle'], 5.0)
    # 200 Hz / 4 groups gives each electrode a 1.25 ms slot, and each pulses at
    # exactly the rate it asked for:
    period = [20.0, 15.0, 10.0, 5.0]
    for e in range(4):
        npt.assert_almost_equal(onsets(stim, e)[0], e * 1.25, decimal=3)
        npt.assert_almost_equal(np.diff(onsets(stim, e)), period[e],
                                decimal=3)
    # Over a 200 ms frame that is floor((200 - pulse) / period) + 1 pulses:
    npt.assert_equal([onsets(stim, e).size for e in range(4)],
                     [10, 14, 20, 40])
    # Still one electrode at a time, even though they are on different rates.
    # This is what the raster cycle buys: without it the four trains would
    # drift onto each other and the stimulator would have to source 200 uA:
    npt.assert_almost_equal(np.abs(stim.data).sum(axis=0).max(), 50.0)
    net = trapezoid(stim.data.astype(np.float64),
                    x=stim.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=4)

    # --- what the model made of it --------------------------------------
    # A temporal model integrates the pulses, so more pulses is brighter even
    # though every pulse carries the same current:
    percept = FadingTemporal().build().predict_percept(implant.stim)
    # One percept frame per video frame -- the image is a single 200 ms frame:
    npt.assert_equal(percept.data.shape, (4, 1, 1))
    bright = percept.data[:, 0, 0]
    npt.assert_equal(np.all(np.diff(bright) > 0), True)
    # Brightness tracks the pulse *count* rather than the amplitude, which is
    # what distinguishes frequency modulation from amplitude modulation. It
    # grows a little slower than the count does, because the model fades
    # between pulses and a slower train spends longer fading:
    counts = np.array([onsets(stim, e).size for e in range(4)],
                      dtype=np.float64)
    npt.assert_allclose(bright / bright[-1], counts / counts[-1], rtol=0.15)
    npt.assert_array_less(bright / bright[-1], counts / counts[-1] + 1e-6)


def test_endtoend_raster_is_what_separates_the_groups():
    # The same four electrodes without a raster: every one of them fires at
    # the start of every period, so the stimulator has to source all of it at
    # once. This is the failure the raster exists to prevent, and it is why
    # `max_current` rejects the unrastered version of the very same image.
    img = ImageStimulus(np.array([[0.25, 0.50], [0.75, 1.00]]))
    implant = make_implant()
    plain = AmplitudeEncoder(implant, amp_range=(0, 50), freq=20,
                             frame_dur=200).encode(img)
    npt.assert_equal(plain.metadata['encoder']['n_schedules'], 1)
    npt.assert_almost_equal(np.abs(plain.data).sum(axis=0).max(), 125.0)

    implant.max_current = 60
    with pytest.raises(ValueError, match='raster'):
        implant.stim = plain
    # Giving the implant the raster is enough -- the encoder picks it up, and
    # the same image now fits inside the current limit:
    implant.raster = one_per_group()
    implant.stim = AmplitudeEncoder(implant, amp_range=(0, 50), freq=20,
                                    frame_dur=200).encode(img)
    npt.assert_almost_equal(np.abs(implant.stim.data).sum(axis=0).max(), 50.0)
