"""End-to-end tests you can check by hand

These tests run the full pipeline (image -> encoder -> implant -> model ->
percept) on a tiny setup where every number has a closed form:

*  Four electrodes on the corners of a 1200 um square, far enough apart that
   each one produces its own phosphene.
*  A 2x2 image, so that ``reshape_stim`` samples exactly one pixel per
   electrode (it maps the image grid linearly onto the electrode bounding box,
   and the electrodes sit on its corners).
*  Distinct, evenly spaced gray levels.
*  One electrode per raster group, so only one electrode is on at a time.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import trapezoid

from pulse2percept.implants import (CustomRaster, DiskElectrode,
                                    ElectrodeArray, Implant)
from pulse2percept.implants.retina import ArgusII
from pulse2percept.models import FadingTemporal, Model
from pulse2percept.models.retina import ScoreboardSpatial
from pulse2percept.stimuli import (AmplitudeEncoder, FrequencyEncoder,
                                   ImageStimulus, Stimulus)
from pulse2percept.utils.constants import DT

# Electrode names in `ElectrodeArray` order, and their positions (um). The
# 2x2 image is sampled at exactly these four points:
#     A = top-left pixel, B = top-right, C = bottom-left, D = bottom-right
NAMES = ['A', 'B', 'C', 'D']
POS = [(-600.0, -600.0), (600.0, -600.0), (-600.0, 600.0), (600.0, 600.0)]


def make_implant(raster=None):
    """Four well-separated electrodes, one per raster group"""
    electrode_array = ElectrodeArray({n: DiskElectrode(x, y, 0, 100)
                                      for n, (x, y) in zip(NAMES, POS)})
    return Implant(electrode_array, raster=raster)


def one_per_group():
    """A raster that drives one electrode at a time"""
    return CustomRaster({n: i for i, n in enumerate(NAMES)})


def onsets(stim, electrode):
    """Return the onset time (ms) of each pulse on one electrode"""
    neg = stim.data[electrode] < 0
    started = neg & ~np.concatenate(([False], neg[:-1]))
    # A pulse ramps up over DT, so the first full-amplitude sample is one DT
    # after onset:
    return stim.time[started] - DT


def at_electrodes(model, implant):
    """Return the grid index nearest each electrode, and the center index"""
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
    implant = make_implant(one_per_group())
    img = ImageStimulus(np.array([[0.25, 0.50], [0.75, 1.00]]))
    stim = implant.prepare_stim(
        AmplitudeEncoder(implant, amp_range=(0, 50), freq=20,
                         frame_dur=200).encode(img))
    npt.assert_equal(list(stim.electrodes), NAMES)

    # --- what the encoder produced --------------------------------------
    # Amplitude is 50 uA * gray level:
    npt.assert_almost_equal(np.abs(stim.data).max(axis=1),
                            [12.5, 25.0, 37.5, 50.0], decimal=4)
    # Every electrode pulses at 20 Hz; the raster only sets the onset within
    # each 50 ms period (four 12.5 ms slots):
    for e in range(4):
        npt.assert_almost_equal(onsets(stim, e)[0], e * 12.5, decimal=3)
        npt.assert_almost_equal(np.diff(onsets(stim, e)), 50.0, decimal=3)
    npt.assert_almost_equal(stim.metadata['encoder']['cycle'], 50.0)
    # Only one electrode is on at a time (all four at once would total
    # 12.5+25+37.5+50 = 125 uA):
    npt.assert_almost_equal(np.abs(stim.data).sum(axis=0).max(), 50.0)
    net = trapezoid(stim.data.astype(np.float64),
                    x=stim.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=4)

    # --- what the model made of it --------------------------------------
    # A spatial model uses the requested amplitude per electrode, without
    # raster timing (see `models.base._spatial_input`). Wrapping the schedule
    # in a plain `Stimulus` passes the delivered pulses instead, so the raster
    # shows up below:
    npt.assert_equal(stim._spatial_view().shape, (4, 1))
    model = ScoreboardSpatial(implant=implant, xrange=(-4, 4), yrange=(-4, 4),
                              step=0.2, rho=200).build()
    percept = model.predict_percept(Stimulus(stim))
    here, middle = at_electrodes(model, implant)
    # Max over time, since no two electrodes are on together:
    env = percept.data.max(axis=-1)
    bright = np.array([env[here[n]] for n in NAMES])

    # One phosphene per electrode, brightness proportional to gray level:
    npt.assert_equal(np.all(np.diff(bright) > 0), True)
    npt.assert_allclose(bright / bright[-1], [0.25, 0.5, 0.75, 1.0], rtol=1e-3)
    # Phosphenes are separate (dark in between):
    npt.assert_equal(env[middle] < 0.01 * bright[0], True)
    npt.assert_equal(np.unravel_index(int(np.argmax(env)), env.shape),
                     here['D'])

    # Size depends on `rho` only, so all four have the same half-max area:
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

    # At most one electrode is lit at any time. The threshold of 1.0 is well
    # above the ~0.006 cross-talk at an unlit electrode and well below the
    # 12.5 of the dimmest lit one:
    lit = np.array([[percept.data[here[n]][t] > 1.0 for n in NAMES]
                    for t in range(percept.data.shape[-1])])
    npt.assert_equal(lit.sum(axis=1).max(), 1)
    # Each electrode is lit at some point:
    npt.assert_equal(lit.any(axis=0), [True] * 4)


def test_endtoend_frequency_modulation():
    # Gray levels give 50, 66.7, 100, and 200 Hz, whose periods are whole
    # multiples of the 5 ms raster cycle (no quantization):
    implant = make_implant(one_per_group())
    img = ImageStimulus(np.array([[0.25, 1 / 3], [0.5, 1.0]]))
    stim = implant.prepare_stim(
        FrequencyEncoder(implant, freq_range=(0, 200), amp=50,
                         frame_dur=200).encode(img))

    # --- what the encoder produced --------------------------------------
    # Same amplitude on every electrode; gray level sets the rate:
    npt.assert_almost_equal(np.abs(stim.data).max(axis=1), 50.0, decimal=4)
    npt.assert_almost_equal(stim.metadata['encoder']['cycle'], 5.0)
    # 5 ms cycle / 4 groups = 1.25 ms slot per electrode, each at its
    # requested rate:
    period = [20.0, 15.0, 10.0, 5.0]
    for e in range(4):
        npt.assert_almost_equal(onsets(stim, e)[0], e * 1.25, decimal=3)
        npt.assert_almost_equal(np.diff(onsets(stim, e)), period[e],
                                decimal=3)
    # floor((200 - pulse_dur) / period) + 1 pulses per 200 ms frame:
    npt.assert_equal([onsets(stim, e).size for e in range(4)],
                     [10, 14, 20, 40])
    # Still one electrode at a time despite different rates (without the
    # raster cycle, the trains would overlap and total 200 uA):
    npt.assert_almost_equal(np.abs(stim.data).sum(axis=0).max(), 50.0)
    net = trapezoid(stim.data.astype(np.float64),
                    x=stim.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=4)

    # --- what the model made of it --------------------------------------
    # The temporal model integrates pulses, so more pulses is brighter at the
    # same current:
    percept = FadingTemporal().build().predict_percept(stim)
    # One percept frame for the single 200 ms image frame:
    npt.assert_equal(percept.data.shape, (4, 1, 1))
    bright = percept.data[:, 0, 0]
    npt.assert_equal(np.all(np.diff(bright) > 0), True)
    # Brightness tracks pulse count, but saturates: each pulse adds less as the
    # percept gets brighter. Normalized to the fastest train, slower trains
    # therefore sit slightly *above* the pulse-count ratio:
    counts = np.array([onsets(stim, e).size for e in range(4)],
                      dtype=np.float64)
    npt.assert_allclose(bright / bright[-1], counts / counts[-1], rtol=0.15)
    npt.assert_array_less(counts / counts[-1] - 1e-6, bright / bright[-1])
    # Closed form (identical pulses, percept = frame peak): each cathodic phase
    # adds `amp (1 - exp(-phase_dur/tau))`, which decays by `exp(-period/tau)`
    # before the next pulse, so the peak after n pulses is a geometric series.
    # This requires a rectified drive (otherwise the anodic phase cancels each
    # pulse). The model samples the stimulus at `dt`, so the effective phase is
    # slightly shorter than `phase_dur` (a few tenths of a percent, within
    # rtol=1e-2):
    tau, phase_dur, amp = 100.0, 0.46, 50.0
    period = np.array(period, dtype=np.float64)
    npt.assert_allclose(
        bright,
        amp * (1 - np.exp(-phase_dur / tau)) *
        (1 - np.exp(-counts * period / tau)) / (1 - np.exp(-period / tau)),
        rtol=1e-2)


@pytest.mark.parametrize('order', [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]])
def test_endtoend_raster_order(order):
    # Reordering raster groups permutes the onsets and nothing else:
    img = ImageStimulus(np.array([[0.25, 0.50], [0.75, 1.00]]))
    implant = make_implant(CustomRaster({n: g
                                         for n, g in zip(NAMES, order)}))
    stim = implant.prepare_stim(
        AmplitudeEncoder(implant, amp_range=(0, 50), freq=20,
                         frame_dur=200).encode(img))

    # Each electrode starts in its group's slot (50 ms / 4 = 12.5 ms):
    npt.assert_almost_equal([onsets(stim, e)[0] for e in range(4)],
                            np.asarray(order) * 12.5, decimal=3)
    # Same amplitudes, rate, and total current regardless of order:
    npt.assert_almost_equal(np.abs(stim.data).max(axis=1),
                            [12.5, 25.0, 37.5, 50.0], decimal=4)
    for e in range(4):
        npt.assert_almost_equal(np.diff(onsets(stim, e)), 50.0, decimal=3)
    npt.assert_almost_equal(np.abs(stim.data).sum(axis=0).max(), 50.0)

    # In the percept, electrodes light up one at a time in raster order, with
    # brightness set by gray level. Uses the delivered pulses (see
    # `_spatial_input`):
    model = ScoreboardSpatial(implant=implant, xrange=(-4, 4), yrange=(-4, 4),
                              step=0.2, rho=200).build()
    percept = model.predict_percept(Stimulus(stim))
    here, _ = at_electrodes(model, implant)
    env = percept.data.max(axis=-1)
    bright = np.array([env[here[n]] for n in NAMES])
    npt.assert_allclose(bright / bright.max(), [0.25, 0.5, 0.75, 1.0],
                        rtol=1e-3)
    lit = np.array([[percept.data[here[n]][t] > 1.0 for n in NAMES]
                    for t in range(percept.data.shape[-1])])
    npt.assert_equal(lit.sum(axis=1).max(), 1)
    # Order of first lighting up matches the raster order:
    first = [int(np.argmax(lit[:, i])) for i in range(4)]
    npt.assert_equal(np.argsort(first).tolist(),
                     np.argsort(order).tolist())


def test_endtoend_raster_is_what_separates_the_groups():
    # Without a raster, all electrodes fire at the start of every period, so
    # `max_current` rejects the stimulus:
    img = ImageStimulus(np.array([[0.25, 0.50], [0.75, 1.00]]))
    implant = make_implant()
    plain = AmplitudeEncoder(implant, amp_range=(0, 50), freq=20,
                             frame_dur=200).encode(img)
    # All electrodes fire at the same times (total 125 uA):
    npt.assert_equal(len(np.unique(np.abs(plain.data) > 0, axis=0)), 1)
    npt.assert_almost_equal(np.abs(plain.data).sum(axis=0).max(), 125.0)

    implant.max_current = 60
    with pytest.raises(ValueError, match='raster'):
        implant.prepare_stim(plain)
    # With a raster (used by the encoder), the same image fits the current
    # limit:
    implant.raster = one_per_group()
    rastered = implant.prepare_stim(
        AmplitudeEncoder(implant, amp_range=(0, 50), freq=20,
                         frame_dur=200).encode(img))
    npt.assert_almost_equal(np.abs(rastered.data).sum(axis=0).max(), 50.0)


def test_endtoend_slow_train_stays_lit_for_the_whole_video(camera_video):
    """A pulse rate well below the frame rate keeps the percept lit

    ``camera_video`` runs at 29.97 fps (1000 / 29.97 ms per frame) and a 6 Hz train
    pulses every 166.67 ms (4.995 frames). Sampling one instant per frame
    drifts through the pulse cycle and can miss the 0.92 ms pulse window
    entirely. A rectified drive keeps brightness between pulses, and reporting
    each frame's peak removes the dependence on sampling phase.
    """
    # Argus II defaults: amplitude encoder at 6 Hz, six-group raster:
    implant = ArgusII()
    with pytest.warns(UserWarning, match='deliver no pulse'):
        # At 6 Hz vs. 29.97 fps, most frames contain no pulse:
        delivered = implant.prepare_stim(camera_video)
    # Pulses span the whole video. The last of six raster groups starts
    # 5 x 2 = 10 ms after the first, so the last onset is 3010.5 ms (vs.
    # 3000.5 ms without a raster):
    onset = delivered.time[np.any(delivered.data < 0, axis=0)]
    npt.assert_almost_equal(onset.max(), 3010.5, decimal=1)

    model = Model(spatial=ScoreboardSpatial(implant, xrange=(-12, 12),
                                            yrange=(-8, 8), step=1),
                  temporal=FadingTemporal(tau=100)).build()
    with pytest.warns(UserWarning, match='deliver no pulse'):
        percept = model.predict_percept(camera_video)
    # One percept frame per video frame, covering the whole video:
    npt.assert_equal(percept.data.shape[-1], 94)
    npt.assert_array_less(3000, percept.time[-1])

    frame = percept.data.reshape(-1, percept.data.shape[-1]).max(axis=0)
    # No frame goes dark, including the second half of the video:
    npt.assert_array_less(0.1 * frame.max(), frame[percept.time > 1000])
    npt.assert_array_less(0.1 * frame.max(), frame.min())
    # Frame-to-frame variation stays small (instantaneous sampling would vary
    # by two orders of magnitude):

    npt.assert_array_less(frame.max() / np.median(frame), 4.0)
