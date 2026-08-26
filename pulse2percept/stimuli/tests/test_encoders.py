import warnings
from copy import copy, deepcopy

import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import trapezoid

from pulse2percept.implants import (ArgusII, CustomRaster, DiskElectrode,
                                    GridImplant, SequentialRaster)
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BiphasicPulseTrain, BostonTrain,
                                   FrequencyEncoder, ImageStimulus,
                                   MonophasicPulse, Stimulus, StimulusEncoder,
                                   VideoStimulus)
from pulse2percept.stimuli import encoders
from pulse2percept.utils.constants import DT
from pulse2percept.utils.testing import assert_warns_msg
from pulse2percept.units import (DimensionMismatchError, Hz, Quantity,
                                 dimensionless, kHz, mA, ms, uA, us)
from pulse2percept.units import s as sec


def n_pulses_of(stim, electrode=0, peak=None):
    """Count the pulses one electrode of an encoded stimulus delivers"""
    row = stim.data[electrode]
    peak = np.abs(row).max() if peak is None else peak
    if peak == 0:
        return 0
    firing = np.abs(row) >= 0.99 * peak
    # Each pulse has a leading and a trailing phase, both at full amplitude:
    return np.count_nonzero(np.diff(firing.astype(int)) > 0) // 2


def pixel_implant(shape, raster=None):
    """An implant with one electrode per pixel of a ``shape`` image

    A raster describes how one particular device takes turns between its
    electrodes, so trying one out means having a device to try it on. This
    implant's electrodes sit exactly on the pixels of a ``shape`` image, so
    sampling one at the other changes nothing about what gets encoded.
    """
    implant = GridImplant(shape, 200, etype=DiskElectrode, r=50)
    implant.raster = raster
    return implant


def n_schedules_of(stim):
    """How many distinct schedules the electrodes of a stimulus are split over

    Two electrodes share a schedule when current flows on them at the same
    times, whatever amplitude it flows at. That is what a raster splits
    electrodes into, and what frequency modulation multiplies. Electrodes
    delivering nothing at all share the empty schedule.
    """
    return len(np.unique(np.abs(stim.data) > 0, axis=0))


def test_StimulusEncoder_is_abstract():
    with pytest.raises(TypeError):
        StimulusEncoder()


def test_StimulusEncoder_warnings_point_at_the_caller(monkeypatch):
    """A warning is only actionable if it names the line that caused it

    All three of these are about the caller's own choice of source, frequency
    or implant, so blaming ``encoders.py`` (which is what a bare
    ``warnings.warn`` does) puts an unhelpful line of p2p's internals in front
    of them.
    """
    monkeypatch.setattr(encoders, '_BIG_STIM', 100)
    monkeypatch.setattr(encoders, '_BIG_TIME', 100)
    vid = VideoStimulus(np.random.rand(8, 8, 4), time=[0, 100, 200, 300])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        AmplitudeEncoder(freq=1000).encode(vid)
    npt.assert_equal(len(caught) > 0, True)
    for warning in caught:
        npt.assert_equal(warning.category, UserWarning)
        npt.assert_equal(warning.filename, __file__)

    # ... including the one about frames that never get a pulse:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        AmplitudeEncoder(freq=2).encode(vid, implant=ArgusII())
    npt.assert_equal([w.filename for w in caught], [__file__] * len(caught))
    npt.assert_equal(any('deliver no pulse' in str(w.message)
                         for w in caught), True)


def test_StimulusEncoder_source():
    enc = AmplitudeEncoder()
    with pytest.raises(TypeError):
        enc.encode(np.random.rand(4, 5))
    with pytest.raises(TypeError):
        enc.encode('not-a-stimulus')


def test_StimulusEncoder_params():
    with pytest.raises(ValueError):
        AmplitudeEncoder(phase_dur=DT / 2)
    with pytest.raises(ValueError):
        AmplitudeEncoder(interphase_dur=-1)
    with pytest.raises(ValueError):
        AmplitudeEncoder(frame_dur=0)
    with pytest.raises(ValueError):
        AmplitudeEncoder(amp_range=(0, 10, 20))
    with pytest.raises(ValueError):
        AmplitudeEncoder(amp_range=(-10, 10))
    with pytest.raises(ValueError):
        AmplitudeEncoder(freq=-1)
    with pytest.raises(TypeError):
        AmplitudeEncoder(pulse={'invalid': 1})
    with pytest.raises(ValueError):
        # A pulse needs a time component:
        AmplitudeEncoder(pulse=Stimulus(3))
    with pytest.raises(ValueError):
        # ... and must live on a single electrode:
        AmplitudeEncoder(pulse=ImageStimulus(np.random.rand(2, 2)))
    with pytest.raises(ValueError):
        AmplitudeEncoder(clock=DT / 2)
    with pytest.raises(ValueError):
        AmplitudeEncoder(n_levels=1)
    with pytest.raises(ValueError):
        FrequencyEncoder(freq_range=(0, 10, 20))
    with pytest.raises(ValueError):
        FrequencyEncoder(freq_range=(-10, 10))
    with pytest.raises(ValueError):
        FrequencyEncoder(amp=-1)
    # A fractional level count would quietly become a fractional step size, so
    # the number of levels you got back would not be the number you asked for:
    with pytest.raises(ValueError):
        AmplitudeEncoder(n_levels=2.5)
    # NaN and infinity slip through every `<` comparison, and would otherwise
    # surface as something inscrutable much further downstream:
    for kwargs in [{'freq': np.nan}, {'amp_range': (0, np.nan)},
                   {'phase_dur': np.nan}, {'interphase_dur': np.inf},
                   {'frame_dur': np.nan}, {'clock': np.nan},
                   {'n_levels': np.nan}]:
        with pytest.raises(ValueError):
            AmplitudeEncoder(**kwargs)
    with pytest.raises(ValueError):
        FrequencyEncoder(amp=np.nan)
    with pytest.raises(ValueError):
        FrequencyEncoder(freq_range=(0, np.inf))
    # Encoders pretty-print their parameters:
    npt.assert_equal('amp_range' in str(AmplitudeEncoder()), True)
    npt.assert_equal('freq_range' in str(FrequencyEncoder()), True)


def test_AmplitudeEncoder():
    # A 6-frame video, 1 ms per frame:
    stim = VideoStimulus(np.random.rand(4, 5, 6))
    npt.assert_almost_equal(np.diff(stim.time), 1)
    enc = AmplitudeEncoder(freq=1000).encode(stim)
    # One electrode per pixel, and the encoded stimulus lasts as long as the
    # video did:
    npt.assert_equal(enc.shape[0], 20)
    npt.assert_almost_equal(enc.time[0], 0)
    npt.assert_almost_equal(enc.time[-1], 6, decimal=3)
    # Time points stay strictly monotonically increasing across frames:
    npt.assert_equal(np.all(np.diff(enc.time) > 0.95 * DT), True)
    # Cathodic first: the biggest excursion of each electrode is negative:
    npt.assert_almost_equal(enc.data.min(axis=1), -50 * stim.data.max(axis=1))
    npt.assert_almost_equal(enc.data.max(axis=1), 50 * stim.data.max(axis=1))
    # Anodic first flips that around, but the magnitude is the same:
    ana = AmplitudeEncoder(freq=1000, cathodic_first=False).encode(stim)
    npt.assert_almost_equal(np.abs(ana.data), np.abs(enc.data))
    npt.assert_almost_equal(ana.data[:, 1], -enc.data[:, 1])
    # Pulses are charge-balanced. Integrated in float64, because a float32
    # time axis cannot resolve the DT-wide pulse edges well enough for
    # `is_charge_balanced` to survive more than a handful of pulses -- which
    # is true of `BiphasicPulseTrain` too, and has nothing to do with encoding:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=4)


@pytest.mark.parametrize('amp_range', [(0, 50), (2, 43), (10, 10)])
def test_AmplitudeEncoder_amp_range(amp_range):
    # Gray levels map onto `amp_range` absolutely: a pixel of a given gray
    # level always encodes to the same amplitude, no matter what the rest of
    # the image looks like:
    lo, hi = amp_range
    for gray in (0.0, 0.25, 1.0):
        img = ImageStimulus(np.full((4, 4), gray))
        enc = AmplitudeEncoder(amp_range=amp_range).encode(img)
        npt.assert_almost_equal(np.abs(enc.data).max(),
                                lo + gray * (hi - lo), decimal=4)
    # Both ends of the range are reached by the extremes of the image:
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    enc = AmplitudeEncoder(amp_range=amp_range).encode(img)
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1).min(), lo, decimal=4)
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1).max(), hi, decimal=4)


def test_AmplitudeEncoder_stretch():
    # A uniform image has an absolute gray level, but no range to stretch:
    img = ImageStimulus(np.full((4, 4), 0.5))
    npt.assert_almost_equal(np.abs(img.encode().data).max(), 25)
    npt.assert_almost_equal(
        np.abs(AmplitudeEncoder(stretch=True).encode(img).data).max(), 0)
    # Stretching pins the darkest pixel to the bottom of the range and the
    # brightest to the top:
    img = ImageStimulus(np.linspace(0.2, 0.6, 16).reshape((4, 4)))
    enc = AmplitudeEncoder(amp_range=(0, 50), stretch=True).encode(img)
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1).min(), 0)
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1).max(), 50)


def test_AmplitudeEncoder_freq():
    # Twice the frequency, twice the pulses per frame:
    img = ImageStimulus(np.ones((2, 2)))
    for freq, n_pulses in [(10, 5), (20, 10), (40, 20)]:
        enc = AmplitudeEncoder(freq=freq).encode(img)
        # Count the cathodic phases of the first electrode:
        cathodic = enc.data[0] <= -49
        npt.assert_equal(np.count_nonzero(np.diff(cathodic.astype(int)) > 0),
                         n_pulses)
    # 0 Hz is silence, but still a stimulus of the right duration:
    enc = AmplitudeEncoder(freq=0).encode(img)
    npt.assert_almost_equal(np.abs(enc.data).max(), 0)
    npt.assert_almost_equal(enc.time[-1], 500)
    # A frequency below the frame rate is realizable -- the pulse clock does
    # not care what the frame rate is -- but wasteful: whole frames then go by
    # without delivering anything, and their gray levels never reach the
    # electrode:
    vid = VideoStimulus(np.ones((2, 2, 5)), metadata={'fps': 30})
    with pytest.warns(UserWarning, match='deliver no pulse'):
        sparse = AmplitudeEncoder(freq=10).encode(vid)
    # 5 frames of 33.3 ms is 166.7 ms, which holds two 100 ms periods:
    npt.assert_equal(n_pulses_of(sparse), 2)
    npt.assert_almost_equal(np.diff(pulse_onsets(sparse)), 100, decimal=3)
    # A pulse that does not fit into a frame is an error, not a warning:
    with pytest.raises(ValueError):
        AmplitudeEncoder(freq=1000, phase_dur=10).encode(img)
    # As is a pulse that does not fit into a pulse train window:
    with pytest.raises(ValueError):
        AmplitudeEncoder(freq=2000, phase_dur=0.46).encode(img)


def test_AmplitudeEncoder_pulse():
    img = ImageStimulus(np.ones((2, 2)))
    # A custom pulse is used in place of the default biphasic one, with its
    # amplitude normalized away:
    pulse = MonophasicPulse(-20, 0.5)
    enc = AmplitudeEncoder(pulse=pulse, freq=100,
                           amp_range=(0, 30)).encode(img)
    npt.assert_almost_equal(np.abs(enc.data).max(), 30)
    # A monophasic pulse is not charge-balanced, and encoding does not make it
    # so: all 50 pulses push charge the same way. Each carries
    # amp * (phase_dur - DT), the DT coming off the two edges the pulse ramps
    # over:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, -30 * (0.5 - DT) * 50, decimal=2)
    # The caller's pulse is left alone (both its data and its time axis):
    pulse = BiphasicPulse(20, 0.2)
    data, time = pulse.data.copy(), pulse.time.copy()
    AmplitudeEncoder(pulse=pulse, freq=100).encode(img)
    npt.assert_almost_equal(pulse.data, data)
    npt.assert_almost_equal(pulse.time, time)


def test_AmplitudeEncoder_image():
    img = ImageStimulus(np.random.rand(4, 5))
    # An image has no time axis of its own, so it is a single frame that lasts
    # 500 ms by default:
    npt.assert_almost_equal(AmplitudeEncoder().encode(img).time[-1], 500)
    npt.assert_almost_equal(
        AmplitudeEncoder(frame_dur=123).encode(img).time[-1], 123)
    # `frame_dur` also overrides the frame rate of a video:
    vid = VideoStimulus(np.random.rand(4, 5, 3))
    npt.assert_almost_equal(
        AmplitudeEncoder(frame_dur=10).encode(vid).time[-1], 30)


def test_AmplitudeEncoder_implant():
    # No raster here: what is being checked is the sampling, and a raster would
    # stagger the onsets away from the pixel-resolution encoding compared with
    # at the end.
    implant = ArgusII(raster=None)
    vid = BostonTrain()
    enc = AmplitudeEncoder(amp_range=(0, 50)).encode(vid, implant=implant)
    # The video is sampled at the electrode locations, so the stimulus has one
    # row per electrode rather than one per pixel:
    npt.assert_equal(enc.shape[0], implant.n_electrodes)
    npt.assert_equal(list(enc.electrodes), list(implant.electrode_names))
    npt.assert_equal(np.abs(enc.data).max() <= 50, True)
    # ... and it is small enough to be worth doing:
    npt.assert_equal(enc.data.nbytes < 1e6, True)
    # It can be assigned to the implant without any further reshaping:
    implant.stim = enc
    npt.assert_equal(implant.stim.shape, enc.shape)
    # Sampling first and encoding second gives the same answer as encoding an
    # already-downsampled video, which is what makes this a pure optimization
    # for amplitude modulation:
    sampled = implant.reshape_stim(vid)
    direct = AmplitudeEncoder(amp_range=(0, 50)).encode(sampled)
    npt.assert_almost_equal(enc.data, direct.data, decimal=4)
    npt.assert_almost_equal(enc.time, direct.time)


def test_AmplitudeEncoder_big_stim_warning(monkeypatch):
    # Encoding at pixel resolution is a memory trap, and the way out is to pass
    # an implant, so say so. The threshold is lowered here rather than actually
    # allocating the hundreds of megabytes it takes to cross it:
    monkeypatch.setattr(encoders, '_BIG_STIM', 100)
    vid = VideoStimulus(np.random.rand(8, 8, 4))
    with pytest.warns(UserWarning, match="Pass 'implant'"):
        AmplitudeEncoder(freq=1000).encode(vid)
    # Passing an implant is the way out, so it does not warn:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        AmplitudeEncoder(freq=1000).encode(vid,
                                          implant=ArgusII(raster=None))


def whole_pulses(freq, frame_dur, pulse_dur=0.92):
    """How many pulses of ``freq`` Hz a frame can start *and finish*"""
    if freq <= 0:
        return 0
    last = np.floor((frame_dur - DT) / DT + 1e-9)
    room = last - round(pulse_dur / DT)
    return int(room // round(1000.0 / freq / DT)) + 1 if room >= 0 else 0


def pulse_onsets(stim, electrode=0):
    """The time (ms) at which each of one electrode's pulses begins

    Assumes a cathodic-first pulse, so that each pulse contributes exactly one
    run of negative samples.
    """
    neg = stim.data[electrode] < 0
    started = neg & ~np.concatenate(([False], neg[:-1]))
    # The first sample at full amplitude sits one tick past the onset, because
    # a pulse ramps up over DT:
    return stim.time[started] - DT


def test_FrequencyEncoder():
    # A gray ramp, so that every electrode ends up on a different frequency:
    grays = np.linspace(0, 1, 16)
    img = ImageStimulus(grays.reshape((4, 4)))
    enc = FrequencyEncoder(freq_range=(0, 100), amp=37,
                           frame_dur=100).encode(img)
    # Every electrode pulses at the same amplitude, cathodic first:
    peaks = np.abs(enc.data).max(axis=1)
    npt.assert_almost_equal(peaks[1:], 37)
    npt.assert_almost_equal(enc.data.min(), -37)
    # A gray level of 0 maps onto 0 Hz, which is silence:
    npt.assert_almost_equal(peaks[0], 0)
    # ... and the number of pulses grows with the gray level:
    counts = [n_pulses_of(enc, e, peak=37) for e in range(16)]
    npt.assert_equal(counts, [whole_pulses(100 * g, 100) for g in grays])
    npt.assert_equal(np.all(np.diff(counts) >= 0), True)
    npt.assert_equal(counts[-1], 10)
    # Electrodes on different frequencies do not share a time axis, so the
    # stimulus has many more time points than amplitude modulation would:
    am = AmplitudeEncoder(freq=100, frame_dur=100).encode(img)
    npt.assert_equal(enc.shape[1] > 4 * am.shape[1], True)
    npt.assert_equal(enc.time[-1], am.time[-1])
    npt.assert_equal(np.all(np.diff(enc.time) > 0.95 * DT), True)
    # Pulses stay charge-balanced no matter how they are interleaved. The
    # tolerance is set by how well a float32 time axis resolves the DT-wide
    # pulse edges, not by the encoding; a single truncated pulse would show up
    # here as several uA*ms, four orders of magnitude above that floor:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=3)


def test_FrequencyEncoder_whole_pulses():
    # A frame delivers only pulses it can finish. At 60 Hz a 33.3 ms frame has
    # room for exactly two 0.92 ms pulses, not two and a fraction of a third:
    img = ImageStimulus(np.ones((2, 2)))
    frame_dur = 1000 / 29.97
    for freq in (30, 60, 90):
        enc = FrequencyEncoder(freq_range=(freq, freq),
                               frame_dur=frame_dur).encode(img)
        npt.assert_equal(n_pulses_of(enc), whole_pulses(freq, frame_dur))
        # Charge balance is what a truncated pulse would break:
        net = trapezoid(enc.data.astype(np.float64),
                        x=enc.time.astype(np.float64))
        npt.assert_almost_equal(net, 0, decimal=3)
    # Amplitude modulation obeys the same rule:
    am = AmplitudeEncoder(freq=60, frame_dur=frame_dur).encode(img)
    npt.assert_equal(n_pulses_of(am), whole_pulses(60, frame_dur))


def test_StimulusEncoder_clock():
    # A clock rounds the pulse period onto a whole number of cycles, so
    # frequencies that differ by less than that collapse onto one schedule:
    img = ImageStimulus(np.linspace(0.5, 1, 16).reshape((4, 4)))
    fine = FrequencyEncoder(freq_range=(0, 300), frame_dur=100).encode(img)
    coarse = FrequencyEncoder(freq_range=(0, 300), frame_dur=100,
                              clock=1).encode(img)
    npt.assert_equal(n_schedules_of(fine), 16)
    npt.assert_equal(n_schedules_of(coarse) < 16, True)
    # ... which is the whole point: far fewer time points to simulate:
    npt.assert_equal(coarse.shape[1] < fine.shape[1] / 2, True)
    # Every pulse still lands on the clock grid, and the fine one does not:
    onsets = pulse_onsets(coarse)
    npt.assert_almost_equal(onsets, np.round(onsets), decimal=3)
    npt.assert_equal(np.allclose(pulse_onsets(fine),
                                 np.round(pulse_onsets(fine)), atol=1e-3),
                     False)
    # A clock that cannot resolve DT is not a clock:
    with pytest.raises(ValueError):
        FrequencyEncoder(clock=DT / 10)


def test_StimulusEncoder_n_levels():
    # Quantizing the gray levels quantizes whatever they are modulated onto:
    img = ImageStimulus(np.linspace(0, 1, 64).reshape((8, 8)))
    am = AmplitudeEncoder(amp_range=(0, 50), n_levels=4).encode(img)
    npt.assert_almost_equal(np.unique(np.abs(am.data).max(axis=1)),
                            [0, 50 / 3, 100 / 3, 50], decimal=4)
    # ... and for frequency modulation that is what keeps the time axis small:
    fm = FrequencyEncoder(freq_range=(0, 300), frame_dur=100).encode(img)
    fm4 = FrequencyEncoder(freq_range=(0, 300), frame_dur=100,
                           n_levels=4).encode(img)
    npt.assert_equal(n_schedules_of(fm4), 4)
    npt.assert_equal(fm4.shape[1] < fm.shape[1], True)


def test_StimulusEncoder_big_time_warning(monkeypatch):
    monkeypatch.setattr(encoders, '_BIG_TIME', 100)
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    with pytest.warns(UserWarning, match='time points'):
        FrequencyEncoder(freq_range=(0, 300), frame_dur=100).encode(img)


def test_FrequencyEncoder_implant():
    # A 300 Hz period is 3.3 ms, which Argus II's own six-group 2 ms raster
    # sweep does not fit into, so this device drives every electrode at once:
    implant = ArgusII(raster=None)
    enc = FrequencyEncoder(freq_range=(0, 300), amp=50, clock=1).encode(
        BostonTrain(), implant=implant)
    npt.assert_equal(enc.shape[0], implant.n_electrodes)
    npt.assert_almost_equal(np.abs(enc.data).max(), 50)
    implant.stim = enc
    npt.assert_equal(implant.stim.shape, enc.shape)
    # The clock is what makes this tractable at all: without one, the same
    # clip needs several times as many time points:
    unclocked = FrequencyEncoder(freq_range=(0, 300), amp=50).encode(
        BostonTrain(), implant=implant)
    npt.assert_equal(enc.shape[1] < unclocked.shape[1] / 5, True)


def test_StimulusEncoder_raster():
    img = ImageStimulus(np.ones((2, 2)))
    implant = pixel_implant((2, 2), SequentialRaster(2, interleave=True))
    enc = AmplitudeEncoder(freq=100, frame_dur=100).encode(img,
                                                           implant=implant)
    # Rastering splits the electrodes across two pulse schedules. The cycle a
    # raster has to get through is the pulse *period*, not the frame, so the
    # two groups are offset by half a period:
    npt.assert_equal(n_schedules_of(enc), 2)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 10)
    onsets = [pulse_onsets(enc, e) for e in (0, 1)]
    npt.assert_almost_equal(onsets[0][0], 0, decimal=3)
    npt.assert_almost_equal(onsets[1][0], 5, decimal=3)
    # Both groups keep the full requested rate -- rastering costs no
    # frequency, it only decides *when* within each period an electrode fires:
    for group in onsets:
        npt.assert_almost_equal(np.diff(group), 10, decimal=3)
    npt.assert_equal(n_pulses_of(enc, 0), 10)
    npt.assert_equal(n_pulses_of(enc, 1), 10)
    # The point of all this: the groups never pulse at the same instant, so
    # the stimulator only ever sources one group's worth of current. Before,
    # the groups shared every onset from 50 ms on and this was 100:
    npt.assert_equal(np.intersect1d(np.round(onsets[0], 3),
                                    np.round(onsets[1], 3)).size, 0)
    # Two of the four electrodes are in each group, so the stimulator sources
    # 2 x 50 uA at a time rather than all 4 x 50 uA:
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 100)
    # Both groups stay charge-balanced:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=3)
    # A clock quantizes the *slot*, and the cycle is rebuilt from it, so the
    # groups keep equal turns. A 4.6 ms slot lands on 5 ms and two of them make
    # a 10 ms cycle -- which here still holds the requested 100 Hz exactly.
    # Rounding each offset and the total cycle independently instead would give
    # a 9 ms cycle made of a 5 ms and a 4 ms turn, and 111 Hz:
    implant.raster = SequentialRaster(2, interleave=True, group_dur=4.6)
    enc = AmplitudeEncoder(freq=100, frame_dur=100, clock=1).encode(
        img, implant=implant)
    npt.assert_almost_equal(pulse_onsets(enc, 1)[0], 5, decimal=3)
    npt.assert_almost_equal(np.diff(pulse_onsets(enc, 1)), 10, decimal=3)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 10)
    # Whether the groups fit is decided on the slot the hardware will actually
    # use. Two 5.1 ms slots do not fit into a 10 ms period, but on a 1 ms clock
    # they are 5 ms slots, and two of those fit exactly:
    implant.raster = SequentialRaster(2, interleave=True, group_dur=5.1)
    enc = AmplitudeEncoder(freq=100, frame_dur=100, clock=1).encode(
        img, implant=implant)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 10)
    npt.assert_almost_equal(pulse_onsets(enc, 1)[0], 5, decimal=3)
    # Without a clock to round it there is nothing to round, and it is an error:
    with pytest.raises(ValueError):
        AmplitudeEncoder(freq=100, frame_dur=100).encode(img, implant=implant)


def test_StimulusEncoder_raster_frequency_modulation():
    # Under frequency modulation electrodes want different periods, so they
    # can only be kept apart by quantizing every period onto a common raster
    # cycle. The fastest electrode pulses once per cycle, slower ones every
    # m-th cycle, and no two groups ever coincide:
    img = ImageStimulus(np.linspace(0.25, 1, 16).reshape((4, 4)))
    implant = pixel_implant((4, 4), SequentialRaster(4, interleave=True))
    enc = FrequencyEncoder(freq_range=(0, 120), amp=10,
                           frame_dur=200).encode(img, implant=implant)
    cycle = enc.metadata['encoder']['cycle']
    npt.assert_almost_equal(cycle, 1000 / 120)
    for e in range(16):
        # Every period is a whole number of raster cycles, which is what keeps
        # two groups from drifting onto each other:
        ratio = np.diff(pulse_onsets(enc, e)) / cycle
        npt.assert_allclose(ratio, np.round(ratio), atol=1e-3)
    # Which is the whole point: the current limit holds instant by instant:
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 4 * 10)
    # Multiplexing a fast train across many groups asks more of a stimulator
    # than it can give, and that is an error rather than a silent collision:
    with pytest.raises(ValueError, match='no room'):
        FrequencyEncoder(freq_range=(0, 300), amp=10, frame_dur=200).encode(
            img, implant=pixel_implant((4, 4), SequentialRaster(6)))


def test_StimulusEncoder_raster_from_implant():
    # The implant is the one place device scheduling is described, so the
    # encoder holds no raster of its own and reads the implant's:
    implant = ArgusII()
    implant.raster = SequentialRaster(6)
    vid = VideoStimulus(np.ones((6, 10, 2)), metadata={'fps': 30})
    enc = AmplitudeEncoder(freq=30).encode(vid, implant=implant)
    npt.assert_equal(n_schedules_of(enc), 6)
    delays = [pulse_onsets(enc, e)[0] for e in (0, 10, 20, 30, 40, 50)]
    npt.assert_almost_equal(delays, np.arange(6) * 1000 / 30 / 6, decimal=2)
    # Trying another one out is a matter of giving it to the implant:
    implant.raster = SequentialRaster(2)
    enc = AmplitudeEncoder(freq=30).encode(vid, implant=implant)
    npt.assert_equal(n_schedules_of(enc), 2)
    # And no raster means every electrode fires at frame onset:
    implant.raster = None
    enc = AmplitudeEncoder(freq=30).encode(vid, implant=implant)
    npt.assert_equal(n_schedules_of(enc), 1)
    # Encoding for no implant at all is pixel resolution and no raster, even
    # though this encoder just encoded for a rastered device:
    bare = AmplitudeEncoder(freq=30).encode(vid)
    npt.assert_equal(n_schedules_of(bare), 1)


def test_StimulusEncoder_raster_current_limit():
    # 60 electrodes at 50 uA is 3000 uA if they all fire at once, but only
    # 500 uA if they take turns ten at a time:
    implant = ArgusII(raster=None)
    implant.max_current = 1000
    vid = VideoStimulus(np.ones((6, 10, 3)), metadata={'fps': 30})
    with pytest.raises(ValueError, match='raster'):
        implant.stim = AmplitudeEncoder(amp_range=(50, 50), freq=30).encode(
            vid, implant=implant)
    implant.raster = SequentialRaster(6)
    implant.stim = AmplitudeEncoder(amp_range=(50, 50), freq=30).encode(
        vid, implant=implant)
    npt.assert_almost_equal(np.abs(implant.stim.data).sum(axis=0).max(), 500)
    # A raster that cannot get through all its groups within a frame is not a
    # usable schedule:
    implant.raster = SequentialRaster(6, group_dur=20)
    with pytest.raises(ValueError):
        AmplitudeEncoder(freq=30).encode(vid, implant=implant)
    # Neither is one whose groups get a turn too short to pulse in. Sixty
    # 0.92 ms pulses take 55 ms, which does not fit into a 33 ms frame, so
    # electrode-at-a-time rastering is impossible here rather than merely
    # dropping the electrodes that come last:
    implant.raster = SequentialRaster(60)
    with pytest.raises(ValueError, match='no room'):
        AmplitudeEncoder(freq=30).encode(vid, implant=implant)
    # Halving the phase duration makes it fit:
    enc = AmplitudeEncoder(freq=30, phase_dur=0.2).encode(vid,
                                                          implant=implant)
    npt.assert_equal(n_schedules_of(enc), 60)
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 50)


@pytest.mark.parametrize('fps', [29.97, 30, 24, 59.94])
@pytest.mark.parametrize('freq', [50, 100])
def test_StimulusEncoder_freq_is_actual_freq(fps, freq):
    # The pulse clock runs independently of the frame clock, so the requested
    # frequency is the frequency delivered -- whatever the frame rate is, and
    # whether or not a frame holds a whole number of periods. Before, each
    # frame restarted the train at its own t=0, and a 29.97 fps video asked for
    # 50 Hz came back at 59.94 pulses/s.
    vid = VideoStimulus(np.ones((2, 2, 20)), metadata={'fps': fps})
    enc = AmplitudeEncoder(freq=freq).encode(vid)
    onsets = pulse_onsets(enc)
    npt.assert_almost_equal(np.diff(onsets), 1000.0 / freq, decimal=3)
    # Pulses stay whole and charge-balanced even where one straddles a frame
    # boundary, which is what the frame clock used to prevent:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=3)
    npt.assert_equal(np.all(np.diff(enc.time) > 0.95 * DT), True)


def fm_onsets(grays, fps=20, **kwargs):
    """Pulse onsets (ms) of a one-pixel video whose gray level changes"""
    vid = VideoStimulus(np.asarray(grays, dtype=float).reshape(1, 1, -1),
                        metadata={'fps': fps})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        enc = FrequencyEncoder(freq_range=(0, 100), amp=10,
                               **kwargs).encode(vid)
    return pulse_onsets(enc), enc


def test_FrequencyEncoder_rate_changes_between_frames():
    # The rate is piecewise constant over the video's frames, and the pulse
    # clock has to pick up a new rate the instant a frame boundary goes by.
    # Scheduling the next pulse a whole period ahead instead would carry the
    # old frame's rate straight across the boundary: a frame asking for 100 Hz
    # sat completely silent because the frame before it asked for 10 Hz and had
    # already booked the next pulse 100 ms out.
    # 50 ms frames: 10 Hz, then 100 Hz. Half a cycle is banked by 50 ms, so the
    # first 100 Hz pulse completes it at 55 ms:
    onsets, enc = fm_onsets([0.1, 1.0])
    npt.assert_almost_equal(onsets, [0, 55, 65, 75, 85, 95], decimal=3)
    # Charge balance is what a pulse cut off by a boundary would break:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=3)
    # The other direction: a fast frame followed by a slow one:
    npt.assert_almost_equal(fm_onsets([1.0, 0.1])[0],
                            [0, 10, 20, 30, 40, 50], decimal=3)
    # Each frame gets the rate it asked for, so the counts follow the video.
    # Note the 50 -> 100 Hz case: at the boundary the 50 Hz frame has banked
    # half a period, and the new rate finishes it 5 ms later at 55 ms, not
    # 20 ms later at 60 ms as the old rate would have:
    for grays, want in [([1.0, 0.5], [0, 10, 20, 30, 40, 50, 70, 90]),
                        ([0.5, 1.0], [0, 20, 40, 55, 65, 75, 85, 95])]:
        npt.assert_almost_equal(fm_onsets(grays)[0], want, decimal=3)
    # A frame at 0 Hz stops the clock rather than swallowing a pulse, and the
    # frame that starts it again comes up at its full rate:
    npt.assert_almost_equal(fm_onsets([1.0, 0.0, 1.0])[0],
                            [0, 10, 20, 30, 40, 100, 110, 120, 130, 140],
                            decimal=3)
    npt.assert_equal(fm_onsets([0.0, 0.0])[0].size, 0)
    npt.assert_almost_equal(fm_onsets([0.0, 1.0])[0],
                            [50, 60, 70, 80, 90], decimal=3)
    # Phase carries through a rate change that lands mid-period: 100, then 50,
    # then 25 Hz. The 50 Hz frame banks half a period by its end, which the
    # 25 Hz frame finishes 20 ms into itself:
    npt.assert_almost_equal(fm_onsets([1.0, 0.5, 0.25])[0],
                            [0, 10, 20, 30, 40, 50, 70, 90, 120], decimal=3)


def test_StimulusEncoder_raster_slots_land_on_the_clock():
    # A stimulator can only start a pulse on a clock edge, so splitting a
    # period evenly between the groups has to be resolved onto that grid.
    # Rounding each group's offset independently could land two of them on the
    # *same* edge -- six groups sharing a 5 ms period on a 1 ms clock have only
    # five edges to go round -- and they would then pulse together, which is
    # the one thing a raster exists to prevent.
    img = ImageStimulus(np.ones((6, 2)))
    implant = pixel_implant((6, 2), SequentialRaster(6))
    with pytest.raises(ValueError, match='clock'):
        AmplitudeEncoder(freq=200, frame_dur=100, clock=1).encode(
            img, implant=implant)
    # Given room, every group gets its own whole number of clock cycles, and
    # the turns come out evenly spaced rather than jittered onto nearby edges:
    enc = AmplitudeEncoder(freq=20, frame_dur=200, clock=1).encode(
        img, implant=implant)
    starts = np.array([pulse_onsets(enc, e)[0] for e in range(0, 12, 2)])
    npt.assert_almost_equal(starts, np.arange(6) * 8.0, decimal=3)
    npt.assert_equal(np.unique(starts).size, 6)
    # The period is untouched: every electrode still runs at the 20 Hz asked
    # for, and no two groups ever coincide.
    for e in range(12):
        npt.assert_almost_equal(np.diff(pulse_onsets(enc, e)), 50, decimal=3)
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 100)


def test_StimulusEncoder_raster_short_slot_keeps_the_rate():
    # A short explicit slot packs the groups into the start of each period.
    # That must not change the rate: when every electrode is on the same
    # period -- which is what amplitude modulation always produces -- two
    # groups a fixed offset apart can never meet, so there is nothing to
    # quantize away. Pinning the period to the cycle anyway turned a requested
    # 20 Hz into 18.5 Hz on Argus II with a 1 ms slot.
    img = ImageStimulus(np.ones((2, 2)))
    implant = pixel_implant((2, 2), SequentialRaster(2, interleave=True,
                                                     group_dur=1.5))
    # 10 ms period against a 2 x 1.5 = 3 ms cycle: quantizing would round the
    # period up to 12 ms (83 Hz):
    enc = AmplitudeEncoder(freq=100, frame_dur=200).encode(img,
                                                           implant=implant)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 3)
    for e, offset in enumerate([0, 1.5, 0, 1.5]):
        npt.assert_almost_equal(pulse_onsets(enc, e)[0], offset, decimal=3)
        npt.assert_almost_equal(np.diff(pulse_onsets(enc, e)), 10, decimal=3)
    # ... and the groups still never pulse at the same instant, which is the
    # only reason the quantization was there:
    npt.assert_equal(np.intersect1d(np.round(pulse_onsets(enc, 0), 3),
                                    np.round(pulse_onsets(enc, 1), 3)).size, 0)
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 100)
    # Frequency modulation still has to quantize, because electrodes on
    # different periods do drift onto each other:
    fm = FrequencyEncoder(freq_range=(50, 100), amp=10,
                          frame_dur=200).encode(
                              ImageStimulus(np.array([[1.0, 0.0],
                                                      [1.0, 0.0]])),
                              implant=implant)
    for e in range(4):
        period = np.diff(pulse_onsets(fm, e))
        npt.assert_allclose(period / 3, np.round(period / 3), atol=1e-3)


def test_FrequencyEncoder_rate_changes_with_raster_offset():
    # A raster group's first legal onset can fall several frames into the video
    # when stimulation is slow relative to it, and the phase accumulator has to
    # start counting in the frame that *contains* that onset. Starting at frame
    # 0 integrated the wrong rates and could even walk the schedule backwards
    # to an earlier frame boundary.
    # 20 ms frames; the top of the range is 10 Hz, so the raster cycle is
    # 100 ms and the second of two groups may only pulse at 50, 150, ... ms:
    implant = pixel_implant((2, 2), SequentialRaster(2, interleave=True))
    vid = VideoStimulus(np.tile(np.array([0, 1, 0, 0, 0, 0], dtype=float),
                                (2, 2, 1)).reshape(2, 2, 6),
                        metadata={'fps': 50})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        enc = FrequencyEncoder(freq_range=(0, 10), amp=10).encode(
            vid, implant=implant)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 100)
    # Only the 20-40 ms frame asks for stimulation, and neither group has a
    # legal slot inside it -- group 0's fall on 0 and 100 ms, group 1's on 50
    # and 150 ms, all of which land in frames asking for 0 Hz. So nothing is
    # delivered at all. Both groups used to fire at their own slot anyway, on
    # the strength of a frame that was nowhere near it:
    for e in range(2):
        npt.assert_equal(pulse_onsets(enc, e).size, 0)
    npt.assert_almost_equal(np.abs(enc.data).max(), 0)
    # Widen the window so each group does get a legal slot, and both fire --
    # in their own slots, a cycle apart:
    vid = VideoStimulus(np.tile(np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                                         dtype=float),
                                (2, 2, 1)).reshape(2, 2, 10),
                        metadata={'fps': 50})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        enc = FrequencyEncoder(freq_range=(0, 10), amp=10).encode(
            vid, implant=implant)
    npt.assert_almost_equal(pulse_onsets(enc, 0), [0], decimal=3)
    npt.assert_almost_equal(pulse_onsets(enc, 1), [50], decimal=3)
    # A pulse is never delivered into a frame that asked for silence:
    for e in range(2):
        for t in pulse_onsets(enc, e):
            npt.assert_equal(vid.data[e, int(t // 20)] > 0, True)


def test_StimulusEncoder_clock_never_speeds_up():
    # Same invariant as the raster, for the stimulator's own time base: a
    # timing constraint may lower the rate an electrode ends up on, never raise
    # it, since raising it delivers more charge than was asked for. Realizable
    # periods are whole clock cycles, so a 3.33 ms period on a 1 ms clock
    # becomes 4 ms (250 Hz), not 3 ms (333 Hz).
    img = ImageStimulus(np.ones((2, 2)))
    for clock, freq, want in [(1, 300, 250.0), (2, 300, 250.0),
                              (3, 300, 1000 / 6), (1, 137, 125.0)]:
        enc = FrequencyEncoder(freq_range=(freq, freq), amp=10, frame_dur=200,
                               clock=clock).encode(img)
        period = np.diff(pulse_onsets(enc))
        npt.assert_almost_equal(1000.0 / period, want, decimal=3)
        # Never faster than requested, and on the clock grid:
        npt.assert_equal(np.all(period >= 1000.0 / freq - 1e-9), True)
        npt.assert_allclose(period / clock, np.round(period / clock),
                            atol=1e-6)


def test_FrequencyEncoder_raster_never_speeds_up():
    # Quantizing a period onto the raster cycle rounds it *up*: an electrode
    # must never come back faster than it was asked for, since that delivers
    # more charge than the caller requested. Against a 10 ms cycle, 67 Hz
    # becomes 50 Hz rather than 100 Hz.
    grays = np.array([1.0, 0.67, 0.4, 0.2])
    img = ImageStimulus(grays.reshape((2, 2)))
    implant = pixel_implant((2, 2), SequentialRaster(4))
    enc = FrequencyEncoder(freq_range=(0, 100), amp=10, frame_dur=200).encode(
        img, implant=implant)
    cycle = enc.metadata['encoder']['cycle']
    npt.assert_almost_equal(cycle, 10)
    for e, gray in enumerate(grays):
        period = np.diff(pulse_onsets(enc, e))
        # Whole cycles, and never shorter than the requested period:
        npt.assert_allclose(period / cycle, np.round(period / cycle),
                            atol=1e-3)
        npt.assert_equal(np.all(period >= 1000.0 / (100 * gray) - 1e-3), True)
    # The fastest electrode still lands exactly on the cycle, so asking for the
    # top of the range costs nothing:
    npt.assert_almost_equal(np.diff(pulse_onsets(enc, 0)), 10, decimal=3)


def test_StimulusEncoder_pulse_offset():
    # `Stimulus` only requires a time axis to be ordered, not to start at zero.
    # What an encoder borrows from a supplied pulse is its *shape*, so a pulse
    # whose time axis is shifted has to encode exactly like an unshifted one.
    # Before, it was measured as 1.01 ms when deciding whether it fit but
    # rendered 5 ms late, which could push it clean out of its own frame:
    shape = np.array([[0, -1, -1, 0]], dtype=float)
    at_zero = Stimulus(shape, time=[0, 0.01, 1, 1.01])
    shifted = Stimulus(shape, time=[5, 5.01, 6, 6.01])
    vid = VideoStimulus(np.ones((1, 2, 3)), metadata={'fps': 50})
    ref = AmplitudeEncoder(pulse=at_zero, freq=1000 / 6).encode(vid)
    enc = AmplitudeEncoder(pulse=shifted, freq=1000 / 6).encode(vid)
    npt.assert_almost_equal(enc.time, ref.time)
    npt.assert_almost_equal(enc.data, ref.data)
    # The time axis stays strictly increasing across frames. A late pulse used
    # to run past the end of its frame and send time backwards at the next one:
    npt.assert_equal(np.all(np.diff(enc.time) > 0.95 * DT), True)
    # The caller's pulse is left alone:
    npt.assert_almost_equal(shifted.time, [5, 5.01, 6, 6.01])


def test_StimulusEncoder_zero_amp():
    # An electrode delivering no current has nothing to schedule, so a dark
    # frame costs no pulses and no time points -- it used to build a full pulse
    # train and then multiply it by zero:
    black = ImageStimulus(np.zeros((2, 2)))
    enc = AmplitudeEncoder(amp_range=(0, 50), freq=100,
                           frame_dur=100).encode(black)
    npt.assert_equal(np.all(enc.data == 0), True)
    npt.assert_equal(enc.shape[1], 2)
    npt.assert_almost_equal(enc.time[-1], 100)
    # A dark electrode alongside bright ones costs nothing either, and does not
    # disturb what the bright ones do:
    half = ImageStimulus(np.array([[0.0, 1.0], [0.0, 1.0]]))
    enc = AmplitudeEncoder(amp_range=(0, 50), freq=100,
                           frame_dur=100).encode(half)
    npt.assert_equal(np.all(enc.data[[0, 2]] == 0), True)
    npt.assert_equal(n_pulses_of(enc, 1), 10)
    # A dark *frame* does not reset the phase of the frames around it: the
    # pulse clock keeps running even where nothing is delivered. Three 25 ms
    # frames at 200 Hz, with the middle one black:
    vid = VideoStimulus(np.array([1.0, 0.0, 1.0]).reshape((1, 1, 3)),
                        metadata={'fps': 40})
    enc = AmplitudeEncoder(amp_range=(0, 50), freq=200).encode(vid)
    onsets = pulse_onsets(enc)
    # Nothing is delivered during the black frame ...
    npt.assert_equal(np.any((onsets >= 25) & (onsets < 50)), False)
    # ... and the train picks back up in phase rather than at the frame edge,
    # so every onset is still a whole number of 5 ms periods from the first:
    npt.assert_almost_equal(np.mod(onsets, 5), 0, decimal=3)
    npt.assert_almost_equal(onsets[[0, -1]], [0, 70], decimal=3)
    # A raster that a stimulus cannot possibly satisfy is a property of the
    # device, not of how bright today's video is, so it is reported either way:
    implant = ArgusII(raster=SequentialRaster(60))
    dark = VideoStimulus(np.zeros((6, 10, 3)), metadata={'fps': 30})
    with pytest.raises(ValueError, match='no room'):
        AmplitudeEncoder(freq=30).encode(dark, implant=implant)
    # ... and a workable one costs a dark video nothing:
    enc = AmplitudeEncoder(amp_range=(0, 50), freq=30, phase_dur=0.2).encode(
        dark, implant=implant)
    npt.assert_equal(np.all(enc.data == 0), True)
    npt.assert_equal(enc.shape[1], 2)


def test_StimulusEncoder_implant_reshape():
    # Passing an implant means "sample the source at the electrode locations".
    # Row count is not a usable test of whether that already happened: a 10x6
    # image and an RGB 4x5 image both have exactly as many rows as Argus II has
    # electrodes, and both used to skip sampling (and, for RGB, `rgb2gray`)
    # while still being labeled with electrode names.
    implant = ArgusII(raster=None)
    for src in [ImageStimulus(np.random.rand(10, 6)),
                ImageStimulus(np.random.rand(4, 5, 3)),
                ImageStimulus(np.random.rand(6, 10)),
                VideoStimulus(np.random.rand(10, 6, 2))]:
        npt.assert_equal(src.data.shape[0], implant.n_electrodes)
        enc = AmplitudeEncoder(amp_range=(0, 50)).encode(src, implant=implant)
        direct = AmplitudeEncoder(amp_range=(0, 50)).encode(
            implant.reshape_stim(src))
        npt.assert_almost_equal(enc.data, direct.data, decimal=4)
        npt.assert_equal(list(enc.electrodes), list(implant.electrode_names))


def test_StimulusEncoder_spatial_view():
    """The modulation half of encoding, with none of the device's timing in it
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    implant = pixel_implant((4, 4), SequentialRaster(4, interleave=True))
    enc = AmplitudeEncoder(amp_range=(0, 50), freq=100, frame_dur=100)
    delivered = enc.encode(img, implant=implant)
    spatial = delivered._spatial_view()
    # Asking for it does not expand the schedule into a waveform:
    npt.assert_equal(_rendered(delivered), False)

    # One row per electrode, one column per frame of the source -- and an
    # image is one frame, so there is no time axis at all:
    npt.assert_equal(spatial.shape, (16, 1))
    npt.assert_equal(spatial.time, None)
    npt.assert_equal(list(spatial.electrodes), list(delivered.electrodes))
    npt.assert_equal(spatial.unit, uA)
    # Gray level maps onto the amplitude range, and that is the whole content:
    npt.assert_almost_equal(spatial.data.ravel(),
                            np.linspace(0, 1, 16) * 50, decimal=4)
    # Which is exactly the peak each electrode reaches in the pulse train the
    # same parameters were assembled into -- the two are descriptions of one
    # stimulus, not two different stimuli:
    npt.assert_almost_equal(np.abs(delivered.data).max(axis=1),
                            spatial.data.ravel(), decimal=4)
    # None of the timing survives: no waveform, no pulse clock, no raster.
    # The delivered train has 4 raster groups and hundreds of time points.
    npt.assert_equal(delivered.time.size > 100, True)
    # Four raster groups, plus the black pixel that delivers nothing:
    npt.assert_equal(n_schedules_of(delivered), 5)
    # The modulation is one column per frame, so there is no schedule in it for
    # the electrodes to be split across -- which is the one thing it does not
    # record. The frame clock is a fact about the source, so it does survive:
    npt.assert_equal('cycle' in spatial.metadata['encoder'], False)
    npt.assert_equal(spatial.metadata['encoder']['frame_dur'],
                     delivered.metadata['encoder']['frame_dur'])
    npt.assert_array_equal(spatial.metadata['encoder']['frame_time'],
                           delivered.metadata['encoder']['frame_time'])

    # A video keeps one column per frame, and a time axis to say when each of
    # them starts (here `frame_dur=100` re-times the source, as it does for
    # `encode`):
    vid = VideoStimulus(np.random.default_rng(0).random((4, 4, 5)),
                        metadata={'fps': 20})
    spatial = enc.encode(vid, implant=implant)._spatial_view()
    npt.assert_equal(spatial.shape, (16, 5))
    npt.assert_almost_equal(spatial.time, np.arange(5) * 100.0)

    # Encoding for no implant at all works the same way, at pixel resolution:
    bare = enc.encode(img)._spatial_view()
    npt.assert_equal(bare.shape, (16, 1))
    npt.assert_almost_equal(bare.data.ravel(), np.linspace(0, 1, 16) * 50,
                            decimal=4)
    # ... and a source that is not a picture is refused here too:
    with pytest.raises(DimensionMismatchError):
        enc.encode(Stimulus([0.5]))


def test_FrequencyEncoder_spatial_view():
    """A spatial reading of rate coding says what it can and no more"""
    grays = np.array([[0.0, 0.5], [0.75, 1.0]])
    img = ImageStimulus(grays)
    enc = FrequencyEncoder(freq_range=(0, 200), amp=30, frame_dur=100)
    delivered = enc.encode(img)
    spatial = delivered._spatial_view()
    # Every electrode that pulses at all pulses at the same amplitude, which
    # is all a reader with no clock can be told, so rate collapses to on/off.
    # An electrode at 0 Hz never pulses, so it delivers no current -- that
    # much is not about time:
    npt.assert_almost_equal(spatial.data.ravel(), [0, 30, 30, 30], decimal=4)
    # The delivered train is where rate becomes visible, as pulse count:
    counts = [n_pulses_of(delivered, e, peak=30) for e in range(4)]
    npt.assert_equal(counts[0], 0)
    npt.assert_equal(np.all(np.diff(counts) > 0), True)


def test_StimulusEncoder_metadata():
    enc = AmplitudeEncoder(freq=50).encode(ImageStimulus(np.ones((2, 2))))
    npt.assert_almost_equal(enc.metadata['encoder']['frame_dur'], 500)
    npt.assert_almost_equal(enc.metadata['encoder']['frame_time'], [0])
    # Amplitude modulation puts every electrode on one schedule; frequency
    # modulation is what makes that number grow:
    npt.assert_equal(n_schedules_of(enc), 1)
    fm = FrequencyEncoder(freq_range=(10, 100), n_levels=4,
                          clock=1).encode(ImageStimulus(np.linspace(
                              0, 1, 16).reshape((4, 4))))
    npt.assert_equal(n_schedules_of(fm), 4)


def test_StimulusEncoder_degenerate_raster_is_no_raster():
    """A raster with one group has nothing to multiplex, so it changes nothing.

    This is the limiting case that says a raster only ever *staggers* onsets:
    with a single group there is nobody to stagger against, and the encoded
    stimulus has to come out bit-for-bit what it would have been without one.
    That holds even with an explicit ``group_dur``, which would otherwise set
    the sweep length.
    """
    implant = ArgusII(raster=None)
    img = ImageStimulus(np.random.default_rng(0).random((6, 10)))
    kwargs = dict(amp_range=(0, 50), freq=20, frame_dur=200)
    plain = AmplitudeEncoder(**kwargs).encode(img, implant=implant)
    npt.assert_equal(plain.metadata['encoder']['cycle'], None)

    names = list(implant.electrode_names)
    for raster in (SequentialRaster(1),
                   SequentialRaster(1, group_dur=3.0),
                   SequentialRaster(1, interleave=True),
                   CustomRaster([names]),
                   CustomRaster({n: 0 for n in names})):
        npt.assert_equal(raster.n_groups, 1)
        implant.raster = raster
        got = AmplitudeEncoder(**kwargs).encode(img, implant=implant)
        npt.assert_array_equal(got.data, plain.data)
        npt.assert_array_equal(got.time, plain.time)
        npt.assert_equal(got.metadata['encoder']['cycle'], None)
        # Nothing is staggered, so every electrode still fires together and the
        # stimulator has to source the whole array at once:
        npt.assert_almost_equal(np.abs(got.data).sum(axis=0).max(),
                                np.abs(plain.data).sum(axis=0).max())


def test_StimulusEncoder_degenerate_ranges():
    """A modulation range of zero width stops the gray levels mattering."""
    implant = ArgusII(raster=None)
    img = ImageStimulus(np.random.default_rng(1).random((6, 10)))
    kwargs = dict(frame_dur=200)

    # One amplitude for every gray level is a constant-amplitude train...
    flat = AmplitudeEncoder(amp_range=(30, 30), freq=20,
                            **kwargs).encode(img, implant=implant)
    npt.assert_almost_equal(np.abs(flat.data).max(axis=1), 30.0, decimal=4)
    # ... and one frequency for every gray level is the very same stimulus,
    # since frequency modulation at a constant rate *is* amplitude modulation
    # at a constant amplitude:
    same = FrequencyEncoder(freq_range=(20, 20), amp=30,
                            **kwargs).encode(img, implant=implant)
    npt.assert_array_equal(same.data, flat.data)
    npt.assert_array_equal(same.time, flat.time)
    npt.assert_equal(n_schedules_of(same), 1)

    # A black image asks for no current at all, at either end of the range:
    black = ImageStimulus(np.zeros((6, 10)))
    for enc in (AmplitudeEncoder(amp_range=(0, 50), **kwargs),
                FrequencyEncoder(freq_range=(0, 200), amp=50, **kwargs)):
        npt.assert_equal(np.any(enc.encode(black, implant=implant).data),
                         False)


def test_StimulusEncoder_n_levels_converges():
    """Quantizing onto enough gray levels is the same as not quantizing."""
    implant = ArgusII()
    img = ImageStimulus(np.random.default_rng(2).random((6, 10)))
    kwargs = dict(amp_range=(0, 50), freq=20, frame_dur=200)
    ref = AmplitudeEncoder(**kwargs).encode(img, implant=implant)
    err = [np.abs(AmplitudeEncoder(n_levels=n, **kwargs).encode(
               img, implant=implant).data - ref.data).max()
           for n in (4, 16, 256, 1 << 16)]
    # Each step of 4x in the level count is a step of ~4x in accuracy, and the
    # finest is close enough to be irrelevant next to a 50 uA range:
    npt.assert_equal(np.all(np.diff(err) < 0), True)
    npt.assert_array_less(err[-1], 1e-2)
    # Two levels is the coarsest allowed, and it is a black-or-white encoding:
    two = AmplitudeEncoder(n_levels=2, **kwargs).encode(img,
                                                        implant=implant)
    npt.assert_array_equal(np.unique(np.abs(two.data).max(axis=1)), [0.0, 50.0])


def test_StimulusEncoder_frame_rate_does_not_move_the_pulses():
    """The pulse clock is independent of the frame clock.

    Re-timing the same frames only changes how long the stimulus lasts and
    which gray level each pulse picks up. It must not change *when* the pulses
    come, which is what a frame-synchronous encoder would get wrong: at 29.97
    fps a frame is not a whole number of 20 Hz periods, so restarting the train
    every frame would silently deliver a different rate.
    """
    implant = ArgusII()
    vid = VideoStimulus(np.random.default_rng(3).random((6, 10, 4)),
                        metadata={'fps': 10})
    onsets = []
    for frame_dur in (100.0, 50.0, 25.0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stim = AmplitudeEncoder(amp_range=(50, 50), freq=20,
                                    frame_dur=frame_dur).encode(
                                        vid, implant=implant)
        npt.assert_almost_equal(stim.time[-1], 4 * frame_dur)
        neg = stim.data[0] < 0
        onsets.append(stim.time[neg & ~np.concatenate(([False], neg[:-1]))])
    # Every onset the shorter stimulus has, the longer one has at the same time:
    for short in onsets[1:]:
        npt.assert_almost_equal(short, onsets[0][:short.size])
    # ... and the requested 20 Hz is delivered whatever the frame rate:
    npt.assert_almost_equal(np.diff(onsets[0]), 50.0, decimal=6)


def test_AmplitudeEncoder_units():
    """Mixed unit spellings must encode to numerically identical stimuli"""
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    bare = AmplitudeEncoder(amp_range=(50, 100), freq=20, phase_dur=0.46,
                            interphase_dur=0.1, clock=1, frame_dur=50)
    unitful = AmplitudeEncoder(amp_range=(50 * uA, 0.1 * mA), freq=0.02 * kHz,
                               phase_dur=460 * us, interphase_dur=0.1 * ms,
                               clock=1000 * us, frame_dur=0.05 * sec)
    # The encoder stores plain numbers in its historical units:
    npt.assert_almost_equal(np.asarray(unitful.amp_range), [50, 100])
    npt.assert_almost_equal(unitful.freq, 20)
    npt.assert_almost_equal(unitful.phase_dur, 0.46)
    npt.assert_almost_equal(unitful.interphase_dur, 0.1)
    npt.assert_almost_equal(unitful.clock, 1)
    npt.assert_almost_equal(unitful.frame_dur, 50)
    for value in (*unitful.amp_range, unitful.freq, unitful.phase_dur,
                  unitful.interphase_dur, unitful.clock, unitful.frame_dur):
        npt.assert_equal(isinstance(value, Quantity), False)
    # ... and encodes identically either way:
    out_bare, out_unitful = bare.encode(img), unitful.encode(img)
    npt.assert_array_equal(out_bare.data, out_unitful.data)
    npt.assert_array_equal(out_bare.time, out_unitful.time)
    # The output is electrical, whatever the inputs were spelled in:
    npt.assert_equal(out_unitful.unit, uA)
    npt.assert_equal(out_unitful.time_unit, ms)
    npt.assert_equal(out_unitful.data.dtype, np.float32)


def test_FrequencyEncoder_units():
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    bare = FrequencyEncoder(freq_range=(20, 300), amp=50, clock=1)
    unitful = FrequencyEncoder(freq_range=(20 * Hz, 0.3 * kHz), amp=0.05 * mA,
                               clock=1000 * us)
    npt.assert_almost_equal(np.asarray(unitful.freq_range), [20, 300])
    npt.assert_almost_equal(unitful.amp, 50)
    for value in (*unitful.freq_range, unitful.amp):
        npt.assert_equal(isinstance(value, Quantity), False)
    out_bare, out_unitful = bare.encode(img), unitful.encode(img)
    npt.assert_array_equal(out_bare.data, out_unitful.data)
    npt.assert_array_equal(out_bare.time, out_unitful.time)
    npt.assert_equal(out_unitful.unit, uA)
    npt.assert_equal(out_unitful.time_unit, ms)


def test_encoder_dimension_errors():
    for kwargs in ({'amp_range': (0, 50 * ms)}, {'amp_range': (0 * ms, 50)},
                   {'freq': 20 * ms}, {'phase_dur': 0.46 * uA},
                   {'interphase_dur': 0.1 * uA}, {'clock': 1 * uA},
                   {'frame_dur': 50 * uA}):
        with pytest.raises(DimensionMismatchError):
            AmplitudeEncoder(**kwargs)
    for kwargs in ({'freq_range': (0, 300 * ms)}, {'amp': 50 * Hz}):
        with pytest.raises(DimensionMismatchError):
            FrequencyEncoder(**kwargs)
    # The message names the offending argument:
    with pytest.raises(DimensionMismatchError) as excinfo:
        AmplitudeEncoder(freq=20 * ms)
    npt.assert_equal("Parameter 'freq' expects frequency (Hz), got time"
                     in str(excinfo.value), True)


def test_encoder_source_must_be_dimensionless():
    """An encoder is where gray levels become current, not the other way"""
    enc = AmplitudeEncoder(amp_range=(0, 50))
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    vid = VideoStimulus(np.ones((2, 2, 3)) * 0.5, time=[0, 20, 40])
    # Pictures, yes:
    for source in (img, vid):
        npt.assert_equal(enc.encode(source).unit, uA)
    # Sampled at an implant's electrodes, still a picture:
    implant = ArgusII()
    sampled = implant.reshape_stim(img)
    npt.assert_equal(sampled.unit, dimensionless)
    npt.assert_equal(enc.encode(sampled).unit, uA)
    # ... and the implant path inside the encoder gives the same answer:
    npt.assert_equal(AmplitudeEncoder(amp_range=(0, 50)).encode(
        img, implant=implant).unit, uA)
    # An electrical stimulus, no: `Stimulus([0.5])` is half a microamp, and
    # reading it as a gray level would clip and re-modulate it silently.
    with pytest.raises(DimensionMismatchError) as excinfo:
        enc.encode(Stimulus([0.5]))
    npt.assert_equal("must be dimensionless" in str(excinfo.value), True)
    for source in (Stimulus([0.5]), BiphasicPulseTrain(20, 50, 0.45),
                   Stimulus(np.ones((2, 2)), time=[0, 1])):
        with pytest.raises(DimensionMismatchError):
            enc.encode(source)


def test_encoder_pulse_template_unit_agnostic():
    """A custom ``pulse`` template only lends its shape, not its unit"""
    img = ImageStimulus(np.linspace(0, 1, 4).reshape((2, 2)))
    shape = np.array([[0, 1, 1, 0, -1, -1, 0]], dtype=float)
    time = [0, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0]
    electrical = Stimulus(shape * 37.0, time=time)
    dimless = Stimulus(VideoStimulus(shape.reshape((1, 1, -1)), time=time))
    npt.assert_equal(electrical.unit, uA)
    npt.assert_equal(dimless.unit, dimensionless)
    # Both templates are accepted, and both give the same encoding: the
    # amplitude is normalized away, so only the shape survives.
    outs = [AmplitudeEncoder(pulse=p, amp_range=(0, 50)).encode(img)
            for p in (electrical, dimless)]
    npt.assert_array_equal(outs[0].data, outs[1].data)
    npt.assert_array_equal(outs[0].time, outs[1].time)
    for out in outs:
        npt.assert_equal(out.unit, uA)
        npt.assert_almost_equal(np.abs(out.data).max(), 50)


def _rendered(stim):
    """Whether the encoded stimulus has expanded its schedule into samples"""
    return stim._Stimulus__stim['data'] is not None


# One case per thing the schedule can do: a still frame, several frames, a
# per-electrode frequency, a device raster, a stimulator clock, a supplied
# pulse, and a source that asks for nothing at all.
ENCODED = [
    ('amplitude', lambda: AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))),
    ('amplitude-video', lambda: AmplitudeEncoder().encode(
        VideoStimulus(np.linspace(0, 1, 192).reshape(8, 8, 3),
                      time=np.arange(3) * 40.0))),
    ('frequency', lambda: FrequencyEncoder(freq_range=(0, 60)).encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))),
    ('rastered', lambda: AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)),
        implant=ArgusII())),
    ('clocked', lambda: AmplitudeEncoder(clock=1.0).encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))),
    ('custom-pulse', lambda: AmplitudeEncoder(
        pulse=BiphasicPulse(1, 0.2, interphase_dur=0.1)).encode(
            ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))),
    ('all-zero', lambda: FrequencyEncoder(freq_range=(0, 60)).encode(
        ImageStimulus(np.zeros((6, 6))))),
]


@pytest.mark.parametrize('name, build', ENCODED, ids=[c[0] for c in ENCODED])
def test_encoded_stimulus_defers_only_the_waveform(name, build):
    stim = build()
    npt.assert_equal(_rendered(stim), False)
    # Everything the schedule already settled, without expanding it:
    npt.assert_equal(len(stim.electrodes) > 0, True)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.time_unit, ms)
    npt.assert_equal(sorted(stim.metadata['encoder']),
                     ['cycle', 'frame_dur', 'frame_time'])
    npt.assert_equal(stim.duration > 0, True)
    repr(stim)
    copies = [copy(stim), deepcopy(stim)]
    npt.assert_equal(_rendered(stim), False)
    for copied in copies:
        npt.assert_equal(_rendered(copied), False)
    # ...and the first read of the waveform is what expands it, once:
    data = stim.data
    npt.assert_equal(_rendered(stim), True)
    for _ in range(3):
        npt.assert_equal(np.shares_memory(stim.data, data), True)
    npt.assert_equal(stim.data.dtype, np.float32)
    npt.assert_equal(stim.time.dtype, np.float64)
    npt.assert_almost_equal(stim.time[-1], stim.duration)


@pytest.mark.parametrize('name, build', ENCODED, ids=[c[0] for c in ENCODED])
def test_encoded_stimulus_holds_no_waveform_sized_array(name, build):
    # The point of the phase: what is retained may scale with electrodes x
    # frames, with the pulse onsets, or with the global time axis -- but not
    # with electrodes x time, which is the array that gets large.
    stim = build()
    n_el, n_time = len(stim.electrodes), stim._ticks.size
    retained = [stim._amp, stim._ticks, stim._sched, stim._pulse_ticks,
                stim._pulse_vals, *stim._onsets, *stim._frames]
    for array in retained:
        npt.assert_equal(array.shape == (n_el, n_time), False)
        # Frozen, like every other piece of a stimulus' scientific state:
        npt.assert_equal(array.flags.writeable, False)
    npt.assert_equal(stim._amp.shape[0], n_el)
    # The matrix appears only when asked for:
    npt.assert_equal(stim.data.shape, (n_el, n_time))


def test_encoded_stimulus_is_independent_of_the_encoder():
    # The schedule is resolved once, at `encode`. Whatever the encoder or its
    # pulse template does afterwards describes some other encoding:
    encoder = AmplitudeEncoder(amp_range=(0, 50), freq=20)
    img = ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8))
    stim = encoder.encode(img)
    encoder.amp_range = (0, 500)
    encoder.freq = 200
    encoder.phase_dur = 4.0
    npt.assert_array_equal(stim.data, encoder.__class__(
        amp_range=(0, 50), freq=20).encode(img).data)


@pytest.mark.parametrize('modify', [lambda s: s + 5, lambda s: s >> 5,
                                    lambda s: s * np.inf,
                                    lambda s: s.pad(s.duration + 10)])
def test_encoded_stimulus_transformations_degrade(modify):
    # A schedule describes when each electrode pulses and how hard on each
    # frame. A DC offset or a shift in time leaves it describing nothing, so
    # what comes back is an ordinary stimulus:
    stim = AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))
    with np.errstate(divide='ignore', invalid='ignore'):
        out = modify(stim)
    npt.assert_equal(type(out), Stimulus)
    npt.assert_equal(out._is_parametric, False)


@pytest.mark.parametrize('factor', [2, 0.5, -1, 1, 0])
def test_encoded_stimulus_scaling_scales_both_descriptions(factor):
    # Scaling changes how hard each electrode is driven, not when. So the
    # schedule survives -- and the waveform and the modulation behind it move
    # together, which is what `find_threshold` varies from trial to trial.
    stim = AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))
    view = stim._spatial_view()
    scaled = stim * factor
    npt.assert_equal(type(scaled), type(stim))
    npt.assert_equal(_rendered(scaled), False)
    npt.assert_allclose(scaled._spatial_view().data, factor * view.data,
                        rtol=1e-6, atol=1e-6)
    npt.assert_allclose(scaled.data, factor * stim.data, rtol=1e-6, atol=1e-6)
    npt.assert_array_equal(scaled.time, stim.time)
    # The frame clock is a fact about the source, and scaling is not:
    npt.assert_equal(scaled.metadata['encoder'], stim.metadata['encoder'])
    npt.assert_allclose(view.data, stim._spatial_view().data)


def test_encoded_stimulus_drops_electrodes_structurally():
    # An implant switching an electrode off must not cost the modulation view,
    # which is the only thing a spatial model can read.
    stim = AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))
    fewer = stim._without_electrodes([0, 3])
    npt.assert_equal(type(fewer), type(stim))
    npt.assert_equal(_rendered(fewer), False)
    npt.assert_equal(len(fewer.electrodes), 62)
    kept_rows = [i for i in range(len(stim.electrodes)) if i not in (0, 3)]
    npt.assert_array_equal(np.asarray(fewer.electrodes),
                           np.asarray(stim.electrodes)[kept_rows])
    # The same rows are gone from both descriptions of it:
    npt.assert_array_equal(fewer._spatial_view().data,
                           stim._spatial_view().data[kept_rows])
    npt.assert_array_equal(fewer.data, stim.data[kept_rows])
    npt.assert_array_equal(fewer.time, stim.time)


def test_encoded_stimulus_validates_while_it_schedules():
    # Scheduling stays eager, so what is wrong with an encoding is still said
    # when the encoding is asked for -- not when its waveform is:
    img = ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8))
    with pytest.raises(ValueError):
        # A pulse longer than the source it has to fit into:
        AmplitudeEncoder(phase_dur=400).encode(img)
    with pytest.raises(ValueError):
        # A raster that cannot get through in one pulse period:
        FrequencyEncoder(freq_range=(0, 300)).encode(img, implant=ArgusII())
    with pytest.raises(ValueError):
        # A pulse whose time points DT cannot resolve:
        AmplitudeEncoder(pulse=Stimulus([[0, 1, 0]],
                                        time=[0, 1e-5, 2e-5])).encode(img)
    # ...and so is the warning about frames that never reach an electrode:
    assert_warns_msg(UserWarning,
                     lambda: AmplitudeEncoder(freq=1).encode(
                         VideoStimulus(np.ones((4, 4, 8)),
                                       time=np.arange(8) * 40.0)),
                     'deliver no pulse at all')


def test_encoded_stimulus_survives_assignment_unrendered():
    # An implant stores a copy of what it is given; that copy has no more
    # reason to hold a waveform than the original did.
    implant = ArgusII()
    implant.stim = AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)), implant=implant)
    npt.assert_equal(_rendered(implant.stim), False)
    npt.assert_equal(len(implant.stim.electrodes), 60)
    # Assigning the picture itself goes through the implant's own encoder, and
    # arrives just as unexpanded:
    encoded = ArgusII(encoder=AmplitudeEncoder())
    encoded.stim = ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8))
    npt.assert_equal(_rendered(encoded.stim), False)
    npt.assert_equal(encoded.stim.data.shape[0], 60)
