import warnings

import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import trapezoid

from pulse2percept.implants import ArgusII, SequentialRaster
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BostonTrain, Encoder, FrequencyEncoder,
                                   ImageStimulus, MonophasicPulse, Stimulus,
                                   VideoStimulus)
from pulse2percept.stimuli import encoders
from pulse2percept.utils.constants import DT


def n_pulses_of(stim, electrode=0, peak=None):
    """Count the pulses one electrode of an encoded stimulus delivers"""
    row = stim.data[electrode]
    peak = np.abs(row).max() if peak is None else peak
    if peak == 0:
        return 0
    firing = np.abs(row) >= 0.99 * peak
    # Each pulse has a leading and a trailing phase, both at full amplitude:
    return np.count_nonzero(np.diff(firing.astype(int)) > 0) // 2


def test_Encoder_is_abstract():
    with pytest.raises(TypeError):
        Encoder()


def test_Encoder_source():
    enc = AmplitudeEncoder()
    with pytest.raises(TypeError):
        enc.encode(np.random.rand(4, 5))
    with pytest.raises(TypeError):
        enc.encode('not-a-stimulus')


def test_Encoder_params():
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
    implant = ArgusII()
    vid = BostonTrain()
    enc = AmplitudeEncoder(implant, amp_range=(0, 50)).encode(vid)
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
        AmplitudeEncoder(ArgusII(), freq=1000).encode(vid)


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


def test_Encoder_clock():
    # A clock rounds the pulse period onto a whole number of cycles, so
    # frequencies that differ by less than that collapse onto one schedule:
    img = ImageStimulus(np.linspace(0.5, 1, 16).reshape((4, 4)))
    fine = FrequencyEncoder(freq_range=(0, 300), frame_dur=100).encode(img)
    coarse = FrequencyEncoder(freq_range=(0, 300), frame_dur=100,
                              clock=1).encode(img)
    npt.assert_equal(fine.metadata['encoder']['n_schedules'], 16)
    npt.assert_equal(coarse.metadata['encoder']['n_schedules'] < 16, True)
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


def test_Encoder_n_levels():
    # Quantizing the gray levels quantizes whatever they are modulated onto:
    img = ImageStimulus(np.linspace(0, 1, 64).reshape((8, 8)))
    am = AmplitudeEncoder(amp_range=(0, 50), n_levels=4).encode(img)
    npt.assert_almost_equal(np.unique(np.abs(am.data).max(axis=1)),
                            [0, 50 / 3, 100 / 3, 50], decimal=4)
    # ... and for frequency modulation that is what keeps the time axis small:
    fm = FrequencyEncoder(freq_range=(0, 300), frame_dur=100).encode(img)
    fm4 = FrequencyEncoder(freq_range=(0, 300), frame_dur=100,
                           n_levels=4).encode(img)
    npt.assert_equal(fm4.metadata['encoder']['n_schedules'], 4)
    npt.assert_equal(fm4.shape[1] < fm.shape[1], True)


def test_Encoder_big_time_warning(monkeypatch):
    monkeypatch.setattr(encoders, '_BIG_TIME', 100)
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    with pytest.warns(UserWarning, match='time points'):
        FrequencyEncoder(freq_range=(0, 300), frame_dur=100).encode(img)


def test_FrequencyEncoder_implant():
    implant = ArgusII()
    enc = FrequencyEncoder(implant, freq_range=(0, 300), amp=50,
                           clock=1).encode(BostonTrain())
    npt.assert_equal(enc.shape[0], implant.n_electrodes)
    npt.assert_almost_equal(np.abs(enc.data).max(), 50)
    implant.stim = enc
    npt.assert_equal(implant.stim.shape, enc.shape)
    # The clock is what makes this tractable at all: without one, the same
    # clip needs several times as many time points:
    unclocked = FrequencyEncoder(implant, freq_range=(0, 300),
                                 amp=50).encode(BostonTrain())
    npt.assert_equal(enc.shape[1] < unclocked.shape[1] / 5, True)


def test_Encoder_raster():
    img = ImageStimulus(np.ones((2, 2)))
    raster = SequentialRaster(2, interleave=True)
    enc = AmplitudeEncoder(freq=100, frame_dur=100,
                           raster=raster).encode(img)
    # Rastering splits the electrodes across two pulse schedules. The cycle a
    # raster has to get through is the pulse *period*, not the frame, so the
    # two groups are offset by half a period:
    npt.assert_equal(enc.metadata['encoder']['n_schedules'], 2)
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
    # A clock snaps the slot offsets along with the period. A 4.6 ms slot lands
    # on 5 ms, and the 10 ms period onto the 9 ms cycle the two slots make:
    enc = AmplitudeEncoder(freq=100, frame_dur=100, clock=1,
                           raster=SequentialRaster(
                               2, interleave=True,
                               group_dur=4.6)).encode(img)
    npt.assert_almost_equal(pulse_onsets(enc, 1)[0], 5, decimal=3)
    npt.assert_almost_equal(np.diff(pulse_onsets(enc, 1)), 9, decimal=3)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 9)


def test_Encoder_raster_frequency_modulation():
    # Under frequency modulation electrodes want different periods, so they
    # can only be kept apart by quantizing every period onto a common raster
    # cycle. The fastest electrode pulses once per cycle, slower ones every
    # m-th cycle, and no two groups ever coincide:
    img = ImageStimulus(np.linspace(0.25, 1, 16).reshape((4, 4)))
    enc = FrequencyEncoder(freq_range=(0, 120), amp=10, frame_dur=200,
                           raster=SequentialRaster(4, interleave=True)).encode(img)
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
        FrequencyEncoder(freq_range=(0, 300), amp=10, frame_dur=200,
                         raster=SequentialRaster(6)).encode(img)


def test_Encoder_raster_from_implant():
    implant = ArgusII()
    implant.raster = SequentialRaster(6)
    vid = VideoStimulus(np.ones((6, 10, 2)), metadata={'fps': 30})
    # The encoder picks up the implant's raster without being told:
    enc = AmplitudeEncoder(implant, freq=30).encode(vid)
    npt.assert_equal(enc.metadata['encoder']['n_schedules'], 6)
    delays = [pulse_onsets(enc, e)[0] for e in (0, 10, 20, 30, 40, 50)]
    npt.assert_almost_equal(delays, np.arange(6) * 1000 / 30 / 6, decimal=2)
    # An explicit raster on the encoder wins, so one can be tried out without
    # modifying the implant:
    enc = AmplitudeEncoder(implant, freq=30,
                           raster=SequentialRaster(2)).encode(vid)
    npt.assert_equal(enc.metadata['encoder']['n_schedules'], 2)
    # And no raster anywhere means every electrode fires at frame onset:
    implant.raster = None
    enc = AmplitudeEncoder(implant, freq=30).encode(vid)
    npt.assert_equal(enc.metadata['encoder']['n_schedules'], 1)


def test_Encoder_raster_current_limit():
    # 60 electrodes at 50 uA is 3000 uA if they all fire at once, but only
    # 500 uA if they take turns ten at a time:
    implant = ArgusII()
    implant.max_current = 1000
    vid = VideoStimulus(np.ones((6, 10, 3)), metadata={'fps': 30})
    with pytest.raises(ValueError, match='raster'):
        implant.stim = AmplitudeEncoder(implant, amp_range=(50, 50),
                                        freq=30).encode(vid)
    implant.raster = SequentialRaster(6)
    implant.stim = AmplitudeEncoder(implant, amp_range=(50, 50),
                                    freq=30).encode(vid)
    npt.assert_almost_equal(np.abs(implant.stim.data).sum(axis=0).max(), 500)
    # A raster that cannot get through all its groups within a frame is not a
    # usable schedule:
    with pytest.raises(ValueError):
        AmplitudeEncoder(implant, freq=30,
                         raster=SequentialRaster(6, group_dur=20)).encode(vid)
    # Neither is one whose groups get a turn too short to pulse in. Sixty
    # 0.92 ms pulses take 55 ms, which does not fit into a 33 ms frame, so
    # electrode-at-a-time rastering is impossible here rather than merely
    # dropping the electrodes that come last:
    with pytest.raises(ValueError, match='no room'):
        AmplitudeEncoder(implant, freq=30,
                         raster=SequentialRaster(60)).encode(vid)
    # Halving the phase duration makes it fit:
    enc = AmplitudeEncoder(implant, freq=30, phase_dur=0.2,
                           raster=SequentialRaster(60)).encode(vid)
    npt.assert_equal(enc.metadata['encoder']['n_schedules'], 60)
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 50)


@pytest.mark.parametrize('fps', [29.97, 30, 24, 59.94])
@pytest.mark.parametrize('freq', [50, 100])
def test_Encoder_freq_is_actual_freq(fps, freq):
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


def test_Encoder_pulse_offset():
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


def test_Encoder_zero_amp():
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
    implant = ArgusII()
    dark = VideoStimulus(np.zeros((6, 10, 3)), metadata={'fps': 30})
    with pytest.raises(ValueError, match='no room'):
        AmplitudeEncoder(implant, freq=30,
                         raster=SequentialRaster(60)).encode(dark)
    # ... and a workable one costs a dark video nothing:
    enc = AmplitudeEncoder(implant, amp_range=(0, 50), freq=30, phase_dur=0.2,
                           raster=SequentialRaster(60)).encode(dark)
    npt.assert_equal(np.all(enc.data == 0), True)
    npt.assert_equal(enc.shape[1], 2)


def test_Encoder_implant_reshape():
    # Passing an implant means "sample the source at the electrode locations".
    # Row count is not a usable test of whether that already happened: a 10x6
    # image and an RGB 4x5 image both have exactly as many rows as Argus II has
    # electrodes, and both used to skip sampling (and, for RGB, `rgb2gray`)
    # while still being labeled with electrode names.
    implant = ArgusII()
    for src in [ImageStimulus(np.random.rand(10, 6)),
                ImageStimulus(np.random.rand(4, 5, 3)),
                ImageStimulus(np.random.rand(6, 10)),
                VideoStimulus(np.random.rand(10, 6, 2))]:
        npt.assert_equal(src.data.shape[0], implant.n_electrodes)
        enc = AmplitudeEncoder(implant, amp_range=(0, 50)).encode(src)
        direct = AmplitudeEncoder(amp_range=(0, 50)).encode(
            implant.reshape_stim(src))
        npt.assert_almost_equal(enc.data, direct.data, decimal=4)
        npt.assert_equal(list(enc.electrodes), list(implant.electrode_names))


def test_Encoder_metadata():
    enc = AmplitudeEncoder(freq=50).encode(ImageStimulus(np.ones((2, 2))))
    npt.assert_equal(enc.metadata['encoder']['kind'], 'AmplitudeEncoder')
    npt.assert_almost_equal(enc.metadata['encoder']['frame_dur'], 500)
    npt.assert_equal(enc.metadata['encoder']['n_frames'], 1)
    # Amplitude modulation puts every electrode on one schedule; frequency
    # modulation is what makes that number grow:
    npt.assert_equal(enc.metadata['encoder']['n_schedules'], 1)
    fm = FrequencyEncoder(freq_range=(10, 100), n_levels=4,
                          clock=1).encode(ImageStimulus(np.linspace(
                              0, 1, 16).reshape((4, 4))))
    npt.assert_equal(fm.metadata['encoder']['kind'], 'FrequencyEncoder')
    npt.assert_equal(fm.metadata['encoder']['n_schedules'], 4)
