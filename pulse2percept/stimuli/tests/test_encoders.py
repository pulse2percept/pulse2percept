import warnings
from copy import copy, deepcopy

import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import trapezoid

from pulse2percept.implants import (CustomRaster, DiskElectrode, GridImplant,
                                    SequentialRaster)
from pulse2percept.implants.retina import ArgusII, PRIMAPivotal
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BiphasicPulseTrain, Encoder,
                                   FrequencyEncoder, ImageStimulus,
                                   ImplantEncoder, MonophasicPulse,
                                   PhotovoltaicEncoder, PRIMAEncoder,
                                   PulseEncoder, Stimulus, VideoStimulus)
from pulse2percept import stimuli as p2p_stimuli
from pulse2percept.stimuli import encoders
from pulse2percept.utils.constants import DT
from pulse2percept.utils.testing import assert_warns_msg
from pulse2percept.units import (DimensionMismatchError, Hz, Quantity, W,
                                 dimensionless, kHz, m, mA, mW, mm, ms, nm,
                                 uA, us, xTh)
from pulse2percept.units import s as sec


def n_pulses_of(stim, electrode=0, peak=None):
    """Return the number of pulses one electrode delivers"""
    row = stim.data[electrode]
    peak = np.abs(row).max() if peak is None else peak
    if peak == 0:
        return 0
    firing = np.abs(row) >= 0.99 * peak
    # Each pulse has a leading and a trailing phase, both at full amplitude:
    return np.count_nonzero(np.diff(firing.astype(int)) > 0) // 2


def pixel_implant(shape, raster=None):
    """Return an implant with one electrode per pixel of a ``shape`` image

    Electrodes sit exactly on the pixels, so sampling at the implant does not
    change the encoding. Used to test rasters.
    """
    implant = GridImplant(shape, 200, electrode_type=DiskElectrode, radius=50)
    implant.raster = raster
    return implant


def n_schedules_of(stim):
    """Return the number of distinct pulse schedules across electrodes

    Two electrodes share a schedule when current flows at the same times,
    regardless of amplitude. Electrodes delivering nothing share the empty
    schedule.
    """
    return len(np.unique(np.abs(stim.data) > 0, axis=0))


def test_PulseEncoder_is_abstract():
    with pytest.raises(TypeError):
        PulseEncoder()
    with pytest.raises(TypeError):
        Encoder()
    with pytest.raises(TypeError):
        ImplantEncoder()


def test_Encoder_hierarchy():
    # Pulse modulation strategies share the electrical pulse machinery:
    for cls in (AmplitudeEncoder, FrequencyEncoder):
        npt.assert_equal(issubclass(cls, PulseEncoder), True)
    npt.assert_equal(issubclass(PulseEncoder, ImplantEncoder), True)
    npt.assert_equal(issubclass(ImplantEncoder, Encoder), True)
    # Photovoltaic encoders are optical, not electrical pulse encoders:
    for cls in (PhotovoltaicEncoder, PRIMAEncoder):
        npt.assert_equal(issubclass(cls, ImplantEncoder), True)
        npt.assert_equal(issubclass(cls, PulseEncoder), False)
    for encoder in (AmplitudeEncoder(), PRIMAEncoder()):
        npt.assert_equal(isinstance(encoder, Encoder), True)
        npt.assert_equal(isinstance(encoder, ImplantEncoder), True)
    npt.assert_equal(issubclass(PRIMAEncoder, PhotovoltaicEncoder), True)
    npt.assert_equal(hasattr(p2p_stimuli, 'StimulusEncoder'), False)


def _unrastered_argus():
    """Return Argus II without a raster (same schedule as no implant)"""
    return ArgusII(raster=None)


@pytest.mark.parametrize('build, make_implant', [
    (lambda i: AmplitudeEncoder(i, amp_range=(0, 30)), _unrastered_argus),
    (lambda i: FrequencyEncoder(i, freq_range=(0, 60)), _unrastered_argus),
    (lambda i: PhotovoltaicEncoder(i, irradiance=4, freq=40, pulse_dur=4,
                                   wavelength=915), PRIMAPivotal),
    (lambda i: PRIMAEncoder(i), PRIMAPivotal),
])
def test_Encoder_implant_is_constructor_state(build, make_implant):
    implant = make_implant()
    encoder = build(implant)
    npt.assert_equal(encoder.implant is implant, True)
    img = ImageStimulus(np.random.default_rng(0).random((12, 12)))
    # Repeated encoding needs no implant argument and samples at electrodes:
    first, second = encoder.encode(img), encoder.encode(img)
    npt.assert_equal(encoder.implant is implant, True)
    npt.assert_equal(list(first.electrodes), list(implant.electrode_names))
    npt.assert_array_equal(first.data, second.data)
    # Sampling at the implant is the same as encoding the sampled picture:
    unbound = build(None)
    npt.assert_equal(unbound.implant, None)
    direct = unbound.encode(implant.reshape_stim(img))
    npt.assert_array_equal(first.data, direct.data)
    npt.assert_array_equal(first.time, direct.time)
    # Without an implant, every pixel is a stimulation site:
    npt.assert_equal(unbound.encode(img).shape[0], img.data.shape[0])
    # The implant is no longer an `encode` argument:
    with pytest.raises(TypeError):
        encoder.encode(img, implant=implant)
    # The implant repr prints its encoder without recursion:
    implant.encoder = encoder
    npt.assert_equal(type(implant).__name__ in str(implant.encoder), True)
    npt.assert_equal('encoder' in str(implant), True)


def test_Encoder_binding_cannot_migrate():
    implant_a, implant_b = ArgusII(encoder=None), ArgusII(encoder=None)
    encoder = AmplitudeEncoder()
    implant_a.encoder = encoder
    # `implant` is read-only:
    with pytest.raises(AttributeError):
        encoder.implant = implant_b
    with pytest.raises(ValueError, match='already bound'):
        encoder._bind(implant_b)
    with pytest.raises(ValueError, match='already bound'):
        implant_b.encoder = encoder
    # Binding to the same implant again is allowed:
    encoder._bind(implant_a)
    implant_a.encoder = encoder
    npt.assert_equal(encoder.implant is implant_a, True)
    npt.assert_equal(implant_a.encoder is encoder, True)
    npt.assert_equal(implant_b.encoder, None)
    # Encoding still samples at implant A's electrodes:
    img = ImageStimulus(np.random.default_rng(0).random((12, 12)))
    npt.assert_array_equal(
        implant_a.prepare_stim(img).data,
        implant_a.prepare_stim(implant_a.reshape_stim(img)).data)


def test_Encoder_rejects_non_implant():
    with pytest.raises(TypeError):
        AmplitudeEncoder((0, 50))
    with pytest.raises(TypeError):
        PRIMAEncoder('PRIMA')
    # Optical parameters are keyword-only:
    with pytest.raises(TypeError):
        PhotovoltaicEncoder(None, 4, 40, 4, 915)


def test_PulseEncoder_warnings_point_at_the_caller(monkeypatch):
    """Warnings about source, frequency, or implant point at the caller's line

    A bare ``warnings.warn`` would point at ``encoders.py``.
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

    # Including the warning about frames that never get a pulse:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        AmplitudeEncoder(ArgusII(), freq=2).encode(vid)
    npt.assert_equal([w.filename for w in caught], [__file__] * len(caught))
    npt.assert_equal(any('deliver no pulse' in str(w.message)
                         for w in caught), True)


def test_PulseEncoder_source():
    enc = AmplitudeEncoder()
    with pytest.raises(TypeError):
        enc.encode(np.random.rand(4, 5))
    with pytest.raises(TypeError):
        enc.encode('not-a-stimulus')


def test_PulseEncoder_params():
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
        # ... and must be on a single electrode:
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
    # A fractional level count would give a fractional step size (and a
    # different number of levels):
    with pytest.raises(ValueError):
        AmplitudeEncoder(n_levels=2.5)
    # NaN and inf pass every `<` comparison, so they must be rejected here:
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
    # One electrode per pixel; the stimulus lasts as long as the video:
    npt.assert_equal(enc.shape[0], 20)
    npt.assert_almost_equal(enc.time[0], 0)
    npt.assert_almost_equal(enc.time[-1], 6, decimal=3)
    # Time points stay strictly monotonically increasing across frames:
    npt.assert_equal(np.all(np.diff(enc.time) > 0.95 * DT), True)
    # Cathodic first: the largest excursion of each electrode is negative:
    npt.assert_almost_equal(enc.data.min(axis=1), -50 * stim.data.max(axis=1))
    npt.assert_almost_equal(enc.data.max(axis=1), 50 * stim.data.max(axis=1))
    # Anodic first flips the sign; the magnitude is the same:
    ana = AmplitudeEncoder(freq=1000, cathodic_first=False).encode(stim)
    npt.assert_almost_equal(np.abs(ana.data), np.abs(enc.data))
    npt.assert_almost_equal(ana.data[:, 1], -enc.data[:, 1])
    # Pulses are charge-balanced. Integrate in float64: a float32 time axis
    # resolves DT-wide pulse edges too poorly for `is_charge_balanced` to hold
    # over more than a few pulses (also true of `BiphasicPulseTrain`):
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=4)


@pytest.mark.parametrize('amp_range', [(0, 50), (2, 43), (10, 10)])
def test_AmplitudeEncoder_amp_range(amp_range):
    # Gray levels map onto `amp_range` absolutely: a given gray level always
    # gives the same amplitude, regardless of the rest of the image:
    lo, hi = amp_range
    for gray in (0.0, 0.25, 1.0):
        img = ImageStimulus(np.full((4, 4), gray))
        enc = AmplitudeEncoder(amp_range=amp_range).encode(img)
        npt.assert_almost_equal(np.abs(enc.data).max(),
                                lo + gray * (hi - lo), decimal=4)
    # The image extremes reach both ends of the range:
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    enc = AmplitudeEncoder(amp_range=amp_range).encode(img)
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1).min(), lo, decimal=4)
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1).max(), hi, decimal=4)


def test_AmplitudeEncoder_stretch():
    # A uniform image has an absolute gray level but no range to stretch:
    img = ImageStimulus(np.full((4, 4), 0.5))
    npt.assert_almost_equal(np.abs(img.encode().data).max(), 25)
    npt.assert_almost_equal(
        np.abs(AmplitudeEncoder(stretch=True).encode(img).data).max(), 0)
    # Stretching maps the darkest pixel to the bottom of the range and the
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
    # 0 Hz is silent but keeps the stimulus duration:
    enc = AmplitudeEncoder(freq=0).encode(img)
    npt.assert_almost_equal(np.abs(enc.data).max(), 0)
    npt.assert_almost_equal(enc.time[-1], 500)
    # A frequency below the frame rate works (the pulse clock is independent
    # of frames) but warns: whole frames deliver nothing, and their gray
    # levels never reach the electrode:
    vid = VideoStimulus(np.ones((2, 2, 5)), metadata={'fps': 30})
    with pytest.warns(UserWarning, match='deliver no pulse'):
        sparse = AmplitudeEncoder(freq=10).encode(vid)
    # 5 frames of 33.3 ms is 166.7 ms, which holds two 100 ms periods:
    npt.assert_equal(n_pulses_of(sparse), 2)
    npt.assert_almost_equal(np.diff(pulse_onsets(sparse)), 100, decimal=3)
    # A pulse that does not fit into a frame raises an error, not a warning:
    with pytest.raises(ValueError):
        AmplitudeEncoder(freq=1000, phase_dur=10).encode(img)
    # So does a pulse that does not fit into a pulse train window:
    with pytest.raises(ValueError):
        AmplitudeEncoder(freq=2000, phase_dur=0.46).encode(img)


def test_AmplitudeEncoder_pulse():
    img = ImageStimulus(np.ones((2, 2)))
    # A custom pulse replaces the default biphasic one; its amplitude is
    # normalized away:
    pulse = MonophasicPulse(-20, 0.5)
    enc = AmplitudeEncoder(pulse=pulse, freq=100,
                           amp_range=(0, 30)).encode(img)
    npt.assert_almost_equal(np.abs(enc.data).max(), 30)
    # A monophasic pulse stays unbalanced: all 50 pulses push charge the same
    # way. Each carries amp * (phase_dur - DT), losing DT to the two ramped
    # edges:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, -30 * (0.5 - DT) * 50, decimal=2)
    # The caller's pulse (data and time axis) is unchanged:
    pulse = BiphasicPulse(20, 0.2)
    data, time = pulse.data.copy(), pulse.time.copy()
    AmplitudeEncoder(pulse=pulse, freq=100).encode(img)
    npt.assert_almost_equal(pulse.data, data)
    npt.assert_almost_equal(pulse.time, time)


def test_AmplitudeEncoder_image():
    img = ImageStimulus(np.random.rand(4, 5))
    # An image has no time axis, so it is a single frame of 500 ms by default:
    npt.assert_almost_equal(AmplitudeEncoder().encode(img).time[-1], 500)
    npt.assert_almost_equal(
        AmplitudeEncoder(frame_dur=123).encode(img).time[-1], 123)
    # `frame_dur` also overrides the frame rate of a video:
    vid = VideoStimulus(np.random.rand(4, 5, 3))
    npt.assert_almost_equal(
        AmplitudeEncoder(frame_dur=10).encode(vid).time[-1], 30)


def test_AmplitudeEncoder_implant(camera_video):
    # No raster: this checks sampling, and a raster would stagger the onsets
    # relative to the pixel-resolution encoding compared at the end.
    implant = ArgusII(raster=None)
    vid = camera_video
    enc = AmplitudeEncoder(implant, amp_range=(0, 50)).encode(vid)
    # The video is sampled at the electrode locations, so the stimulus has one
    # row per electrode instead of one per pixel:
    npt.assert_equal(enc.shape[0], implant.n_electrodes)
    npt.assert_equal(list(enc.electrodes), list(implant.electrode_names))
    npt.assert_equal(np.abs(enc.data).max() <= 50, True)
    # ... and it is much smaller:
    npt.assert_equal(enc.data.nbytes < 1e6, True)
    # The implant can prepare it without further reshaping:
    npt.assert_equal(implant.prepare_stim(enc).shape, enc.shape)
    # Sampling then encoding equals encoding a downsampled video, so for
    # amplitude modulation this is a pure optimization:
    sampled = implant.reshape_stim(vid)
    direct = AmplitudeEncoder(amp_range=(0, 50)).encode(sampled)
    npt.assert_almost_equal(enc.data, direct.data, decimal=4)
    npt.assert_almost_equal(enc.time, direct.time)


def test_AmplitudeEncoder_big_stim_warning(monkeypatch):
    # Pixel-resolution encoding warns about memory and suggests passing an
    # implant. The threshold is lowered to avoid allocating hundreds of MB:
    monkeypatch.setattr(encoders, '_BIG_STIM', 100)
    vid = VideoStimulus(np.random.rand(8, 8, 4))
    with pytest.warns(UserWarning, match="with an 'implant'"):
        AmplitudeEncoder(freq=1000).encode(vid)
    # Passing an implant does not warn:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        AmplitudeEncoder(ArgusII(raster=None), freq=1000).encode(vid)


def whole_pulses(freq, frame_dur, pulse_dur=0.92):
    """Return how many ``freq`` Hz pulses can start and finish in a frame"""
    if freq <= 0:
        return 0
    last = np.floor((frame_dur - DT) / DT + 1e-9)
    room = last - round(pulse_dur / DT)
    return int(room // round(1000.0 / freq / DT)) + 1 if room >= 0 else 0


def pulse_onsets(stim, electrode=0):
    """Return the onset times (ms) of one electrode's pulses

    Assumes cathodic-first pulses (one run of negative samples per pulse).
    """
    neg = stim.data[electrode] < 0
    started = neg & ~np.concatenate(([False], neg[:-1]))
    # The first full-amplitude sample is one tick past the onset, because a
    # pulse ramps up over DT:
    return stim.time[started] - DT


def test_FrequencyEncoder():
    # A gray ramp, so every electrode gets a different frequency:
    grays = np.linspace(0, 1, 16)
    img = ImageStimulus(grays.reshape((4, 4)))
    enc = FrequencyEncoder(freq_range=(0, 100), amp=37,
                           frame_dur=100).encode(img)
    # Every electrode pulses at the same amplitude, cathodic first:
    peaks = np.abs(enc.data).max(axis=1)
    npt.assert_almost_equal(peaks[1:], 37)
    npt.assert_almost_equal(enc.data.min(), -37)
    # Gray level 0 maps onto 0 Hz (silence):
    npt.assert_almost_equal(peaks[0], 0)
    # ... and the pulse count grows with gray level:
    counts = [n_pulses_of(enc, e, peak=37) for e in range(16)]
    npt.assert_equal(counts, [whole_pulses(100 * g, 100) for g in grays])
    npt.assert_equal(np.all(np.diff(counts) >= 0), True)
    npt.assert_equal(counts[-1], 10)
    # Electrodes at different frequencies do not share a time axis, so the
    # stimulus has many more time points than with amplitude modulation:
    am = AmplitudeEncoder(freq=100, frame_dur=100).encode(img)
    npt.assert_equal(enc.shape[1] > 4 * am.shape[1], True)
    npt.assert_equal(enc.time[-1], am.time[-1])
    npt.assert_equal(np.all(np.diff(enc.time) > 0.95 * DT), True)
    # Pulses stay charge-balanced. The tolerance reflects how well a float32
    # time axis resolves DT-wide pulse edges; a truncated pulse would show up
    # as several uA*ms, four orders of magnitude above that floor:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=3)


def test_FrequencyEncoder_whole_pulses():
    # A frame delivers only pulses it can finish. At 60 Hz a 33.3 ms frame
    # fits exactly two 0.92 ms pulses:
    img = ImageStimulus(np.ones((2, 2)))
    frame_dur = 1000 / 29.97
    for freq in (30, 60, 90):
        enc = FrequencyEncoder(freq_range=(freq, freq),
                               frame_dur=frame_dur).encode(img)
        npt.assert_equal(n_pulses_of(enc), whole_pulses(freq, frame_dur))
        # A truncated pulse would break charge balance:
        net = trapezoid(enc.data.astype(np.float64),
                        x=enc.time.astype(np.float64))
        npt.assert_almost_equal(net, 0, decimal=3)
    # Same for amplitude modulation:
    am = AmplitudeEncoder(freq=60, frame_dur=frame_dur).encode(img)
    npt.assert_equal(n_pulses_of(am), whole_pulses(60, frame_dur))


def test_PulseEncoder_clock():
    # A clock rounds the pulse period to whole cycles, so frequencies closer
    # than that collapse onto one schedule:
    img = ImageStimulus(np.linspace(0.5, 1, 16).reshape((4, 4)))
    fine = FrequencyEncoder(freq_range=(0, 300), frame_dur=100).encode(img)
    coarse = FrequencyEncoder(freq_range=(0, 300), frame_dur=100,
                              clock=1).encode(img)
    npt.assert_equal(n_schedules_of(fine), 16)
    npt.assert_equal(n_schedules_of(coarse) < 16, True)
    # ... giving far fewer time points to simulate:
    npt.assert_equal(coarse.shape[1] < fine.shape[1] / 2, True)
    # Every pulse lands on the clock grid; without a clock they do not:
    onsets = pulse_onsets(coarse)
    npt.assert_almost_equal(onsets, np.round(onsets), decimal=3)
    npt.assert_equal(np.allclose(pulse_onsets(fine),
                                 np.round(pulse_onsets(fine)), atol=1e-3),
                     False)
    # A clock that cannot resolve DT is rejected:
    with pytest.raises(ValueError):
        FrequencyEncoder(clock=DT / 10)


def test_PulseEncoder_n_levels():
    # Quantizing gray levels quantizes the modulated parameter:
    img = ImageStimulus(np.linspace(0, 1, 64).reshape((8, 8)))
    am = AmplitudeEncoder(amp_range=(0, 50), n_levels=4).encode(img)
    npt.assert_almost_equal(np.unique(np.abs(am.data).max(axis=1)),
                            [0, 50 / 3, 100 / 3, 50], decimal=4)
    # ... which, for frequency modulation, keeps the time axis small:
    fm = FrequencyEncoder(freq_range=(0, 300), frame_dur=100).encode(img)
    fm4 = FrequencyEncoder(freq_range=(0, 300), frame_dur=100,
                           n_levels=4).encode(img)
    npt.assert_equal(n_schedules_of(fm4), 4)
    npt.assert_equal(fm4.shape[1] < fm.shape[1], True)


def test_PulseEncoder_big_time_warning(monkeypatch):
    monkeypatch.setattr(encoders, '_BIG_TIME', 100)
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    with pytest.warns(UserWarning, match='time points'):
        FrequencyEncoder(freq_range=(0, 300), frame_dur=100).encode(img)


def test_FrequencyEncoder_implant(camera_video):
    # A 300 Hz period (3.3 ms) is too short for Argus II's six-group 2 ms
    # raster sweep, so this device drives every electrode at once:
    implant = ArgusII(raster=None)
    enc = FrequencyEncoder(implant, freq_range=(0, 300), amp=50,
                           clock=1).encode(camera_video)
    npt.assert_equal(enc.shape[0], implant.n_electrodes)
    npt.assert_almost_equal(np.abs(enc.data).max(), 50)
    npt.assert_equal(implant.prepare_stim(enc).shape, enc.shape)
    # Without a clock, the same clip needs several times as many time points:
    unclocked = FrequencyEncoder(implant, freq_range=(0, 300), amp=50).encode(
        camera_video)
    npt.assert_equal(enc.shape[1] < unclocked.shape[1] / 5, True)


def test_PulseEncoder_raster():
    img = ImageStimulus(np.ones((2, 2)))
    implant = pixel_implant((2, 2), SequentialRaster(2, interleave=True))
    enc = AmplitudeEncoder(implant, freq=100, frame_dur=100).encode(img)
    # The raster splits the electrodes into two pulse schedules. A raster
    # cycle spans the pulse period (not the frame), so the groups are offset
    # by half a period:
    npt.assert_equal(n_schedules_of(enc), 2)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 10)
    onsets = [pulse_onsets(enc, e) for e in (0, 1)]
    npt.assert_almost_equal(onsets[0][0], 0, decimal=3)
    npt.assert_almost_equal(onsets[1][0], 5, decimal=3)
    # Both groups keep the full requested rate; rastering only sets when
    # within each period an electrode fires:
    for group in onsets:
        npt.assert_almost_equal(np.diff(group), 10, decimal=3)
    npt.assert_equal(n_pulses_of(enc, 0), 10)
    npt.assert_equal(n_pulses_of(enc, 1), 10)
    # The groups never pulse at the same instant, so the stimulator sources
    # only one group's current at a time:
    npt.assert_equal(np.intersect1d(np.round(onsets[0], 3),
                                    np.round(onsets[1], 3)).size, 0)
    # Each group has two of the four electrodes: 2 x 50 uA at a time, not
    # 4 x 50 uA:
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 100)
    # Both groups stay charge-balanced:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=3)
    # A clock quantizes the slot, and the cycle is rebuilt from it, so groups
    # keep equal turns: a 4.6 ms slot becomes 5 ms, two slots make a 10 ms
    # cycle, and 100 Hz is kept exactly. Rounding offsets and cycle separately
    # would give a 9 ms cycle (5 ms + 4 ms turns) and 111 Hz:
    implant.raster = SequentialRaster(2, interleave=True, group_dur=4.6)
    enc = AmplitudeEncoder(implant, freq=100, frame_dur=100, clock=1).encode(
        img)
    npt.assert_almost_equal(pulse_onsets(enc, 1)[0], 5, decimal=3)
    npt.assert_almost_equal(np.diff(pulse_onsets(enc, 1)), 10, decimal=3)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 10)
    # Fit is checked on the quantized slot: two 5.1 ms slots do not fit a
    # 10 ms period, but on a 1 ms clock they become 5 ms slots, which do:
    implant.raster = SequentialRaster(2, interleave=True, group_dur=5.1)
    enc = AmplitudeEncoder(implant, freq=100, frame_dur=100, clock=1).encode(
        img)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 10)
    npt.assert_almost_equal(pulse_onsets(enc, 1)[0], 5, decimal=3)
    # Without a clock, the slots do not fit and raise an error:
    with pytest.raises(ValueError):
        AmplitudeEncoder(implant, freq=100, frame_dur=100).encode(img)


def test_PulseEncoder_raster_frequency_modulation():
    # Under frequency modulation electrodes have different periods, so every
    # period is quantized onto a common raster cycle. The fastest electrode
    # pulses once per cycle, slower ones every m-th cycle, and no two groups
    # coincide:
    img = ImageStimulus(np.linspace(0.25, 1, 16).reshape((4, 4)))
    implant = pixel_implant((4, 4), SequentialRaster(4, interleave=True))
    enc = FrequencyEncoder(implant, freq_range=(0, 120), amp=10,
                           frame_dur=200).encode(img)
    cycle = enc.metadata['encoder']['cycle']
    npt.assert_almost_equal(cycle, 1000 / 120)
    for e in range(16):
        # Every period is a whole number of raster cycles, so groups cannot
        # drift onto each other:
        ratio = np.diff(pulse_onsets(enc, e)) / cycle
        npt.assert_allclose(ratio, np.round(ratio), atol=1e-3)
    # The current limit holds at every instant:
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 4 * 10)
    # A fast train split across too many groups exceeds the stimulator and
    # raises an error:
    with pytest.raises(ValueError, match='no room'):
        FrequencyEncoder(pixel_implant((4, 4), SequentialRaster(6)),
                         freq_range=(0, 300), amp=10,
                         frame_dur=200).encode(img)


def test_PulseEncoder_raster_from_implant():
    # The encoder has no raster of its own and uses the implant's:
    implant = ArgusII()
    implant.raster = SequentialRaster(6)
    vid = VideoStimulus(np.ones((6, 10, 2)), metadata={'fps': 30})
    enc = AmplitudeEncoder(implant, freq=30).encode(vid)
    npt.assert_equal(n_schedules_of(enc), 6)
    delays = [pulse_onsets(enc, e)[0] for e in (0, 10, 20, 30, 40, 50)]
    npt.assert_almost_equal(delays, np.arange(6) * 1000 / 30 / 6, decimal=2)
    # To try another raster, set it on the implant:
    implant.raster = SequentialRaster(2)
    enc = AmplitudeEncoder(implant, freq=30).encode(vid)
    npt.assert_equal(n_schedules_of(enc), 2)
    # Without a raster, every electrode fires at frame onset:
    implant.raster = None
    enc = AmplitudeEncoder(implant, freq=30).encode(vid)
    npt.assert_equal(n_schedules_of(enc), 1)
    # Encoding without an implant uses pixel resolution and no raster, even
    # after encoding for a rastered device:
    bare = AmplitudeEncoder(freq=30).encode(vid)
    npt.assert_equal(n_schedules_of(bare), 1)


def test_PulseEncoder_raster_current_limit():
    # 60 electrodes at 50 uA is 3000 uA if they all fire at once, but only
    # 500 uA if they take turns ten at a time:
    implant = ArgusII(raster=None)
    implant.max_current = 1000
    vid = VideoStimulus(np.ones((6, 10, 3)), metadata={'fps': 30})
    with pytest.raises(ValueError, match='raster'):
        implant.prepare_stim(AmplitudeEncoder(implant, amp_range=(50, 50),
                                              freq=30).encode(vid))
    implant.raster = SequentialRaster(6)
    stim = implant.prepare_stim(
        AmplitudeEncoder(implant, amp_range=(50, 50), freq=30).encode(vid))
    npt.assert_almost_equal(np.abs(stim.data).sum(axis=0).max(), 500)
    # A raster whose groups cannot all fire within a frame is rejected:
    implant.raster = SequentialRaster(6, group_dur=20)
    with pytest.raises(ValueError):
        AmplitudeEncoder(implant, freq=30).encode(vid)
    # So is one whose group turns are too short for a pulse. Sixty 0.92 ms
    # pulses take 55 ms, which does not fit a 33 ms frame, so
    # electrode-at-a-time rastering fails instead of dropping the last
    # electrodes:
    implant.raster = SequentialRaster(60)
    with pytest.raises(ValueError, match='no room'):
        AmplitudeEncoder(implant, freq=30).encode(vid)
    # Halving the phase duration makes it fit:
    enc = AmplitudeEncoder(implant, freq=30, phase_dur=0.2).encode(vid)
    npt.assert_equal(n_schedules_of(enc), 60)
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 50)


@pytest.mark.parametrize('fps', [29.97, 30, 24, 59.94])
@pytest.mark.parametrize('freq', [50, 100])
def test_PulseEncoder_freq_is_actual_freq(fps, freq):
    # The pulse clock is independent of the frame clock, so the requested
    # frequency is delivered at any frame rate, whether or not a frame holds a
    # whole number of periods (e.g., 50 Hz at 29.97 fps):
    vid = VideoStimulus(np.ones((2, 2, 20)), metadata={'fps': fps})
    enc = AmplitudeEncoder(freq=freq).encode(vid)
    onsets = pulse_onsets(enc)
    npt.assert_almost_equal(np.diff(onsets), 1000.0 / freq, decimal=3)
    # Pulses stay whole and charge-balanced even when straddling a frame
    # boundary:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=3)
    npt.assert_equal(np.all(np.diff(enc.time) > 0.95 * DT), True)


def fm_onsets(grays, fps=20, **kwargs):
    """Return pulse onsets (ms) of a one-pixel video with varying gray level"""
    vid = VideoStimulus(np.asarray(grays, dtype=float).reshape(1, 1, -1),
                        metadata={'fps': fps})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        enc = FrequencyEncoder(freq_range=(0, 100), amp=10,
                               **kwargs).encode(vid)
    return pulse_onsets(enc), enc


def test_FrequencyEncoder_rate_changes_between_frames():
    # The rate is piecewise constant over frames, and the pulse clock switches
    # rate at each frame boundary. Scheduling the next pulse a full period
    # ahead would carry the old rate across the boundary (e.g., a 100 Hz frame
    # after a 10 Hz frame would stay silent).
    # 50 ms frames: 10 Hz, then 100 Hz. Half a cycle is accumulated by 50 ms,
    # so the first 100 Hz pulse completes it at 55 ms:
    onsets, enc = fm_onsets([0.1, 1.0])
    npt.assert_almost_equal(onsets, [0, 55, 65, 75, 85, 95], decimal=3)
    # A pulse cut off at a boundary would break charge balance:
    net = trapezoid(enc.data.astype(np.float64),
                    x=enc.time.astype(np.float64))
    npt.assert_almost_equal(net, 0, decimal=3)
    # Fast frame followed by a slow one:
    npt.assert_almost_equal(fm_onsets([1.0, 0.1])[0],
                            [0, 10, 20, 30, 40, 50], decimal=3)
    # Pulse counts follow the per-frame rates. For 50 -> 100 Hz, the 50 Hz
    # frame has accumulated half a period, which the new rate completes 5 ms
    # later (55 ms), not 20 ms later (60 ms) as at the old rate:
    for grays, want in [([1.0, 0.5], [0, 10, 20, 30, 40, 50, 70, 90]),
                        ([0.5, 1.0], [0, 20, 40, 55, 65, 75, 85, 95])]:
        npt.assert_almost_equal(fm_onsets(grays)[0], want, decimal=3)
    # A 0 Hz frame stops the clock without losing a pulse; the next frame
    # resumes at its full rate:
    npt.assert_almost_equal(fm_onsets([1.0, 0.0, 1.0])[0],
                            [0, 10, 20, 30, 40, 100, 110, 120, 130, 140],
                            decimal=3)
    npt.assert_equal(fm_onsets([0.0, 0.0])[0].size, 0)
    npt.assert_almost_equal(fm_onsets([0.0, 1.0])[0],
                            [50, 60, 70, 80, 90], decimal=3)
    # Phase carries through a mid-period rate change: 100, 50, then 25 Hz.
    # The 50 Hz frame accumulates half a period, which the 25 Hz frame
    # completes 20 ms in:
    npt.assert_almost_equal(fm_onsets([1.0, 0.5, 0.25])[0],
                            [0, 10, 20, 30, 40, 50, 70, 90, 120], decimal=3)


def test_PulseEncoder_raster_slots_land_on_the_clock():
    # Pulses can only start on a clock edge. Rounding each group's offset
    # separately could put two groups on the same edge (six groups sharing a
    # 5 ms period on a 1 ms clock have only five edges), so they would pulse
    # together:
    img = ImageStimulus(np.ones((6, 2)))
    implant = pixel_implant((6, 2), SequentialRaster(6))
    with pytest.raises(ValueError, match='clock'):
        AmplitudeEncoder(implant, freq=200, frame_dur=100, clock=1).encode(
            img)
    # With enough room, each group gets a whole number of clock cycles and
    # the turns are evenly spaced:
    enc = AmplitudeEncoder(implant, freq=20, frame_dur=200, clock=1).encode(
        img)
    starts = np.array([pulse_onsets(enc, e)[0] for e in range(0, 12, 2)])
    npt.assert_almost_equal(starts, np.arange(6) * 8.0, decimal=3)
    npt.assert_equal(np.unique(starts).size, 6)
    # The period is unchanged: every electrode runs at the requested 20 Hz,
    # and no two groups coincide.
    for e in range(12):
        npt.assert_almost_equal(np.diff(pulse_onsets(enc, e)), 50, decimal=3)
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 100)


def test_PulseEncoder_raster_short_slot_keeps_the_rate():
    # A short explicit slot packs the groups into the start of each period
    # without changing the rate: with a common period (always the case for
    # amplitude modulation), groups at a fixed offset never meet, so the
    # period is not quantized onto the cycle (which would turn 20 Hz into
    # 18.5 Hz on Argus II with a 1 ms slot).
    img = ImageStimulus(np.ones((2, 2)))
    implant = pixel_implant((2, 2), SequentialRaster(2, interleave=True,
                                                     group_dur=1.5))
    # 10 ms period vs. a 2 x 1.5 = 3 ms cycle: quantizing would round the
    # period up to 12 ms (83 Hz):
    enc = AmplitudeEncoder(implant, freq=100, frame_dur=200).encode(img)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 3)
    for e, offset in enumerate([0, 1.5, 0, 1.5]):
        npt.assert_almost_equal(pulse_onsets(enc, e)[0], offset, decimal=3)
        npt.assert_almost_equal(np.diff(pulse_onsets(enc, e)), 10, decimal=3)
    # ... and the groups still never pulse at the same instant:
    npt.assert_equal(np.intersect1d(np.round(pulse_onsets(enc, 0), 3),
                                    np.round(pulse_onsets(enc, 1), 3)).size, 0)
    npt.assert_almost_equal(np.abs(enc.data).sum(axis=0).max(), 100)
    # Frequency modulation still quantizes, because electrodes with different
    # periods drift onto each other:
    fm = FrequencyEncoder(implant, freq_range=(50, 100), amp=10,
                          frame_dur=200).encode(
                              ImageStimulus(np.array([[1.0, 0.0],
                                                      [1.0, 0.0]])))
    for e in range(4):
        period = np.diff(pulse_onsets(fm, e))
        npt.assert_allclose(period / 3, np.round(period / 3), atol=1e-3)


def test_FrequencyEncoder_rate_changes_with_raster_offset():
    # With slow stimulation, a raster group's first valid onset can fall
    # several frames into the video. The phase accumulator must start in the
    # frame containing that onset, not at frame 0.
    # 20 ms frames; the top of the range is 10 Hz, so the raster cycle is
    # 100 ms and the second of two groups can only pulse at 50, 150, ... ms:
    implant = pixel_implant((2, 2), SequentialRaster(2, interleave=True))
    vid = VideoStimulus(np.tile(np.array([0, 1, 0, 0, 0, 0], dtype=float),
                                (2, 2, 1)).reshape(2, 2, 6),
                        metadata={'fps': 50})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        enc = FrequencyEncoder(implant, freq_range=(0, 10), amp=10).encode(
            vid)
    npt.assert_almost_equal(enc.metadata['encoder']['cycle'], 100)
    # Only the 20-40 ms frame requests stimulation, and neither group has a
    # valid slot in it (group 0: 0 and 100 ms; group 1: 50 and 150 ms, all in
    # 0 Hz frames), so nothing is delivered:
    for e in range(2):
        npt.assert_equal(pulse_onsets(enc, e).size, 0)
    npt.assert_almost_equal(np.abs(enc.data).max(), 0)
    # With a wider window each group gets a valid slot, and both fire, a
    # cycle apart:
    vid = VideoStimulus(np.tile(np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                                         dtype=float),
                                (2, 2, 1)).reshape(2, 2, 10),
                        metadata={'fps': 50})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        enc = FrequencyEncoder(implant, freq_range=(0, 10), amp=10).encode(
            vid)
    npt.assert_almost_equal(pulse_onsets(enc, 0), [0], decimal=3)
    npt.assert_almost_equal(pulse_onsets(enc, 1), [50], decimal=3)
    # No pulse is delivered in a frame that requested silence:
    for e in range(2):
        for t in pulse_onsets(enc, e):
            npt.assert_equal(vid.data[e, int(t // 20)] > 0, True)


def test_PulseEncoder_clock_never_speeds_up():
    # Timing constraints (here the stimulator clock) may lower an electrode's
    # rate, never raise it, since a higher rate delivers more charge.
    # Periods are whole clock cycles, so a 3.33 ms period on a 1 ms clock
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
    # Quantizing onto the raster cycle rounds the period up, since a faster
    # rate would deliver more charge than requested. With a 10 ms cycle,
    # 67 Hz becomes 50 Hz, not 100 Hz.
    grays = np.array([1.0, 0.67, 0.4, 0.2])
    img = ImageStimulus(grays.reshape((2, 2)))
    implant = pixel_implant((2, 2), SequentialRaster(4))
    enc = FrequencyEncoder(implant, freq_range=(0, 100), amp=10,
                           frame_dur=200).encode(img)
    cycle = enc.metadata['encoder']['cycle']
    npt.assert_almost_equal(cycle, 10)
    for e, gray in enumerate(grays):
        period = np.diff(pulse_onsets(enc, e))
        # Whole cycles, and never shorter than the requested period:
        npt.assert_allclose(period / cycle, np.round(period / cycle),
                            atol=1e-3)
        npt.assert_equal(np.all(period >= 1000.0 / (100 * gray) - 1e-3), True)
    # The fastest electrode lands exactly on the cycle, so the top of the
    # range is delivered exactly:
    npt.assert_almost_equal(np.diff(pulse_onsets(enc, 0)), 10, decimal=3)


def test_PulseEncoder_pulse_offset():
    # `Stimulus` time axes need not start at zero. The encoder uses only the
    # pulse shape, so a time-shifted pulse must encode exactly like an
    # unshifted one (same duration for the fit check, no onset delay):
    shape = np.array([[0, -1, -1, 0]], dtype=float)
    at_zero = Stimulus(shape, time=[0, 0.01, 1, 1.01])
    shifted = Stimulus(shape, time=[5, 5.01, 6, 6.01])
    vid = VideoStimulus(np.ones((1, 2, 3)), metadata={'fps': 50})
    ref = AmplitudeEncoder(pulse=at_zero, freq=1000 / 6).encode(vid)
    enc = AmplitudeEncoder(pulse=shifted, freq=1000 / 6).encode(vid)
    npt.assert_almost_equal(enc.time, ref.time)
    npt.assert_almost_equal(enc.data, ref.data)
    # The time axis stays strictly increasing across frames (a late pulse
    # would run past its frame):
    npt.assert_equal(np.all(np.diff(enc.time) > 0.95 * DT), True)
    # The caller's pulse is unchanged:
    npt.assert_almost_equal(shifted.time, [5, 5.01, 6, 6.01])


def test_PulseEncoder_zero_amp():
    # An electrode with zero current has nothing to schedule, so a dark frame
    # adds no pulses and no time points:
    black = ImageStimulus(np.zeros((2, 2)))
    enc = AmplitudeEncoder(amp_range=(0, 50), freq=100,
                           frame_dur=100).encode(black)
    npt.assert_equal(np.all(enc.data == 0), True)
    npt.assert_equal(enc.shape[1], 2)
    npt.assert_almost_equal(enc.time[-1], 100)
    # A dark electrode next to bright ones adds nothing and does not change
    # the bright ones:
    half = ImageStimulus(np.array([[0.0, 1.0], [0.0, 1.0]]))
    enc = AmplitudeEncoder(amp_range=(0, 50), freq=100,
                           frame_dur=100).encode(half)
    npt.assert_equal(np.all(enc.data[[0, 2]] == 0), True)
    npt.assert_equal(n_pulses_of(enc, 1), 10)
    # A dark frame does not reset the phase: the pulse clock keeps running
    # while nothing is delivered. Three 25 ms frames at 200 Hz, middle one
    # black:
    vid = VideoStimulus(np.array([1.0, 0.0, 1.0]).reshape((1, 1, 3)),
                        metadata={'fps': 40})
    enc = AmplitudeEncoder(amp_range=(0, 50), freq=200).encode(vid)
    onsets = pulse_onsets(enc)
    # Nothing is delivered during the black frame:
    npt.assert_equal(np.any((onsets >= 25) & (onsets < 50)), False)
    # ... and the train resumes in phase: every onset is a whole number of
    # 5 ms periods from the first:
    npt.assert_almost_equal(np.mod(onsets, 5), 0, decimal=3)
    npt.assert_almost_equal(onsets[[0, -1]], [0, 70], decimal=3)
    # An infeasible raster is a device property, so it raises an error
    # regardless of video brightness:
    implant = ArgusII(raster=SequentialRaster(60))
    dark = VideoStimulus(np.zeros((6, 10, 3)), metadata={'fps': 30})
    with pytest.raises(ValueError, match='no room'):
        AmplitudeEncoder(implant, freq=30).encode(dark)
    # ... and a feasible one adds nothing for a dark video:
    enc = AmplitudeEncoder(implant, amp_range=(0, 50), freq=30,
                           phase_dur=0.2).encode(dark)
    npt.assert_equal(np.all(enc.data == 0), True)
    npt.assert_equal(enc.shape[1], 2)


def test_PulseEncoder_implant_reshape():
    # Passing an implant samples the source at the electrode locations. Row
    # count cannot tell whether that already happened: a 10x6 image and an
    # RGB 4x5 image both have as many rows as Argus II has electrodes, yet
    # must still be sampled (and, for RGB, converted with `rgb2gray`).
    implant = ArgusII(raster=None)
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


def test_PulseEncoder_spatial_view():
    """_spatial_view returns the amplitude modulation without timing"""
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    implant = pixel_implant((4, 4), SequentialRaster(4, interleave=True))
    params = dict(amp_range=(0, 50), freq=100, frame_dur=100)
    enc = AmplitudeEncoder(implant, **params)
    delivered = enc.encode(img)
    spatial = delivered._spatial_view()
    # Computing it does not render the waveform:
    npt.assert_equal(_rendered(delivered), False)

    # One row per electrode, one column per source frame; an image is one
    # frame, so there is no time axis:
    npt.assert_equal(spatial.shape, (16, 1))
    npt.assert_equal(spatial.time, None)
    npt.assert_equal(list(spatial.electrodes), list(delivered.electrodes))
    npt.assert_equal(spatial.unit, uA)
    # Gray level maps linearly onto the amplitude range:
    npt.assert_almost_equal(spatial.data.ravel(),
                            np.linspace(0, 1, 16) * 50, decimal=4)
    # This equals the peak each electrode reaches in the rendered pulse
    # train:
    npt.assert_almost_equal(np.abs(delivered.data).max(axis=1),
                            spatial.data.ravel(), decimal=4)
    # No timing information is kept (no waveform, pulse clock, or raster),
    # although the delivered train has 4 raster groups and hundreds of time
    # points.
    npt.assert_equal(delivered.time.size > 100, True)
    # Four raster groups, plus the black pixel that delivers nothing:
    npt.assert_equal(n_schedules_of(delivered), 5)
    # One column per frame has no schedule to split electrodes across. The
    # source frame clock is kept:
    npt.assert_equal('cycle' in spatial.metadata['encoder'], False)
    npt.assert_equal(spatial.metadata['encoder']['frame_dur'],
                     delivered.metadata['encoder']['frame_dur'])
    npt.assert_array_equal(spatial.metadata['encoder']['frame_time'],
                           delivered.metadata['encoder']['frame_time'])

    # A video keeps one column per frame and a time axis of frame onsets
    # (`frame_dur=100` re-times the source, as in `encode`):
    vid = VideoStimulus(np.random.default_rng(0).random((4, 4, 5)),
                        metadata={'fps': 20})
    spatial = enc.encode(vid)._spatial_view()
    npt.assert_equal(spatial.shape, (16, 5))
    npt.assert_almost_equal(spatial.time, np.arange(5) * 100.0)

    # Encoding without an implant works the same way, at pixel resolution:
    bare = AmplitudeEncoder(**params).encode(img)._spatial_view()
    npt.assert_equal(bare.shape, (16, 1))
    npt.assert_almost_equal(bare.data.ravel(), np.linspace(0, 1, 16) * 50,
                            decimal=4)
    # ... and a non-picture source is rejected here too:
    with pytest.raises(DimensionMismatchError):
        enc.encode(Stimulus([0.5]))


def test_FrequencyEncoder_spatial_view():
    """_spatial_view of rate coding reduces to on/off at a fixed amplitude"""
    grays = np.array([[0.0, 0.5], [0.75, 1.0]])
    img = ImageStimulus(grays)
    enc = FrequencyEncoder(freq_range=(0, 200), amp=30, frame_dur=100)
    delivered = enc.encode(img)
    spatial = delivered._spatial_view()
    # Every electrode that pulses has the same amplitude, so without a time
    # axis rate collapses to on/off. An electrode at 0 Hz delivers no current:
    npt.assert_almost_equal(spatial.data.ravel(), [0, 30, 30, 30], decimal=4)
    # The delivered train encodes rate as pulse count:
    counts = [n_pulses_of(delivered, e, peak=30) for e in range(4)]
    npt.assert_equal(counts[0], 0)
    npt.assert_equal(np.all(np.diff(counts) > 0), True)


def test_PulseEncoder_metadata():
    enc = AmplitudeEncoder(freq=50).encode(ImageStimulus(np.ones((2, 2))))
    npt.assert_almost_equal(enc.metadata['encoder']['frame_dur'], 500)
    npt.assert_almost_equal(enc.metadata['encoder']['frame_time'], [0])
    # Amplitude modulation uses one schedule for all electrodes; frequency
    # modulation increases the count:
    npt.assert_equal(n_schedules_of(enc), 1)
    fm = FrequencyEncoder(freq_range=(10, 100), n_levels=4,
                          clock=1).encode(ImageStimulus(np.linspace(
                              0, 1, 16).reshape((4, 4))))
    npt.assert_equal(n_schedules_of(fm), 4)


def test_PulseEncoder_degenerate_raster_is_no_raster():
    """A single-group raster leaves the encoding unchanged

    A raster only staggers onsets, so with one group the encoded stimulus is
    bit-for-bit identical to having no raster, even with an explicit
    ``group_dur`` (which would otherwise set the sweep length).
    """
    implant = ArgusII(raster=None)
    img = ImageStimulus(np.random.default_rng(0).random((6, 10)))
    kwargs = dict(amp_range=(0, 50), freq=20, frame_dur=200)
    plain = AmplitudeEncoder(implant, **kwargs).encode(img)
    npt.assert_equal(plain.metadata['encoder']['cycle'], None)

    names = list(implant.electrode_names)
    for raster in (SequentialRaster(1),
                   SequentialRaster(1, group_dur=3.0),
                   SequentialRaster(1, interleave=True),
                   CustomRaster([names]),
                   CustomRaster({n: 0 for n in names})):
        npt.assert_equal(raster.n_groups, 1)
        implant.raster = raster
        got = AmplitudeEncoder(implant, **kwargs).encode(img)
        npt.assert_array_equal(got.data, plain.data)
        npt.assert_array_equal(got.time, plain.time)
        npt.assert_equal(got.metadata['encoder']['cycle'], None)
        # Nothing is staggered, so every electrode fires together and the
        # stimulator sources the whole array at once:
        npt.assert_almost_equal(np.abs(got.data).sum(axis=0).max(),
                                np.abs(plain.data).sum(axis=0).max())


def test_PulseEncoder_degenerate_ranges():
    """A zero-width modulation range makes gray levels irrelevant"""
    implant = ArgusII(raster=None)
    img = ImageStimulus(np.random.default_rng(1).random((6, 10)))
    kwargs = dict(frame_dur=200)

    # A single amplitude for all gray levels gives a constant-amplitude train:
    flat = AmplitudeEncoder(implant, amp_range=(30, 30), freq=20,
                            **kwargs).encode(img)
    npt.assert_almost_equal(np.abs(flat.data).max(axis=1), 30.0, decimal=4)
    # A single frequency for all gray levels gives the same stimulus, since
    # frequency modulation at a constant rate equals amplitude modulation at
    # a constant amplitude:
    same = FrequencyEncoder(implant, freq_range=(20, 20), amp=30,
                            **kwargs).encode(img)
    npt.assert_array_equal(same.data, flat.data)
    npt.assert_array_equal(same.time, flat.time)
    npt.assert_equal(n_schedules_of(same), 1)

    # A black image requests no current at either end of the range:
    black = ImageStimulus(np.zeros((6, 10)))
    for enc in (AmplitudeEncoder(implant, amp_range=(0, 50), **kwargs),
                FrequencyEncoder(implant, freq_range=(0, 200), amp=50,
                                 **kwargs)):
        npt.assert_equal(np.any(enc.encode(black).data), False)


def test_PulseEncoder_n_levels_converges():
    """Quantizing onto enough gray levels matches no quantization"""
    implant = ArgusII()
    img = ImageStimulus(np.random.default_rng(2).random((6, 10)))
    kwargs = dict(amp_range=(0, 50), freq=20, frame_dur=200)
    ref = AmplitudeEncoder(implant, **kwargs).encode(img)
    err = [np.abs(AmplitudeEncoder(implant, n_levels=n, **kwargs).encode(
               img).data - ref.data).max()
           for n in (4, 16, 256, 1 << 16)]
    # Each 4x increase in level count gives ~4x better accuracy; the finest is
    # negligible relative to a 50 uA range:
    npt.assert_equal(np.all(np.diff(err) < 0), True)
    npt.assert_array_less(err[-1], 1e-2)
    # Two levels (the minimum) gives a black-or-white encoding:
    two = AmplitudeEncoder(implant, n_levels=2, **kwargs).encode(img)
    npt.assert_array_equal(np.unique(np.abs(two.data).max(axis=1)), [0.0, 50.0])


def test_PulseEncoder_frame_rate_does_not_move_the_pulses():
    """The pulse clock is independent of the frame clock

    Re-timing the same frames changes only the stimulus duration and the gray
    level each pulse uses, not the pulse onsets. At 29.97 fps a frame is not a
    whole number of 20 Hz periods, so restarting the train every frame would
    deliver a different rate.
    """
    implant = ArgusII()
    vid = VideoStimulus(np.random.default_rng(3).random((6, 10, 4)),
                        metadata={'fps': 10})
    onsets = []
    for frame_dur in (100.0, 50.0, 25.0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stim = AmplitudeEncoder(implant, amp_range=(50, 50), freq=20,
                                    frame_dur=frame_dur).encode(
                                        vid)
        npt.assert_almost_equal(stim.time[-1], 4 * frame_dur)
        neg = stim.data[0] < 0
        onsets.append(stim.time[neg & ~np.concatenate(([False], neg[:-1]))])
    # Every onset of the shorter stimulus also occurs in the longer one:
    for short in onsets[1:]:
        npt.assert_almost_equal(short, onsets[0][:short.size])
    # ... and 20 Hz is delivered at any frame rate:
    npt.assert_almost_equal(np.diff(onsets[0]), 50.0, decimal=6)


def test_AmplitudeEncoder_units():
    """Mixed unit spellings encode to numerically identical stimuli"""
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    bare = AmplitudeEncoder(amp_range=(50, 100), freq=20, phase_dur=0.46,
                            interphase_dur=0.1, clock=1, frame_dur=50)
    unitful = AmplitudeEncoder(amp_range=(50 * uA, 0.1 * mA), freq=0.02 * kHz,
                               phase_dur=460 * us, interphase_dur=0.1 * ms,
                               clock=1000 * us, frame_dur=0.05 * sec)
    # The encoder stores plain numbers in its default units:
    npt.assert_almost_equal(np.asarray(unitful.amp_range), [50, 100])
    npt.assert_almost_equal(unitful.freq, 20)
    npt.assert_almost_equal(unitful.phase_dur, 0.46)
    npt.assert_almost_equal(unitful.interphase_dur, 0.1)
    npt.assert_almost_equal(unitful.clock, 1)
    npt.assert_almost_equal(unitful.frame_dur, 50)
    for value in (*unitful.amp_range, unitful.freq, unitful.phase_dur,
                  unitful.interphase_dur, unitful.clock, unitful.frame_dur):
        npt.assert_equal(isinstance(value, Quantity), False)
    # ... and encodes identically:
    out_bare, out_unitful = bare.encode(img), unitful.encode(img)
    npt.assert_array_equal(out_bare.data, out_unitful.data)
    npt.assert_array_equal(out_bare.time, out_unitful.time)
    # The output is electrical, whatever the input units:
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
    """Encoders accept only dimensionless (gray-level) sources"""
    enc = AmplitudeEncoder(amp_range=(0, 50))
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    vid = VideoStimulus(np.ones((2, 2, 3)) * 0.5, time=[0, 20, 40])
    # Images and videos are accepted:
    for source in (img, vid):
        npt.assert_equal(enc.encode(source).unit, uA)
    # A picture sampled at an implant's electrodes is still dimensionless:
    implant = ArgusII()
    sampled = implant.reshape_stim(img)
    npt.assert_equal(sampled.unit, dimensionless)
    npt.assert_equal(enc.encode(sampled).unit, uA)
    # ... and the encoder's implant path gives the same result:
    npt.assert_equal(AmplitudeEncoder(implant, amp_range=(0, 50)).encode(
        img).unit, uA)
    # Electrical stimuli are rejected: `Stimulus([0.5])` is 0.5 uA and would
    # otherwise be clipped and re-modulated as a gray level.
    with pytest.raises(DimensionMismatchError) as excinfo:
        enc.encode(Stimulus([0.5]))
    npt.assert_equal("must be dimensionless" in str(excinfo.value), True)
    for source in (Stimulus([0.5]), BiphasicPulseTrain(20, 50, 0.45),
                   Stimulus(np.ones((2, 2)), time=[0, 1])):
        with pytest.raises(DimensionMismatchError):
            enc.encode(source)


def test_encoder_pulse_template_unit_agnostic():
    """A custom ``pulse`` template contributes its shape, not its unit"""
    img = ImageStimulus(np.linspace(0, 1, 4).reshape((2, 2)))
    shape = np.array([[0, 1, 1, 0, -1, -1, 0]], dtype=float)
    time = [0, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0]
    electrical = Stimulus(shape * 37.0, time=time)
    dimless = Stimulus(VideoStimulus(shape.reshape((1, 1, -1)), time=time))
    npt.assert_equal(electrical.unit, uA)
    npt.assert_equal(dimless.unit, dimensionless)
    # Both templates give the same encoding: the amplitude is normalized away,
    # so only the shape matters.
    outs = [AmplitudeEncoder(pulse=p, amp_range=(0, 50)).encode(img)
            for p in (electrical, dimless)]
    npt.assert_array_equal(outs[0].data, outs[1].data)
    npt.assert_array_equal(outs[0].time, outs[1].time)
    for out in outs:
        npt.assert_equal(out.unit, uA)
        npt.assert_almost_equal(np.abs(out.data).max(), 50)


def _rendered(stim):
    """Return True if the encoded stimulus has rendered its waveform"""
    return stim._Stimulus__stim['data'] is not None


# One case per schedule feature: a still frame, several frames,
# per-electrode frequency, a device raster, a stimulator clock, a custom
# pulse, and an all-zero source.
ENCODED = [
    ('amplitude', lambda: AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))),
    ('amplitude-video', lambda: AmplitudeEncoder().encode(
        VideoStimulus(np.linspace(0, 1, 192).reshape(8, 8, 3),
                      time=np.arange(3) * 40.0))),
    ('frequency', lambda: FrequencyEncoder(freq_range=(0, 60)).encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))),
    ('rastered', lambda: AmplitudeEncoder(ArgusII()).encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))),
    ('clocked', lambda: AmplitudeEncoder(clock=1.0).encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))),
    ('custom-pulse', lambda: AmplitudeEncoder(
        pulse=BiphasicPulse(1, 0.2, interphase_dur=0.1)).encode(
            ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))),
    ('all-zero', lambda: FrequencyEncoder(freq_range=(0, 60)).encode(
        ImageStimulus(np.zeros((6, 6))))),
]


def test_electrical_source_clock_follows_the_source_time_axis():
    frames = np.linspace(0, 1, 192).reshape(8, 8, 3)
    # One timed frame still has a source clock:
    one = AmplitudeEncoder().encode(VideoStimulus(frames[..., :1],
                                                  time=[0]))
    npt.assert_array_equal(one.metadata['encoder']['source_frame_time'], [0])
    video = AmplitudeEncoder().encode(
        VideoStimulus(frames, time=np.arange(3) * 40.0))
    npt.assert_array_equal(video.metadata['encoder']['source_frame_time'],
                           [0, 40, 80])
    npt.assert_array_equal(
        video._spatial_view().metadata['encoder']['source_frame_time'],
        [0, 40, 80])
    npt.assert_array_equal(
        (video * 0.5).metadata['encoder']['source_frame_time'], [0, 40, 80])
    # An image has no source clock; `frame_dur` retimes a video:
    still = AmplitudeEncoder().encode(ImageStimulus(frames[..., 0]))
    retimed = AmplitudeEncoder(frame_dur=50).encode(
        VideoStimulus(frames, time=np.arange(3) * 40.0))
    for stim in (still, retimed):
        npt.assert_equal('source_frame_time' in stim.metadata['encoder'],
                         False)


@pytest.mark.parametrize('name, build', ENCODED, ids=[c[0] for c in ENCODED])
def test_encoded_stimulus_defers_only_the_waveform(name, build):
    stim = build()
    npt.assert_equal(_rendered(stim), False)
    # Schedule properties do not require rendering.
    npt.assert_equal(len(stim.electrodes) > 0, True)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(stim.time_unit, ms)
    # A video also records its source clock (here equal to the frame clock).
    video = [] if 'video' not in name else ['source_frame_dur',
                                            'source_frame_time']
    npt.assert_equal(sorted(stim.metadata['encoder']),
                     ['cycle', 'frame_dur', 'frame_time'] + video)
    npt.assert_equal(stim.duration > 0, True)
    repr(stim)
    copies = [copy(stim), deepcopy(stim)]
    npt.assert_equal(_rendered(stim), False)
    for copied in copies:
        npt.assert_equal(_rendered(copied), False)
    # The first read of the waveform renders it, once:
    data = stim.data
    npt.assert_equal(_rendered(stim), True)
    for _ in range(3):
        npt.assert_equal(np.shares_memory(stim.data, data), True)
    npt.assert_equal(stim.data.dtype, np.float32)
    npt.assert_equal(stim.time.dtype, np.float64)
    npt.assert_almost_equal(stim.time[-1], stim.duration)


@pytest.mark.parametrize('name, build', ENCODED, ids=[c[0] for c in ENCODED])
def test_encoded_stimulus_holds_no_waveform_sized_array(name, build):
    # Stored state may scale with electrodes x frames, with pulse onsets, or
    # with the global time axis, but not with electrodes x time (the large
    # array).
    stim = build()
    n_el, n_time = len(stim.electrodes), stim._ticks.size
    retained = [stim._amp, stim._ticks, stim._sched, stim._pulse_ticks,
                stim._pulse_vals, *stim._onsets, *stim._frames]
    for array in retained:
        npt.assert_equal(array.shape == (n_el, n_time), False)
        # Immutable, like the rest of a stimulus' state:
        npt.assert_equal(array.flags.writeable, False)
    npt.assert_equal(stim._amp.shape[0], n_el)
    # The matrix is built only on request:
    npt.assert_equal(stim.data.shape, (n_el, n_time))


def test_encoded_stimulus_is_independent_of_the_encoder():
    # The schedule is resolved at `encode`. Later changes to the encoder or
    # its pulse template do not affect it:
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
    # A schedule stores when each electrode pulses and how hard per frame. A
    # DC offset or a time shift invalidates it, so the result is a plain
    # stimulus:
    stim = AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))
    with np.errstate(divide='ignore', invalid='ignore'):
        out = modify(stim)
    npt.assert_equal(type(out), Stimulus)
    npt.assert_equal(out._is_parametric, False)


@pytest.mark.parametrize('factor', [2, 0.5, -1, 1, 0])
def test_encoded_stimulus_scaling_scales_both_descriptions(factor):
    # Scaling changes amplitude, not timing, so the schedule is kept; the
    # waveform and modulation scale together, so spatial and spatiotemporal
    # models see the same change.
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
    # The frame clock describes the source and is not scaled:
    npt.assert_equal(scaled.metadata['encoder'], stim.metadata['encoder'])
    npt.assert_allclose(view.data, stim._spatial_view().data)


def test_encoded_stimulus_drops_electrodes_structurally():
    # Deactivating an electrode must keep the modulation view, which is the
    # only input a spatial model uses.
    stim = AmplitudeEncoder().encode(
        ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8)))
    fewer = stim._without_electrodes([0, 3])
    npt.assert_equal(type(fewer), type(stim))
    npt.assert_equal(_rendered(fewer), False)
    npt.assert_equal(len(fewer.electrodes), 62)
    kept_rows = [i for i in range(len(stim.electrodes)) if i not in (0, 3)]
    npt.assert_array_equal(np.asarray(fewer.electrodes),
                           np.asarray(stim.electrodes)[kept_rows])
    # The same rows are removed from both:
    npt.assert_array_equal(fewer._spatial_view().data,
                           stim._spatial_view().data[kept_rows])
    npt.assert_array_equal(fewer.data, stim.data[kept_rows])
    npt.assert_array_equal(fewer.time, stim.time)


def test_encoded_stimulus_validates_while_it_schedules():
    # Scheduling is eager, so encoding errors appear at `encode`, not when the
    # waveform is read:
    img = ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8))
    with pytest.raises(ValueError):
        # A pulse longer than the source:
        AmplitudeEncoder(phase_dur=400).encode(img)
    with pytest.raises(ValueError):
        # A raster that does not fit into one pulse period:
        FrequencyEncoder(ArgusII(), freq_range=(0, 300)).encode(img)
    with pytest.raises(ValueError):
        # A pulse whose time points DT cannot resolve:
        AmplitudeEncoder(pulse=Stimulus([[0, 1, 0]],
                                        time=[0, 1e-5, 2e-5])).encode(img)
    # ... and the warning about frames that never reach an electrode:
    assert_warns_msg(UserWarning,
                     lambda: AmplitudeEncoder(freq=1).encode(
                         VideoStimulus(np.ones((4, 4, 8)),
                                       time=np.arange(8) * 40.0)),
                     'deliver no pulse at all')


def test_encoded_stimulus_survives_preparation_unrendered():
    # Preparation returns a copy, which does not render the waveform either.
    implant = ArgusII()
    img = ImageStimulus(np.linspace(0, 1, 64).reshape(8, 8))
    stim = implant.prepare_stim(AmplitudeEncoder(implant).encode(img))
    npt.assert_equal(_rendered(stim), False)
    npt.assert_equal(len(stim.electrodes), 60)
    # Presenting the picture goes through the implant's encoder and is not
    # rendered either:
    stim = ArgusII(encoder=AmplitudeEncoder()).prepare_stim(img)
    npt.assert_equal(_rendered(stim), False)
    npt.assert_equal(stim.data.shape[0], 60)


# -----------------------------------------------------------------------------
# PhotovoltaicEncoder
# -----------------------------------------------------------------------------

def test_PhotovoltaicEncoder():
    encoder = PhotovoltaicEncoder(irradiance=4, freq=40, pulse_dur=4,
                                  wavelength=915)
    npt.assert_almost_equal(encoder.irradiance, 4)
    npt.assert_almost_equal(encoder.freq, 40)
    npt.assert_almost_equal(encoder.pulse_dur, 4)
    npt.assert_almost_equal(encoder.wavelength, 915)
    npt.assert_almost_equal(encoder.period, 25)
    # Normalized drive is referenced to a fully lit pixel at these settings.
    npt.assert_almost_equal(encoder.ref_drive, 4 * 4 * 40 / 1000)
    npt.assert_equal(encoder.grayscale, True)
    npt.assert_equal(isinstance(encoder, Encoder), True)
    npt.assert_equal(isinstance(encoder, PulseEncoder), False)
    npt.assert_equal('PhotovoltaicEncoder' in str(encoder), True)
    npt.assert_equal('wavelength' in str(encoder), True)

    # Unitful and bare parameters are equivalent.
    unitful = PhotovoltaicEncoder(irradiance=4000 * W / m ** 2,
                                  freq=0.04 * kHz, pulse_dur=4000 * us,
                                  wavelength=915 * nm)
    npt.assert_almost_equal(unitful.irradiance, 4)
    npt.assert_almost_equal(unitful.freq, 40)
    npt.assert_almost_equal(unitful.pulse_dur, 4)
    npt.assert_almost_equal(unitful.wavelength, 915)


@pytest.mark.parametrize('kwargs', [
    {'irradiance': 0}, {'irradiance': -1}, {'irradiance': np.inf},
    {'freq': 0}, {'freq': -40}, {'freq': np.nan},
    {'pulse_dur': -4}, {'pulse_dur': np.inf},
    {'pulse_dur': 30},        # does not fit into a 25 ms period
    {'wavelength': 0}, {'wavelength': -915},
    {'threshold': -0.1}, {'threshold': 1.5},
])
def test_PhotovoltaicEncoder_rejects(kwargs):
    settings = {'irradiance': 4, 'freq': 40, 'pulse_dur': 4,
                'wavelength': 915}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        PhotovoltaicEncoder(**settings)


def test_PhotovoltaicEncoder_needs_a_wavelength():
    """wavelength has no default; each device sets its own"""
    with pytest.raises(TypeError):
        PhotovoltaicEncoder(irradiance=4, freq=40, pulse_dur=4)


@pytest.mark.parametrize('pulse_dur, freq', [(4, 40), (10, 2)])
def test_PhotovoltaicEncoder_is_not_bound_to_the_PRIMA_grid(pulse_dur, freq):
    """PhotovoltaicEncoder accepts durations off the 0.7 ms PRIMA grid"""
    with pytest.raises(ValueError):
        PRIMAEncoder(pulse_dur=pulse_dur, freq=freq)
    encoder = PhotovoltaicEncoder(PRIMAPivotal(), irradiance=4, freq=freq,
                                  pulse_dur=pulse_dur, wavelength=915)
    stim = encoder.encode(ImageStimulus(np.ones((16, 16))))
    npt.assert_almost_equal(stim.pulse_dur.max(), pulse_dur)
    npt.assert_almost_equal(on_intervals(stim)[0], pulse_dur, decimal=6)


def test_PhotovoltaicEncoder_grayscale_is_continuous():
    """Gray levels scale ON duration linearly, without quantization"""
    implant = PRIMAPivotal()
    ramp = np.tile(np.linspace(0, 1, 64), (64, 1))
    encoder = PhotovoltaicEncoder(implant, irradiance=4, freq=40, pulse_dur=4,
                                  wavelength=915)
    stim = encoder.encode(ImageStimulus(ramp))
    gray = implant.reshape_stim(ImageStimulus(ramp)).data
    npt.assert_almost_equal(stim.pulse_dur[:, :1], gray * 4, decimal=6)
    # More distinct durations than PRIMA's 14 duration levels allow:
    npt.assert_equal(np.unique(np.round(stim.pulse_dur, 6)).size > 14, True)
    # Gray level changes duration, not peak irradiance.
    lit = stim.data[stim.data > 0]
    npt.assert_almost_equal(np.unique(np.round(lit, 6)), np.array([4.0]))

    # Binary mode lights a pixel for the full duration or not at all.
    binary = PhotovoltaicEncoder(implant, irradiance=4, freq=40, pulse_dur=4,
                                 wavelength=915, grayscale=False,
                                 threshold=0.5)
    dur = binary.encode(ImageStimulus(ramp)).pulse_dur
    npt.assert_almost_equal(np.unique(np.round(dur, 6)), np.array([0.0, 4.0]))


def test_PhotovoltaicEncoder_spatial_view():
    implant = PRIMAPivotal()
    ramp = np.tile(np.linspace(0, 1, 64), (64, 1))
    encoder = PhotovoltaicEncoder(implant, irradiance=4, freq=40, pulse_dur=4,
                                  wavelength=915)
    view = encoder.encode(ImageStimulus(ramp))._spatial_view()
    npt.assert_equal(view.unit, dimensionless)
    npt.assert_almost_equal(view.data.min(), 0)
    npt.assert_almost_equal(view.data.max(), 1, decimal=6)
    # 1.0 is a fully lit pixel at these settings, so halving the gray level
    # halves the drive, but halving the irradiance does not change the scale.
    half = encoder.encode(ImageStimulus(np.full((8, 8), 0.5)))._spatial_view()
    npt.assert_almost_equal(half.data.max(), 0.5, decimal=6)
    dim = PhotovoltaicEncoder(implant, irradiance=2, freq=40, pulse_dur=4,
                              wavelength=915).encode(
        ImageStimulus(np.ones((8, 8))))._spatial_view()
    npt.assert_almost_equal(dim.data.max(), 1, decimal=6)


# -----------------------------------------------------------------------------
# PRIMAEncoder
# -----------------------------------------------------------------------------

def on_intervals(stim, electrode=0):
    """Return durations (ms) of a pixel's intervals at peak irradiance"""
    row, time = stim.data[electrode], stim.time
    lit = row >= 0.99 * row.max() if row.max() > 0 else row > np.inf
    edges = np.diff(np.concatenate(([0], lit.astype(int), [0])))
    starts, stops = np.flatnonzero(edges > 0), np.flatnonzero(edges < 0) - 1
    # Include one-DT rise and fall edges.
    return [time[b] - time[a] + 2 * DT for a, b in zip(starts, stops)]


def test_PRIMAEncoder():
    encoder = PRIMAEncoder()
    npt.assert_almost_equal(encoder.irradiance, 3.5)
    npt.assert_almost_equal(encoder.freq, 30)
    npt.assert_almost_equal(encoder.pulse_dur, 9.8)
    # Grayscale PWM is the default; threshold applies only in binary mode.
    npt.assert_equal(encoder.grayscale, True)
    npt.assert_almost_equal(encoder.threshold, 0.5)
    npt.assert_almost_equal(encoder.period, 1000 / 30)
    npt.assert_almost_equal(encoder.wavelength, 880)
    npt.assert_equal(encoder.n_levels, 14)
    npt.assert_equal(isinstance(encoder, Encoder), True)
    # PRIMAEncoder is not an electrical PulseEncoder.
    npt.assert_equal(isinstance(encoder, PulseEncoder), False)
    npt.assert_equal('PRIMAEncoder' in str(encoder), True)

    # Unitful and bare parameters are equivalent.
    unitful = PRIMAEncoder(irradiance=3500 * W / m ** 2, freq=0.03 * kHz,
                           pulse_dur=9800 * us)
    npt.assert_almost_equal(unitful.irradiance, 3.5)
    npt.assert_almost_equal(unitful.freq, 30)
    npt.assert_almost_equal(unitful.pulse_dur, 9.8)


@pytest.mark.parametrize('kwargs', [
    {'irradiance': 0}, {'irradiance': -1}, {'irradiance': np.inf},
    {'freq': 0}, {'freq': -30}, {'freq': np.nan},
    {'pulse_dur': -0.7}, {'pulse_dur': np.inf},
    {'pulse_dur': 1.0},        # off the 0.7 ms hardware grid
    {'pulse_dur': 10.5},       # a grid step, but past the documented maximum
    {'freq': 200},             # a 9.8 ms pulse does not fit a 5 ms period
    {'threshold': -0.1}, {'threshold': 1.5},
])
def test_PRIMAEncoder_rejects(kwargs):
    with pytest.raises(ValueError):
        PRIMAEncoder(**kwargs)


def test_PRIMAEncoder_takes_a_picture():
    implant = PRIMAPivotal()
    with pytest.raises(DimensionMismatchError):
        PRIMAEncoder(implant).encode(Stimulus([[1, 0]] * uA, time=[0, 10]))
    with pytest.raises(TypeError):
        PRIMAEncoder(implant).encode(np.ones((4, 4)))
    # RGB input is converted to gray during implant sampling.
    rgb = ImageStimulus(np.ones((8, 8, 3)))
    npt.assert_equal(PRIMAEncoder(implant).encode(rgb).shape[0], 378)


@pytest.mark.parametrize('grayscale', [True, False])
@pytest.mark.parametrize('gray, n_lit', [(1.0, 378), (0.0, 0)])
def test_PRIMAEncoder_extremes(gray, n_lit, grayscale):
    # Black is off; white uses the maximum duration.
    implant = PRIMAPivotal()
    stim = PRIMAEncoder(implant, grayscale=grayscale).encode(
        ImageStimulus(np.full((16, 16), gray)))
    npt.assert_equal(stim.shape[0], 378)
    npt.assert_equal(np.count_nonzero(stim.data.max(axis=1)), n_lit)
    # No raster limits simultaneous illumination.
    npt.assert_equal(implant.raster, None)


@pytest.mark.parametrize('threshold', [0.25, 0.5, 0.75])
def test_PRIMAEncoder_threshold(threshold):
    # Binary mode lights pixels at or above threshold.
    ramp = np.tile(np.linspace(0, 1, 32), (32, 1))
    implant = PRIMAPivotal()
    encoder = PRIMAEncoder(implant, grayscale=False, threshold=threshold)
    gray = implant.reshape_stim(ImageStimulus(ramp)).data.ravel()
    stim = encoder.encode(ImageStimulus(ramp))
    dur = stim.pulse_dur
    # Static images repeat across projector periods.
    npt.assert_equal(dur.shape, (378, 15))
    npt.assert_equal(np.all(dur == dur[:, :1]), True)
    dur = dur[:, 0]
    npt.assert_array_equal(dur > 0, gray >= threshold)
    # Binary mode uses only 0 and the full pulse duration.
    npt.assert_array_equal(np.unique(dur), np.array([0.0, 9.8]))


def test_PRIMAEncoder_optical_waveform():
    implant = PRIMAPivotal()
    stim = PRIMAEncoder(implant).encode(ImageStimulus(np.ones((16, 16))))
    # Output is optical irradiance.
    npt.assert_equal(stim.unit, mW / mm ** 2)
    npt.assert_equal(stim.unit.dimension, (W / m ** 2).dimension)
    npt.assert_equal(stim.time_unit, ms)
    npt.assert_almost_equal(stim.data.max(), 3.5)
    npt.assert_almost_equal(stim.data.min(), 0)
    # Static images use the standard presentation duration.
    npt.assert_almost_equal(stim.duration, 500)
    intervals = on_intervals(stim)
    npt.assert_equal(len(intervals), 15)
    npt.assert_almost_equal(intervals, 9.8, decimal=6)
    # 30 Hz: the pulses are one projector period apart.
    onsets = stim.time[np.flatnonzero(np.diff(stim.data[0]) > 0)]
    npt.assert_almost_equal(np.diff(onsets), 1000 / 30, decimal=3)
    npt.assert_almost_equal(stim.metadata['encoder']['frame_dur'], 500)


def test_PRIMAEncoder_irradiance_is_not_modulated():
    implant = PRIMAPivotal()
    ramp = np.tile(np.linspace(0, 1, 32), (32, 1))
    stim = PRIMAEncoder(implant, grayscale=True, irradiance=2.0).encode(
        ImageStimulus(ramp))
    lit = stim.data[stim.data > 0]
    # Gray level changes duration, not peak irradiance.
    npt.assert_almost_equal(np.unique(np.round(lit, 6)), np.array([2.0]))


def test_PRIMAEncoder_grayscale():
    implant = PRIMAPivotal()
    ramp = np.tile(np.linspace(0, 1, 64), (64, 1))
    stim = PRIMAEncoder(implant, grayscale=True).encode(ImageStimulus(ramp))
    dur = stim.pulse_dur[:, :1]
    # Durations lie on the 0.7 ms hardware grid.
    npt.assert_almost_equal(dur / 0.7, np.round(dur / 0.7))
    npt.assert_almost_equal(dur.min(), 0)
    npt.assert_almost_equal(dur.max(), 9.8)
    npt.assert_equal(np.unique(np.round(dur, 6)).size > 2, True)
    npt.assert_equal(stim.grayscale, True)
    # Gray levels map linearly to duration levels.
    gray = implant.reshape_stim(ImageStimulus(ramp)).data
    npt.assert_almost_equal(dur, np.round(gray * 14) * 0.7, decimal=6)

    # pulse_dur limits the available duration levels.
    coarse = PRIMAEncoder(implant, grayscale=True, pulse_dur=2.1).encode(
        ImageStimulus(ramp))
    levels = np.unique(coarse.pulse_dur)
    npt.assert_almost_equal(levels, np.array([0.0, 0.7, 1.4, 2.1]))


def test_PRIMAEncoder_records_its_settings():
    stim = PRIMAEncoder(PRIMAPivotal(), grayscale=True).encode(
        ImageStimulus(np.ones((8, 8))))
    # Projector settings are available without rendering.
    npt.assert_almost_equal(stim.wavelength, 880)
    npt.assert_almost_equal(stim.irradiance, 3.5)
    npt.assert_almost_equal(stim.freq, 30)
    npt.assert_almost_equal(stim.pulse_dur.max(), 9.8)
    npt.assert_almost_equal(stim.duty_cycle.max(), 0.294)
    npt.assert_equal(stim.grayscale, True)
    npt.assert_equal(sorted(stim.metadata['encoder']),
                     ['frame_dur', 'frame_time'])


def test_PRIMAEncoder_spatial_view():
    implant = PRIMAPivotal()
    ramp = np.tile(np.linspace(0, 1, 64), (64, 1))
    view = PRIMAEncoder(implant, grayscale=True).encode(
        ImageStimulus(ramp))._spatial_view()
    # Normalized drive ranges from 0 to 1.
    npt.assert_equal(view.unit, dimensionless)
    npt.assert_equal(view.time, None)
    npt.assert_equal(view.shape, (378, 1))
    npt.assert_almost_equal(view.data.min(), 0)
    npt.assert_almost_equal(view.data.max(), 1, decimal=6)
    # Drive is proportional to ON duration at default settings.
    dur = PRIMAEncoder(implant, grayscale=True).encode(
        ImageStimulus(ramp)).pulse_dur[:, 0]
    npt.assert_almost_equal(view.data.ravel(), dur / 9.8, decimal=6)

    # Drive scales with irradiance, frequency, and pulse duration.
    half = PRIMAEncoder(implant, irradiance=1.75).encode(
        ImageStimulus(np.ones((8, 8))))._spatial_view()
    npt.assert_almost_equal(half.data.max(), 0.5, decimal=6)
    half = PRIMAEncoder(implant, freq=15).encode(
        ImageStimulus(np.ones((8, 8))))._spatial_view()
    npt.assert_almost_equal(half.data.max(), 0.5, decimal=6)
    half = PRIMAEncoder(implant, pulse_dur=4.9).encode(
        ImageStimulus(np.ones((8, 8))))._spatial_view()
    npt.assert_almost_equal(half.data.max(), 0.5, decimal=6)


def test_PRIMAEncoder_video():
    implant = PRIMAPivotal()
    # Three 40 ms source frames.
    frames = np.stack([np.ones((8, 8)), np.zeros((8, 8)), np.ones((8, 8))],
                      axis=-1)
    video = VideoStimulus(frames, time=np.arange(3) * 40.0)
    stim = PRIMAEncoder(implant).encode(video)
    # A 30 Hz projector samples the 120 ms source four times.
    npt.assert_almost_equal(stim.duration, 120)
    npt.assert_almost_equal(stim.metadata['encoder']['frame_dur'], 1000 / 30)
    npt.assert_equal(stim.pulse_dur.shape, (378, 4))
    npt.assert_array_almost_equal(stim.pulse_dur.max(axis=0),
                                  [9.8, 9.8, 0, 9.8])
    view = stim._spatial_view()
    npt.assert_equal(view.shape, (378, 4))
    npt.assert_almost_equal(view.data.max(axis=0), [1, 1, 0, 1], decimal=6)
    npt.assert_almost_equal(view.time, np.arange(4) * (1000 / 30), decimal=3)


@pytest.mark.parametrize('fps, n_source', [(15, 15), (30, 30), (60, 60)])
def test_PRIMAEncoder_samples_a_video_without_retiming_it(fps, n_source):
    # Source frame rate does not change the 30 Hz projector clock.
    implant = PRIMAPivotal()
    gray = np.zeros((8, 8, n_source))
    gray[..., ::max(1, n_source // 5)] = 1.0
    video = VideoStimulus(gray, time=np.arange(n_source) * (1000.0 / fps))
    stim = PRIMAEncoder(implant).encode(video)
    npt.assert_almost_equal(stim.duration, 1000)
    npt.assert_equal(stim.pulse_dur.shape[1], 30)
    npt.assert_almost_equal(np.diff(stim._spatial_view().time), 1000 / 30,
                            decimal=3)


@pytest.mark.parametrize('fps', [15, 60])
def test_PRIMAEncoder_repeats_slow_frames_and_skips_fast_ones(fps):
    implant = PRIMAPivotal()
    gray = np.zeros((8, 8, fps))
    gray[..., ::2] = 1.0
    video = VideoStimulus(gray, time=np.arange(fps) * (1000.0 / fps))
    stim = PRIMAEncoder(implant).encode(video)
    lit = stim.pulse_dur.max(axis=0) > 0
    npt.assert_equal(lit.size, 30)
    if fps == 15:
        # At 15 fps, each source frame is sampled twice.
        npt.assert_array_equal(lit.reshape(-1, 2)[:, 0],
                               lit.reshape(-1, 2)[:, 1])
    else:
        npt.assert_equal(lit.all(), True)


@pytest.mark.parametrize('fps', [15, 29.97, 60])
def test_PRIMAEncoder_keeps_the_source_clock_apart(fps):
    video = VideoStimulus(np.ones((8, 8, 6)),
                          time=np.arange(6) * (1000 / fps))
    stim = PRIMAEncoder(PRIMAPivotal()).encode(video)
    meta = stim.metadata['encoder']
    # The projector clock is unchanged:
    npt.assert_allclose(meta['frame_time'], stim.pulse_time, atol=DT)
    npt.assert_almost_equal(meta['frame_dur'], 1000 / 30)
    # The source clock is recorded next to it:
    npt.assert_array_equal(meta['source_frame_time'], video.time)
    npt.assert_almost_equal(meta['source_frame_dur'], 1000 / fps)
    for other in (stim * 0.5, stim._spatial_view()):
        npt.assert_array_equal(
            other.metadata['encoder']['source_frame_time'], video.time)


def test_PRIMAEncoder_starts_where_the_source_does():
    implant = PRIMAPivotal()
    video = VideoStimulus(np.ones((8, 8, 3)),
                          time=100 + np.arange(3) * (1000 / 30))
    stim = PRIMAEncoder(implant).encode(video)
    npt.assert_almost_equal(stim.duration, 200, decimal=6)
    npt.assert_equal(stim.pulse_dur.shape[1], 3)
    npt.assert_almost_equal(stim._spatial_view().time[0], 100, decimal=6)
    # Dark until the source starts:
    npt.assert_almost_equal(stim.data[:, stim.time < 100].max(), 0)

    gray = np.zeros((8, 8, 9))
    gray[..., 3:6] = 1.0
    cropped = VideoStimulus(gray, time=np.arange(9) * (1000 / 30)).crop(
        front=3, back=3, bottom=1, right=1)
    npt.assert_almost_equal(cropped.time[0], 100, decimal=6)
    stim = PRIMAEncoder(implant).encode(cropped)
    npt.assert_almost_equal(stim._spatial_view().time[0], 100, decimal=6)
    npt.assert_almost_equal(stim.data[:, stim.time < 100].max(), 0)


def test_PRIMAEncoder_finishes_every_pulse_it_starts():
    # A 9.8 ms pulse does not fit in the final 6.7 ms of this source.
    implant = PRIMAPivotal()
    video = VideoStimulus(np.ones((8, 8, 2)), time=[0.0, 20.0])
    stim = PRIMAEncoder(implant).encode(video)
    npt.assert_almost_equal(stim.duration, 40)
    npt.assert_array_almost_equal(stim.pulse_dur.max(axis=0), [9.8, 0])
    npt.assert_almost_equal(stim.data[:, -1].max(), 0)
    kept = stim.pulse_dur[stim.pulse_dur > 0]
    npt.assert_almost_equal(kept / 0.7, np.round(kept / 0.7))


def test_PRIMAEncoder_defers_the_waveform():
    implant = PRIMAPivotal()
    stim = PRIMAEncoder(implant).encode(ImageStimulus(np.ones((16, 16))))
    npt.assert_equal(_rendered(stim), False)
    npt.assert_almost_equal(stim.duration, 500)
    npt.assert_equal(stim.unit, mW / mm ** 2)
    npt.assert_equal(len(stim.electrodes), 378)
    npt.assert_almost_equal(stim.pulse_dur.max(), 9.8)
    npt.assert_equal(_rendered(stim._spatial_view()), True)
    npt.assert_equal(_rendered(stim), False)
    npt.assert_equal(_rendered(implant.prepare_stim(stim)), False)
    implant.deactivate('A5')
    prepared = implant.prepare_stim(ImageStimulus(np.ones((16, 16))))
    npt.assert_equal(_rendered(prepared), False)
    npt.assert_equal(len(prepared.electrodes), 377)
    npt.assert_equal(stim.data.shape, (378, stim.time.size))
    npt.assert_equal(_rendered(stim), True)


def test_PRIMAEncoder_scales_the_power():
    stim = PRIMAEncoder(PRIMAPivotal()).encode(ImageStimulus(np.ones((8, 8))))
    scaled = stim * 0.5
    npt.assert_equal(_rendered(scaled), False)
    npt.assert_almost_equal(scaled.irradiance, 1.75)
    npt.assert_array_almost_equal(scaled.pulse_dur, stim.pulse_dur)
    npt.assert_almost_equal(scaled.data.max(), 1.75)


def test_AmplitudeEncoder_amp_range_unit():
    for amp_range in ((0, 50), (0, 50 * uA), (0 * uA, 0.05 * mA)):
        encoder = AmplitudeEncoder(amp_range=amp_range)
        npt.assert_equal(encoder.amp_unit, uA)
        npt.assert_almost_equal(np.asarray(encoder.amp_range), [0, 50])
    encoder = AmplitudeEncoder(amp_range=(0 * xTh, 3 * xTh))
    npt.assert_equal(encoder.amp_unit, xTh)
    npt.assert_almost_equal(np.asarray(encoder.amp_range), [0, 3])
    npt.assert_equal(isinstance(encoder.amp_range[1], Quantity), False)
    npt.assert_equal(
        AmplitudeEncoder(amp_range=np.array([0, 3]) * xTh).amp_unit, xTh)


@pytest.mark.parametrize('amp_range', [(0, 3 * xTh), (0 * xTh, 3),
                                       (0 * uA, 3 * xTh), (0 * xTh, 50 * uA)])
def test_AmplitudeEncoder_amp_range_rejects_mixed_units(amp_range):
    with pytest.raises(DimensionMismatchError):
        AmplitudeEncoder(amp_range=amp_range)


@pytest.mark.parametrize('amp_range', [(0 * ms, 3 * ms), (0 * mW, 3 * mW)])
def test_AmplitudeEncoder_amp_range_rejects_wrong_dimension(amp_range):
    with pytest.raises(DimensionMismatchError):
        AmplitudeEncoder(amp_range=amp_range)


@pytest.mark.parametrize('bad', [(-1 * xTh, 3 * xTh),
                                 (0 * xTh, np.inf * xTh),
                                 (0 * xTh, np.nan * xTh)])
def test_AmplitudeEncoder_amp_range_xTh_is_validated(bad):
    with pytest.raises(ValueError):
        AmplitudeEncoder(amp_range=bad)


def test_AmplitudeEncoder_encodes_threshold_multiples():
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    stim = AmplitudeEncoder(amp_range=(0 * xTh, 3 * xTh)).encode(img)
    npt.assert_equal(stim.unit, xTh)
    npt.assert_equal(stim.time_unit, ms)
    npt.assert_almost_equal(np.abs(stim.data).max(axis=1).min(), 0, decimal=4)
    npt.assert_almost_equal(np.abs(stim.data).max(axis=1).max(), 3, decimal=4)
    npt.assert_equal(stim.unit, xTh)
    current = AmplitudeEncoder(amp_range=(0, 3)).encode(img)
    npt.assert_equal(current.unit, uA)
    npt.assert_array_equal(current.data, stim.data)


def test_AmplitudeEncoder_xTh_survives_the_schedule_operations():
    stim = AmplitudeEncoder(amp_range=(0 * xTh, 3 * xTh)).encode(
        ImageStimulus(np.ones((4, 4))))
    npt.assert_equal(stim._spatial_view().unit, xTh)
    npt.assert_almost_equal(stim._spatial_view().data.max(), 3)
    npt.assert_equal(stim._scaled(2).unit, xTh)
    npt.assert_almost_equal(stim._scaled(2)._spatial_view().data.max(), 6)
    npt.assert_equal(stim._without_electrodes([stim.electrodes[0]]).unit, xTh)
    npt.assert_equal(_rendered(stim), False)


def test_AmplitudeEncoder_xTh_is_calibrated_by_the_implant():
    img = ImageStimulus(np.ones((16, 16)))
    encoder = lambda: AmplitudeEncoder(amp_range=(0 * xTh, 2 * xTh))
    plain = ArgusII(encoder=encoder()).prepare_stim(img)
    npt.assert_equal(plain.unit, xTh)
    npt.assert_almost_equal(plain._spatial_view().data.max(), 2)
    calibrated = ArgusII(thresholds=80, encoder=encoder()).prepare_stim(img)
    npt.assert_equal(calibrated.unit, uA)
    npt.assert_equal(_rendered(calibrated), False)
    npt.assert_almost_equal(calibrated._spatial_view().data.max(), 160)
    implant = ArgusII(thresholds={'A1': 40, 'A2': 80}, encoder=encoder())
    implant.thresholds = {**implant.thresholds,
                          **{name: 60 for name in implant.electrode_names
                             if name not in ('A1', 'A2')}}
    view = implant.prepare_stim(img)._spatial_view()
    amps = dict(zip(view.electrodes, np.abs(view.data).max(axis=1)))
    npt.assert_almost_equal(amps['A1'], 80)
    npt.assert_almost_equal(amps['A2'], 160)


def test_AmplitudeEncoder_xTh_partial_calibration_raises():
    implant = ArgusII(thresholds={'A1': 80},
                      encoder=AmplitudeEncoder(amp_range=(0 * xTh, 2 * xTh)))
    with pytest.raises(DimensionMismatchError) as err:
        implant.prepare_stim(ImageStimulus(np.ones((16, 16))))
    npt.assert_equal('threshold multiples' in str(err.value), True)


def test_AmplitudeEncoder_xTh_zero_amplitude_needs_no_threshold():
    implant = ArgusII(thresholds={'A1': 80},
                      encoder=AmplitudeEncoder(amp_range=(0 * xTh, 2 * xTh)))
    stim = implant.prepare_stim(ImageStimulus(np.zeros((16, 16))))
    npt.assert_equal(stim.unit, xTh)
    npt.assert_almost_equal(np.abs(stim.data).max(), 0)
    img = np.zeros((6, 10))
    img[0, 0] = 1
    stim = implant.prepare_stim(ImageStimulus(img))
    npt.assert_equal(stim.unit, uA)
    npt.assert_almost_equal(np.abs(stim.data).max(), 160)


def test_AmplitudeEncoder_xTh_fails_the_electrical_safety_checks():
    def encoder():
        return AmplitudeEncoder(amp_range=(0 * xTh, 2 * xTh))

    img = ImageStimulus(np.ones((16, 16)))
    limited = ArgusII(encoder=encoder())
    limited.max_current = 1000
    for implant in (ArgusII(encoder=encoder(), safe_mode=True), limited):
        with pytest.raises(DimensionMismatchError):
            implant.prepare_stim(img)
    implant = ArgusII(encoder=encoder(), thresholds=80, safe_mode=True)
    npt.assert_equal(implant.prepare_stim(img).unit, uA)
