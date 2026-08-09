import warnings

import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import trapezoid

from pulse2percept.implants import ArgusII
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BostonTrain, Encoder, ImageStimulus,
                                   MonophasicPulse, Stimulus, VideoStimulus)
from pulse2percept.stimuli import encoders
from pulse2percept.utils.constants import DT


class FreqEncoder(Encoder):
    """Stand-in for the frequency encoder, to exercise the base class"""

    def _modulate(self, gray):
        return 20, 10 + 10 * gray


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
    # Encoders pretty-print their parameters:
    npt.assert_equal('amp_range' in str(AmplitudeEncoder()), True)


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
    # A frequency below the frame rate cannot be realized, because a frame
    # cannot hold less than one pulse:
    with pytest.warns(UserWarning, match='slower than the frame rate'):
        AmplitudeEncoder(freq=10, frame_dur=20).encode(img)
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


def test_Encoder_nonseparable():
    # Electrodes that do not share a frequency do not share a time axis. That
    # is the general (union time axis) path, which frequency modulation and
    # raster groups will need:
    img = ImageStimulus(np.random.rand(4, 4))
    with pytest.raises(NotImplementedError):
        FreqEncoder().encode(img)
    # A subclass that happens to return one frequency for every electrode does
    # go through the separable path:
    enc = FreqEncoder().encode(ImageStimulus(np.zeros((4, 4))))
    npt.assert_almost_equal(np.abs(enc.data).max(), 20)


def test_Encoder_metadata():
    enc = AmplitudeEncoder(freq=50).encode(ImageStimulus(np.ones((2, 2))))
    npt.assert_equal(enc.metadata['encoder']['kind'], 'AmplitudeEncoder')
    npt.assert_almost_equal(enc.metadata['encoder']['freq'], 50)
    npt.assert_almost_equal(enc.metadata['encoder']['frame_dur'], 500)
    npt.assert_equal(enc.metadata['encoder']['n_frames'], 1)
