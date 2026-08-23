"""Waveform characterization for pulses, pulse trains and encoders

These tests pin down numbers that no other test asserts directly -- the exact
tick each signal edge sits on, the dtypes the data and time axes are stored
in, and the shape of the container -- so that a change in how a stimulus is
*represented* cannot quietly change what it *delivers*.
"""
import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import trapezoid

from pulse2percept.implants import ArgusII
from pulse2percept.stimuli import (AmplitudeEncoder,
                                   AsymmetricBiphasicPulse,
                                   AsymmetricBiphasicPulseTrain,
                                   BiphasicPulse, BiphasicPulseTrain,
                                   BiphasicTripletTrain, ImageStimulus,
                                   MonophasicPulse, PulseTrain, Stimulus,
                                   VideoStimulus)


def _fake_pulse():
    """A three-point pulse, the minimum `PulseTrain` will tile"""
    return Stimulus([[0, -1, 0]], time=[0, 0.1, 0.2])


def _digest(stim):
    """A compact fingerprint of a waveform

    Records where the signal changes rather than every sample, so that a long
    train is described by a handful of numbers instead of a frozen array. The
    two integrals catch a changed amplitude, and ``net_area`` doubles as the
    charge-balance result.
    """
    data, time = stim.data, stim.time
    edges = time[1:][np.any(np.diff(data, axis=1) != 0, axis=0)]
    return {'shape': data.shape,
            't_end': float(time[-1]),
            'n_edges': int(edges.size),
            'first_edges': [round(float(e), 6) for e in edges[:5]],
            'last_edges': [round(float(e), 6) for e in edges[-2:]],
            'abs_area': round(float(trapezoid(np.abs(data), time).sum()), 6),
            'net_area': round(float(trapezoid(data, time).sum()), 9),
            'peak': (round(float(data.min()), 6),
                     round(float(data.max()), 6))}


# One entry per (build, expected digest). Between them these cover a zero
# frequency, a nonzero interphase gap, a delay, an explicit pulse count, a
# 20-second train and a multi-electrode collection:
WAVEFORMS = [
    (lambda: MonophasicPulse(-20, 1, delay_dur=2, stim_dur=10),
     {'shape': (1, 6), 't_end': 10.0, 'n_edges': 2,
      'first_edges': [2.001, 3.0], 'last_edges': [2.001, 3.0],
      'abs_area': 19.98, 'net_area': -19.979999542, 'peak': (-20.0, 0.0)}),
    (lambda: MonophasicPulse(13, 0.45),
     {'shape': (1, 4), 't_end': 0.45, 'n_edges': 2,
      'first_edges': [0.001, 0.45], 'last_edges': [0.001, 0.45],
      'abs_area': 5.836999, 'net_area': 5.836999416, 'peak': (0.0, 13.0)}),
    (lambda: BiphasicPulse(-20, 1, delay_dur=2, stim_dur=10),
     {'shape': (1, 9), 't_end': 10.0, 'n_edges': 4,
      'first_edges': [2.001, 3.0, 3.001, 4.0], 'last_edges': [3.001, 4.0],
      'abs_area': 39.959999, 'net_area': 0.0, 'peak': (-20.0, 20.0)}),
    (lambda: BiphasicPulse(20, 0.45, interphase_dur=0.5,
                           cathodic_first=False),
     {'shape': (1, 8), 't_end': 1.4, 'n_edges': 4,
      'first_edges': [0.001, 0.45, 0.951, 1.4], 'last_edges': [0.951, 1.4],
      'abs_area': 17.960001, 'net_area': 2.29e-07, 'peak': (-20.0, 20.0)}),
    (lambda: AsymmetricBiphasicPulse(-40, 10, 1, 4, interphase_dur=1,
                                     delay_dur=2, stim_dur=15),
     {'shape': (1, 10), 't_end': 15.0, 'n_edges': 4,
      'first_edges': [2.001, 3.0, 4.001, 8.0], 'last_edges': [4.001, 8.0],
      'abs_area': 79.949997, 'net_area': 0.030002594, 'peak': (-40.0, 10.0)}),
    (lambda: AsymmetricBiphasicPulse(40, 10, 0.45, 0.9,
                                     cathodic_first=False),
     {'shape': (1, 7), 't_end': 1.35, 'n_edges': 4,
      'first_edges': [0.001, 0.45, 0.451, 1.35], 'last_edges': [0.451, 1.35],
      'abs_area': 26.949999, 'net_area': 8.970002174, 'peak': (-10.0, 40.0)}),
    (lambda: PulseTrain(10, _fake_pulse(), n_pulses=3, electrode='A4'),
     {'shape': (1, 10), 't_end': 1000.0, 'n_edges': 6,
      'first_edges': [0.1, 0.2, 100.1, 100.2, 200.1],
      'last_edges': [200.1, 200.2], 'abs_area': 0.3,
      'net_area': -0.300000012, 'peak': (-1.0, 0.0)}),
    (lambda: PulseTrain(3, _fake_pulse(), stim_dur=11),
     {'shape': (1, 4), 't_end': 11.0, 'n_edges': 2,
      'first_edges': [0.1, 0.2], 'last_edges': [0.1, 0.2], 'abs_area': 0.1,
      'net_area': -0.100000001, 'peak': (-1.0, 0.0)}),
    (lambda: BiphasicPulseTrain(20, 50, 0.45, stim_dur=1000),
     {'shape': (1, 141), 't_end': 1000.0, 'n_edges': 80,
      'first_edges': [0.001, 0.45, 0.451, 0.9, 50.001],
      'last_edges': [950.451, 950.9], 'abs_area': 898.000061,
      'net_area': 3.82e-07, 'peak': (-50.0, 50.0)}),
    (lambda: BiphasicPulseTrain(0, 50, 0.45, stim_dur=100),
     {'shape': (1, 2), 't_end': 100.0, 'n_edges': 0, 'first_edges': [],
      'last_edges': [], 'abs_area': 0.0, 'net_area': 0.0,
      'peak': (0.0, 0.0)}),
    (lambda: BiphasicPulseTrain(23.456, -3, 2, interphase_dur=np.pi,
                                delay_dur=np.e, stim_dur=657.456,
                                cathodic_first=False),
     {'shape': (1, 145), 't_end': 657.456, 'n_edges': 64,
      'first_edges': [2.719282, 4.718282, 7.860874, 9.859874, 45.352297],
      'last_edges': [647.3561, 649.3551], 'abs_area': 191.903992,
      'net_area': 0.0, 'peak': (-3.0, 3.0)}),
    (lambda: BiphasicPulseTrain(500, 30, 0.05, n_pulses=4, stim_dur=19),
     {'shape': (1, 29), 't_end': 19.0, 'n_edges': 16,
      'first_edges': [0.001, 0.05, 0.051, 0.1, 2.001],
      'last_edges': [6.051, 6.1], 'abs_area': 11.760001,
      'net_area': -1.05e-07, 'peak': (-30.0, 30.0)}),
    (lambda: BiphasicPulseTrain(50, 50, 0.46, stim_dur=20000),
     {'shape': (1, 7001), 't_end': 20000.0, 'n_edges': 4000,
      'first_edges': [0.001, 0.46, 0.461, 0.92, 20.001],
      'last_edges': [19980.461, 19980.92], 'abs_area': 45900.007812,
      'net_area': 1.3351e-05, 'peak': (-50.0, 50.0)}),
    (lambda: AsymmetricBiphasicPulseTrain(20, -1, 4, 1, 2, interphase_dur=1,
                                          delay_dur=6, stim_dur=500),
     {'shape': (1, 91), 't_end': 500.0, 'n_edges': 40,
      'first_edges': [6.001, 7.0, 8.001, 10.0, 56.001],
      'last_edges': [458.001, 460.0], 'abs_area': 89.949997,
      'net_area': 69.969993591, 'peak': (-1.0, 4.0)}),
    (lambda: BiphasicTripletTrain(20, -3, 1, interphase_dur=1,
                                  interpulse_dur=1, delay_dur=4,
                                  stim_dur=500),
     {'shape': (1, 341), 't_end': 500.0, 'n_edges': 120,
      'first_edges': [4.001, 5.0, 6.001, 7.0, 12.001],
      'last_edges': [472.001, 473.0], 'abs_area': 179.819992,
      'net_area': 0.0, 'peak': (-3.0, 3.0)}),
    (lambda: Stimulus({'A1': BiphasicPulseTrain(20, 50, 0.45, stim_dur=200),
                       'B2': BiphasicPulseTrain(30, -20, 0.45,
                                                stim_dur=200)}),
     {'shape': (2, 57), 't_end': 200.0, 'n_edges': 32,
      'first_edges': [0.001, 0.45, 0.451, 0.9, 33.334333],
      'last_edges': [167.117667, 167.566667], 'abs_area': 287.359985,
      'net_area': -4.58e-07, 'peak': (-50.0, 50.0)}),
]


@pytest.mark.parametrize('build, expected', WAVEFORMS)
def test_waveform_characterization(build, expected):
    npt.assert_equal(_digest(build()), expected)


@pytest.mark.parametrize('build, expected', WAVEFORMS)
def test_waveform_container_invariants(build, expected):
    # Every model in the library reads the data as a C-contiguous float32
    # matrix, and the time axis has to be float64: float32 cannot resolve two
    # points a DT step apart past t = 8.4 s, which is well inside a 20 s train.
    stim = build()
    npt.assert_equal(stim.data.dtype, np.float32)
    npt.assert_equal(stim.time.dtype, np.float64)
    npt.assert_equal(stim.data.flags['C_CONTIGUOUS'], True)


def test_pulse_train_electrode_names():
    # A train names the electrode it was built for, and a collection of them
    # keeps the names it was keyed by:
    npt.assert_equal(BiphasicPulseTrain(20, 50, 0.45, electrode='C3'
                                        ).electrodes, ['C3'])
    npt.assert_equal(PulseTrain(10, _fake_pulse(), n_pulses=2,
                                electrode='A4').electrodes, ['A4'])
    stim = Stimulus({'A1': BiphasicPulseTrain(20, 50, 0.45, stim_dur=200),
                     'B2': BiphasicPulseTrain(30, -20, 0.45, stim_dur=200)})
    npt.assert_equal(list(stim.electrodes), ['A1', 'B2'])


def _encoded():
    """The encoder cases the refactor has to reproduce exactly"""
    implant = ArgusII()
    rng = np.random.RandomState(0)
    img = ImageStimulus(rng.rand(60, 60).astype(np.float32))
    vid = VideoStimulus(rng.rand(60, 60, 4).astype(np.float32),
                        time=np.arange(4) * 50.0)
    return [('amp-image', AmplitudeEncoder(freq=20).encode(img,
                                                           implant=implant)),
            ('amp-video', AmplitudeEncoder(freq=20).encode(vid,
                                                           implant=implant))]


@pytest.mark.parametrize('name, expected', [
    ('amp-image', {'shape': (60, 421), 't_end': 500.0, 'n_frames': 1,
                   'frame_dur': 500.0}),
    ('amp-video', {'shape': (60, 169), 't_end': 200.0, 'n_frames': 4,
                   'frame_dur': 50.0}),
])
def test_encoder_characterization(name, expected):
    stim = dict(_encoded())[name]
    meta = stim.metadata['encoder']
    npt.assert_equal({'shape': stim.data.shape,
                      't_end': float(stim.time[-1]),
                      'n_frames': int(meta['frame_time'].size),
                      'frame_dur': float(meta['frame_dur'])}, expected)
    npt.assert_equal(stim.data.dtype, np.float32)
    npt.assert_equal(stim.time.dtype, np.float64)
