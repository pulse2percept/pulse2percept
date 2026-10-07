import numpy as np
import numpy.testing as npt
import pytest
import copy

from pulse2percept.implants import (DiskElectrode, PointSource,
                                    ElectrodeArray, Implant)
from pulse2percept.implants.retina import ArgusI
from pulse2percept.stimuli import BiphasicPulseTrain, Stimulus
from pulse2percept.percepts import Percept
from pulse2percept.models.retina import (Nanduri2012Model, Nanduri2012Spatial,
                                         Nanduri2012Temporal)
from pulse2percept.utils import FreezeError


def test_Nanduri2012Spatial():
    # Nanduri2012Spatial automatically sets `atten_a`:
    model = Nanduri2012Spatial(implant=ArgusI(), step=5)

    # User can set `atten_a`:
    model.atten_a = 12345
    npt.assert_equal(model.atten_a, 12345)
    model.build(atten_a=987)
    npt.assert_equal(model.atten_a, 987)

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(None), None)

    # Zero in = zero out:
    percept = model.predict_percept(np.zeros(16))
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [1])
    npt.assert_almost_equal(percept.data, 0)

    # Only works for DiskElectrode arrays (checked at build time):
    with pytest.raises(TypeError):
        Nanduri2012Spatial(
            implant=Implant(ElectrodeArray(PointSource(0, 0, 0)))
        ).build()
    with pytest.raises(TypeError):
        Nanduri2012Spatial(implant=Implant(ElectrodeArray(
            [DiskElectrode(0, 0, 0, 100), PointSource(100, 100, 0)]))).build()

    # Also checked at predict time, in case the array is swapped after build:
    model = Nanduri2012Spatial(implant=Implant(ElectrodeArray(
        DiskElectrode(0, 0, 0, 100))), step=5).build()
    model.implant.electrode_array = ElectrodeArray(PointSource(0, 0, 0))
    with pytest.raises(TypeError):
        model.predict_percept({0: 1})

    # Multiple frames are processed independently:
    model = Nanduri2012Spatial(implant=ArgusI(), atten_a=14000, step=5,
                               xrange=(-20, 20), yrange=(-15, 15))
    model.build()
    percept = model.predict_percept({'A1': [1, 2]})
    npt.assert_equal(percept.shape, list(model.grid.x.shape) + [2])
    pmax = percept.data.max(axis=(0, 1))
    npt.assert_almost_equal(percept.data[2, 3, :], pmax)
    npt.assert_almost_equal(pmax[1] / pmax[0], 2.0)

    # Nanduri model uses a linear dva_to_ret conversion factor:
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(model.visual_field_map.dva_to_ret(factor,
                                                                  factor),
                                (280.0 * factor, -280.0 * factor))
    for factor in [0.0, 1.0, 2.0]:
        npt.assert_almost_equal(
            model.visual_field_map.ret_to_dva(280.0 * factor,
                                              -280.0 * factor),
            (factor, factor))


def test_eq_Nanduri2012Spatial():
    nanduri_spatial = Nanduri2012Spatial(implant=ArgusI())

    # Assert not equal for differing classes
    npt.assert_equal(nanduri_spatial == int, False)

    # Assert equal to itself
    npt.assert_equal(nanduri_spatial == nanduri_spatial, True)

    # Assert equal for shallow references
    copied = nanduri_spatial
    npt.assert_equal(nanduri_spatial == copied, True)

    # Assert deep copies are equal
    copied = copy.deepcopy(nanduri_spatial)
    npt.assert_equal(nanduri_spatial == copied, True)

    # Assert differing objects aren't equal
    differing_model = Nanduri2012Model(implant=ArgusI())
    differing_model.spatial.xrange = (-10, 10)
    npt.assert_equal(nanduri_spatial == differing_model, False)


def test_deepcopy_Nanduri2012Spatial():
    original = Nanduri2012Spatial(implant=ArgusI())
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)
    npt.assert_equal(original == copied, True)

    # Assert changing the original doesn't affect the copied
    original.verbose = False
    npt.assert_equal(original != copied, True)

@pytest.mark.parametrize('scale_out', (1, 2))
def test_Nanduri2012Temporal(scale_out):
    model = Nanduri2012Temporal(scale_out=scale_out)
    # User can set their own params:
    model.dt = 0.1
    npt.assert_equal(model.dt, 0.1)
    model.build(dt=1e-4)
    npt.assert_equal(model.dt, 1e-4)
    # User cannot add more model parameters:
    with pytest.raises(FreezeError):
        model.rho = 100

    # Nothing in, None out:
    npt.assert_equal(model.predict_percept(ArgusI().prepare_stim(None)), None)

    # Zero in = zero out:
    stim = ArgusI().prepare_stim(np.zeros((16, 100)))
    percept = model.predict_percept(stim, t_percept=[0, 1, 2])
    npt.assert_equal(isinstance(percept, Percept), True)
    npt.assert_equal(percept.shape, (16, 1, 3))
    npt.assert_almost_equal(percept.data, 0)

    # Can't request the same time twice (the Cython loop increments
    # `idx_frame` after each write):
    with pytest.raises(ValueError):
        model.predict_percept(ArgusI().prepare_stim(np.ones((16, 100))),
                              t_percept=[0.2, 0.2])

    # Brightness scales differently with amplitude vs frequency:
    model = Nanduri2012Temporal(dt=5e-3, scale_out=scale_out)
    model.build()
    sdur = 1000.0  # stimulus duration (ms)
    pdur = 0.45  # (ms)
    t_percept = np.arange(0, sdur, 5)
    implant = Implant(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
    bright_amp = []
    for amp in np.linspace(0, 50, 5):
        stim = implant.prepare_stim(
            BiphasicPulseTrain(20, amp, pdur, interphase_dur=pdur,
                               stim_dur=sdur))
        percept = model.predict_percept(stim, t_percept=t_percept)
        bright_amp.append(percept.data.max())
    # Reference values changed by <1.5% in 0.10.0 (kernel time-stepping fix):
    bright_amp_ref = np.array([0.0, 0.00881, 0.06477, 0.14975, 0.16964])
    npt.assert_almost_equal(bright_amp, scale_out * bright_amp_ref, decimal=3)

    bright_freq = []
    for freq in np.linspace(0, 100, 5):
        stim = implant.prepare_stim(
            BiphasicPulseTrain(freq, 20, pdur, interphase_dur=pdur,
                               stim_dur=sdur))
        percept = model.predict_percept(stim, t_percept=t_percept)
        bright_freq.append(percept.data.max())
    bright_freq_ref = np.array([0.0, 0.03892, 0.07297, 0.10686, 0.13841])
    npt.assert_almost_equal(bright_freq, scale_out * bright_freq_ref,
                            decimal=3)


def test_deepcopy_Nanduri2012Temporal():
    original = Nanduri2012Temporal()
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original.__dict__, copied.__dict__)

    # Assert changing the original doesn't affect the copied
    original.verbose = False
    npt.assert_equal(original != copied, True)

def test_Nanduri2012Model():
    model = Nanduri2012Model(implant=ArgusI(), step=5)
    npt.assert_equal(hasattr(model, 'has_time'), True)
    npt.assert_equal(model.has_time, True)

    # User can set `dt`:
    model.temporal.dt = 1e-5
    npt.assert_almost_equal(model.temporal.dt, 1e-5)
    model.temporal.build(dt=3e-4)
    npt.assert_almost_equal(model.temporal.dt, 3e-4)

    # User cannot add more model parameters:
    with pytest.raises(FreezeError):
        model.temporal.rho = 100

    # `thresh_percept` is passed to both components:
    th = 0.512
    both = Nanduri2012Model(implant=ArgusI(), thresh_percept=th)
    npt.assert_almost_equal(both.spatial.thresh_percept, th)
    npt.assert_almost_equal(both.temporal.thresh_percept, th)
    # Each component then has its own copy:
    both.temporal.thresh_percept = 2 * th
    npt.assert_almost_equal(both.spatial.thresh_percept, th)
    npt.assert_almost_equal(both.temporal.thresh_percept, 2 * th)


def test_deepcopy_Nanduri2012Model():
    original = Nanduri2012Model(implant=ArgusI())
    copied = copy.deepcopy(original)

    # Assert these are two different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent
    npt.assert_equal(original == copied, True)

    # Assert changing the original doesn't affect the copied
    original.spatial.verbose = False
    npt.assert_equal(original != copied, True)


def test_Nanduri2012Model_predict_percept():
    # Nothing in = nothing out:
    model = Nanduri2012Model(implant=ArgusI(), xrange=(0, 0), yrange=(0, 0))
    model.build()
    npt.assert_equal(model.predict_percept(None), None)
    npt.assert_almost_equal(model.predict_percept(np.zeros(16)).data, 0)

    # Single-pixel model same as TemporalModel:
    single = Implant(DiskElectrode(0, 0, 0, 100))
    model = Nanduri2012Model(implant=single, xrange=(0, 0), yrange=(0, 0))
    model.build()
    train = BiphasicPulseTrain(20, 20, 0.45, interphase_dur=0.45)
    t_percept = [0, 0.01, 1.0]
    percept = model.predict_percept(train, t_percept=t_percept)
    temp = Nanduri2012Temporal().build()
    temp = temp.predict_percept(single.prepare_stim(train),
                                t_percept=t_percept)
    npt.assert_almost_equal(percept.data, temp.data, decimal=4)

    # Only works for DiskElectrode arrays (checked at build time):

    with pytest.raises(TypeError):
        Nanduri2012Model(
            implant=Implant(ElectrodeArray(PointSource(0, 0, 0))),
            xrange=(0, 0), yrange=(0, 0)).build()
    with pytest.raises(TypeError):
        Nanduri2012Model(implant=Implant(ElectrodeArray(
            [DiskElectrode(0, 0, 0, 100), PointSource(100, 100, 0)])),
            xrange=(0, 0), yrange=(0, 0)).build()

    # Requested times must be multiples of model.dt:
    implant = Implant(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
    model = Nanduri2012Model(implant=implant, xrange=(0, 0), yrange=(0, 0))
    model.build()
    train = BiphasicPulseTrain(20, 20, 0.45)
    model.temporal.dt = 0.1
    with pytest.raises(ValueError):
        model.predict_percept(train, t_percept=[0.01])
    with pytest.raises(ValueError):
        model.predict_percept(train, t_percept=[0.01, 1.0])
    with pytest.raises(ValueError):
        model.predict_percept(train, t_percept=np.arange(0, 0.5, 0.101))
    model.predict_percept(train, t_percept=np.arange(0, 0.5, 1.0000001))

    # Can't request the same time twice (the Cython loop increments
    # `idx_frame` after each write):
    with pytest.raises(ValueError):
        model.predict_percept(train, t_percept=[0.2, 0.2])

    # It's ok to extrapolate beyond `stim` if the `extrapolate` flag is set:
    model.temporal.dt = 1e-2
    npt.assert_almost_equal(model.predict_percept(train,
                                                  t_percept=10000).data, 0)

    # Output shape must be determined by t_percept:
    npt.assert_equal(model.predict_percept(train, t_percept=0).shape,
                     (1, 1, 1))
    npt.assert_equal(model.predict_percept(train, t_percept=[0, 1]).shape,
                     (1, 1, 2))

    # Brightness vs. size (use values from Nanduri paper):
    implant = Implant(ElectrodeArray(DiskElectrode(0, 0, 0, 260)))
    model = Nanduri2012Model(implant=implant, step=0.5, xrange=(-4, 4),
                             yrange=(-4, 4))
    model.build()
    amp_th = 30
    bright_th = 0.107
    stim_dur = 1000.0
    pdur = 0.45
    t_percept = np.arange(0, stim_dur, 5)
    amp_factors = [1, 6]
    frames_amp = []
    for amp_f in amp_factors:
        train = BiphasicPulseTrain(20, amp_f * amp_th, pdur,
                                   interphase_dur=pdur, stim_dur=stim_dur)
        percept = model.predict_percept(train, t_percept=t_percept)
        idx_frame = np.argmax(np.max(percept.data, axis=(0, 1)))
        brightest_frame = percept.data[..., idx_frame]
        frames_amp.append(brightest_frame)
    npt.assert_equal([np.sum(f > bright_th) for f in frames_amp], [0, 161])
    freqs = [20, 120]
    frames_freq = []
    for freq in freqs:
        train = BiphasicPulseTrain(freq, 1.25 * amp_th, pdur,
                                   interphase_dur=pdur, stim_dur=stim_dur)
        percept = model.predict_percept(train, t_percept=t_percept)
        idx_frame = np.argmax(np.max(percept.data, axis=(0, 1)))
        brightest_frame = percept.data[..., idx_frame]
        frames_freq.append(brightest_frame)
    npt.assert_equal([np.sum(f > bright_th) for f in frames_freq], [21, 49])


def _nanduri_temporal_reference(data, t_stim, t_percept, model):
    """Return the Nanduri cascade, computed per location and time step"""
    f = np.float32
    dt, tau1, tau2, tau3 = (f(getattr(model, p))
                            for p in ('dt', 'tau1', 'tau2', 'tau3'))
    # `eps` was fit with a microsecond time step:
    eps = f(f(model.eps) / f(1000.0))
    asymptote, shift, slope, scale_out, thresh = (
        f(getattr(model, p)) for p in ('asymptote', 'shift', 'slope',
                                       'scale_out', 'thresh_percept'))
    idx_p = np.round(np.asarray(t_percept) / model.dt).astype(int)
    n_sim = idx_p[-1] + 1
    out = np.zeros((data.shape[0], len(idx_p)), dtype=np.float32)
    for s in range(data.shape[0]):
        # Pass 1: rectified fast response and its peak:
        ca = r1 = r2 = f(0.0)
        r3 = np.zeros(n_sim, dtype=np.float32)
        max_r3 = f(1e-37)
        idx_stim = 0
        for i in range(n_sim):
            # Several frames may start within one step:
            while (idx_stim + 1 < len(t_stim) and
                   f(i) * dt >= t_stim[idx_stim + 1]):
                idx_stim += 1
            amp = f(data[s, idx_stim])
            r1 = f(r1 + dt * (amp - r1) / tau1)
            ca = f(ca + dt * max(amp, f(0.0)))
            r2 = f(r2 + dt * (ca - r2) / tau2)
            r3[i] = max(f(r1 - eps * r2), f(0.0))
            max_r3 = max(max_r3, r3[i])
        # Logistic gain on the peak:
        scale = f(asymptote / (1 + np.exp(-(max_r3 - shift) / slope)) /
                  max_r3)
        # Pass 2: slow cascade:
        r4a = r4b = r4c = f(0.0)
        frame = 0
        for i in range(n_sim):
            r4a = f(r4a + dt * (r3[i] * scale - r4a) / tau3)
            r4b = f(r4b + dt * (r4a - r4b) / tau3)
            r4c = f(r4c + dt * (r4b - r4c) / tau3)
            if i == idx_p[frame]:
                # Legacy: thresholding resets the state, too:
                if abs(r4c) < thresh:
                    r4c = f(0.0)
                out[s, frame] = r4c * scale_out
                frame += 1
    return out


@pytest.mark.parametrize('thresh_percept, scale_out', [(0, 1), (0.01, 2)])
def test_Nanduri2012Temporal_matches_reference(thresh_percept, scale_out):
    rng = np.random.default_rng(0)
    data = ((rng.random((3, 6)) - 0.5) * 100).astype(np.float32)
    # 1-us edges put two frames within one dt=10 us step:
    t_stim = np.array([0, 4, 4.001, 8, 12, 16], dtype=np.float32)
    t_percept = np.arange(0, 20, 2.0)
    model = Nanduri2012Temporal(dt=0.01, thresh_percept=thresh_percept,
                                scale_out=scale_out).build()
    got = model.predict_percept(Stimulus(data, time=t_stim),
                                t_percept=t_percept).data.reshape(3, -1)
    want = _nanduri_temporal_reference(data, t_stim, t_percept, model)
    # The threshold must zero some, but not all, outputs:
    assert (0 < np.mean(want == 0) < 1) == (thresh_percept > 0)
    npt.assert_array_equal(got == 0, want == 0)
    npt.assert_allclose(got, want, rtol=1e-4, atol=1e-6 * np.abs(want).max())


def test_Nanduri2012Temporal_threshold_resets_state():
    # Legacy: an output below `thresh_percept` also zeroes the slow state,
    # so requesting an earlier output lowers later ones:
    stim = Stimulus(np.array([[0, 30, 0]], dtype=np.float32),
                    time=[0, 1, 1.5])
    model = Nanduri2012Temporal(dt=0.01, thresh_percept=0.05).build()
    alone = model.predict_percept(stim, t_percept=[40]).data.ravel()
    t_percept = [5, 10, 15, 40]
    after = model.predict_percept(stim, t_percept=t_percept).data.ravel()
    npt.assert_equal(after[:3], 0)
    npt.assert_array_less(after[3], 0.9 * alone[0])
    npt.assert_allclose(after, _nanduri_temporal_reference(
        stim.data, stim.time, t_percept, model).ravel(), rtol=1e-4)


def _disk_spatial(electrodes, **params):
    """Return a built Nanduri2012Spatial with a row of grid points on the
    x axis, 0-560 um from the origin."""
    return Nanduri2012Spatial(
        Implant(ElectrodeArray(electrodes)), xrange=(0, 2), yrange=(0, 0),
        step=0.5, **params).build()


def _disk_weight(model, x_el, y_el, z_el, r_el):
    """Return Eq. 2 of [Nanduri2012]_ at the grid points, in float64."""
    s = np.hypot(model.grid.ret.x.ravel() - x_el,
                 model.grid.ret.y.ravel() - y_el)
    d = np.hypot(np.maximum(s - r_el, 0), z_el)
    return model.atten_a / (model.atten_a + d ** model.atten_n)


def _spatial_paths(model, amps):
    """Return the public and tensor responses to static amplitudes."""
    import torch
    names = model.implant.electrode_names
    public = model.predict_percept(
        Stimulus(np.array(amps, dtype=float)[:, None], electrodes=names))
    tensor = model._predict_tensor(
        torch.tensor(amps, dtype=torch.float32)[:, None], None)
    return public.data.ravel(), tensor.data.numpy().ravel()


@pytest.mark.parametrize('z_el', [0, 50])
@pytest.mark.parametrize('amp', [20, -20])
def test_Nanduri2012Spatial_current_spread(z_el, amp):
    # Points 0 and 140 um are beneath the 200-um disk; 280-560 um lie outside:
    model = _disk_spatial(DiskElectrode(0, 0, z_el, 200))
    want = amp * _disk_weight(model, 0, 0, z_el, 200)
    if z_el == 0:
        # Uniform beneath the disk:
        npt.assert_equal(want[:2], amp)
    for got in _spatial_paths(model, [amp]):
        npt.assert_allclose(got, want, rtol=1e-6)


def test_Nanduri2012Spatial_sums_electrodes():
    model = _disk_spatial([DiskElectrode(0, 0, 0, 100),
                           DiskElectrode(400, 0, 30, 150)])
    want = (10 * _disk_weight(model, 0, 0, 0, 100) -
            25 * _disk_weight(model, 400, 0, 30, 150))
    for got in _spatial_paths(model, [10, -25]):
        npt.assert_allclose(got, want, rtol=1e-6)


def test_Nanduri2012Spatial_threshold():
    model = _disk_spatial(DiskElectrode(0, 0, 0, 200))
    full = _spatial_paths(model, [20])[0]
    # Values equal to the threshold are kept; smaller ones are zeroed:
    model.thresh_percept = full[3]
    for got in _spatial_paths(model, [20]):
        npt.assert_equal(got[:4], full[:4])
        npt.assert_equal(got[4:], 0)


def test_Nanduri2012Spatial_nan_grid_point_is_zero():
    model = _disk_spatial(DiskElectrode(0, 0, 0, 200))
    model.grid.ret.x[0, 1] = np.nan
    for got in _spatial_paths(model, [20]):
        npt.assert_equal(got[1], 0)
        npt.assert_equal(np.all(got[[0, 2, 3, 4]] > 0), True)


def test_Nanduri2012Spatial_sign_of_z():
    # Current spread depends on distance, so z = +20 and -20 agree, also
    # beneath the disk:
    above, below = (_spatial_paths(_disk_spatial(DiskElectrode(0, 0, z, 200)),
                                   [20]) for z in (20, -20))
    for got, want in zip(below, above):
        npt.assert_equal(np.isfinite(got).all(), True)
        npt.assert_array_equal(got, want)
