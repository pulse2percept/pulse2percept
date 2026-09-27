import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree

from pulse2percept.implants import DiskElectrode, ElectrodeArray, Implant
from pulse2percept.implants.cortex import Orion
from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.models.retina import ScoreboardSpatial
from pulse2percept.stimuli import Encoder, PulseEncoder, TraceEncoder
from pulse2percept.topography import VisualFieldMap
from pulse2percept.units import (DimensionMismatchError, deg, dva, mm, ms, uA,
                                 um, xTh)


class LinearMap(VisualFieldMap):
    """1 dva = 1000 um, so dva and um differ by a known factor"""

    def dva_to_lin(self, x, y):
        return (1000 * np.asarray(x, dtype=float),
                1000 * np.asarray(y, dtype=float))

    def lin_to_dva(self, x, y):
        return (np.asarray(x, dtype=float) / 1000,
                np.asarray(y, dtype=float) / 1000)

    def from_dva(self):
        return {'lin': self.dva_to_lin}

    def to_dva(self):
        return {'lin': self.lin_to_dva}


class TwoRegionMap(LinearMap):
    """'lin' is mapped; 'other' has no transform"""

    def dva_to_other(self, x, y):
        raise NotImplementedError

    def from_dva(self):
        return {'lin': self.dva_to_lin, 'other': self.dva_to_other}

    def to_dva(self):
        return {'lin': self.lin_to_dva, 'other': self.dva_to_other}


class NoForwardMap(LinearMap):
    def from_dva(self):
        raise NotImplementedError


class NoInverseMap(LinearMap):
    def to_dva(self):
        raise NotImplementedError


class EdgeMap(LinearMap):
    """The mapped region ends at x = 1500 um"""

    def lin_to_dva(self, x, y):
        x, y = super().lin_to_dva(x, y)
        return np.where(x > 1.5, np.nan, x), y


class BatchMap(LinearMap):
    """Shifts x = 0 toward the batch mean, like Polimeni2006Map"""

    def dva_to_lin(self, x, y):
        x = np.array(x, dtype=float)
        x[x == 0] += np.copysign(1.2, np.mean(x))
        return super().dva_to_lin(x, y)


def line_implant():
    """Electrodes A, B, C at x = 0, 1000, 2000 um"""
    return Implant(ElectrodeArray({
        'A': DiskElectrode(0, 0, 0, 100),
        'B': DiskElectrode(1000, 0, 0, 100),
        'C': DiskElectrode(2000, 0, 0, 100)}))


def line_model(implant=None, **params):
    return ScoreboardSpatial(line_implant() if implant is None else implant,
                             visual_field_map=LinearMap(), **params)


def sequence(encoder, trace):
    return encoder.electrode_sequence(trace)


def test_TraceEncoder_is_an_Encoder():
    model = line_model()
    encoder = TraceEncoder(model)
    npt.assert_equal(isinstance(encoder, Encoder), True)
    npt.assert_equal(isinstance(encoder, PulseEncoder), False)
    npt.assert_equal(encoder.implant is model.implant, True)
    npt.assert_equal(encoder.model is model, True)
    # Nonrecursive: the model is printed by class name only.
    npt.assert_equal("model='ScoreboardSpatial'" in str(encoder), True)


def test_TraceEncoder_requires_model_context():
    with pytest.raises(TypeError, match='implant'):
        TraceEncoder(object())


def test_TraceEncoder_rejects_rebound_implant():
    model = line_model()
    encoder = TraceEncoder(model)
    model.implant = line_implant()
    with pytest.raises(ValueError, match='rebound'):
        encoder.encode([[0, 0]])


@pytest.mark.parametrize('trace', [np.zeros(2), np.zeros((3, 3)),
                                   np.zeros((0, 2)), np.zeros((2, 2, 1))])
def test_TraceEncoder_rejects_bad_shape(trace):
    with pytest.raises(ValueError, match=r'\(N, 2\)'):
        TraceEncoder(line_model()).encode(trace)


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_TraceEncoder_rejects_nonfinite_trace(bad):
    with pytest.raises(ValueError, match='finite'):
        TraceEncoder(line_model()).encode([[0, 0], [bad, 0]])


def test_TraceEncoder_units():
    encoder = TraceEncoder(line_model())
    npt.assert_equal(sequence(encoder, [[1, 0]] * dva), ['B'])
    # Bare numbers are dva:
    npt.assert_equal(sequence(encoder, [[1, 0]]), ['B'])
    for unit in (um, ms):
        with pytest.raises(DimensionMismatchError):
            encoder.encode(np.array([[1, 0]]) * unit)
    with pytest.raises(DimensionMismatchError):
        TraceEncoder(line_model(), amp=5 * ms)


def test_TraceEncoder_regions():
    npt.assert_equal(sequence(TraceEncoder(line_model(), region='lin'),
                              [[2, 0]]), ['C'])
    with pytest.raises(ValueError, match='not available'):
        TraceEncoder(line_model(), region='v1').encode([[0, 0]])
    two = ScoreboardSpatial(line_implant(), visual_field_map=TwoRegionMap())
    with pytest.raises(ValueError, match="Pass 'region'"):
        TraceEncoder(two).encode([[0, 0]])
    npt.assert_equal(sequence(TraceEncoder(two, region='lin'), [[1, 0]]),
                     ['B'])
    with pytest.raises(NotImplementedError, match="'other'"):
        TraceEncoder(two, region='other').encode([[0, 0]])
    none = ScoreboardSpatial(line_implant(), visual_field_map=NoForwardMap())
    with pytest.raises(NotImplementedError, match='does not map dva'):
        TraceEncoder(none).encode([[0, 0]])
    oneway = ScoreboardSpatial(line_implant(),
                               visual_field_map=NoInverseMap())
    with pytest.raises(NotImplementedError, match='invertible'):
        TraceEncoder(oneway).encode([[0, 0]])


def test_TraceEncoder_skips_electrodes_outside_mapped_region():
    # C (2000 um) is outside the region; the nearest candidate to 2 dva is B:
    model = ScoreboardSpatial(line_implant(), visual_field_map=EdgeMap())
    encoder = TraceEncoder(model)
    npt.assert_equal(sequence(encoder, [[0, 0], [2, 0]]), ['A', 'B'])
    # No candidate left:
    model.implant_position = (5000, 0)
    with pytest.raises(ValueError, match='None of the 3'):
        encoder.encode([[0, 0]])


def test_TraceEncoder_maps_pointwise():
    # A, B, C at 0.5, 1.5, 2.5 dva. Alone, x = 0 shifts to +1.2 dva (nearest
    # B); in a batch with a negative mean it would shift to -1.2 dva (A).
    model = ScoreboardSpatial(line_implant(), visual_field_map=BatchMap(),
                              implant_position=(500, 0))
    encoder = TraceEncoder(model)
    npt.assert_equal(sequence(encoder, [[0, 0]]), ['B'])
    npt.assert_equal(sequence(encoder, [[0, 0], [-3, 0]]), ['B', 'A'])
    npt.assert_equal(sequence(encoder, [[0, 0], [3, 0]]), ['B', 'C'])


def test_TraceEncoder_rejects_unmappable_trace():
    # Polimeni maps eccentricities beyond 90 dva to NaN:
    implant = Orion()
    model = DynaphosModel(implant, implant_position=(20, -5) * mm)
    with pytest.raises(ValueError, match=r'sample\(s\) \[1\]'):
        TraceEncoder(model).encode([[-3, -2], [-120, 0]])


def test_TraceEncoder_rejects_location_noise():
    with pytest.raises(NotImplementedError, match='location_noise'):
        TraceEncoder(line_model(location_noise=0.5))
    model = line_model()
    encoder = TraceEncoder(model)
    model.location_noise = 0.5
    with pytest.raises(NotImplementedError, match='location_noise'):
        encoder.encode([[0, 0]])
    # Disabled noise is accepted:
    TraceEncoder(line_model(location_noise=0)).encode([[0, 0]])


def test_TraceEncoder_requires_active_electrodes():
    implant = line_implant()
    implant.deactivate(['A', 'B', 'C'])
    with pytest.raises(ValueError, match='no activated'):
        TraceEncoder(line_model(implant)).encode([[0, 0]])


def test_TraceEncoder_invalid_params():
    for kwargs in ({'amp': 0}, {'amp': -1}, {'freq': 0}, {'amp': np.nan},
                   {'step_dur': 0}, {'phase_dur': 0}):
        with pytest.raises(ValueError):
            TraceEncoder(line_model(), **kwargs)


def test_TraceEncoder_selects_in_tissue_space():
    encoder = TraceEncoder(line_model())
    # Exact electrode locations:
    npt.assert_equal(sequence(encoder, [[0, 0], [1, 0], [2, 0]]),
                     ['A', 'B', 'C'])
    # 1.6 dva is 1600 um, nearest to C. Read as 1.6 um it would be A:
    npt.assert_equal(sequence(encoder, [[1.6, 0]]), ['C'])


def test_TraceEncoder_translation():
    # The device-local origin sits at 2000 um, so A is at 2 dva:
    encoder = TraceEncoder(line_model(implant_position=(2000, 0)))
    npt.assert_equal(sequence(encoder, [[2, 0], [3, 0], [4, 0]]),
                     ['A', 'B', 'C'])
    npt.assert_equal(sequence(encoder, [[0, 0]]), ['A'])


def test_TraceEncoder_rotation():
    # Rotating by 90 deg moves the array onto the +y axis:
    encoder = TraceEncoder(line_model(implant_rotation=90 * deg))
    npt.assert_equal(sequence(encoder, [[0, 1], [0, 2], [0, 0]]),
                     ['B', 'C', 'A'])
    npt.assert_equal(sequence(encoder, [[1.2, 0]]), ['A'])


def test_TraceEncoder_skips_deactivated():
    implant = line_implant()
    implant.deactivate('B')
    encoder = TraceEncoder(line_model(implant))
    npt.assert_equal(sequence(encoder, [[0.9, 0], [1.1, 0]]), ['A', 'C'])
    stim = encoder.encode([[0.9, 0], [1.1, 0]])
    npt.assert_equal(list(stim.electrodes), ['A', 'C'])


def test_TraceEncoder_sequence():
    amp, freq, phase_dur, step_dur = 80, 250, 0.2, 40
    encoder = TraceEncoder(line_model(), amp=amp * uA, freq=freq,
                           phase_dur=phase_dur * ms, step_dur=step_dur * ms)
    # Nearest electrodes: A, A, B, B, C, B, then A again after a gap:
    trace = [[0, 0], [0.2, 0], [0.9, 0], [1.1, 0], [2, 0], [1.2, 0], [0, 0]]
    npt.assert_equal(sequence(encoder, trace),
                     ['A', 'B', 'C', 'B', 'A'])
    stim = encoder.encode(trace)
    npt.assert_equal(stim.unit, uA)
    npt.assert_equal(list(stim.electrodes), ['A', 'B', 'C'])
    view = stim._spatial_view()
    npt.assert_equal(view.data, amp * np.array([[1, 0, 0, 0, 1],
                                                [0, 1, 0, 1, 0],
                                                [0, 0, 1, 0, 0]]))
    npt.assert_almost_equal(stim.metadata['encoder']['frame_time'],
                            step_dur * np.arange(5))
    npt.assert_almost_equal(stim.metadata['encoder']['frame_dur'], step_dur)
    npt.assert_almost_equal(stim.duration, 5 * step_dur)
    npt.assert_almost_equal(stim._freq, freq)
    npt.assert_almost_equal(stim._phase_dur, phase_dur)
    # Delivered waveform: one electrode at a time, each in its own steps.
    data, time = stim.data, stim.time
    active = np.abs(data) > 0
    npt.assert_equal(np.all(active.sum(axis=0) <= 1), True)
    step = np.minimum((time // step_dur).astype(int), 4)
    owner = np.argmax(view.data, axis=0)
    for row in range(3):
        npt.assert_equal(np.all(owner[step[active[row]]] == row), True)
    npt.assert_almost_equal(np.abs(data).max(), amp)


def test_TraceEncoder_single_step():
    stim = TraceEncoder(line_model(), step_dur=50 * ms).encode(
        [[1, 0], [1.1, 0]])
    npt.assert_equal(list(stim.electrodes), ['B'])
    npt.assert_almost_equal(stim.duration, 50)


def test_TraceEncoder_threshold_amp():
    stim = TraceEncoder(line_model(), amp=2 * xTh).encode([[0, 0], [2, 0]])
    npt.assert_equal(stim.unit, xTh)
    npt.assert_almost_equal(stim._spatial_view().data, [[2, 0], [0, 2]])


def test_TraceEncoder_Polimeni_round_trip():
    # Placed electrode -> to_dva -> TraceEncoder must return that electrode.
    implant = Orion()
    model = DynaphosModel(implant, implant_position=(20, -5) * mm,
                          implant_rotation=30 * deg)
    names = implant.electrode_names
    x, y, _ = model._electrode_coords(implant.electrode_array, None,
                                      electrodes=names)
    vfmap = model.visual_field_map
    xdva, ydva = vfmap.to_dva()['v1'](x, y)
    encoder = TraceEncoder(model)
    # '96' and '90' lie outside Polimeni's V1 wedge; `to_dva` wraps them into
    # the other hemifield. They are not candidates:
    off_v1 = ['96', '90']
    on_v1 = [n not in off_v1 for n in names]
    candidates, _ = encoder._sequence([[-3, -2]])
    npt.assert_equal(sorted(set(names) - set(candidates)), sorted(off_v1))
    trace = np.column_stack([xdva, ydva])[on_v1]
    npt.assert_equal(sequence(encoder, trace),
                     [n for n, ok in zip(names, on_v1) if ok])
    # Valid targets along the upper vertical meridian lie closest in tissue
    # to the off-V1 electrodes, but select on-V1 ones:
    edge = np.column_stack([np.full(50, -0.01), np.linspace(1.3, 1.8, 50)])
    tissue = np.column_stack(vfmap.from_dva()['v1'](edge[:, 0], edge[:, 1]))
    _, nearest = cKDTree(np.column_stack([x, y])).query(tissue)
    npt.assert_equal(set(off_v1) <= {names[i] for i in nearest}, True)
    npt.assert_equal(set(sequence(encoder, edge)) & set(off_v1), set())
    # Deactivated electrodes are replaced by an active neighbor:
    implant.deactivate(names[10])
    got = sequence(encoder, [[xdva[10], ydva[10]]])
    npt.assert_equal(got != [names[10]], True)


def test_TraceEncoder_Dynaphos():
    implant = Orion()
    model = DynaphosModel(implant, implant_position=(20, -5) * mm,
                          xrange=(-6, 0), yrange=(-2, 4), step=0.1)
    # Well-separated phosphenes, so their blobs do not overlap:
    names = ['70', '58', '46']
    x, y, _ = model._electrode_coords(implant.electrode_array, None,
                                      electrodes=names)
    xdva, ydva = model.visual_field_map.to_dva()['v1'](x, y)
    step_dur = 100
    encoder = TraceEncoder(model, amp=500 * uA, freq=model.freq,
                           phase_dur=model.p_dur, step_dur=step_dur * ms)
    trace = np.repeat(np.column_stack([xdva, ydva]), 3, axis=0)
    stim = encoder.encode(trace)
    npt.assert_equal(sequence(encoder, trace), names)
    npt.assert_almost_equal(stim.duration, 3 * step_dur)
    percept = model.predict_percept(stim, t_percept=step_dur * np.arange(4))
    npt.assert_equal(np.all(np.isfinite(percept.data)), True)
    xg, yg = model.grid['dva'].x, model.grid['dva'].y
    for k in range(3):
        # The largest brightness increase during step k lies at electrode k's
        # phosphene location:
        gain = percept.data[..., k + 1] - percept.data[..., k]
        peak = np.unravel_index(np.argmax(gain), gain.shape)
        npt.assert_equal(gain[peak] > 0, True)
        npt.assert_equal(np.hypot(xg[peak] - xdva[k],
                                  yg[peak] - ydva[k]) < 0.25, True)
