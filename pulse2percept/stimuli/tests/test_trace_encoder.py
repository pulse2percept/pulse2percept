import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from pulse2percept.implants import DiskElectrode, ElectrodeArray, Implant
from pulse2percept.implants.cortex import Orion
from pulse2percept.models.cortex import DynaphosModel
from pulse2percept.models.retina import ScoreboardSpatial
from pulse2percept.stimuli import (Encoder, ImageStimulus, ImplantEncoder,
                                   PulseEncoder, Stimulus, TraceEncoder,
                                   VideoStimulus)
from pulse2percept.stimuli.encoders import _ordered_path
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
    npt.assert_equal(isinstance(encoder, ImplantEncoder), False)
    npt.assert_equal(isinstance(encoder, PulseEncoder), False)
    # Not installable on an implant; the rejected assignment changes nothing:
    with pytest.raises(TypeError, match='ImplantEncoder'):
        model.implant.encoder = encoder
    npt.assert_equal(model.implant.encoder, None)
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


def test_TraceEncoder_trajectory():
    encoder = TraceEncoder(line_model())
    xy = encoder.trajectory([[1, 0], [2, 0.5]] * dva)
    npt.assert_equal(isinstance(xy, np.ndarray), True)
    npt.assert_almost_equal(xy, [[1, 0], [2, 0.5]])
    # 'extent' belongs to image targets only:
    for method in (encoder.trajectory, encoder.electrode_sequence,
                   encoder.encode):
        with pytest.raises(ValueError, match="'extent' applies"):
            method([[0, 0]], extent=(-1, 1, -1, 1) * dva)
    # Other stimuli are not targets:
    for stim in (Stimulus([[1]]), VideoStimulus(np.zeros((2, 2, 3)))):
        with pytest.raises(TypeError, match='ImageStimulus'):
            encoder.trajectory(stim)


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


def image(shape, pixels):
    """ImageStimulus that is 1 at (row, col) ``pixels`` and 0 elsewhere"""
    img = np.zeros(shape)
    img[tuple(np.array(pixels).T)] = 1
    return ImageStimulus(img)


def test_TraceEncoder_image_pixel_centers():
    # 8 x 4 dva over 2 x 4 pixels: 2 dva pixels, centers at x = -3, -1, 1,
    # 3 and y = 1 (row 0), -1 (row 1):
    encoder = TraceEncoder(line_model())
    extent = (-4, 4, -2, 2) * dva
    npt.assert_almost_equal(
        encoder.trajectory(image((2, 4), [(1, 0), (1, 1), (1, 2), (1, 3)]),
                           extent=extent),
        [[-3, -1], [-1, -1], [1, -1], [3, -1]])
    npt.assert_almost_equal(
        encoder.trajectory(image((2, 4), [(0, 0), (1, 1)]), extent=extent),
        [[-3, 1], [-1, -1]])
    # Start is the endpoint first in row-major order:
    npt.assert_almost_equal(
        encoder.trajectory(image((2, 4), [(1, 2), (0, 3)]), extent=extent),
        [[3, 1], [1, -1]])
    # A single pixel is a one-sample trajectory:
    npt.assert_almost_equal(
        encoder.trajectory(image((2, 4), [(0, 2)]), extent=extent), [[1, 1]])


@pytest.mark.parametrize('pixels, expected', [
    # Horizontal, left to right:
    ([(2, c) for c in range(1, 6)], [(2, c) for c in range(1, 6)]),
    # Vertical, top to bottom:
    ([(r, 3) for r in range(1, 6)], [(r, 3) for r in range(1, 6)]),
    # Diagonal, from the top:
    ([(5 - i, 1 + i) for i in range(5)], [(1 + i, 5 - i) for i in range(5)]),
    # Right-angle bend; skeletonize removes the corner pixel:
    ([(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)],
     [(1, 1), (1, 2), (2, 3), (3, 3)]),
])
def test_TraceEncoder_image_simple_paths(pixels, expected):
    # extent puts pixel (row, col) at x = col, y = -row:
    extent = (-0.5, 6.5, -6.5, 0.5) * dva
    xy = TraceEncoder(line_model()).trajectory(image((7, 7), pixels),
                                               extent=extent)
    npt.assert_almost_equal(xy, [(c, -r) for r, c in expected])


def test_ordered_path_bend():
    # The diagonal (1, 2)-(2, 3) has bridge pixel (1, 3), so the corner is a
    # path, not a branch:
    skeleton = np.zeros((5, 5), dtype=bool)
    bend = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
    skeleton[tuple(np.array(bend).T)] = True
    npt.assert_equal(_ordered_path(skeleton), bend)
    # A lone diagonal step still connects:
    skeleton[1, 3] = False
    npt.assert_equal(_ordered_path(skeleton),
                     [(1, 1), (1, 2), (2, 3), (3, 3)])


@pytest.mark.parametrize('pixels, match', [
    ([], 'no trace'),
    ([(1, 1), (1, 2), (4, 4), (4, 5)], 'disconnected'),
    ([(3, c) for c in range(1, 6)] + [(1, 3), (2, 3), (4, 3), (5, 3)],
     'branches'),
    ([(1, c) for c in range(1, 6)] + [(5, c) for c in range(1, 6)] +
     [(r, 1) for r in range(2, 5)] + [(r, 5) for r in range(2, 5)],
     'closed loop'),
])
def test_TraceEncoder_image_rejects_topology(pixels, match):
    target = (image((7, 7), pixels) if pixels
              else ImageStimulus(np.zeros((7, 7))))
    with pytest.raises(ValueError, match=match):
        TraceEncoder(line_model()).trajectory(target,
                                              extent=(-1, 1, -1, 1) * dva)


def test_TraceEncoder_image_rejects_bad_input():
    encoder = TraceEncoder(line_model())
    target = image((3, 3), [(1, 0), (1, 1), (1, 2)])
    with pytest.raises(ValueError, match='extent'):
        encoder.trajectory(target)
    for extent in [(-1, 1, -1) * dva, (1, -1, -1, 1) * dva,
                   (-1, 1, 1, -1) * dva, (-1, 1, -1, np.inf) * dva]:
        with pytest.raises(ValueError, match='extent'):
            encoder.encode(target, extent=extent)
    for unit in (deg, um):
        with pytest.raises(DimensionMismatchError):
            encoder.trajectory(target, extent=(-1, 1, -1, 1) * unit)
    rgb = ImageStimulus(np.ones((3, 3, 3)))
    with pytest.raises(ValueError, match='as_gray=True'):
        encoder.trajectory(rgb, extent=(-1, 1, -1, 1) * dva)
    img = np.zeros((3, 3))
    img[1] = 1
    with pytest.raises(ValueError, match='compress=False'):
        encoder.trajectory(ImageStimulus(img, compress=True),
                           extent=(-1, 1, -1, 1) * dva)
    for threshold in (-0.1, 1.1, np.nan, [0.2, 0.5]):
        with pytest.raises(ValueError, match='threshold'):
            TraceEncoder(line_model(), threshold=threshold)


def test_TraceEncoder_image_threshold():
    img = np.zeros((3, 5))
    img[1, :] = 0.6
    target = ImageStimulus(img)
    extent = (-1, 1, -1, 1) * dva
    npt.assert_equal(len(TraceEncoder(line_model()).trajectory(
        target, extent=extent)), 5)
    # Strictly above threshold:
    with pytest.raises(ValueError, match='no trace'):
        TraceEncoder(line_model(), threshold=0.6).trajectory(target,
                                                             extent=extent)


def thick_z(n=40, width=5, margin=4):
    """A white letter Z on black, strokes ``width`` pixels thick"""
    img = np.zeros((n, n))
    img[margin:margin + width, margin:n - margin] = 1
    img[n - margin - width:n - margin, margin:n - margin] = 1
    for row in range(margin, n - margin):
        col = n - 1 - row
        img[row, max(col - width // 2, 0):col + width // 2 + 1] = 1
    return img


@pytest.mark.parametrize('width', [3, 5, 8])
def test_TraceEncoder_image_thick_Z(width):
    n = 40
    img = thick_z(n, width)
    # extent puts pixel (row, col) at x = col + 0.5, y = -(row + 0.5):
    extent = (0, n, -n, 0) * dva
    # Strict mode rejects the corner spurs of a thick stroke:
    with pytest.raises(ValueError, match='branches'):
        TraceEncoder(line_model()).trajectory(ImageStimulus(img),
                                              extent=extent)
    xy = TraceEncoder(line_model(), prune_spurs=True).trajectory(
        ImageStimulus(img), extent=extent)
    rc = np.column_stack([-xy[:, 1] - 0.5, xy[:, 0] - 0.5])
    npt.assert_almost_equal(rc, np.round(rc))
    rc = np.round(rc).astype(int)
    # Distinct skeleton pixels, each a neighbor of the next:
    npt.assert_equal(np.all(skeletonize(img > 0.5)[tuple(rc.T)]), True)
    npt.assert_equal(len({tuple(p) for p in rc}), len(rc))
    npt.assert_equal(np.abs(np.diff(rc, axis=0)).max(axis=1), 1)
    # Top-left to bottom-right:
    npt.assert_equal(np.all(rc[0] < n // 4), True)
    npt.assert_equal(np.all(rc[-1] >= 3 * n // 4), True)
    # The middle of the path runs from upper right to lower left:
    diagonal = rc[(rc[:, 0] > n // 4) & (rc[:, 0] < 3 * n // 4)]
    npt.assert_equal(np.all(np.diff(diagonal[:, 0]) >= 0), True)
    npt.assert_equal(np.all(np.diff(diagonal[:, 1]) <= 0), True)
    npt.assert_equal(diagonal[0, 1] > 3 * n // 5, True)
    npt.assert_equal(diagonal[-1, 1] < 2 * n // 5, True)


def test_TraceEncoder_image_thick_T():
    # A fork with two short arms is rejected even with pruning:
    img = np.zeros((30, 30))
    img[3:8, 3:27] = 1
    img[3:27, 13:18] = 1
    for prune_spurs in (False, True):
        with pytest.raises(ValueError, match='branches'):
            TraceEncoder(line_model(), prune_spurs=prune_spurs).trajectory(
                ImageStimulus(img), extent=(0, 30, -30, 0) * dva)


def test_TraceEncoder_image_asymmetric_fork():
    # A thick bar with one short arm:
    img = np.zeros((30, 40))
    img[13:18, 3:37] = 1
    img[8:13, 18:23] = 1
    target, extent = ImageStimulus(img), (0, 40, -30, 0) * dva
    with pytest.raises(ValueError, match='branches'):
        TraceEncoder(line_model()).trajectory(target, extent=extent)
    # Known limitation: pruning cannot tell this arm from a corner spur, and
    # removes it:
    xy = TraceEncoder(line_model(), prune_spurs=True).trajectory(
        target, extent=extent)
    npt.assert_equal(np.all(xy[:, 1] < -13), True)


def test_TraceEncoder_image_end_to_end():
    # One row of 7 pixels, centered at x = 0, 1/3, ..., 2 dva and y = 0:
    amp, step_dur = 60, 30
    encoder = TraceEncoder(line_model(), amp=amp * uA, step_dur=step_dur * ms)
    target = image((3, 7), [(1, c) for c in range(7)])
    extent = (-1 / 6, 13 / 6, -0.5, 0.5) * dva
    npt.assert_almost_equal(encoder.trajectory(target, extent=extent),
                            np.column_stack([np.arange(7) / 3, np.zeros(7)]))
    # Nearest: A, A, B, B, B, C, C, collapsed to A, B, C:
    npt.assert_equal(encoder.electrode_sequence(target, extent=extent),
                     ['A', 'B', 'C'])
    stim = encoder.encode(target, extent=extent)
    npt.assert_equal(list(stim.electrodes), ['A', 'B', 'C'])
    npt.assert_almost_equal(stim._spatial_view().data, amp * np.eye(3))
    npt.assert_almost_equal(stim.metadata['encoder']['frame_time'],
                            step_dur * np.arange(3))
    npt.assert_almost_equal(stim.duration, 3 * step_dur)
    npt.assert_almost_equal(np.abs(stim.data).max(), amp)


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
