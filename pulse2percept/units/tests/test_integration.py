"""Library-wide acceptance tests for unit handling

Per-module tests check that each API converts its own arguments. These tests
check, across the library:

*  every public object that accepts a quantity gives the same result for
   every equivalent spelling, and stores plain numbers;
*  every dimension mismatch the unit system should catch is caught (one
   matrix, in one place).
"""
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import (DiskElectrode, ElectrodeGrid,
                                    EnsembleImplant, Implant)
from pulse2percept.implants.retina import ArgusII
from pulse2percept.implants.cortex import NeuroPortArray
from pulse2percept.models import AlphaTemporal, FadingTemporal, Model
from pulse2percept.models.retina import (AxonMapSpatial, Nanduri2012Spatial,
                                         ScoreboardSpatial)
from pulse2percept.models.cortex import (ScoreboardSpatial as
                                         CortexScoreboardSpatial)
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BiphasicPulseTrain, ImageStimulus,
                                   Stimulus)
from pulse2percept.topography import Grid2D
from pulse2percept.topography.cortex import Polimeni2006Map
from pulse2percept.topography.retina import Watson2014Map
from pulse2percept.units import (DimensionMismatchError, Quantity, Unit, cm,
                                 dimensionless, dva, mA, mm, ms, nA, s, uA, um,
                                 us)

# The same physical quantity in different units. Each row is
# (bare, [equivalent unitful spellings]).
CURRENTS = (41.7, [0.0417 * mA, 41700 * nA])
TIMES = (20, [0.02 * s, 20000 * us])
LENGTHS = (575, [0.575 * mm, 0.0575 * cm])
ANGLES = (2, [2 * dva])

#: Attributes that are allowed to hold a
#: :py:class:`~pulse2percept.units.Unit`. All other state must be plain
#: numeric data.
_UNIT_SLOTS = ('_unit', '_time_unit')


def _state(obj):
    """Returns every value an object stores, from __dict__ and __slots__"""
    state = dict(getattr(obj, '__dict__', {}) or {})
    for klass in type(obj).__mro__:
        for name in getattr(klass, '__slots__', ()) or ():
            if isinstance(name, str) and name not in state:
                try:
                    state[name] = getattr(obj, name)
                except AttributeError:
                    pass
    return state


def assert_stores_plain_numbers(obj, label, _seen=None, _depth=0):
    """Asserts that no Quantity is stored anywhere in an object's state

    Units are stripped at the Python boundary, so NumPy, Cython, and pickle
    never see a Quantity. A Unit is allowed where an object records what its
    numbers mean (``Stimulus._unit`` etc.); a Quantity is not, and neither is
    an object-dtype array (a Quantity inside an array).
    """
    if _seen is None:
        _seen = set()
    if id(obj) in _seen or _depth > 6:
        return
    _seen.add(id(obj))
    if isinstance(obj, Quantity):
        raise AssertionError(f'{label} is a Quantity ({obj})')
    if isinstance(obj, np.ndarray):
        assert obj.dtype != object, f'{label} is an object-dtype array'
        return
    if isinstance(obj, (str, bytes, bool, int, float, np.number, Unit)) \
            or obj is None:
        return
    if isinstance(obj, dict):
        for key, val in obj.items():
            assert_stores_plain_numbers(val, f'{label}[{key!r}]', _seen,
                                        _depth + 1)
        return
    if isinstance(obj, (list, tuple, set)):
        for i, val in enumerate(obj):
            assert_stores_plain_numbers(val, f'{label}[{i}]', _seen,
                                        _depth + 1)
        return
    for name, val in _state(obj).items():
        if name in _UNIT_SLOTS:
            assert isinstance(val, Unit), f'{label}.{name} is not a Unit'
            continue
        assert_stores_plain_numbers(val, f'{label}.{name}', _seen, _depth + 1)


def _same(build, bare, spellings, extract, label, rtol=1e-12):
    """Asserts that every spelling of a quantity builds the same object

    The extracted result must be nonzero, since comparing two all-zero percepts
    would pass trivially.
    """
    reference = build(bare)
    assert_stores_plain_numbers(reference, f'{label}(bare)')
    expected = extract(reference)
    assert np.any(np.asarray(expected, dtype=float)), \
        f'{label} produced nothing to compare (all zero)'
    for spelling in spellings:
        got = build(spelling)
        assert_stores_plain_numbers(got, f'{label}({spelling})')
        npt.assert_allclose(extract(got), expected, rtol=rtol,
                            err_msg=f'{label} disagreed for {spelling}')


def test_every_spelling_builds_the_same_object():
    """Equivalent spellings of a quantity give the same result and store no
    Quantity"""
    amp, amps = CURRENTS
    dur, durs = TIMES
    length, lengths = LENGTHS
    angle, angles = ANGLES

    # --- Current ----------------------------------------------------------
    _same(lambda a: BiphasicPulse(a, 0.45, stim_dur=20), amp, amps,
          lambda p: p.data, 'BiphasicPulse.amp')
    _same(lambda a: BiphasicPulseTrain(20, a, 0.45, stim_dur=100), amp, amps,
          lambda p: p.data, 'BiphasicPulseTrain.amp', rtol=1e-6)
    _same(lambda a: Stimulus([a]), amp, amps, lambda s: s.data,
          'Stimulus', rtol=1e-6)
    _same(lambda a: Implant(ArgusII().electrode_array, max_current=a),
          amp, amps, lambda p: p.max_current, 'Implant.max_current')
    _same(lambda a: AmplitudeEncoder(ArgusII(), amp_range=(0, a),
                                     freq=20).encode(
              ImageStimulus(np.linspace(0, 1, 36).reshape((6, 6)))),
          amp, amps, lambda s: s.data, 'AmplitudeEncoder.amp_range',
          rtol=1e-6)

    # --- Time -------------------------------------------------------------
    _same(lambda t: BiphasicPulse(41.7, 0.45, stim_dur=t), dur, durs,
          lambda p: p.time, 'BiphasicPulse.stim_dur')
    _same(lambda t: BiphasicPulseTrain(20, 41.7, 0.45, stim_dur=5 * t),
          dur, durs, lambda p: p.time, 'BiphasicPulseTrain.stim_dur')
    _same(lambda t: Percept(np.zeros((2, 2, 2)), time=[0, t]), dur, durs,
          lambda p: p.time, 'Percept.time')
    _same(lambda t: FadingTemporal(tau=t).build(), dur, durs,
          lambda m: m.tau, 'FadingTemporal.tau')
    _same(lambda t: AlphaTemporal(tau=t).build(), dur, durs,
          lambda m: m.tau, 'AlphaTemporal.tau')
    pulse = BiphasicPulseTrain(20, 41.7, 0.45, stim_dur=100)
    temporal = FadingTemporal().build()
    _same(lambda t: temporal.predict_percept(pulse, t_percept=[0, t]),
          dur, durs, lambda p: p.data, 'TemporalModel.t_percept')

    # --- Length -----------------------------------------------------------
    _same(lambda x: DiskElectrode(x, 0, 0, 100), length, lengths,
          lambda e: e.x, 'DiskElectrode.x')
    _same(lambda r: DiskElectrode(0, 0, 0, r), length, lengths,
          lambda e: e.radius, 'DiskElectrode.radius')
    _same(lambda z: ArgusII(z=z), length, lengths,
          lambda i: np.array([[e.x, e.y, e.z]
                              for e in i.electrode_array.electrode_objects]),
          'ArgusII.z')
    _same(lambda sp: ElectrodeGrid((2, 3), sp), length, lengths,
          lambda g: np.array([[e.x, e.y] for e in g.electrode_objects]),
          'ElectrodeGrid.spacing')
    # A central electrode and a grid large enough to contain its phosphene, so
    # the percepts are nonzero:
    implant = ArgusII()
    source = {'C5': BiphasicPulseTrain(20, 41.7, 0.45, stim_dur=100)}
    grid = dict(implant=implant, xrange=(-8, 8), yrange=(-8, 8), step=2)
    _same(lambda r: ScoreboardSpatial(rho=r, **grid).build(), length, lengths,
          lambda m: m.predict_percept(source).data, 'ScoreboardSpatial.rho')
    _same(lambda lam: AxonMapSpatial(lam=lam, rho=575, n_axons=100,
                                     n_ax_segments=50, **grid).build(),
          length, lengths,
          lambda m: m.predict_percept(source).data, 'AxonMapSpatial.lam')

    # --- Visual angle -----------------------------------------------------
    _same(lambda a: Grid2D((-a, a), (-a, a), step=0.5), angle, angles,
          lambda g: g.x, 'Grid2D.x_range')
    _same(lambda a: ScoreboardSpatial(implant=implant, rho=575,
                                      xrange=(-4 * a, 4 * a),
                                      yrange=(-4 * a, 4 * a),
                                      step=a).build(), angle, angles,
          lambda m: m.predict_percept(source).data,
          'ScoreboardSpatial.xrange')
    _same(lambda a: EnsembleImplant.from_visual_field_map(
        NeuroPortArray, Polimeni2006Map(), xrange=(-a, a), yrange=(-a, a),
        step=2 * a), angle, angles,
        lambda e: np.array([[el.x, el.y]
                            for el in e.electrode_array.electrode_objects]),
        'EnsembleImplant.from_visual_field_map')
    _same(lambda a: Watson2014Map().dva_to_ret(a, a), angle, angles,
          lambda xy: np.asarray(xy, dtype=float), 'Watson2014Map.dva_to_ret')

    # --- A whole pipeline with unitful arguments --------------------------
    imp_bare = ArgusII(z=575)
    imp_unit = ArgusII(z=0.575 * mm)
    bare = Model(spatial=ScoreboardSpatial(imp_bare, rho=575,
                                           xrange=(-8, 8), yrange=(-8, 8),
                                           step=2),
                 temporal=FadingTemporal(tau=20)).build()
    unitful = Model(spatial=ScoreboardSpatial(imp_unit, rho=0.575 * mm,
                                              xrange=(-8 * dva, 8 * dva),
                                              yrange=(-8 * dva, 8 * dva),
                                              step=2 * dva),
                    temporal=FadingTemporal(tau=0.02 * s)).build()
    src_bare = {'C5': BiphasicPulseTrain(20, 41.7, 0.45, stim_dur=100)}
    src_unit = {'C5': BiphasicPulseTrain(
        0.02 * (1 / ms), 0.0417 * mA, 450 * us, stim_dur=0.1 * s)}
    p_bare = bare.predict_percept(src_bare, t_percept=[0, 20, 40])
    p_unit = unitful.predict_percept(src_unit, t_percept=[0, 0.02 * s,
                                                          40000 * us])
    npt.assert_equal(np.any(p_bare.data), True)
    npt.assert_allclose(p_unit.data, p_bare.data, rtol=1e-6)
    npt.assert_allclose(p_unit.time, p_bare.time, rtol=1e-12)
    assert_stores_plain_numbers(p_unit, 'percept')
    assert_stores_plain_numbers(imp_unit, 'implant')
    assert_stores_plain_numbers(unitful, 'model')


def test_the_whole_rejection_matrix():
    """Every dimension mismatch the unit system should catch

    Most are also tested in their own modules; this matrix makes gaps visible.
    """
    img = ImageStimulus(np.linspace(0, 1, 36).reshape((6, 6)))
    current = Stimulus({'A1': BiphasicPulseTrain(20, 50, 0.45, stim_dur=100)})
    model = ScoreboardSpatial(implant=ArgusII(), xrange=(-2, 2),
                              yrange=(-2, 2), step=1).build()

    # dimensionless -> implant: an implant delivers current, so a picture is
    # rejected by prepare_stim. (Argus II has a default encoder, which would
    # convert the picture to current.)
    with pytest.raises(DimensionMismatchError):
        ArgusII(preprocess=False, encoder=None).prepare_stim(img)

    # dimensionless -> model: gray levels are not currents. Tested with an
    # implant that delivers dimensionless stimuli, to get past prepare_stim.
    class Projector(ArgusII):
        stimulus_unit = dimensionless

    with pytest.raises(DimensionMismatchError):
        Nanduri2012Spatial(implant=Projector(preprocess=False), xrange=(-2, 2),
                           yrange=(-2, 2), step=1).build().predict_percept(img)
    # Except for a scale-free spatial model without a temporal stage, which
    # reads gray levels as relative electrode drive:
    npt.assert_equal(
        ScoreboardSpatial(implant=Projector(preprocess=False), xrange=(-2, 2),
                          yrange=(-2, 2), step=1).predict_percept(img) is None,
        False)

    # current -> encoder: an encoder converts pictures to current.
    with pytest.raises(DimensionMismatchError):
        AmplitudeEncoder(ArgusII(), amp_range=(0, 50)).encode(current)

    # visual angle -> physical coordinate: requires a visual field map.
    with pytest.raises(DimensionMismatchError):
        DiskElectrode(2 * dva, 0, 0, 100)
    with pytest.raises(DimensionMismatchError):
        ArgusII(z=2 * dva)
    with pytest.raises(DimensionMismatchError):
        Watson2014Map().ret_to_dva(2 * dva, 2 * dva)

    # length -> visual field: same, in the other direction.
    with pytest.raises(DimensionMismatchError):
        Grid2D((-2 * mm, 2 * mm), (-2, 2))
    with pytest.raises(DimensionMismatchError):
        Watson2014Map().dva_to_ret(575 * um, 575 * um)
    # A retinal model converts a physical `xrange` through its own map (see
    # `SpatialModel._retinal_range_to_dva`) as a shorthand for the visual field
    # extent; no other parameter does:
    with pytest.raises(DimensionMismatchError):
        ScoreboardSpatial(implant=ArgusII(), step=100 * um)
    with pytest.raises(DimensionMismatchError):
        CortexScoreboardSpatial(implant=NeuroPortArray(),
                                xrange=(-2 * mm, 2 * mm))

    # current -> time, and time -> current.
    with pytest.raises(DimensionMismatchError):
        BiphasicPulse(50, 0.45 * uA)
    with pytest.raises(DimensionMismatchError):
        BiphasicPulse(50 * ms, 0.45)
    with pytest.raises(DimensionMismatchError):
        model.predict_percept(current, t_percept=[0, 20] * uA)
    with pytest.raises(DimensionMismatchError):
        Implant(ArgusII().electrode_array, max_current=5 * ms)

    # dimensionless -> safety check: a picture has no charge.
    with pytest.raises(DimensionMismatchError):
        Implant(ArgusII().electrode_array, safe_mode=True,
                preprocess=False).prepare_stim(img)
    with pytest.raises(DimensionMismatchError):
        Implant(ArgusII().electrode_array, max_current=20,
                preprocess=False).prepare_stim(img)

    # A bare number is always accepted, so none of the above needs a
    # deprecation cycle:
    for build in (lambda: DiskElectrode(575, 0, 0, 100),
                  lambda: ArgusII(z=575),
                  lambda: Grid2D((-2, 2), (-2, 2)),
                  lambda: BiphasicPulse(50, 0.45),
                  lambda: Implant(ArgusII().electrode_array, max_current=20),
                  lambda: Watson2014Map().dva_to_ret(2, 2)):
        build()
