"""Cross-cutting acceptance tests for the units boundary.

The per-module test files check that each API normalizes its own arguments.
These two check the properties that only make sense across the whole library:

*  every public object that claims to accept a quantity gives the same result
   for every spelling of it, and stores ordinary numbers afterwards;
*  every dimensional confusion the design is meant to catch is caught, in one
   place, so the matrix has no accidental gaps.
"""
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.implants import (ArgusII, DiskElectrode, ElectrodeGrid,
                                    EnsembleImplant, ProsthesisSystem)
from pulse2percept.implants.cortex import Cortivis
from pulse2percept.models import (AlphaTemporal, AxonMapSpatial,
                                  FadingTemporal, Model, ScoreboardSpatial)
from pulse2percept.models.cortex import (ScoreboardSpatial as
                                         CortexScoreboardSpatial)
from pulse2percept.percepts import Percept
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulse,
                                   BiphasicPulseTrain, ImageStimulus,
                                   Stimulus)
from pulse2percept.topography import Grid2D, Polimeni2006Map, Watson2014Map
from pulse2percept.units import (DimensionMismatchError, Quantity, Unit, cm,
                                 dimensionless, dva, mA, mm, ms, nA, s, uA, um,
                                 us)

# The same physical quantity, spelled every awkward way the unit system
# allows. Each row is (bare, [equivalent unitful spellings]).
CURRENTS = (41.7, [0.0417 * mA, 41700 * nA])
TIMES = (20, [0.02 * s, 20000 * us])
LENGTHS = (575, [0.575 * mm, 0.0575 * cm])
ANGLES = (2, [2 * dva])

#: Attributes that hold a :py:class:`~pulse2percept.units.Unit` on purpose.
#: Everything else in an object's state has to be plain numeric data.
_UNIT_SLOTS = ('_unit', '_time_unit')


def _state(obj):
    """Every value an object stores, whether it uses __dict__ or __slots__"""
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
    """No Quantity survives construction, anywhere in an object's state

    This is the invariant the whole design rests on: units are stripped at the
    Python boundary, so nothing downstream -- NumPy, Cython, pickle -- ever
    meets one. A Unit is allowed where an object records what its numbers
    mean (``Stimulus._unit`` and friends); a Quantity is not allowed anywhere,
    and neither is an object-dtype array, which is what a Quantity that slipped
    into an array would look like.
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
    """Every spelling of a quantity builds the same object

    The extracted result has to contain something other than zero. Comparing
    an all-zero percept against an all-zero percept passes for the wrong
    reason, which is exactly the failure mode a test like this invites.
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
    """One quantity, many spellings, one result -- and no Quantity left over"""
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
    _same(lambda a: ProsthesisSystem(ArgusII().earray, max_current=a),
          amp, amps, lambda p: p.max_current, 'ProsthesisSystem.max_current')
    _same(lambda a: AmplitudeEncoder(amp_range=(0, a), freq=20).encode(
              ImageStimulus(np.linspace(0, 1, 36).reshape((6, 6))),
              implant=ArgusII()),
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
          lambda e: e.r, 'DiskElectrode.r')
    _same(lambda x: ArgusII(x=x), length, lengths,
          lambda i: np.array([[e.x, e.y, e.z]
                              for e in i.earray.electrode_objects]),
          'ArgusII.x')
    _same(lambda sp: ElectrodeGrid((2, 3), sp), length, lengths,
          lambda g: np.array([[e.x, e.y] for e in g.electrode_objects]),
          'ElectrodeGrid.spacing')
    # A central electrode and a grid wide enough to hold its phosphene, so
    # that the comparisons below have something in them:
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
    _same(lambda a: EnsembleImplant.from_cortical_map(
        Cortivis, Polimeni2006Map(), xrange=(-a, a), yrange=(-a, a),
        step=2 * a), angle, angles,
        lambda e: np.array([[el.x, el.y]
                            for el in e.earray.electrode_objects]),
        'EnsembleImplant.from_cortical_map')
    _same(lambda a: Watson2014Map().dva_to_ret(a, a), angle, angles,
          lambda xy: np.asarray(xy, dtype=float), 'Watson2014Map.dva_to_ret')

    # --- And a whole pipeline, spelled unitfully end to end ---------------
    imp_bare = ArgusII(x=575)
    imp_unit = ArgusII(x=0.575 * mm)
    bare = Model(implant=imp_bare,
                 spatial=ScoreboardSpatial(rho=575, xrange=(-8, 8),
                                           yrange=(-8, 8), step=2),
                 temporal=FadingTemporal(tau=20)).build()
    unitful = Model(implant=imp_unit,
                    spatial=ScoreboardSpatial(rho=0.575 * mm,
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
    """Every dimensional confusion the design is meant to catch, in one place

    Most of these are also tested where they live; this is the matrix, so that
    a gap shows up as a gap rather than as a missing file.
    """
    img = ImageStimulus(np.linspace(0, 1, 36).reshape((6, 6)))
    current = Stimulus({'A1': BiphasicPulseTrain(20, 50, 0.45, stim_dur=100)})
    model = ScoreboardSpatial(implant=ArgusII(), xrange=(-2, 2),
                              yrange=(-2, 2), step=1).build()

    # dimensionless -> implant: an implant delivers current, so a picture is
    # refused where it is prepared rather than where it is eventually read.
    # (Argus II ships with an encoder, which would otherwise turn the picture
    # into the current the implant does deliver.)
    with pytest.raises(DimensionMismatchError):
        ArgusII(preprocess=False, encoder=None).prepare_stim(img)

    # dimensionless -> model: gray levels are not small currents. The implant
    # above is the outer boundary; this is the one behind it, so it is reached
    # through an implant that claims to deliver something else. Scoreboard does
    # read one dimensionless quantity -- the normalized optical drive of a
    # photovoltaic implant (see `PRIMAEncoder`) -- and still refuses a picture,
    # because a picture does not claim to be an encoded drive.
    class Projector(ArgusII):
        stimulus_unit = dimensionless

    with pytest.raises(DimensionMismatchError):
        ScoreboardSpatial(implant=Projector(preprocess=False), xrange=(-2, 2),
                          yrange=(-2, 2), step=1).build().predict_percept(img)

    # current -> encoder: an encoder is what *makes* current out of pictures.
    with pytest.raises(DimensionMismatchError):
        AmplitudeEncoder(amp_range=(0, 50)).encode(current, implant=ArgusII())

    # visual angle -> physical coordinate: retinotopy is a map, not a factor.
    with pytest.raises(DimensionMismatchError):
        DiskElectrode(2 * dva, 0, 0, 100)
    with pytest.raises(DimensionMismatchError):
        ArgusII(x=2 * dva)
    with pytest.raises(DimensionMismatchError):
        Watson2014Map().ret_to_dva(2 * dva, 2 * dva)

    # length -> visual field: and the same map in the other direction.
    with pytest.raises(DimensionMismatchError):
        Grid2D((-2 * mm, 2 * mm), (-2, 2))
    with pytest.raises(DimensionMismatchError):
        Watson2014Map().dva_to_ret(575 * um, 575 * um)
    # A retinal model does resolve a physical `xrange` through its own map
    # (see `SpatialModel._retinal_range_to_dva`), but that is shorthand for a
    # visual field extent, not a conversion, and it is offered nowhere else:
    with pytest.raises(DimensionMismatchError):
        ScoreboardSpatial(implant=ArgusII(), step=100 * um)
    with pytest.raises(DimensionMismatchError):
        CortexScoreboardSpatial(implant=ArgusII(),
                                xrange=(-2 * mm, 2 * mm))

    # current -> time, and time -> current.
    with pytest.raises(DimensionMismatchError):
        BiphasicPulse(50, 0.45 * uA)
    with pytest.raises(DimensionMismatchError):
        BiphasicPulse(50 * ms, 0.45)
    with pytest.raises(DimensionMismatchError):
        model.predict_percept(current, t_percept=[0, 20] * uA)
    with pytest.raises(DimensionMismatchError):
        ProsthesisSystem(ArgusII().earray, max_current=5 * ms)

    # dimensionless -> safety check: there is no charge in a picture.
    with pytest.raises(DimensionMismatchError):
        ProsthesisSystem(ArgusII().earray, safe_mode=True,
                         preprocess=False).prepare_stim(img)
    with pytest.raises(DimensionMismatchError):
        ProsthesisSystem(ArgusII().earray, max_current=20,
                         preprocess=False).prepare_stim(img)

    # A bare number is never rejected, anywhere. That is the other half of the
    # contract, and the reason none of the above needs a deprecation cycle:
    for build in (lambda: DiskElectrode(575, 0, 0, 100),
                  lambda: ArgusII(x=575),
                  lambda: Grid2D((-2, 2), (-2, 2)),
                  lambda: BiphasicPulse(50, 0.45),
                  lambda: ProsthesisSystem(ArgusII().earray, max_current=20),
                  lambda: Watson2014Map().dva_to_ret(2, 2)):
        build()
