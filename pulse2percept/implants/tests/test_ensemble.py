
import numpy as np
import numpy.testing as npt
from pulse2percept.units import DimensionMismatchError, mm, ms, um
from pulse2percept.units import dva
import pytest
from pulse2percept.implants import (EnsembleImplant, GridImplant, Implant,
                                    PointSource)
from pulse2percept.implants.cortex import NeuroPortArray, Orion
from pulse2percept.implants.retina import ArgusI
from pulse2percept.topography.cortex import Polimeni2006Map
from pulse2percept.topography.retina import Curcio1990Map
from pulse2percept.models.cortex import ScoreboardModel
from pulse2percept.stimuli import BiphasicPulseTrain, MonophasicPulse
from pulse2percept.utils.constants import DT


def _shifted(implant_type, dx, dy):
    """Return a constituent translated in ensemble coordinates."""
    implant = implant_type()
    for elec in implant.electrode_array.electrode_objects:
        elec.x += dx
        elec.y += dy
    return implant


def _orion_pair(**kwargs):
    """Return two Orion arrays side by side."""
    return EnsembleImplant([Orion(), _shifted(Orion, -35000, 0)], **kwargs)


def test_EnsembleImplant():
    # Invalid instantiations:
    with pytest.raises(TypeError):
        EnsembleImplant(implants="this can't happen")
    with pytest.raises(TypeError):
        EnsembleImplant(implants=[3,NeuroPortArray()])
    with pytest.raises(TypeError):
        EnsembleImplant(implants={'1': NeuroPortArray(), '2': 'abcd'})

    # Instantiate with list
    p1 = Implant(PointSource(0,0,0))
    p2 = Implant(PointSource(1,1,1))
    ensemble = EnsembleImplant(implants=[p1,p2])
    npt.assert_equal(ensemble.n_electrodes, 2)
    npt.assert_equal(ensemble[0], p1[0])
    npt.assert_equal(ensemble[1], p2[0])
    npt.assert_equal(ensemble.electrode_names, ['0-0','1-0'])

    # Instantiate with dict
    ensemble = EnsembleImplant(implants={'A': p2, 'B': p1})
    npt.assert_equal(ensemble.n_electrodes, 2)
    npt.assert_equal(ensemble[0], p2[0])
    npt.assert_equal(ensemble[1], p1[0])
    npt.assert_equal(ensemble.electrode_names, ['A-0','B-0'])

    # predict_percept smoke test
    model = ScoreboardModel(implant=ensemble).build()
    model.predict_percept([1, 1])

# The ensemble sets electrode names; electrode placement comes from the
# constituent implants (tested elsewhere), but is checked here too:
def test_ensemble_neuroport():
    neuroport = NeuroPortArray()

    ensemble = EnsembleImplant.from_coords(NeuroPortArray,
                                           locs=np.array([(0, 0),
                                                          (10000, 0)]))

    # Each device keeps its own geometry, offset into the ensemble frame:
    npt.assert_equal(ensemble['0-1'].x, neuroport['1'].x)
    npt.assert_equal(ensemble['0-1'].y, neuroport['1'].y)
    npt.assert_equal(ensemble['1-1'].x, neuroport['1'].x + 10000)
    npt.assert_equal(ensemble['1-1'].y, neuroport['1'].y)

# test from_coords initialization (physical coords in um)
def test_from_coords():
    locs = np.array([(0,0), (10000,0)])

    # check invalid instantiations
    with pytest.raises(TypeError):
        EnsembleImplant.from_coords(NeuroPortArray(0), locs=locs)

    locs = np.array([(0,0), (10000,0), (0, 10000)])

    device = NeuroPortArray()
    ensemble = EnsembleImplant.from_coords(NeuroPortArray, locs=locs)

    # Each device has the same geometry, shifted to its location:
    for i, (dx, dy) in enumerate(locs):
        npt.assert_equal(ensemble[f'{i}-1'].x, device['1'].x + dx)
        npt.assert_equal(ensemble[f'{i}-1'].y, device['1'].y + dy)
        npt.assert_equal(ensemble[f'{i}-1'].z, device['1'].z)


class _Grid2x2(GridImplant):
    """A constituent whose constructor has `x`/`y` arguments"""

    def __init__(self, x=0, y=0):
        super().__init__((2, 2), 400, x=x, y=y)


def test_from_coords_translates_every_kind_of_constituent():
    """from_coords translates constituents with or without `x`/`y` arguments"""
    locs = np.array([(0, 0), (10000, -4000)])
    for implant_type in (NeuroPortArray, _Grid2x2):
        device = implant_type()
        name = device.electrode_names[0]
        ensemble = EnsembleImplant.from_coords(implant_type, locs=locs)
        for i, (dx, dy) in enumerate(locs):
            npt.assert_almost_equal(ensemble[f'{i}-{name}'].x,
                                    device[name].x + dx)
            npt.assert_almost_equal(ensemble[f'{i}-{name}'].y,
                                    device[name].y + dy)
        # The prototype device is unchanged:
        npt.assert_almost_equal(device[name].x, implant_type()[name].x)


# test from_visual_field_map initialization (vf coords in dva)
def test_from_visual_field_map():
    visual_field_map = Polimeni2006Map()

    locs = np.array([(2000,2000), (10000,0), (5000, 5000)]).astype(np.float64)

    # find locations in dva
    dva_x, dva_y = visual_field_map.to_dva()['v1'](locs[:,0], locs[:,1])
    dva_list = [(x,y) for x,y in zip(dva_x, dva_y)]
    dva_locs = np.array(dva_list)

    device = NeuroPortArray()

    # use dva coords to create ensemble
    ensemble = EnsembleImplant.from_visual_field_map(
        NeuroPortArray, visual_field_map, dva_locs)

    # The dva locations map back to the original physical locations:
    for i, (dx, dy) in enumerate(locs):
        npt.assert_approx_equal(ensemble[f'{i}-1'].x, device['1'].x + dx, 5)
        npt.assert_approx_equal(ensemble[f'{i}-1'].y, device['1'].y + dy, 5)
        npt.assert_approx_equal(ensemble[f'{i}-1'].z, device['1'].z, 5)


def test_from_visual_field_map_works_for_a_retinal_map():
    """from_visual_field_map also works with a retinal map

    A retinal map has a single region, so ``region`` can be omitted.
    """
    visual_field_map = Curcio1990Map()
    locs = np.array([[-2., 0.], [0., 0.], [3., 1.]])
    ensemble = EnsembleImplant.from_visual_field_map(ArgusI, visual_field_map,
                                                     locs=locs)
    npt.assert_equal(len(ensemble.implants), 3)
    device = ArgusI()
    x_ret, y_ret = visual_field_map.dva_to_ret(locs[:, 0].copy(),
                                               locs[:, 1].copy())
    for i, (dx, dy) in enumerate(zip(x_ret, y_ret)):
        npt.assert_almost_equal(ensemble[f'{i}-A1'].x, device['A1'].x + dx)
        npt.assert_almost_equal(ensemble[f'{i}-A1'].y, device['A1'].y + dy)
        npt.assert_almost_equal(ensemble[f'{i}-A1'].z, device['A1'].z)
    # Ranges are in dva; the result is in retinal um:
    ranged = EnsembleImplant.from_visual_field_map(
        ArgusI, visual_field_map, xrange=(-2, 2), yrange=(0, 0), step=2)
    npt.assert_equal(len(ranged.implants), 3)
    npt.assert_allclose(
        ranged.electrode_array.coordinates(),
        EnsembleImplant.from_visual_field_map(
            ArgusI, visual_field_map, xrange=(-2 * dva, 2 * dva),
            yrange=(0 * dva, 0 * dva),
            step=2 * dva).electrode_array.coordinates(), rtol=1e-12)


def test_prepare_stim_merges_per_implant_input():
    """A dict keyed by implant gives each constituent its own source"""
    ensemble = _orion_pair()
    npt.assert_equal(ensemble.prepare_stim(None), None)
    npt.assert_equal(ensemble.prepare_stim({}), None)
    # A missing key contributes zeros:
    stim = ensemble.prepare_stim({0: np.ones(60)})
    npt.assert_equal(stim.data.shape, (120, 1))
    npt.assert_equal(stim.electrodes, ensemble.electrode_names)
    stim = ensemble.prepare_stim({0: np.ones(60), 1: np.ones(60) * 2})
    npt.assert_equal(stim.data.shape, (120, 1))
    npt.assert_equal(stim.electrodes, ensemble.electrode_names)
    npt.assert_equal(stim.data[:60], 1)
    npt.assert_equal(stim.data[60:], 2)

    # with time
    stim = ensemble.prepare_stim({0: np.ones((60, 5)),
                                  1: np.ones((60, 2)) * 2})
    npt.assert_equal(stim.data.shape, (120, 5))
    npt.assert_equal(stim.data[:60], 1)
    npt.assert_equal(stim.data[60:, :2], 2)
    npt.assert_equal(stim.data[60:, 2:], 0)
    # A merge of sampled waveforms has no structured sources:
    npt.assert_equal(stim._structured_sources(), None)

    # biphasic pulse trains
    names = Orion().electrode_names
    stim = ensemble.prepare_stim(
        {0: {e: BiphasicPulseTrain(50, 1, .45) for e in names},
         1: {e: BiphasicPulseTrain(20, 2, .85) for e in names}})
    # Checked before `.data`, which would render the waveform. Each child
    # electrode keeps its own pulse train under its ensemble name, so a model
    # sees two clocks:
    sources = stim._structured_sources()
    npt.assert_equal([e for e, _ in sources], ensemble.electrode_names)
    sources = dict(sources)
    npt.assert_equal((sources['0-96'].freq, sources['0-96'].phase_dur),
                     (50, .45))
    npt.assert_equal((sources['1-96'].freq, sources['1-96'].phase_dur),
                     (20, .85))
    npt.assert_equal(stim.data.shape, (120, 471))
    # Time points of the two implants accumulate differently, so merging their
    # time axes uses a tolerance:
    npt.assert_equal(np.all(np.diff(stim.time) > 0.95 * DT), True)

    # with NeuroPortArray and Orion
    mixed = EnsembleImplant([Orion(), _shifted(NeuroPortArray, 10000, 0)])
    npt.assert_equal(
        mixed.prepare_stim({0: np.ones(60), 1: np.ones(96) * 2}).data.shape,
        (156, 1))

    # A source not keyed by implant is applied to the whole array:
    stim = ensemble.prepare_stim(np.ones(120) * 3)
    npt.assert_equal(stim.data.shape, (120, 1))
    npt.assert_equal(stim.data, 3)


def _driven(stim):
    """Returns a dict of each driven electrode's peak amplitude"""
    data = np.atleast_2d(np.asarray(stim.data))
    return {str(e): float(np.abs(row).max())
            for e, row in zip(stim.electrodes, data) if np.any(row)}


def test_prepare_stim_sparse_per_implant_input():
    """Sparse child input drives the named electrodes, not the first rows"""
    ensemble = _orion_pair()
    stim = ensemble.prepare_stim(
        {0: {'70': BiphasicPulseTrain(20, 100, 0.45, stim_dur=100)}})
    npt.assert_equal(list(_driven(stim)), ['0-70'])
    npt.assert_almost_equal(_driven(stim)['0-70'], 100)
    # The pulse train is kept, so models can read its clock:
    (name, train), = stim._structured_sources()
    npt.assert_equal((name, train.freq), ('0-70', 20))
    # Scalars, in a different order than the implant's, across both children:
    stim = ensemble.prepare_stim({0: {'50': 7, '70': 5}, 1: {'41': 3}})
    npt.assert_equal(_driven(stim), {'0-70': 5, '0-50': 7, '1-41': 3})
    # Sparse, time-varying input with its own time axis:
    stim = ensemble.prepare_stim(
        {1: {'41': MonophasicPulse(-10, 1), '96': MonophasicPulse(-20, 1)}})
    npt.assert_equal(_driven(stim), {'1-41': 10, '1-96': 20})


def test_prepare_stim_merged_goes_through_the_ensemble_pipeline():
    """Merged per-implant input goes through ensemble preprocessing and
    safety checks"""
    ensemble = _orion_pair(preprocess=lambda s: s * -2)
    stim = ensemble.prepare_stim({0: np.ones(60), 1: np.ones(60) * 2})
    npt.assert_almost_equal(stim.data[:60], -2)
    npt.assert_almost_equal(stim.data[60:], -4)

    # The ensemble safety check applies, even though neither child uses
    # safe_mode:
    unsafe = _orion_pair(safe_mode=True)
    with pytest.raises(ValueError, match='charge-balanced'):
        unsafe.prepare_stim({0: {'96': MonophasicPulse(20, 0.45)}})


def test_EnsembleImplant_from_coords_units():
    """`from_coords` accepts length units"""
    locs = np.array([[0., 0.], [10000., -5000.]])
    bare = EnsembleImplant.from_coords(NeuroPortArray, locs=locs)
    unitful = EnsembleImplant.from_coords(NeuroPortArray,
                                          locs=locs / 1000 * mm)
    npt.assert_allclose(unitful.electrode_array.coordinates(),
                        bare.electrode_array.coordinates(), rtol=1e-12)
    # Also for the range form:
    ranged = EnsembleImplant.from_coords(NeuroPortArray,
                                         xrange=(-10 * mm, 10 * mm),
                                         yrange=(0, 0), step=10000 * um)
    npt.assert_allclose(
        ranged.electrode_array.coordinates(),
        EnsembleImplant.from_coords(NeuroPortArray, xrange=(-10000, 10000),
                                    yrange=(0, 0),
                                    step=10000).electrode_array.coordinates(),
        rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        EnsembleImplant.from_coords(NeuroPortArray, locs=locs * ms)
    with pytest.raises(DimensionMismatchError):
        EnsembleImplant.from_coords(NeuroPortArray, xrange=(0, 1 * ms),
                                    yrange=(0, 0), step=1)


def test_EnsembleImplant_from_coords_needs_a_specification():
    """`from_coords` requires locations or a complete grid

    There is no physical default equivalent to the ``(-3, 3)`` dva default of
    `from_visual_field_map`, since it depends on the visual field map.
    """
    with pytest.raises(ValueError):
        EnsembleImplant.from_coords(NeuroPortArray)
    # A partial grid is rejected:
    with pytest.raises(ValueError) as excinfo:
        EnsembleImplant.from_coords(NeuroPortArray, xrange=(-1 * mm, 1 * mm),
                                    step=500 * um)
    npt.assert_equal('yrange' in str(excinfo.value), True)
    for kwargs in ({'yrange': (0, 0), 'step': 1000},
                   {'xrange': (0, 0), 'step': 1000},
                   {'xrange': (0, 0), 'yrange': (0, 0)}):
        with pytest.raises(ValueError):
            EnsembleImplant.from_coords(NeuroPortArray, **kwargs)


def test_EnsembleImplant_from_visual_field_map_units():
    """`from_visual_field_map` accepts dva units"""
    bare = EnsembleImplant.from_visual_field_map(
        NeuroPortArray, Polimeni2006Map(), xrange=(-2, 2), yrange=(0, 0),
        step=2)
    unitful = EnsembleImplant.from_visual_field_map(
        NeuroPortArray, Polimeni2006Map(), xrange=(-2 * dva, 2 * dva),
        yrange=(0 * dva, 0 * dva), step=2 * dva)
    npt.assert_allclose(unitful.electrode_array.coordinates(),
                        bare.electrode_array.coordinates(), rtol=1e-12)
    # Also for locations:
    locs = np.array([[-2.0, 0.0], [2.0, 0.0]])
    unitful = EnsembleImplant.from_visual_field_map(
        NeuroPortArray, Polimeni2006Map(), locs=locs * dva)
    bare = EnsembleImplant.from_visual_field_map(
        NeuroPortArray, Polimeni2006Map(), locs=locs)
    npt.assert_allclose(unitful.electrode_array.coordinates(),
                        bare.electrode_array.coordinates(), rtol=1e-12)
    # Length units are rejected (these are dva):
    for kwargs in ({'xrange': (-2 * mm, 2 * mm)}, {'step': 2 * um},
                   {'locs': locs * um}):
        with pytest.raises(DimensionMismatchError):
            EnsembleImplant.from_visual_field_map(
                NeuroPortArray, Polimeni2006Map(),
                **{'xrange': (-2, 2), 'yrange': (0, 0), 'step': 2, **kwargs})


def test_EnsembleImplant_from_coords_is_physical():
    """`from_coords` builds its mesh in um, not dva

    `Grid2D` reads its ranges as dva, so `from_coords` does not use it.
    """
    # A range and the equivalent explicit locations agree:
    ranged = EnsembleImplant.from_coords(NeuroPortArray,
                                         xrange=(-10000, 10000),
                                         yrange=(0, 0), step=10000)
    listed = EnsembleImplant.from_coords(
        NeuroPortArray, locs=np.array([[-10000., 0.], [0., 0.], [10000., 0.]]))
    npt.assert_equal(len(ranged.implants), 3)
    npt.assert_allclose(ranged.electrode_array.coordinates(),
                        listed.electrode_array.coordinates(), rtol=1e-12)
    # Length units are accepted and dva rejected (the reverse of
    # `from_visual_field_map`):
    npt.assert_allclose(
        EnsembleImplant.from_coords(
            NeuroPortArray, xrange=(-10 * mm, 10 * mm), yrange=(0, 0),
            step=10000 * um).electrode_array.coordinates(),
        ranged.electrode_array.coordinates(), rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        EnsembleImplant.from_coords(NeuroPortArray, xrange=(-2 * dva, 2 * dva),
                                    yrange=(0, 0), step=1)


def _generic(dx=0):
    """Returns an implant with no anatomical target"""
    return GridImplant((2, 2), 500, x=dx, electrode_type=PointSource)


def test_EnsembleImplant_anatomical_target():
    # Same target (possibly different device types), or generic constituents:
    for ensemble in [EnsembleImplant([ArgusI(), _shifted(ArgusI, 5000, 0)]),
                     EnsembleImplant([NeuroPortArray(), Orion()]),
                     EnsembleImplant([ArgusI(), _generic()]),
                     EnsembleImplant([NeuroPortArray(), _generic()]),
                     EnsembleImplant([_generic(), _generic(5000)])]:
        npt.assert_equal(len(ensemble.implants), 2)
    with pytest.raises(TypeError, match='retinal and cortical'):
        EnsembleImplant([ArgusI(), NeuroPortArray()])
    # Nested ensembles resolve recursively:
    with pytest.raises(TypeError, match='retinal and cortical'):
        EnsembleImplant([EnsembleImplant([ArgusI()]), NeuroPortArray()])


def test_EnsembleImplant_rejected_reassignment_keeps_constituents():
    ensemble = EnsembleImplant([NeuroPortArray(), Orion()])
    before = list(ensemble.implants.values())
    names = list(ensemble.electrode_names)
    with pytest.raises(TypeError, match='retinal and cortical'):
        ensemble.implants = [NeuroPortArray(), ArgusI()]
    npt.assert_equal([i is j for i, j in
                      zip(ensemble.implants.values(), before)], [True, True])
    npt.assert_equal(list(ensemble.electrode_names), names)
