
import numpy as np
import numpy.testing as npt
from pulse2percept.units import DimensionMismatchError, mm, ms, um
from pulse2percept.units import dva
import pytest
from pulse2percept.implants import (EnsembleImplant, GridImplant, Implant,
                                    PointSource)
from pulse2percept.implants.cortex import Cortivis, Orion
from pulse2percept.topography import Polimeni2006Map
from pulse2percept.models.cortex.base import ScoreboardModel
from pulse2percept.stimuli import BiphasicPulseTrain, MonophasicPulse
from pulse2percept.utils.constants import DT


def _shifted(implant_type, dx, dy):
    """A device translated inside the ensemble's own coordinate frame

    Named implants describe hardware about their own origin; where several of
    them sit relative to one another is the ensemble's geometry.
    """
    implant = implant_type()
    for elec in implant.electrode_array.electrode_objects:
        elec.x += dx
        elec.y += dy
    return implant


def _orion_pair(**kwargs):
    """Two Orions side by side, the way an ensemble would be built"""
    return EnsembleImplant([Orion(), _shifted(Orion, -35000, 0)], **kwargs)


def test_EnsembleImplant():
    # Invalid instantiations:
    with pytest.raises(TypeError):
        EnsembleImplant(implants="this can't happen")
    with pytest.raises(TypeError):
        EnsembleImplant(implants=[3,Cortivis()])
    with pytest.raises(TypeError):
        EnsembleImplant(implants={'1': Cortivis(), '2': 'abcd'})

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

# we essentially just need to make sure that electrode names are
# set properly, the rest of the EnsembleImplant functionality 
# (electrode placement, etc) is determined by the implants passed in
# and thus already tested
# but we'll test it again just to make sure
def test_ensemble_cortivis():
    cortivis = Cortivis()

    ensemble = EnsembleImplant.from_coords(Cortivis,
                                           locs=np.array([(0, 0),
                                                          (10000, 0)]))

    # Each device keeps its own geometry, offset into the ensemble's frame:
    npt.assert_equal(ensemble['0-1'].x, cortivis['1'].x)
    npt.assert_equal(ensemble['0-1'].y, cortivis['1'].y)
    npt.assert_equal(ensemble['1-1'].x, cortivis['1'].x + 10000)
    npt.assert_equal(ensemble['1-1'].y, cortivis['1'].y)

# test from_coords initialization (physical coords in um)
def test_from_coords():
    locs = np.array([(0,0), (10000,0)])

    # check invalid instantiations
    with pytest.raises(TypeError):
        EnsembleImplant.from_coords(Cortivis(0), locs=locs)

    locs = np.array([(0,0), (10000,0), (0, 10000)])

    device = Cortivis()
    ensemble = EnsembleImplant.from_coords(Cortivis, locs=locs)

    # Every device is the same hardware, shifted to its own location:
    for i, (dx, dy) in enumerate(locs):
        npt.assert_equal(ensemble[f'{i}-1'].x, device['1'].x + dx)
        npt.assert_equal(ensemble[f'{i}-1'].y, device['1'].y + dy)
        npt.assert_equal(ensemble[f'{i}-1'].z, device['1'].z)


class _Grid2x2(GridImplant):
    """A constituent whose constructor happens to expose `x`/`y`"""

    def __init__(self, x=0, y=0):
        super().__init__((2, 2), 400, x=x, y=y)


def test_from_coords_translates_every_kind_of_constituent():
    """Placement in an ensemble does not depend on a constructor's spelling"""
    locs = np.array([(0, 0), (10000, -4000)])
    for implant_type in (Cortivis, _Grid2x2):
        device = implant_type()
        name = device.electrode_names[0]
        ensemble = EnsembleImplant.from_coords(implant_type, locs=locs)
        for i, (dx, dy) in enumerate(locs):
            npt.assert_almost_equal(ensemble[f'{i}-{name}'].x,
                                    device[name].x + dx)
            npt.assert_almost_equal(ensemble[f'{i}-{name}'].y,
                                    device[name].y + dy)
        # And the prototype device is untouched:
        npt.assert_almost_equal(device[name].x, implant_type()[name].x)


# test from_cortical_map initialization (vf coords in dva)
def test_from_cortical_map():
    visual_field_map = Polimeni2006Map()

    locs = np.array([(2000,2000), (10000,0), (5000, 5000)]).astype(np.float64)

    # find locations in dva
    dva_x, dva_y = visual_field_map.to_dva()['v1'](locs[:,0], locs[:,1])
    dva_list = [(x,y) for x,y in zip(dva_x, dva_y)]
    dva_locs = np.array(dva_list)

    device = Cortivis()

    # use dva coords to create ensemble
    ensemble = EnsembleImplant.from_cortical_map(Cortivis, visual_field_map,
                                                 dva_locs)

    # The dva locations round-trip back to the physical ones they came from:
    for i, (dx, dy) in enumerate(locs):
        npt.assert_approx_equal(ensemble[f'{i}-1'].x, device['1'].x + dx, 5)
        npt.assert_approx_equal(ensemble[f'{i}-1'].y, device['1'].y + dy, 5)
        npt.assert_approx_equal(ensemble[f'{i}-1'].z, device['1'].z, 5)


def test_prepare_stim_merges_per_implant_input():
    """A dict keyed by implant gives each constituent its own source"""
    ensemble = _orion_pair()
    npt.assert_equal(ensemble.prepare_stim(None), None)
    npt.assert_equal(ensemble.prepare_stim({}), None)
    # A key left out contributes zeros, but the rest still merge:
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
    # A merge of sampled waveforms has no structure to keep:
    npt.assert_equal(stim._structured_sources(), None)

    # biphasic pulse trains
    names = Orion().electrode_names
    stim = ensemble.prepare_stim(
        {0: {e: BiphasicPulseTrain(50, 1, .45) for e in names},
         1: {e: BiphasicPulseTrain(20, 2, .85) for e in names}})
    # Asked before `.data`: reading the waveform must not be what builds it.
    # Each child electrode keeps the train that drives it, under the name the
    # ensemble gives it, so a model still sees two clocks and not one array:
    sources = stim._structured_sources()
    npt.assert_equal([e for e, _ in sources], ensemble.electrode_names)
    sources = dict(sources)
    npt.assert_equal((sources['0-96'].freq, sources['0-96'].phase_dur),
                     (50, .45))
    npt.assert_equal((sources['1-96'].freq, sources['1-96'].phase_dur),
                     (20, .85))
    npt.assert_equal(stim.data.shape, (120, 471))
    # Two implants that pulse at the same instant get there by accumulating
    # their own way, so merging their time axes needs a tolerance:
    npt.assert_equal(np.all(np.diff(stim.time) > 0.95 * DT), True)

    # with cortivis and orion
    mixed = EnsembleImplant([Orion(), _shifted(Cortivis, 10000, 0)])
    npt.assert_equal(
        mixed.prepare_stim({0: np.ones(60), 1: np.ones(96) * 2}).data.shape,
        (156, 1))

    # A source that is not keyed by implant is laid out on the whole array:
    stim = ensemble.prepare_stim(np.ones(120) * 3)
    npt.assert_equal(stim.data.shape, (120, 1))
    npt.assert_equal(stim.data, 3)


def test_prepare_stim_merged_goes_through_the_ensemble_pipeline():
    """Merging is how per-implant input becomes one stimulus, not a way around

    The children prepare their own halves, but what the ensemble delivers is
    still the ensemble's to preprocess and to check.
    """
    ensemble = _orion_pair(preprocess=lambda s: s * -2)
    stim = ensemble.prepare_stim({0: np.ones(60), 1: np.ones(60) * 2})
    npt.assert_almost_equal(stim.data[:60], -2)
    npt.assert_almost_equal(stim.data[60:], -4)

    # ... and an ensemble-level safety check still refuses what it should,
    # even though neither child enforces charge balance of its own:
    unsafe = _orion_pair(safe_mode=True)
    with pytest.raises(ValueError, match='charge-balanced'):
        unsafe.prepare_stim({0: {'96': MonophasicPulse(20, 0.45)}})


def test_EnsembleImplant_from_coords_units():
    """`from_coords` takes physical coordinates, so they may be unitful"""
    locs = np.array([[0., 0.], [10000., -5000.]])
    bare = EnsembleImplant.from_coords(Cortivis, locs=locs)
    unitful = EnsembleImplant.from_coords(Cortivis,
                                          locs=locs / 1000 * mm)
    npt.assert_allclose(unitful.electrode_array.coordinates(),
                        bare.electrode_array.coordinates(), rtol=1e-12)
    # ... and so may the range form:
    ranged = EnsembleImplant.from_coords(Cortivis,
                                         xrange=(-10 * mm, 10 * mm),
                                         yrange=(0, 0), step=10000 * um)
    npt.assert_allclose(
        ranged.electrode_array.coordinates(),
        EnsembleImplant.from_coords(Cortivis, xrange=(-10000, 10000),
                                    yrange=(0, 0),
                                    step=10000).electrode_array.coordinates(),
        rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        EnsembleImplant.from_coords(Cortivis, locs=locs * ms)
    with pytest.raises(DimensionMismatchError):
        EnsembleImplant.from_coords(Cortivis, xrange=(0, 1 * ms),
                                    yrange=(0, 0), step=1)


def test_EnsembleImplant_from_coords_needs_a_specification():
    """Locations or a complete grid, but never a guessed physical default

    There is no universal physical equivalent of the ``(-3, 3)`` dva that
    `from_cortical_map` defaults to: how far a degree reaches depends on the
    visual field map.
    """
    with pytest.raises(ValueError):
        EnsembleImplant.from_coords(Cortivis)
    # A partial grid is not a grid:
    with pytest.raises(ValueError) as excinfo:
        EnsembleImplant.from_coords(Cortivis, xrange=(-1 * mm, 1 * mm),
                                    step=500 * um)
    npt.assert_equal('yrange' in str(excinfo.value), True)
    for kwargs in ({'yrange': (0, 0), 'step': 1000},
                   {'xrange': (0, 0), 'step': 1000},
                   {'xrange': (0, 0), 'yrange': (0, 0)}):
        with pytest.raises(ValueError):
            EnsembleImplant.from_coords(Cortivis, **kwargs)


def test_EnsembleImplant_from_cortical_map_units():
    """`from_cortical_map` places implants by visual field location (dva)"""
    bare = EnsembleImplant.from_cortical_map(
        Cortivis, Polimeni2006Map(), xrange=(-2, 2), yrange=(0, 0), step=2)
    unitful = EnsembleImplant.from_cortical_map(
        Cortivis, Polimeni2006Map(), xrange=(-2 * dva, 2 * dva),
        yrange=(0 * dva, 0 * dva), step=2 * dva)
    npt.assert_allclose(unitful.electrode_array.coordinates(),
                        bare.electrode_array.coordinates(), rtol=1e-12)
    # Locations, too:
    locs = np.array([[-2.0, 0.0], [2.0, 0.0]])
    unitful = EnsembleImplant.from_cortical_map(Cortivis, Polimeni2006Map(),
                                                locs=locs * dva)
    bare = EnsembleImplant.from_cortical_map(Cortivis, Polimeni2006Map(),
                                             locs=locs)
    npt.assert_allclose(unitful.electrode_array.coordinates(),
                        bare.electrode_array.coordinates(), rtol=1e-12)
    # These are degrees, not microns: the whole point of the map is that the
    # two are not interchangeable.
    for kwargs in ({'xrange': (-2 * mm, 2 * mm)}, {'step': 2 * um},
                   {'locs': locs * um}):
        with pytest.raises(DimensionMismatchError):
            EnsembleImplant.from_cortical_map(
                Cortivis, Polimeni2006Map(),
                **{'xrange': (-2, 2), 'yrange': (0, 0), 'step': 2, **kwargs})


def test_EnsembleImplant_from_coords_is_physical():
    """`from_coords` lays out its own micron mesh, not a visual field one

    The two factories take the same argument names and mean different things
    by them, which is why `from_coords` no longer borrows a `Grid2D`: a
    `Grid2D` reads its ranges as degrees.
    """
    # A range and the equivalent explicit locations must agree:
    ranged = EnsembleImplant.from_coords(Cortivis, xrange=(-10000, 10000),
                                         yrange=(0, 0), step=10000)
    listed = EnsembleImplant.from_coords(
        Cortivis, locs=np.array([[-10000., 0.], [0., 0.], [10000., 0.]]))
    npt.assert_equal(len(ranged.implants), 3)
    npt.assert_allclose(ranged.electrode_array.coordinates(),
                        listed.electrode_array.coordinates(), rtol=1e-12)
    # A micron range is fine here and a dva one is not -- the mirror image of
    # `from_cortical_map`:
    npt.assert_allclose(
        EnsembleImplant.from_coords(
            Cortivis, xrange=(-10 * mm, 10 * mm), yrange=(0, 0),
            step=10000 * um).electrode_array.coordinates(),
        ranged.electrode_array.coordinates(), rtol=1e-12)
    with pytest.raises(DimensionMismatchError):
        EnsembleImplant.from_coords(Cortivis, xrange=(-2 * dva, 2 * dva),
                                    yrange=(0, 0), step=1)
