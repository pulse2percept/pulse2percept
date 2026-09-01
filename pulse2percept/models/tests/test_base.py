from contextlib import contextmanager
import copy
import multiprocessing
import warnings

import numpy as np
import pytest
import numpy.testing as npt
from matplotlib.axes import Subplot
import matplotlib.pyplot as plt
import time

from pulse2percept.implants import ArgusI, ArgusII
from pulse2percept.implants.cortex import Cortivis
from pulse2percept.stimuli import (AmplitudeEncoder, BiphasicPulseTrain,
                                   BostonTrain, ImageStimulus, LogoBVL,
                                   Stimulus, VideoStimulus)
from pulse2percept.percepts import Percept
from pulse2percept.models import (AxonMapModel, AxonMapSpatial, BaseModel,
                                  BiphasicAxonMapModel, FadingTemporal,
                                  Horsager2009Model, Model, Nanduri2012Model,
                                  ScoreboardModel, ScoreboardSpatial,
                                  SpatialModel, TemporalModel,
                                  Thompson2003Model)
from pulse2percept.models.base import _blend_meridian
from pulse2percept.models.cortex import (DynaphosModel,
                                         ScoreboardModel as
                                         CortexScoreboardModel,
                                         ScoreboardSpatial as
                                         CortexScoreboardSpatial)
from pulse2percept.units import (DimensionMismatchError, Quantity,
                                 dimensionless, dva, mA, mm, ms, s, uA, um,
                                 us)
from pulse2percept.utils import FreezeError, frame_interval
from pulse2percept.topography import (Curcio1990Map, Grid2D,
                                      Polimeni2006Map, RetinalMap,
                                      Watson2014Map)


class ValidBaseModel(BaseModel):

    def get_default_params(self):
        return {'a': 1, 'b': 2}


def test_BaseModel():
    # Test PrettyPrint:
    model = ValidBaseModel()
    npt.assert_equal(str(model), 'ValidBaseModel(a=1, b=2)')

    # Can overwrite default values:
    model = ValidBaseModel(b=3)
    npt.assert_almost_equal(model.b, 3)

    # Use the sklearn syntax:
    model.set_params(a=5, b=5)
    npt.assert_almost_equal(model.a, 5)
    npt.assert_almost_equal(model.b, 5)

    # Cannot add more attributes:
    with pytest.raises(FreezeError):
        model.c = 3

    # Check the build switch:
    npt.assert_equal(model.is_built, False)
    model.build(a=3)
    npt.assert_almost_equal(model.a, 3)
    npt.assert_equal(model.is_built, True)

    # Attributes must be in `get_default_params`:
    with pytest.raises(AttributeError):
        ValidBaseModel(c=3)
    with pytest.raises(AttributeError):
        ValidBaseModel().is_built = True


class ValidSpatialModel(SpatialModel):
    """A spatial model that predicts nothing"""

    def get_default_params(self):
        params = super(ValidSpatialModel, self).get_default_params()
        params.update({'vfmap': Watson2014Map()})
        return params

    def _predict_spatial(self, earray, stim):
        n_time = 1 if stim.time is None else stim.time.size
        return np.zeros((self.grid.x.size, n_time), dtype=np.float32)


def test_SpatialModel():
    implant = ArgusI()
    # Build grid:
    model = ValidSpatialModel(implant)
    npt.assert_equal(model.grid, None)
    npt.assert_equal(model.is_built, False)
    model.build()
    npt.assert_equal(model.is_built, True)
    npt.assert_equal(isinstance(model.grid, Grid2D), True)
    npt.assert_equal(isinstance(model.grid.ret.x, np.ndarray), True)

    # Can overwrite default values:
    model = ValidSpatialModel(implant, step=1.234)
    npt.assert_almost_equal(model.step, 1.234)
    model.build(step=2.345)
    npt.assert_almost_equal(model.step, 2.345)

    # Cannot add more attributes:
    with pytest.raises(AttributeError):
        ValidSpatialModel(ArgusI(), newparam=1)
    with pytest.raises(FreezeError):
        model.newparam = 1

    # Returns Percept object of proper size:
    npt.assert_equal(model.predict_percept(None), None)
    for source in [np.ones(16), np.zeros(16), {'A1': 2}, np.ones((16, 2))]:
        percept = model.predict_percept(source)
        npt.assert_equal(isinstance(percept, Percept), True)
        stim = implant.prepare_stim(source)
        n_time = 1 if stim.time is None else len(stim.time)
        npt.assert_equal(percept.shape, (model.grid.x.shape[0],
                                         model.grid.x.shape[1],
                                         n_time))
        npt.assert_almost_equal(percept.data, 0)

    # Invalid calls:
    with pytest.raises(ValueError):
        # stim.time==None but requesting t_percept != None
        model.predict_percept(np.ones(16), t_percept=[0, 1, 2])
    # An unbuilt model builds itself rather than refusing:
    fresh = ValidSpatialModel(implant)
    npt.assert_equal(fresh.predict_percept(np.ones(16)) is not None, True)
    npt.assert_equal(fresh.is_built, True)
    with pytest.raises(TypeError):
        # the implant must be an Implant
        ValidSpatialModel(Stimulus(3))
    with pytest.raises(TypeError):
        ValidSpatialModel(None)
    with pytest.raises(TypeError):
        # ... and it cannot be unbound afterwards either
        model.implant = None


def test_SpatialModel_predict_percept_time_order():
    # Before predicting, identical stimulus frames are collapsed with
    # np.unique, which orders what it returns by stimulus value rather than by
    # time. The de-duplicated stimulus handed to _predict_spatial must still
    # run forwards in time, and the frames must be put back where they belong.
    seen_time = []

    class RecordingSpatialModel(ValidSpatialModel):

        def _predict_spatial(self, earray, stim):
            seen_time.append(np.asarray(stim.time))
            # Hand back the first electrode's amplitude at every grid point,
            # so the caller can tell which frame ended up where:
            return np.tile(stim.data[0], (self.grid.x.size, 1))

    model = RecordingSpatialModel(implant=ArgusI(), step=2).build()
    # Amplitudes chosen so that sorting the frames by value shuffles them with
    # respect to time (sorted: 1, 2, 3 -> frames 1, 2, 0):
    with warnings.catch_warnings():
        # A shuffled time axis makes Stimulus warn:
        warnings.simplefilter('error', UserWarning)
        percept = model.predict_percept({'A1': [3, 1, 2]})

    # The model saw time running forwards:
    npt.assert_equal(len(seen_time), 1)
    npt.assert_almost_equal(seen_time[0], [0, 1, 2])
    # ...and every frame was restored to its original position:
    npt.assert_almost_equal(percept.time, [0, 1, 2])
    for idx, amp in enumerate([3, 1, 2]):
        npt.assert_almost_equal(percept.data[..., idx], amp)


def test_SpatialModel_predict_percept_deduplicates_frames():
    # Repeated frames must be computed once and then handed back to every time
    # point they belong to.
    n_calls = []

    class CountingSpatialModel(ValidSpatialModel):

        def _predict_spatial(self, earray, stim):
            n_calls.append(stim.data.shape[1])
            return np.tile(stim.data[0], (self.grid.x.size, 1))

    model = CountingSpatialModel(implant=ArgusI(), step=2).build()
    # Four time points, but only two distinct frames:
    percept = model.predict_percept({'A1': [2, 5, 2, 5]})
    # _predict_spatial was called once, on the two unique frames only:
    npt.assert_equal(n_calls, [2])
    npt.assert_almost_equal(percept.time, [0, 1, 2, 3])
    for idx, amp in enumerate([2, 5, 2, 5]):
        npt.assert_almost_equal(percept.data[..., idx], amp)


def test_SpatialModel_predict_percept_keeps_metadata():
    # `_predict_spatial` only ever sees the de-duplicated copy of the
    # stimulus, so the caller's metadata has to survive the trip:
    seen_metadata = []

    class RecordingSpatialModel(ValidSpatialModel):

        def _predict_spatial(self, earray, stim):
            seen_metadata.append(stim.metadata)
            return np.zeros((self.grid.x.size, stim.data.shape[1]),
                            dtype=np.float32)

    model = RecordingSpatialModel(implant=ArgusI(), step=2).build()
    source = Stimulus({'A1': BiphasicPulseTrain(20, 10, 0.45, stim_dur=20)},
                      metadata='mine')
    model.predict_percept(source)
    npt.assert_equal(len(seen_metadata), 1)
    npt.assert_equal(seen_metadata[0]['user'], 'mine')
    # ...also when the caller asks for time points of their own:
    seen_metadata.clear()
    model.predict_percept(source, t_percept=[0, 5, 10])
    npt.assert_equal(seen_metadata[0]['user'], 'mine')


@pytest.mark.parametrize('param, value', [('engine', 'serial'),
                                          ('scheduler', 'dask'),
                                          ('xystep', 2)])
def test_SpatialModel_removed_params(param, value):
    # `engine` chose the Cython vs pure-Python axon-growth path and
    # `scheduler` drove the joblib/dask backends, both removed in 0.10.0;
    # `xystep` was renamed to `step` in 0.10.0 and removed in 0.11.0. All
    # three are now unknown parameters:
    with pytest.raises(AttributeError):
        ValidSpatialModel(ArgusI(), **{param: value})
    with pytest.raises(AttributeError):
        ValidSpatialModel(ArgusI()).set_params(**{param: value})


def test_Model_has_no_parameter_namespace():
    """Component parameters live on the component, not on the composite"""
    model = Model(spatial=ValidSpatialModel(ArgusI(), step=1),
                  temporal=ValidTemporalModel())
    # The composite accepts no component parameters:
    with pytest.raises(TypeError):
        Model(spatial=ValidSpatialModel(ArgusI()), step=2)
    npt.assert_equal(hasattr(model, 'set_params'), False)
    with pytest.raises(TypeError):
        model.build(step=2)
    # Attributes are not forwarded to components:
    with pytest.raises(AttributeError):
        model.step
    with pytest.raises(FreezeError):
        model.step = 2
    npt.assert_almost_equal(model.spatial.step, 1)


def test_eq_SpatialModel():
    valid = ValidSpatialModel(ArgusI())

    # Assert not equal for differing classes
    npt.assert_equal(valid == ValidBaseModel(), False)

    # Assert equal to itself
    npt.assert_equal(valid == valid, True)

    # Assert equal for shallow references
    copied = valid
    npt.assert_equal(valid == copied, True)

    # Assert deep copies are equal
    copied = copy.deepcopy(valid)
    npt.assert_equal(valid == copied, True)

    # Assert different models do not equal each other
    differing_model = ValidSpatialModel(ArgusI(), xrange=(-10, 10))
    npt.assert_equal(valid != differing_model, True)

def test_deepcopy_SpatialModel():
    original = ValidSpatialModel(ArgusI())
    copied = copy.deepcopy(original)

    # Assert they are different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent to each other
    npt.assert_equal(original == copied, True)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    npt.assert_equal(original != copied, True)

    # Change the copied attribute by "destroying" the vfmap attribute
    # which should be unique to each SpatialModel object
    copied = copy.deepcopy(original)
    copied.vfmap = None
    npt.assert_equal(original.vfmap is not None, True)
    npt.assert_equal(original != copied, True)

    # Assert "destroying" the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)


def test_SpatialModel_plot():
    model = ValidSpatialModel(ArgusI())
    model.build()
    # Simulated area might be larger than that:
    model = ValidSpatialModel(ArgusI(), xrange=(-20.5, 20.5),
                              yrange=(-16.1, 16.1))
    model.build()
    ax = model.plot(use_dva=True)
    npt.assert_almost_equal(ax.get_xlim(), (-22.55, 22.55))
    ax = model.plot(use_dva=False)
    npt.assert_almost_equal(ax.get_xlim(), (-6122.87, 6122.87), decimal=2)
    npt.assert_almost_equal(ax.get_ylim(), (-4808.7, 4808.7), decimal=2)

    # Figure size can be changed:
    ax = model.plot(figsize=(8, 7))
    npt.assert_almost_equal(ax.figure.get_size_inches(), (8, 7))


class ValidTemporalModel(TemporalModel):

    def _predict_temporal(self, stim, t_percept):
        return np.zeros((stim.data.shape[0], len(t_percept)), dtype=np.float32)


def test_Model_units():
    # A unitful value normalizes wherever a component parameter is reached:
    # assignment, `set_params`, and `build`.
    model = Model(temporal=FadingTemporal())
    model.temporal.tau = 0.1 * s
    npt.assert_almost_equal(model.temporal.tau, 100)
    npt.assert_equal(isinstance(model.temporal.tau, float), True)
    model.temporal.build(tau=0.2 * s)
    npt.assert_almost_equal(model.temporal.tau, 200)
    model.temporal.set_params(tau=300 * ms)
    npt.assert_almost_equal(model.temporal.tau, 300)
    with pytest.raises(DimensionMismatchError):
        model.temporal.tau = 5 * uA
    with pytest.raises(DimensionMismatchError):
        Model(temporal=FadingTemporal()).temporal.build(dt=5 * um)
    # Both halves of a spatial+temporal model normalize independently:
    model = Model(spatial=ScoreboardSpatial(implant=ArgusII()), temporal=FadingTemporal())
    model.spatial.set_params(rho=0.3 * mm)
    model.temporal.set_params(tau=0.15 * s, dt=0.01 * ms)
    npt.assert_almost_equal(model.spatial.rho, 300)
    npt.assert_almost_equal(model.temporal.tau, 150)
    npt.assert_almost_equal(model.temporal.dt, 0.01)


def test_SpatialModel_units():
    # A range can be given as a quantity wrapping a pair...
    model = ScoreboardSpatial(implant=ArgusII(), xrange=(-5, 5) * dva, yrange=(-4, 4) * dva,
                              step=1 * dva)
    npt.assert_almost_equal(model.xrange, [-5, 5])
    npt.assert_almost_equal(model.yrange, [-4, 4])
    npt.assert_almost_equal(model.step, 1)
    # ... or as a pair of quantities, which keeps the tuple it was given:
    model = ScoreboardSpatial(implant=ArgusII(), xrange=(-5 * dva, 5 * dva))
    npt.assert_equal(model.xrange, (-5, 5))
    # Either way it grids identically to the bare-number spelling:
    bare = ScoreboardSpatial(implant=ArgusII(), xrange=(-5, 5), yrange=(-5, 5), step=1).build()
    for unitful in (ScoreboardSpatial(implant=ArgusII(), xrange=(-5, 5) * dva,
                                      yrange=(-5, 5) * dva, step=1 * dva),
                    ScoreboardSpatial(implant=ArgusII(), xrange=(-5 * dva, 5 * dva),
                                      yrange=(-5 * dva, 5 * dva),
                                      step=1 * dva)):
        unitful.build()
        npt.assert_almost_equal(bare.grid.x, unitful.grid.x)
        npt.assert_almost_equal(bare.grid.y, unitful.grid.y)
    # A range in the wrong dimension is caught elementwise too. A *length* is
    # the one exception, and means something specific; see
    # `test_SpatialModel_retinal_range`:
    with pytest.raises(DimensionMismatchError):
        ScoreboardSpatial(implant=ArgusII(), xrange=(-5 * ms, 5 * ms))
    with pytest.raises(DimensionMismatchError):
        ScoreboardSpatial(implant=ArgusII(), xrange=(-5, 5) * uA)
    # The grid spacing takes dva and nothing else: a grid spaced evenly on the
    # retina is a different grid, not a different spelling of this one.
    with pytest.raises(DimensionMismatchError):
        ScoreboardSpatial(implant=ArgusII(), step=100 * um)


def test_SpatialModel_retinal_range():
    """A retinal extent is shorthand for the visual field range it covers"""
    # Curcio1990Map puts 280 um to the degree, so 2.8 mm is 10 dva:
    model = ScoreboardSpatial(implant=ArgusII(), xrange=(-2.8 * mm, 2.8 * mm),
                              yrange=(-1.4 * mm, 1.4 * mm),
                              vfmap=Curcio1990Map(), step=1)
    npt.assert_allclose(model.xrange, (-10, 10), rtol=1e-12)
    npt.assert_allclose(model.yrange, (-5, 5), rtol=1e-12)
    # What is stored is plain dva, not a quantity, and it grids exactly like
    # the dva spelling does:
    for value in (model.xrange, model.yrange):
        npt.assert_equal(isinstance(value, Quantity), False)
    bare = ScoreboardSpatial(implant=ArgusII(), xrange=(-10, 10), yrange=(-5, 5),
                             vfmap=Curcio1990Map(), step=1).build()
    npt.assert_almost_equal(bare.grid.x, model.build().grid.x)
    npt.assert_almost_equal(bare.grid.y, model.grid.y)
    # Which map is installed decides the answer, so the user's map has to be
    # applied first however the parameters were ordered:
    for order in ({'xrange': (-2.8 * mm, 2.8 * mm), 'vfmap': Curcio1990Map()},
                  {'vfmap': Curcio1990Map(), 'xrange': (-2.8 * mm, 2.8 * mm)}):
        npt.assert_allclose(ScoreboardSpatial(implant=ArgusII(), step=1, **order).xrange,
                            (-10, 10), rtol=1e-12)
        npt.assert_allclose(
            ScoreboardModel(implant=ArgusII(), step=1, **order).spatial.xrange,
            (-10, 10), rtol=1e-12)
        npt.assert_allclose(
            ScoreboardSpatial(implant=ArgusII(), step=1).build(**order).xrange, (-10, 10),
            rtol=1e-12)
        npt.assert_allclose(
            ScoreboardModel(implant=ArgusII(),
                            step=1).spatial.build(**order).xrange, (-10, 10),
            rtol=1e-12)
    # A quantity wrapping a pair says the same thing:
    npt.assert_allclose(
        ScoreboardSpatial(implant=ArgusII(), xrange=(-2.8, 2.8) * mm, vfmap=Curcio1990Map(),
                          step=1).xrange, (-10, 10), rtol=1e-12)
    # The retinal y axis points the other way, so the pair comes back sorted
    # rather than reversed:
    yrange = ScoreboardSpatial(implant=ArgusII(), yrange=(1.4 * mm, -1.4 * mm),
                               vfmap=Curcio1990Map(), step=1).yrange
    npt.assert_allclose(yrange, (-5, 5), rtol=1e-12)
    # Resolved once, at assignment: a later map does not reinterpret it.
    model = ScoreboardSpatial(implant=ArgusII(), xrange=(-2.8 * mm, 2.8 * mm),
                              vfmap=Curcio1990Map(), step=1)
    model.vfmap = Watson2014Map()
    npt.assert_allclose(model.xrange, (-10, 10), rtol=1e-12)
    # Direct assignment is sequential, and uses the map in place at the time:
    model = ScoreboardSpatial(implant=ArgusII(), step=1)
    model.vfmap = Curcio1990Map()
    model.xrange = (-2.8 * mm, 2.8 * mm)
    npt.assert_allclose(model.xrange, (-10, 10), rtol=1e-12)
    # It must be a pair, whatever the units:
    with pytest.raises(ValueError):
        ScoreboardSpatial(implant=ArgusII(), xrange=2.8 * mm, vfmap=Curcio1990Map())


def test_SpatialModel_retinal_range_nonlinear_map():
    """The motivating case: an axon map model sized in millimeters

    Every other test here uses ``Curcio1990Map``, where the transform is a
    single factor and so cannot tell a real conversion apart from a lucky one.
    ``AxonMapModel`` defaults to ``Watson2014Map``, whose inverse is a quartic
    polynomial in the eccentricity, so this pins the answer to the map rather
    than to a scale factor. No ``build``: growing axon bundles is expensive
    and has nothing to do with what the range came out as.
    """
    model = AxonMapModel(implant=ArgusII(), xrange=(-4 * mm, 4 * mm),
                         yrange=(-2 * mm, 2 * mm)).spatial
    npt.assert_equal(isinstance(model.vfmap, Watson2014Map), True)
    # Each range is resolved along its own meridian, which is what makes the
    # two answers independent of one another:
    watson = Watson2014Map()
    npt.assert_allclose(model.xrange,
                        (watson.ret_to_dva(-4000, 0)[0],
                         watson.ret_to_dva(4000, 0)[0]), rtol=1e-12)
    # The retinal y axis points the other way, so the pair comes back sorted
    # rather than in the order the eccentricities were given:
    npt.assert_allclose(model.yrange,
                        sorted((watson.ret_to_dva(0, -2000)[1],
                                watson.ret_to_dva(0, 2000)[1])), rtol=1e-12)
    # And the map really is consulted: a linear 280 um/dva reading would put
    # the edge of the x range half a degree away from where Watson does.
    npt.assert_equal(abs(model.xrange[1] - 4000 / 280.0) > 0.4, True)
    # The spatial model alone answers identically, and is what the named model
    # constructed:
    npt.assert_allclose(AxonMapSpatial(implant=ArgusII(), xrange=(-4 * mm, 4 * mm)).xrange,
                        model.xrange, rtol=1e-12)


def test_SpatialModel_retinal_range_needs_a_retinal_map():
    """Only a retinal map can say what visual field an extent covers"""
    # A cortical map is not one, whether it was passed explicitly ...
    with pytest.raises(DimensionMismatchError) as excinfo:
        ScoreboardSpatial(implant=ArgusII(), xrange=(-2 * mm, 2 * mm), vfmap=Polimeni2006Map())
    npt.assert_equal('in dva instead' in str(excinfo.value), True)
    # ... or is the model's own default, which a cortical model installs only
    # after its parameters have been applied:
    with pytest.raises(DimensionMismatchError):
        CortexScoreboardSpatial(Cortivis(), yrange=(-2 * mm, 2 * mm))

    # A retinal map without an inverse cannot answer either, and says so:
    class NoInverse(RetinalMap):
        def dva_to_ret(self, xdva, ydva):
            return 280.0 * xdva, -280.0 * ydva

    with pytest.raises(NotImplementedError) as excinfo:
        ScoreboardSpatial(implant=ArgusII(), xrange=(-2 * mm, 2 * mm), vfmap=NoInverse())
    npt.assert_equal('in dva instead' in str(excinfo.value), True)


def test_TemporalModel():
    # Build grid:
    model = ValidTemporalModel()
    npt.assert_equal(model.is_built, False)
    model.build()
    npt.assert_equal(model.is_built, True)

    # Can overwrite default values:
    model = ValidTemporalModel(dt=2e-5)
    npt.assert_almost_equal(model.dt, 2e-5)
    model.build(dt=1.234)
    npt.assert_almost_equal(model.dt, 1.234)

    # Cannot add more attributes:
    with pytest.raises(AttributeError):
        ValidTemporalModel(newparam=1)
    with pytest.raises(FreezeError):
        model.newparam = 1

    # Returns Percept object of proper size:
    implant = ArgusI()
    npt.assert_equal(model.predict_percept(implant.prepare_stim(None)), None)
    model.dt = 1
    for source in [np.ones((16, 3)), np.zeros((16, 3)),
                   {'A1': [1, 2]}, np.ones((16, 2))]:
        stim = implant.prepare_stim(source)
        percept = model.predict_percept(stim)
        # By default, percept is output every 20ms. If stimulus is too short,
        # output at t=[0, 20]. This is mentioned in the docs - for really short
        # stimuli, users should specify the desired time points manually.
        n_time = 1 if stim.time is None else 2
        npt.assert_equal(percept.shape, (stim.shape[0], 1, n_time))
        npt.assert_almost_equal(percept.data, 0)

    # t_percept is automatically sorted:
    model.dt = 0.1
    percept = model.predict_percept(Stimulus(np.zeros((3, 17))),
                                    t_percept=[0.1, 0.8, 0.6])
    npt.assert_almost_equal(percept.time, [0.1, 0.6, 0.8])

    # Invalid calls:
    with pytest.raises(ValueError):
        # Cannot request t_percepts that are not multiples of dt:
        model.predict_percept(Stimulus(np.ones((3, 9))), t_percept=[0.1, 0.11])
    with pytest.raises(ValueError):
        # Has temporal model but stim.time is None:
        ValidTemporalModel().predict_percept(Stimulus(3))
    with pytest.raises(ValueError):
        # stim.time==None but requesting t_percept != None
        ValidTemporalModel().predict_percept(Stimulus(3), t_percept=[0, 1, 2])
    # An unbuilt model builds itself rather than refusing:
    fresh = ValidTemporalModel()
    npt.assert_equal(
        fresh.predict_percept(Stimulus({'A1': [[1, 1]]}, time=[0, 1]))
        is not None, True)
    npt.assert_equal(fresh.is_built, True)
    with pytest.raises(TypeError):
        # Must pass a stimulus:
        ValidTemporalModel().build().predict_percept(ArgusI())


def test_eq_TemporalModel():
    valid = ValidTemporalModel()

    # Assert not equal for differing classes
    npt.assert_equal(valid == ValidBaseModel(), False)

    # Assert equal to itself
    npt.assert_equal(valid == valid, True)

    # Assert equal for shallow references
    copied = valid
    npt.assert_equal(valid == copied, True)

    # Assert deep copies are equal
    copied = copy.deepcopy(valid)
    npt.assert_equal(valid == copied, True)

    # Assert different models do not equal each other
    differing_model = ValidSpatialModel(ArgusI(), xrange=(-10, 10))
    npt.assert_equal(valid != differing_model, True)


def test_deepcopy_TemporalModel():
    original = ValidTemporalModel()
    copied = copy.deepcopy(original)

    # Assert they are different objects
    npt.assert_equal(id(original) != id(copied), True)

    # Assert the objects are equivalent to each other
    npt.assert_equal(original == copied, True)

    # Assert building one object does not affect the copied
    original.build()
    npt.assert_equal(copied.is_built, False)
    npt.assert_equal(original != copied, True)

    # Change the copied attribute by resetting the verbose attribute
    copied = copy.deepcopy(original)
    copied.verbose = False
    npt.assert_equal(original.verbose, True)
    npt.assert_equal(original != copied, True)

    # Assert "destroying" the original doesn't affect the copied
    original = None
    npt.assert_equal(copied is not None, True)


@pytest.mark.parametrize('cls, ImplantType',
                         [(ValidSpatialModel, ArgusI),
                          (ValidTemporalModel, None)])
def test_n_jobs_aliases_n_threads(cls, ImplantType):
    # Only the spatial model takes an implant, and it is required there:
    extra = {} if ImplantType is None else {'implant': ImplantType()}
    # `n_jobs` and `n_threads` are two names for the OpenMP thread count, and
    # must never disagree:
    model = cls(**extra)
    npt.assert_equal(model.n_jobs, model.n_threads)
    # Setting either name moves both, whether in the constructor...
    model = cls(n_jobs=3, **extra)
    npt.assert_equal(model.n_threads, 3)
    npt.assert_equal(model.n_jobs, 3)
    # ...or afterwards, by attribute or by set_params:
    model.n_jobs = 5
    npt.assert_equal(model.n_threads, 5)
    model.n_threads = 7
    npt.assert_equal(model.n_jobs, 7)
    model.set_params(n_jobs=2)
    npt.assert_equal(model.n_threads, 2)
    # The default must not quietly drop us to a single thread:
    npt.assert_equal(cls(**extra).n_threads, multiprocessing.cpu_count())
    # None and -1 both mean "every core", following scikit-learn:
    npt.assert_equal(cls(n_jobs=None, **extra).n_threads,
                     multiprocessing.cpu_count())
    npt.assert_equal(cls(n_jobs=-1, **extra).n_threads,
                     multiprocessing.cpu_count())
    # Nonsense is rejected rather than silently ignored:
    for bad in (0, -2, 2.5, 'many'):
        with pytest.raises(ValueError):
            cls(n_jobs=bad, **extra)
    # It is an alias, not a deprecation -- it must not warn:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cls(n_jobs=4, **extra)


def test_Model_n_jobs_aliases_n_threads():
    # Each component of a composite carries its own thread count:
    model = Model(spatial=ValidSpatialModel(ArgusI(), n_jobs=3),
                  temporal=ValidTemporalModel(n_jobs=3))
    npt.assert_equal(model.spatial.n_threads, 3)
    npt.assert_equal(model.temporal.n_threads, 3)
    model.spatial.n_jobs = 6
    model.temporal.n_jobs = 6
    npt.assert_equal(model.spatial.n_threads, 6)
    npt.assert_equal(model.temporal.n_threads, 6)


def test_Model():
    # A component-free Model is meaningless:
    with pytest.raises(TypeError):
        Model()

    # Wrong model type, and the class instead of an instance:
    with pytest.raises(TypeError):
        Model(spatial=ValidTemporalModel())
    with pytest.raises(TypeError):
        Model(temporal=ValidSpatialModel(ArgusI()))
    with pytest.raises(TypeError, match='not the class itself'):
        Model(temporal=ValidTemporalModel)

    # SpatialModel, but no TemporalModel:
    model = Model(spatial=ValidSpatialModel(ArgusI()))
    npt.assert_equal(model.has_space, True)
    npt.assert_equal(model.has_time, False)
    npt.assert_almost_equal(model.spatial.step, 0.25)
    model.spatial.step = 2
    npt.assert_almost_equal(model.spatial.step, 2)
    # Cannot add more attributes:
    with pytest.raises(AttributeError):
        model.a
    with pytest.raises(FreezeError):
        model.a = 1

    # TemporalModel, but no SpatialModel:
    model = Model(temporal=ValidTemporalModel())
    npt.assert_equal(model.has_space, False)
    npt.assert_equal(model.has_time, True)
    npt.assert_almost_equal(model.temporal.dt, 5e-3)
    model.temporal.dt = 1
    npt.assert_almost_equal(model.temporal.dt, 1)

    # SpatialModel and TemporalModel, positionally:
    model = Model(ValidSpatialModel(ArgusI()), ValidTemporalModel())
    npt.assert_equal(model.has_space, True)
    npt.assert_equal(model.has_time, True)
    npt.assert_almost_equal(model.spatial.step, 0.25)
    npt.assert_almost_equal(model.temporal.dt, 5e-3)
    # A name both components declare stays two separate parameters:
    model.spatial.thresh_percept = 1.234
    npt.assert_almost_equal(model.spatial.thresh_percept, 1.234)
    npt.assert_almost_equal(model.temporal.thresh_percept, 0)


class ValidCompositeModel(Model):
    """A Model subclass that owns an attribute of its own"""

    def __init__(self, implant, *, step=0.25):
        super().__init__(spatial=ValidSpatialModel(implant, step=step),
                         temporal=None)
        self.n_calls = 0
        self.history = []


def test_Model_subclass_constructor_owns_its_attributes():
    model = ValidCompositeModel(ArgusI())
    npt.assert_equal(model.n_calls, 0)
    npt.assert_equal('n_calls' in model.__dict__, True)
    npt.assert_equal(hasattr(model.spatial, 'n_calls'), False)
    # Nothing new may be added once the constructor is done:
    with pytest.raises(FreezeError):
        model.n_misses = 0


def test_Model_subclass_params_go_to_the_component_model():
    # Named-model constructor parameters are stored only on the component:
    model = ValidCompositeModel(ArgusI(), step=2.5)
    npt.assert_almost_equal(model.spatial.step, 2.5)
    npt.assert_equal('step' in model.__dict__, False)
    model.spatial.step = 0.5
    npt.assert_almost_equal(model.spatial.step, 0.5)
    npt.assert_equal('step' in model.__dict__, False)
    # Unknown constructor parameters are rejected:
    with pytest.raises(TypeError):
        ValidCompositeModel(ArgusI(), nonexistent=1)


def test_Model_build():
    # SpatialModel, but no TemporalModel:
    model = Model(spatial=ValidSpatialModel(ArgusI()))
    npt.assert_equal(model.is_built, False)
    model.build()
    npt.assert_equal(model.is_built, True)

    # TemporalModel, but no SpatialModel:
    model = Model(temporal=ValidTemporalModel())
    npt.assert_equal(model.is_built, False)
    model.build()
    npt.assert_equal(model.is_built, True)

    # SpatialModel and TemporalModel:
    model = Model(spatial=ValidSpatialModel(ArgusI()),
                  temporal=ValidTemporalModel())
    npt.assert_equal(model.is_built, False)
    model.build()
    npt.assert_equal(model.is_built, True)


def test_a_new_parameter_value_un_builds_the_model():
    """Which parameters a build depends on is not enumerated anywhere"""
    model = ValidSpatialModel(ArgusI(), step=1).build()
    # Assigning the value it already has changes nothing, so the build stands:
    model.step = model.step
    model.thresh_percept = 0
    npt.assert_equal(model.is_built, True)
    # A different value un-builds it, whether or not the grid depends on it:
    model.step = 2
    npt.assert_equal(model.is_built, False)
    model.build()
    model.thresh_percept = 0.5
    npt.assert_equal(model.is_built, False)
    # Attributes that are not parameters are not the build's business:
    model.build()
    model.grid = model.grid
    npt.assert_equal(model.is_built, True)


def test_a_composite_un_builds_the_stage_the_parameter_belongs_to():
    model = Model(spatial=ValidSpatialModel(ArgusI(), step=1),
                  temporal=ValidTemporalModel()).build()
    model.spatial.step = 2
    npt.assert_equal(model.spatial.is_built, False)
    npt.assert_equal(model.temporal.is_built, True)
    npt.assert_equal(model.is_built, False)


def test_predict_percept_rebuilds_only_the_stale_stage():
    """A sweep over one stage's parameter must not rebuild the other

    The spatial build is the expensive one -- growing an axon map, loading a
    pickle -- so a loop over a temporal parameter that quietly regrew it would
    cost far more than the sweep itself.
    """
    builds = {'spatial': 0, 'temporal': 0}

    class CountingSpatial(ValidSpatialModel):
        def _build(self):
            builds['spatial'] += 1

    class CountingTemporal(FadingTemporal):
        def _build(self):
            super()._build()
            builds['temporal'] += 1

    model = Model(spatial=CountingSpatial(ArgusI()),
                  temporal=CountingTemporal()).build()
    npt.assert_equal(builds, {'spatial': 1, 'temporal': 1})

    model.temporal.tau = 50
    model.predict_percept({'A1': np.ones(10)})
    npt.assert_equal(builds, {'spatial': 1, 'temporal': 2})

    model.spatial.step = 2
    model.predict_percept({'A1': np.ones(10)})
    npt.assert_equal(builds, {'spatial': 2, 'temporal': 2})

    # Nothing is stale, so predicting again builds nothing:
    model.predict_percept({'A1': np.ones(10)})
    npt.assert_equal(builds, {'spatial': 2, 'temporal': 2})

    # `build` stays the way to force a full rebuild:
    model.build()
    npt.assert_equal(builds, {'spatial': 3, 'temporal': 3})


@pytest.mark.parametrize('make_model', [
    lambda: ValidSpatialModel(ArgusI()),
    lambda: Model(spatial=ValidSpatialModel(ArgusI())),
    lambda: Model(spatial=ValidSpatialModel(ArgusI()),
                  temporal=FadingTemporal()),
])
def test_predict_percept_builds_what_it_needs(make_model):
    model = make_model()
    npt.assert_equal(model.is_built, False)
    npt.assert_equal(model.predict_percept({'A1': np.ones(10)}) is not None,
                     True)
    npt.assert_equal(model.is_built, True)


@pytest.mark.parametrize('ModelClass, ImplantType', [
    (ScoreboardModel, ArgusII), (AxonMapModel, ArgusII),
    (Thompson2003Model, ArgusII), (Nanduri2012Model, ArgusII),
    (BiphasicAxonMapModel, ArgusII), (CortexScoreboardModel, Cortivis),
    (DynaphosModel, Cortivis)])
def test_a_standalone_spatial_model_needs_an_implant(ModelClass, ImplantType):
    """Which device is modeled is not something to fill in later"""
    with pytest.raises(TypeError):
        ModelClass()
    with pytest.raises(TypeError):
        ModelClass(None)
    implant = ImplantType()
    for model in (ModelClass(implant), ModelClass(implant=implant)):
        npt.assert_equal(model.implant is implant, True)
        with pytest.raises(TypeError):
            model.implant = None


def test_a_temporal_only_model_takes_no_implant():
    """`Horsager2009Model` has no electrodes to place"""
    npt.assert_equal(Horsager2009Model().has_space, False)
    with pytest.raises(TypeError, match='implant'):
        Horsager2009Model(implant=ArgusII())


def test_Model_takes_a_bound_spatial_instance():
    """`Model` composes spatial components; it does not construct them"""
    with pytest.raises(TypeError, match='not the class itself'):
        Model(spatial=ValidSpatialModel, temporal=ValidTemporalModel())
    # `implant` is constructor context for the spatial model, not something
    # the composite takes:
    with pytest.raises(TypeError):
        Model(spatial=ValidSpatialModel(ArgusI()), implant=ArgusII())
    # Reading and rebinding delegate to the spatial model, and rebinding
    # invalidates its build:
    model = Model(spatial=ValidSpatialModel(ArgusI())).build()
    model.implant = ArgusII()
    npt.assert_equal(isinstance(model.implant, ArgusII), True)
    npt.assert_equal(model.spatial.implant is model.implant, True)
    npt.assert_equal(model.spatial.is_built, False)
    # A temporal-only model has no electrodes to place:
    temporal = Model(temporal=ValidTemporalModel())
    npt.assert_equal(temporal.implant, None)
    with pytest.raises(AttributeError):
        temporal.implant = ArgusII()


def test_Model_components_stay_swappable_but_valid():
    """Replacing a component is allowed; breaking the composite is not"""
    model = Model(ValidSpatialModel(ArgusI()), ValidTemporalModel())
    # Components may be replaced or removed when another remains:
    model.temporal = FadingTemporal(tau=50)
    npt.assert_almost_equal(model.temporal.tau, 50)
    model.temporal = None
    npt.assert_equal(model.has_time, False)
    # Each slot accepts only its corresponding component type:
    for value in (42, 'oops', ValidTemporalModel(), ValidSpatialModel):
        with pytest.raises(TypeError):
            model.spatial = value
    # At least one component is required:
    with pytest.raises(TypeError):
        model.spatial = None
    npt.assert_equal(model.has_space, True)


def test_Model_plots_its_spatial_model():
    """Plotting a model means plotting where it samples the visual field"""
    model = Model(ValidSpatialModel(ArgusI()), ValidTemporalModel())
    _, ax = plt.subplots()
    npt.assert_equal(model.plot(ax=ax) is ax, True)
    plt.close()
    with pytest.raises(ValueError):
        Model(temporal=ValidTemporalModel()).plot()


def test_Model_predict_percept():
    # Just the spatial model:
    model = Model(spatial=ValidSpatialModel(ArgusI())).build()
    npt.assert_equal(model.predict_percept(None), None)
    # Just the temporal model:
    model = Model(temporal=ValidTemporalModel()).build()
    npt.assert_equal(model.predict_percept(None), None)
    # Both spatial and temporal:

    # Invalid calls:
    model = Model(spatial=ValidSpatialModel(ArgusI()),
                  temporal=ValidTemporalModel())
    model.build()
    with pytest.raises(ValueError):
        # Cannot request t_percepts that are not multiples of dt:
        model.predict_percept({'A1': np.ones(16)}, t_percept=[0.1, 0.11])
    with pytest.raises(ValueError):
        # Has temporal model but stim.time is None:
        ValidTemporalModel().predict_percept(Stimulus(3))
    with pytest.raises(ValueError):
        # stim.time==None but requesting t_percept != None
        model.predict_percept(np.ones(16), t_percept=[0, 1, 2])


@pytest.mark.parametrize('fps', [29.97, 30, 24])
def test_Model_predict_percept_frame_clock(fps):
    # A stimulus that came out of an encoder knows the frame rate of the video
    # behind it, and that is the rate worth reporting a percept at: one percept
    # frame per video frame. The pulse train's own time points are far finer
    # and carry no extra picture, and the hardcoded 20 ms default has nothing
    # to do with the source.
    implant = ArgusI()
    vid = VideoStimulus(np.random.rand(4, 4, 6), metadata={'fps': fps})
    stim = AmplitudeEncoder(amp_range=(0, 50), freq=60).encode(
        vid, implant=implant)
    model = Model(temporal=ValidTemporalModel()).build()
    percept = model.predict_percept(stim)
    npt.assert_equal(percept.data.shape[-1], 6)
    # Evenly spaced, and on the model's dt grid -- 1000/29.97 ms is neither a
    # whole number of dt nor, if rounded point by point, evenly spaced:
    npt.assert_almost_equal(np.diff(percept.time),
                            np.diff(percept.time)[0])
    ratio = percept.time / model.temporal.dt
    npt.assert_allclose(ratio, np.round(ratio), atol=1e-3)
    # Close enough to the source's own frame rate to animate at it:
    npt.assert_almost_equal(frame_interval(percept.time), 1000.0 / fps,
                            decimal=1)
    # The spatial model hands the temporal one a Percept rather than a
    # Stimulus, so the frame clock has to survive that hop too. This leg needs
    # real models: `ValidTemporalModel` returns one row per electrode, not one
    # per grid point, so it cannot consume a spatial percept.
    both = Model(spatial=ScoreboardSpatial(implant, xrange=(-2, 2),
                                           yrange=(-2, 2), step=1),
                 temporal=FadingTemporal()).build()
    npt.assert_equal(both.predict_percept(stim).data.shape[-1], 6)
    # An explicit `t_percept` still wins:
    npt.assert_equal(
        model.predict_percept(stim, t_percept=[0, 1, 2]).data.shape[-1], 3)
    # ... and a stimulus that did not come from an encoder keeps the 20 ms
    # default it always had:
    plain = Stimulus(np.ones((16, 2)), time=[0, 100])
    npt.assert_almost_equal(model.predict_percept(plain).time,
                            np.arange(0, 101, 20))


def test_Model_predict_percept_frame_peak():
    # Electrical stimulation is pulsatile, so brightness rises and falls within
    # a video frame. Reporting the single instant a frame happened to end on
    # says more about where in the pulse cycle that instant fell than about the
    # frame: at 20 Hz against a 29.97 fps video the period (50 ms) and the
    # frame (33.37 ms) are incommensurate, so the sampling phase walks through
    # the cycle and neighbouring frames came out two orders of magnitude apart.
    implant = ArgusI()
    rng = np.random.default_rng(0)
    vid = VideoStimulus(rng.random((4, 4, 16)), metadata={'fps': 29.97})
    stim = AmplitudeEncoder(amp_range=(0, 50), freq=20).encode(
        vid, implant=implant)
    model = Model(temporal=FadingTemporal(tau=100)).build()
    peak = model.predict_percept(stim)
    # Same frames, but sampled only at the instant each one ends:
    at_end = model.predict_percept(stim, t_percept=peak.time)
    npt.assert_equal(peak.data.shape, at_end.data.shape)
    npt.assert_almost_equal(peak.time, at_end.time)

    # The default reports the peak each frame reached, so it is never below the
    # value at the instant the frame ended, and for most frames it is above it:
    npt.assert_array_less(at_end.data - 1e-6, peak.data)
    npt.assert_array_less(0.5, np.mean(peak.data > at_end.data + 1e-7))
    # ... which is what stops the frame-to-frame swing from being an artifact
    # of the sampling phase rather than a property of the video:
    swing = lambda d: d.max() / np.median(d)
    npt.assert_equal(swing(peak.data.max(axis=(0, 1))) <
                     swing(at_end.data.max(axis=(0, 1))), True)
    # A percept is still one frame per video frame, on an evenly spaced axis:
    npt.assert_equal(peak.data.shape[-1], 16)
    npt.assert_almost_equal(np.diff(peak.time), np.diff(peak.time)[0])

    # `reduce='last'` asks for the closing instant instead, which is what every
    # version before 0.10.0 reported:
    last = Model(temporal=FadingTemporal(tau=100, reduce='last')).build()
    npt.assert_array_equal(last.predict_percept(stim).data, at_end.data)


def test_Model_predict_percept_correctly_parallelizes():
    # setup and time spatial model with 1 thread
    one_thread_spatial = Model(
        spatial=ValidSpatialModel(ArgusI(), n_threads=1)).build()
    start_time_one_thread_spatial = time.perf_counter()
    one_thread_spatial.predict_percept(np.ones(16))
    one_thread_spatial_predict_time = time.perf_counter() - start_time_one_thread_spatial

    # setup and time spatial model with 2 threads
    two_thread_spatial = Model(
        spatial=ValidSpatialModel(ArgusI(), n_threads=2)).build()
    start_time_two_thread_spatial = time.perf_counter()
    two_thread_spatial.predict_percept(np.ones(16))
    two_threaded_spatial_predict_time = time.perf_counter() - start_time_two_thread_spatial

    # we expect roughly a linear decrease in time as thread count increases
    npt.assert_almost_equal(actual=two_threaded_spatial_predict_time, desired=one_thread_spatial_predict_time / 2, decimal=1e-5)

    # setup and time temporal model with 1 thread
    one_thread_temporal = Model(temporal=ValidTemporalModel(n_threads=1)).build()
    start_time_one_thread_temporal = time.perf_counter()
    one_thread_temporal.predict_percept(Stimulus(np.ones((16, 2)),
                                                  time=[0, 100]))
    one_thread_temporal_predict_time = time.perf_counter() - start_time_one_thread_temporal

    # setup and time temporal model with 2 threads
    two_thread_temporal = Model(temporal=ValidTemporalModel(n_threads=2)).build()
    start_time_two_thread_temporal = time.perf_counter()
    two_thread_temporal.predict_percept(Stimulus(np.ones((16, 2)),
                                                  time=[0, 100]))
    two_thread_temporal_predict_time = time.perf_counter() - start_time_two_thread_temporal

    # we expect roughly a linear decrease in time as thread count increases
    npt.assert_almost_equal(actual=two_thread_temporal_predict_time, desired=one_thread_temporal_predict_time / 2, decimal=1e-5)


def test_Model_deepcopy_memo():
    model = Model(spatial=ValidSpatialModel(ArgusI()))

    # Called directly, without a memo dict:
    copied = model.__deepcopy__()
    npt.assert_equal(isinstance(copied, Model), True)
    npt.assert_equal(id(copied) != id(model), True)

    # An object already in the memo is returned as-is, not re-copied:
    sentinel = 'already copied'
    npt.assert_equal(model.__deepcopy__({id(model): sentinel}), sentinel)

    # Shared references are copied once, not duplicated:
    shared = ValidSpatialModel(ArgusI())
    pair = copy.deepcopy({'a': shared, 'b': shared})
    npt.assert_equal(pair['a'] is pair['b'], True)


def test_SpatialModel_deepcopy_memo():
    model = ValidSpatialModel(ArgusI())
    npt.assert_equal(model.__deepcopy__() == model, True)
    sentinel = 'already copied'
    npt.assert_equal(model.__deepcopy__({id(model): sentinel}), sentinel)


def test_Model_eq_and_hash():
    model = Model(spatial=ValidSpatialModel(ArgusI()))

    # Identity short-circuit:
    npt.assert_equal(model == model, True)
    # Different type:
    npt.assert_equal(model == 'not a model', False)
    npt.assert_equal(model == ValidSpatialModel(ArgusI()), False)

    # Hashable, so models can go in sets/dicts:
    npt.assert_equal(isinstance(hash(model), int), True)
    npt.assert_equal(len({model, model}), 1)


def test_Model_pprint_params():
    both = Model(spatial=ValidSpatialModel(ArgusI()),
                 temporal=ValidTemporalModel())
    params = both._pprint_params()
    # Only the components are exposed at the Model level:
    npt.assert_equal(sorted(params), ['spatial', 'temporal'])
    npt.assert_equal(isinstance(str(both), str), True)


def test_Model_deepcopy_preserves_submodels_and_params():
    # A plain Model takes spatial/temporal as constructor arguments and does
    # not recreate them, so they must survive the copy:
    model = Model(spatial=ValidSpatialModel(ArgusI()),
                  temporal=ValidTemporalModel())
    copied = copy.deepcopy(model)
    npt.assert_equal(isinstance(copied.spatial, ValidSpatialModel), True)
    npt.assert_equal(isinstance(copied.temporal, ValidTemporalModel), True)
    npt.assert_equal(copied == model, True)
    # ... as real copies, not as shared references:
    npt.assert_equal(id(copied.spatial) != id(model.spatial), True)
    npt.assert_equal(id(copied.temporal) != id(model.temporal), True)

    # Deepcopy preserves component parameter values:
    model = Model(spatial=ValidSpatialModel(ArgusI(), step=5,
                                            thresh_percept=0.5))
    copied = copy.deepcopy(model)
    npt.assert_almost_equal(copied.spatial.step, 5)
    npt.assert_almost_equal(copied.spatial.thresh_percept, 0.5)
    npt.assert_equal(copied == model, True)

    # The copy is independent of the original:
    copied.spatial.step = 3
    npt.assert_almost_equal(model.spatial.step, 5)

    # The implant is context, and is shared rather than duplicated:
    npt.assert_equal(copied.implant is model.implant, True)

    # A built model stays built:
    built = Model(spatial=ValidSpatialModel(ArgusI(), step=5)).build()
    npt.assert_equal(copy.deepcopy(built).is_built, True)

    # A named model copies the same way, components and all:
    named = ValidCompositeModel(ArgusI(), step=5)
    named.n_calls = 3
    twin = copy.deepcopy(named)
    npt.assert_equal(isinstance(twin, ValidCompositeModel), True)
    npt.assert_almost_equal(twin.spatial.step, 5)
    npt.assert_equal(twin.n_calls, 3)
    npt.assert_equal(id(twin.spatial) != id(named.spatial), True)
    # Subclass-owned mutable state is deep-copied:
    twin.history.append('predict')
    npt.assert_equal(named.history, [])


def test_model_unit_contract():
    """Every model states the units its numerical implementation works in"""
    spatial = ScoreboardSpatial(implant=ArgusII(), xrange=(-2, 2), yrange=(-2, 2), step=1)
    temporal = FadingTemporal()
    for model in (spatial, temporal, Model(spatial=spatial),
                  Model(spatial=spatial, temporal=temporal)):
        npt.assert_equal(model.stimulus_unit, uA)
        npt.assert_equal(model.time_unit, ms)
    npt.assert_equal(spatial.space_unit, um)


def test_model_electrode_coords_follow_the_stimulus():
    """Coordinates come back per row of the stimulus, not per electrode

    A stimulus need not name every electrode of the implant, and a dict
    stimulus need not name them in array order, so the model looks the
    coordinates up by name rather than taking the array as it stands.
    """
    implant = ArgusII()
    model = ScoreboardSpatial(implant=implant, xrange=(-2, 2), yrange=(-2, 2),
                              step=1)
    # A subset, in an order that is not the array's:
    stim = Stimulus({'F10': 1, 'A1': 2, 'C5': 3})
    x, y, z = model._electrode_coords(implant.earray, stim)
    npt.assert_equal(len(x), len(stim.electrodes))
    npt.assert_almost_equal(x, [implant[e].x for e in stim.electrodes])
    npt.assert_almost_equal(y, [implant[e].y for e in stim.electrodes])
    npt.assert_almost_equal(z, [implant[e].z for e in stim.electrodes])
    # Contiguous float32, which is what the Cython kernels take:
    for arr in (x, y, z):
        npt.assert_equal(arr.dtype, np.float32)
        npt.assert_equal(arr.flags['C_CONTIGUOUS'], True)
    # End to end: a one-electrode stimulus lights up the same place whether it
    # is the only row or one of several.
    model.build()
    only = model.predict_percept({'F10': 3})
    among = model.predict_percept({'F10': 3, 'A1': 0, 'C5': 0})
    npt.assert_almost_equal(only.data, among.data)


def test_model_t_percept_units():
    """`t_percept` is a time, spelled however the caller likes"""
    implant = ArgusII()
    source = BiphasicPulseTrain(20, 50, 0.45, stim_dur=100)
    spatial = ScoreboardSpatial(implant=implant, xrange=(-2, 2),
                                yrange=(-2, 2), step=1).build()
    temporal = FadingTemporal().build()
    composite = Model(spatial=ScoreboardSpatial(implant, xrange=(-2, 2),
                                                yrange=(-2, 2), step=1),
                      temporal=FadingTemporal()).build()
    for model, stim in [(spatial, source),
                        (temporal, implant.prepare_stim(source)),
                        (composite, source)]:
        bare = model.predict_percept(stim, t_percept=[0, 20, 40])
        for spelling in ([0, 20, 40] * ms, np.array([0, .02, .04]) * s,
                         [0 * ms, 20000 * us, 0.04 * s]):
            unitful = model.predict_percept(stim, t_percept=spelling)
            npt.assert_allclose(unitful.data, bare.data, rtol=1e-12)
            npt.assert_allclose(unitful.time, [0, 20, 40], rtol=1e-12)
        with pytest.raises(DimensionMismatchError):
            model.predict_percept(stim, t_percept=[0, 20] * uA)
    # A single unitful time point, not just a list:
    single = spatial.predict_percept(source, t_percept=0.02 * s)
    npt.assert_allclose(single.time, [20], rtol=1e-12)


def test_model_requires_a_current_stimulus():
    """A model reads current, so a picture has to be encoded first

    Rejecting on *dimension*, not on unit: an amplitude spelled ``0.05 * mA``
    is already 50 uA by the time a model sees it, because `Stimulus`
    canonicalizes when it is built. Gray levels are a different quantity
    altogether, and reading them as microamps is the silent reinterpretation
    `stimulus_unit` exists to declare away.
    """
    img = ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4)))
    # `implant.prepare_stim(img)` is either encoded by the implant or refused
    # by it (see `Implant.stimulus_unit`), so the model-side guard is
    # reached through an implant that claims to deliver something else. Both
    # are needed: the implant one catches the call that was actually wrong,
    # and this one is what no model may be talked out of.
    class Projector(ArgusII):
        stimulus_unit = dimensionless

    implant = Projector(preprocess=False)
    spatial = ScoreboardSpatial(implant=implant, xrange=(-2, 2),
                                yrange=(-2, 2), step=1).build()
    temporal = FadingTemporal().build()
    composite = Model(spatial=ScoreboardSpatial(implant, xrange=(-2, 2),
                                                yrange=(-2, 2), step=1),
                      temporal=FadingTemporal()).build()
    with pytest.raises(DimensionMismatchError):
        ArgusII(preprocess=False, encoder=None).prepare_stim(img)
    projected = implant.prepare_stim(img)
    npt.assert_equal(projected.unit, dimensionless)
    for model in (spatial, composite):
        with pytest.raises(DimensionMismatchError) as excinfo:
            model.predict_percept(img)
        npt.assert_equal('electric current' in str(excinfo.value), True)
        npt.assert_equal('dimensionless' in str(excinfo.value), True)
    with pytest.raises(DimensionMismatchError):
        temporal.predict_percept(projected)

    # Encoded, it goes through:
    encoded = AmplitudeEncoder(amp_range=(0, 50)).encode(
        img, implant=ArgusII(raster=None))
    npt.assert_equal(encoded.unit, uA)
    argus = ArgusII(encoder=None, raster=None)
    for model in (ScoreboardSpatial(implant=argus, xrange=(-2, 2),
                                    yrange=(-2, 2), step=1).build(),
                  Model(spatial=ScoreboardSpatial(argus, xrange=(-2, 2),
                                                  yrange=(-2, 2), step=1),
                        temporal=FadingTemporal()).build()):
        npt.assert_equal(model.predict_percept(encoded) is None, False)

    # A current spelled in another unit is converted, not refused. `Stimulus`
    # has already done the converting, which is the point:
    npt.assert_equal(Stimulus([0.05 * mA]).unit, uA)
    npt.assert_allclose(Stimulus([0.05 * mA]).data, 50, rtol=1e-12)
    electrical = ScoreboardSpatial(implant=ArgusII(), xrange=(-2, 2),
                                   yrange=(-2, 2), step=1).build()
    bare = {'A1': BiphasicPulseTrain(20, 50, 0.45, stim_dur=100)}
    unitful = {'A1': BiphasicPulseTrain(20, 0.05 * mA, 0.45, stim_dur=100)}
    npt.assert_allclose(electrical.predict_percept(unitful).data,
                        electrical.predict_percept(bare).data, rtol=1e-12)

    # A Percept is brightness, not current, and is not checked -- it is what a
    # spatial model hands a temporal one:
    percept = electrical.predict_percept(bare)
    npt.assert_equal(temporal.predict_percept(percept) is None, False)


class RecordingSpatial(SpatialModel):
    """A spatial model that reports what crossed the numerical boundary"""
    stimulus_unit = uA
    space_unit = um

    def get_default_params(self):
        return {**super().get_default_params(), 'seen': None}

    def _predict_spatial(self, earray, stim):
        x, y, z = self._electrode_coords(earray, stim)
        self.seen = {'amp': self._stim_values(stim),
                     'time': self._stim_times(stim), 'x': x, 'y': y, 'z': z}
        return np.zeros((self.grid.x.size, stim.data.shape[1]),
                        dtype=np.float32)


class MilliSpatial(RecordingSpatial):
    """The same model, declaring milli-units instead"""
    stimulus_unit = mA
    space_unit = mm
    time_unit = s


def test_model_units_are_a_numerical_contract():
    """Declaring a unit has to deliver numbers in it, not just document one

    Every model p2p ships works in uA/um/ms and every conversion below is the
    identity for them, which is the point: the helpers are the boundary, so a
    model that declares something else gets something else instead of
    silently receiving microamps and being off by a thousand.
    """
    # A ramp, so that a time point picked out of it has an exact expected
    # value and neighbouring columns never coincide:
    ramp = Stimulus(np.arange(10, dtype=float).reshape((1, -1)),
                    electrodes=['A1'], time=np.arange(10, dtype=float))
    implant = ArgusII()
    canonical = RecordingSpatial(implant=implant, xrange=(-2, 2),
                                 yrange=(-2, 2), step=1).build()
    milli = MilliSpatial(implant=implant, xrange=(-2, 2), yrange=(-2, 2),
                         step=1).build()
    canonical.predict_percept(ramp)
    milli.predict_percept(ramp)
    a, m = canonical.seen, milli.seen

    # Stimulus values: uA in, mA out
    npt.assert_allclose(m['amp'], a['amp'] / 1000, rtol=1e-6)
    npt.assert_allclose(a['amp'].max(), 9, rtol=1e-6)
    npt.assert_allclose(m['amp'].max(), 0.009, rtol=1e-6)
    # Coordinates: um in, mm out
    for axis in ('x', 'y', 'z'):
        npt.assert_allclose(m[axis], a[axis] / 1000, rtol=1e-6)
    npt.assert_allclose(a['x'][0], implant['A1'].x, rtol=1e-6)
    npt.assert_allclose(m['x'][0], implant['A1'].x / 1000, rtol=1e-6)
    # Time: ms in, s out
    npt.assert_allclose(m['time'], a['time'] / 1000, rtol=1e-12)
    npt.assert_allclose(a['time'], ramp.time, rtol=0, atol=0)
    # ... and the canonical model really is the zero-conversion path:
    npt.assert_allclose(a['amp'], ramp.data, rtol=0, atol=0)

    # `t_percept` is read in the model's own unit, and converted back to the
    # stimulus' unit to index it, so 0.005 s and 5 ms pick the same sample:
    milli.predict_percept(ramp, t_percept=[0.0, 0.005])
    canonical.predict_percept(ramp, t_percept=[0.0, 5.0])
    # rtol at float32 precision: `Stimulus.data` is float32, so 0.005 mA is
    # only good to about seven digits however exact the conversion was.
    npt.assert_allclose(canonical.seen['amp'], [[0, 5]], rtol=1e-6)
    npt.assert_allclose(milli.seen['amp'], [[0, 0.005]], rtol=1e-6)
    # The percept is labelled in the model's unit, not in milliseconds:
    npt.assert_allclose(
        milli.predict_percept(ramp, t_percept=[0.0, 0.005]).time,
        [0.0, 0.005], rtol=1e-12)
    # A quantity is normalized into that unit too:
    npt.assert_allclose(
        milli.predict_percept(ramp, t_percept=[0 * ms, 5 * ms]).time,
        [0.0, 0.005], rtol=1e-12)

    # The dimension guard reads the declared unit: mA and uA are the same
    # dimension, so an ordinary stimulus is fine and a picture is not. An
    # implant with an encoder never carries one, so this needs an implant that
    # claims to deliver gray levels:
    class Projector(ArgusII):
        stimulus_unit = dimensionless

    projected = MilliSpatial(implant=Projector(preprocess=False),
                             xrange=(-2, 2), yrange=(-2, 2), step=1).build()
    with pytest.raises(DimensionMismatchError):
        projected.predict_percept(
            ImageStimulus(np.linspace(0, 1, 16).reshape((4, 4))))


class SecondSpatial(RecordingSpatial):
    """A spatial model that labels its percept in seconds"""
    time_unit = s


class MilliTemporal(TemporalModel):
    """A temporal model that reports what crossed the numerical boundary

    Milliseconds, which is the default and what every model p2p ships uses.
    Named for what it is so that the pairing with ``SecondSpatial`` reads.
    """
    time_unit = ms

    def get_default_params(self):
        return {**super().get_default_params(), 'seen': None}

    def _predict_temporal(self, stim, t_percept):
        self.seen = {'values': self._stim_values(stim),
                     'time': self._stim_times(stim)}
        n_space = int(np.prod(stim.data.shape[:-1]))
        return np.zeros((n_space, len(t_percept)), dtype=np.float32)


def test_percept_time_crosses_model_boundary():
    """A percept's time axis carries the unit it was written in

    Before this, a percept's time was bare numbers whose meaning was implied
    by whichever model happened to produce them -- so a spatial model counting
    in seconds handed a temporal model counting in milliseconds a time axis a
    thousand times too long, and neither of them could tell.
    """
    ramp = Stimulus(np.arange(21, dtype=float).reshape((1, -1)),
                    electrodes=['A1'], time=np.arange(21, dtype=float))
    spatial = SecondSpatial(implant=ArgusII(), xrange=(-2, 2), yrange=(-2, 2),
                            step=1).build()
    temporal = MilliTemporal().build()
    npt.assert_equal(spatial.time_unit, s)
    npt.assert_equal(temporal.time_unit, ms)

    # The spatial model labels its percept in its own unit, and `t_percept`
    # was read in that unit too:
    percept = spatial.predict_percept(ramp, t_percept=[0, .005, .010])
    npt.assert_equal(percept.time_unit, s)
    npt.assert_allclose(percept.time, [0, .005, .010], rtol=1e-12)
    npt.assert_allclose(percept.times(ms), [0, 5, 10], rtol=1e-12)
    # `times()` with no unit is the stored array itself, unconverted:
    npt.assert_allclose(percept.times(), percept.time, rtol=0, atol=0)
    npt.assert_equal(percept.time_quantity.unit, s)
    npt.assert_allclose(percept.time_quantity.to_value(ms), [0, 5, 10],
                        rtol=1e-12)

    # ... and the temporal model reads that percept in *its* unit. This is the
    # crossing: the kernel sees milliseconds, not the seconds it was labelled
    # with.
    temporal.predict_percept(percept, t_percept=[0, 5, 10])
    npt.assert_allclose(temporal.seen['time'], [0, 5, 10], rtol=1e-12)
    # Brightness has no unit and is handed over exactly as it stands:
    npt.assert_allclose(temporal.seen['values'], percept.data, rtol=0, atol=0)

    # The same crossing through a composite, which is where it really happens:
    model = Model(spatial=SecondSpatial(ArgusII(), xrange=(-2, 2),
                                        yrange=(-2, 2), step=1),
                  temporal=MilliTemporal()).build()
    # A composite reports the unit of the stage that reads `t_percept` and
    # writes the percept, which is the temporal model:
    npt.assert_equal(model.time_unit, ms)
    npt.assert_equal(model.spatial.time_unit, s)
    out = model.predict_percept(ramp, t_percept=[0, 5, 10])
    npt.assert_equal(out.time_unit, ms)
    npt.assert_allclose(out.time, [0, 5, 10], rtol=1e-12)
    # The spatial model ran at every stimulus time point, in seconds, and the
    # temporal model got those same instants back in milliseconds:
    npt.assert_allclose(model.temporal.seen['time'], ramp.time, rtol=1e-12)
    # Spelling `t_percept` unitfully changes nothing:
    npt.assert_allclose(
        model.predict_percept(ramp, t_percept=[0 * ms, 5 * ms,
                                               10 * ms]).time,
        [0, 5, 10], rtol=1e-12)
    npt.assert_allclose(
        model.predict_percept(ramp, t_percept=[0, .005 * s,
                                               .010 * s]).time,
        [0, 5, 10], rtol=1e-12)


def test_Model_units_follow_their_component():
    """A composite's declared units are the ones its numbers are really in"""
    # One component: its own.
    space_only = Model(spatial=MilliSpatial(ArgusII()))
    npt.assert_equal(space_only.stimulus_unit, mA)
    npt.assert_equal(space_only.space_unit, mm)
    npt.assert_equal(space_only.time_unit, s)
    time_only = Model(temporal=MilliTemporal())
    npt.assert_equal(time_only.stimulus_unit, uA)
    npt.assert_equal(time_only.time_unit, ms)
    # Both, disagreeing: the stimulus goes into the spatial model and the
    # percept comes out of the temporal one, so they are read off different
    # components rather than merged into the dict `__getattr__` would return.
    both = Model(spatial=MilliSpatial(ArgusII()), temporal=MilliTemporal())
    npt.assert_equal(both.stimulus_unit, mA)
    npt.assert_equal(both.space_unit, mm)
    npt.assert_equal(both.time_unit, ms)


def test_TemporalModel_default_frame_rate_is_50Hz():
    """The fallback output rate is a rate, not the number 20

    With no `t_percept` and no encoder frame clock to follow, a temporal model
    reports at 50 Hz. That is 20 ms, which used to be written down as the
    number 20 -- one frame every 20 *seconds* for a model counting in seconds.
    """
    # Cathodic, so that the stub kernel's all-zero output is not mistaken for
    # a polarity problem and warned about:
    stim = Stimulus(-np.ones((1, 3)), electrodes=['A1'], time=[0, 50, 100])
    milli = MilliTemporal().build()
    npt.assert_allclose(milli.predict_percept(stim).time,
                        [0, 20, 40, 60, 80, 100], rtol=1e-12)

    class SecondTemporal(MilliTemporal):
        time_unit = s

        def get_default_params(self):
            # dt in seconds too, so that `t_percept` still lands on its grid:
            return {**super().get_default_params(), 'dt': 5e-6}

    second = SecondTemporal().build()
    percept = second.predict_percept(stim)
    npt.assert_equal(percept.time_unit, s)
    npt.assert_allclose(percept.time, [0, .02, .04, .06, .08, .10], rtol=1e-9)
    # Same instants, same number of frames -- the rate is the same physical
    # rate however the model counts:
    npt.assert_allclose(percept.times(ms), milli.predict_percept(stim).time,
                        rtol=1e-9)

    # The clock also stops at the stimulus rather than past it. `arange`'s
    # half-open end used to be nudged by the literal 1 -- a whole millisecond,
    # enough to add a frame after the stimulus was over:
    ragged = Stimulus(-np.ones((1, 3)), electrodes=['A1'],
                      time=[0, 60, 119.5])
    late = milli.predict_percept(ragged)
    npt.assert_allclose(late.time, [0, 20, 40, 60, 80, 100], rtol=1e-12)
    npt.assert_equal(late.time[-1] <= ragged.time[-1], True)

    # The one exception, and it is deliberate: a stimulus shorter than a
    # single frame still gets one, so its time point does fall after the end
    # of the stimulus. Reporting a 10 ms pulse only at t=0 would describe it
    # before it had had any effect, and brightness outlives the stimulus that
    # caused it -- so the frame containing it is what is worth reporting.
    for dur in (5, 10, 20):
        brief = Stimulus(-np.ones((1, 3)), electrodes=['A1'],
                         time=[0, dur / 2, dur])
        percept = FadingTemporal().build().predict_percept(brief)
        npt.assert_allclose(percept.time, [0, 20], rtol=1e-12)
        # ... and the extra frame is not an empty one:
        npt.assert_equal(percept.data.ravel()[-1] > 0.1, True)
    # The floor is one *frame*, not one unit of anything, so it means the same
    # thing to a model counting in seconds:
    brief = Stimulus(-np.ones((1, 3)), electrodes=['A1'], time=[0, 5, 10])
    npt.assert_allclose(SecondTemporal().build().predict_percept(brief).time,
                        [0, 0.02], rtol=1e-9)


def test_spatial_model_reads_modulation_not_pulses():
    """Spatial models see modulation frames; temporal models see pulses.

    A pulse train says *when* current flows and a raster says which electrodes
    may flow together. Both are facts about time, and a model with no temporal
    component has no way to express either: handed the delivered train, it
    reports the stimulus one instant at a time, so an encoded image comes back
    as a sequence of raster slots rather than as the image. Argus II rasters
    six groups by default, which is exactly the case that showed it.
    """
    logo = LogoBVL()
    implant = ArgusII()
    delivered = implant.prepare_stim(logo)
    spatial = ScoreboardSpatial(implant=implant, xrange=(-12, 12),
                                yrange=(-8, 8), step=1).build()

    # One frame in, one frame out -- not one per pulse edge:
    percept = spatial.predict_percept(logo)
    npt.assert_equal(delivered.time.size > 50, True)
    npt.assert_equal(percept.data.shape[-1], 1)
    # ... and every electrode the image lights is lit in it, rather than the
    # one raster group that happened to be firing at the sampled instant:
    lit = delivered._spatial_view().data.ravel() > 0
    npt.assert_equal(lit.sum() > 10, True)
    groups = implant.raster.groups(implant.electrode_names)
    # No instant of the delivered train ever holds more than one group, so
    # anything above one is more than a raster slot's worth of picture:
    npt.assert_equal(len(np.unique(groups[lit])) > 1, True)
    for column in delivered.data.T:
        npt.assert_equal(np.unique(groups[column != 0]).size <= 1, True)
    # Which is what the percept says too: it is the same picture the
    # modulation asked for, run through the model.
    bare = ScoreboardSpatial(implant=ArgusII(encoder=None), xrange=(-12, 12),
                             yrange=(-8, 8), step=1).build()
    direct = bare.predict_percept(delivered._spatial_view())
    npt.assert_almost_equal(percept.data, direct.data)

    # A video reports one percept frame per *video* frame:
    with pytest.warns(UserWarning, match='deliver no pulse'):
        n_frames = spatial.predict_percept(BostonTrain()).data.shape[-1]
    npt.assert_equal(n_frames, 94)

    # A model with a temporal component is the opposite case: the pulses are
    # what it integrates, so it has to see them, and the spatial stage it is
    # built on must not quietly swap them out.
    seen = []

    class Recording(ScoreboardSpatial):
        def _predict_spatial(self, earray, stim):
            # A pulse train has cathodic phases in it; modulation amplitudes
            # are never negative. So the sign says which one arrived:
            seen.append(float(stim.data.min()))
            return super()._predict_spatial(earray, stim)

    both = Model(spatial=Recording(implant, xrange=(-12, 12), yrange=(-8, 8),
                                   step=1),
                 temporal=FadingTemporal(tau=100)).build()
    both.predict_percept(logo)
    npt.assert_array_less(seen[-1], 0)
    # ... and the caller's own picture is untouched by that:
    npt.assert_equal(implant.prepare_stim(logo)._has_spatial_view, True)
    # Spatial-only, the same model class reads the modulation instead:
    seen.clear()
    Recording(implant=implant, xrange=(-12, 12), yrange=(-8, 8),
              step=1).build().predict_percept(logo)
    npt.assert_array_less(-1e-12, seen[-1])

    # Nothing changes for a stimulus that was presented as current: there is
    # no modulation behind it, so there is nothing to prefer.
    plain = implant.prepare_stim(
        {'A1': BiphasicPulseTrain(20, 50, 0.45, stim_dur=100)})
    npt.assert_equal(plain._has_spatial_view, False)
    npt.assert_equal(spatial.predict_percept(plain).data.shape[-1],
                     plain.time.size)


def _blend_grid(step=0.5, extent=6):
    """A plain dva grid to hand `_blend_meridian`"""
    grid = Grid2D((-extent, extent), (-extent, extent), step=step)
    grid.build(Curcio1990Map())
    return grid


def _step_across(grid, meridian, n_time=1):
    """A response that jumps from 0 to 1 across one meridian

    Shaped the way `_predict_spatial` returns it: space x time, with the
    spatial axes flattened in the grid's own C order.
    """
    coord = grid.x if meridian == 'vertical' else grid.y
    resp = np.where(coord > 0, 1.0, 0.0).astype(np.float32)
    return np.repeat(resp.reshape(-1, 1), n_time, axis=1)


def _as_grid(resp, grid):
    """Undo the flattening, for reading rows and columns back out"""
    return resp.reshape(grid.x.shape + (-1,))


def _normal_profile(resp, grid, meridian):
    """The response normal to the meridian, sorted by distance"""
    out = _as_grid(resp, grid)
    if meridian == 'vertical':
        coord, profile = grid.x[0, :], out[out.shape[0] // 2, :, 0]
    else:
        coord, profile = grid.y[:, 0], out[:, out.shape[1] // 2, 0]
    order = np.argsort(coord)
    return coord[order], profile[order]


@pytest.mark.parametrize('meridian', ['vertical', 'horizontal'])
@pytest.mark.parametrize('width', [None, 0])
def test_blend_meridian_off(meridian, width):
    # Disabled blending returns the original object:
    grid = _blend_grid()
    resp = _step_across(grid, meridian)
    npt.assert_equal(_blend_meridian(resp, grid, meridian, width) is resp,
                     True)


@pytest.mark.parametrize('meridian', ['vertical', 'horizontal'])
def test_blend_meridian_smooths_the_seam(meridian):
    # A hard step across the meridian is what the blend exists to soften.
    grid = _blend_grid()
    resp = _step_across(grid, meridian)
    blended = _blend_meridian(resp, grid, meridian, 1.0)
    npt.assert_equal(blended.shape, resp.shape)
    npt.assert_equal(blended.dtype, resp.dtype)

    _, was = _normal_profile(resp, grid, meridian)
    _, profile = _normal_profile(blended, grid, meridian)
    npt.assert_almost_equal(np.abs(np.diff(was)).max(), 1.0)
    npt.assert_array_less(np.abs(np.diff(profile)).max(), 0.5)
    # Still monotonic, so the seam was smoothed rather than rippled over:
    npt.assert_array_less(-1e-6, np.diff(profile))
    # And the two ends are untouched, so nothing was globally blurred:
    npt.assert_allclose(profile[0], was[0], atol=1e-6)
    npt.assert_allclose(profile[-1], was[-1], atol=1e-6)


@pytest.mark.parametrize('meridian', ['vertical', 'horizontal'])
def test_blend_meridian_leaves_the_far_field_alone(meridian):
    # The weight is a Gaussian centered on the meridian, so several widths
    # away the response has to come back as it went in.
    grid = _blend_grid()
    rng = np.random.default_rng(42)
    resp = rng.random((grid.x.size, 3)).astype(np.float32)
    width = 0.5
    blended = _blend_meridian(resp, grid, meridian, width)
    dist = grid.x if meridian == 'vertical' else grid.y
    far = np.abs(dist).ravel() > 5 * width
    npt.assert_equal(np.any(far), True)
    npt.assert_allclose(blended[far], resp[far], atol=1e-6)
    # ...while the seam itself did move:
    near = np.abs(dist).ravel() < width
    npt.assert_array_less(1e-3, np.abs(blended[near] - resp[near]).max())


def test_blend_meridian_is_one_dimensional():
    # The blur runs normal to the meridian and nowhere else, so a step that
    # runs the *other* way is invisible to it.
    grid = _blend_grid()
    # A vertical blend smooths along x, so a step across y survives it:
    across_y = _step_across(grid, 'horizontal')
    npt.assert_array_equal(_blend_meridian(across_y, grid, 'vertical', 1.0),
                           across_y)
    # ...and a horizontal blend smooths along y, so a step across x survives:
    across_x = _step_across(grid, 'vertical')
    npt.assert_array_equal(_blend_meridian(across_x, grid, 'horizontal', 1.0),
                           across_x)


def test_blend_meridian_time_points_are_independent():
    # Each frame is blended on its own; nothing leaks along the time axis.
    grid = _blend_grid()
    rng = np.random.default_rng(0)
    frames = [rng.random((grid.x.size, 1)).astype(np.float32)
              for _ in range(4)]
    together = _blend_meridian(np.hstack(frames), grid, 'vertical', 0.8)
    for t, frame in enumerate(frames):
        alone = _blend_meridian(frame, grid, 'vertical', 0.8)
        npt.assert_allclose(together[:, [t]], alone, atol=1e-6)
    # A frame of zeros stays zero even sitting between bright ones:
    frames[2][:] = 0
    mixed = _blend_meridian(np.hstack(frames), grid, 'vertical', 0.8)
    npt.assert_array_equal(mixed[:, 2], 0)


def test_blend_meridian_is_a_distance_not_a_pixel_count():
    # A fixed dva width should be resolution-independent:
    at = np.linspace(-3, 3, 25)
    profiles = {}
    for step in (0.5, 0.25, 0.125):
        grid = _blend_grid(step=step)
        # A step edge that sits at x=0 on every grid, rather than half a
        # sample to one side of it: otherwise the three inputs differ before
        # any blending happens and the comparison measures that instead.
        resp = (0.5 * (np.sign(grid.x) + 1)).astype(np.float32).reshape(-1, 1)
        blended = _blend_meridian(resp, grid, 'vertical', 1.0)
        coord, profile = _normal_profile(blended, grid, 'vertical')
        profiles[step] = np.interp(at, coord, profile)
    # The coarsest grid samples the Gaussian with only two points per width,
    # so it is allowed a little more slack than the two finer ones:
    npt.assert_allclose(profiles[0.5], profiles[0.25], atol=0.02)
    npt.assert_allclose(profiles[0.25], profiles[0.125], atol=0.005)


@pytest.mark.parametrize('meridian', ['vertical', 'horizontal'])
@pytest.mark.parametrize('half', [(0, 6), (-6, 0)])
def test_blend_meridian_needs_both_sides(meridian, half):
    # One-sided grids have no meridian seam and are left unchanged:
    if meridian == 'vertical':
        grid = Grid2D(half, (-6, 6), step=0.5)
    else:
        grid = Grid2D((-6, 6), half, step=0.5)
    grid.build(Curcio1990Map())
    rng = np.random.default_rng(7)
    resp = rng.random((grid.x.size, 2)).astype(np.float32)
    npt.assert_equal(_blend_meridian(resp, grid, meridian, 1.0) is resp, True)
    # The same response on a grid that does straddle the meridian is blended,
    # so it is the one-sidedness doing this and not the width or the data:
    both = _blend_grid(step=0.5)
    resp = rng.random((both.x.size, 2)).astype(np.float32)
    npt.assert_equal(_blend_meridian(resp, both, meridian, 1.0) is resp, False)


def test_blend_meridian_keeps_precision():
    grid = _blend_grid()
    for dtype in (np.float32, np.float64):
        resp = _step_across(grid, 'vertical').astype(dtype)
        blended = _blend_meridian(resp, grid, 'vertical', 1.0)
        npt.assert_equal(blended.dtype, np.dtype(dtype))
    # Blending float32 and then widening agrees with blending in float64, so
    # the narrower working precision costs nothing that float32 could hold:
    resp = _step_across(grid, 'vertical')
    npt.assert_allclose(
        _blend_meridian(resp, grid, 'vertical', 1.0).astype(np.float64),
        _blend_meridian(resp.astype(np.float64), grid, 'vertical', 1.0),
        atol=1e-6)


def test_blend_meridian_bad_input():
    grid = _blend_grid()
    resp = _step_across(grid, 'vertical')
    with pytest.raises(ValueError):
        _blend_meridian(resp, grid, 'diagonal', 1.0)
    with pytest.raises(ValueError):
        _blend_meridian(resp, grid, 'vertical', -1.0)
    # A grid with a single sample normal to the meridian has nothing to blur
    # along, and says so by doing nothing:
    flat = Grid2D((0, 0), (-3, 3), step=0.5)
    flat.build(Curcio1990Map())
    thin = np.ones((flat.x.size, 1), dtype=np.float32)
    npt.assert_equal(_blend_meridian(thin, flat, 'vertical', 1.0) is thin,
                     True)


def test_postprocess_spatial_hook():
    # The hook is a no-op by default, and a model that overrides it sees the
    # finished response, at every requested time point.
    plain = ScoreboardSpatial(implant=ArgusII(), xrange=(-2, 2), yrange=(-2, 2), step=1).build()
    resp = np.arange(12, dtype=np.float32).reshape(4, 3)
    npt.assert_equal(plain._postprocess_spatial(resp) is resp, True)

    seen = []

    class Doubling(ScoreboardSpatial):
        def _postprocess_spatial(self, resp):
            seen.append(resp.shape)
            return 2 * resp

    source = Stimulus({'A5': [1, 2, 3], 'F7': [3, 2, 1]})
    doubled = Doubling(implant=ArgusII(), xrange=(-2, 2), yrange=(-2, 2),
                       step=1).build()
    expected = plain.predict_percept(source)
    got = doubled.predict_percept(source)
    npt.assert_equal(len(seen), 1)
    npt.assert_equal(seen[0], (plain.grid.x.size, expected.data.shape[-1]))
    npt.assert_allclose(got.data, 2 * expected.data)


def test_models_accept_read_only_stimulus_data():
    # Every Cython kernel receives `Stimulus.data` as it is stored, and a
    # stimulus stores it read-only. A memoryview that is not declared `const`
    # rejects such an array outright ("buffer source array is read-only"),
    # which is a failure no numerical test would catch on its own.
    from pulse2percept.models import Nanduri2012Spatial, Nanduri2012Temporal
    implant = ArgusII()
    source = {'A1': BiphasicPulseTrain(20, 50, 0.45, stim_dur=20)}
    stim = implant.prepare_stim(source)
    npt.assert_equal(stim.data.flags.writeable, False)
    spatial = Nanduri2012Spatial(implant=implant, xrange=(-1, 1),
                                 yrange=(-1, 1), step=1).build()
    resp = spatial.predict_percept(source)
    npt.assert_equal(np.all(np.isfinite(resp.data)), True)
    temporal = Nanduri2012Temporal().build()
    npt.assert_equal(temporal.predict_percept(stim) is not None, True)
    # And the same for the axon map / scoreboard kernels:
    for cls in (ScoreboardSpatial, AxonMapSpatial):
        model = cls(implant=implant, xrange=(-1, 1), yrange=(-1, 1),
                    step=1).build()
        npt.assert_equal(np.all(np.isfinite(
            model.predict_percept(source).data)), True)


@contextmanager
def _no_schedule_expansion():
    """Make expanding an encoder schedule into a waveform an error

    The only way to state "this path never needs the pulses" as a test: if
    anything reaches for them, it fails loudly.
    """
    from pulse2percept.stimuli.encoders import _EncodedStimulus
    original = _EncodedStimulus._render

    def refuse(self):
        raise AssertionError('expanded an encoder schedule')
    _EncodedStimulus._render = refuse
    try:
        yield
    finally:
        _EncodedStimulus._render = original


@pytest.mark.parametrize('source', ['image', 'video'])
def test_spatial_model_predicts_without_expanding_the_schedule(source):
    # A spatial model reads one amplitude per electrode per frame. Nothing
    # about that needs the pulse train the schedule would expand into.
    implant = ArgusII()
    spatial = ScoreboardSpatial(implant=implant, xrange=(-12, 12),
                                yrange=(-8, 8), step=1).build()
    picture = (LogoBVL() if source == 'image' else
               VideoStimulus(np.random.default_rng(0).random((6, 10, 4)),
                             metadata={'fps': 20}))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepared = implant.prepare_stim(picture)
    with _no_schedule_expansion():
        percept = spatial.predict_percept(prepared)
    npt.assert_equal(np.any(percept.data), True)
    npt.assert_equal(percept.data.shape[-1], 1 if source == 'image' else 4)


def test_combined_model_still_integrates_the_delivered_pulses():
    # The opposite case: a temporal stage integrates pulses, so the spatial
    # stage under it has to be handed them. That reading is unchanged, and the
    # implant it was asked of keeps its own schedule.
    implant = ArgusII()
    delivered = implant.prepare_stim(LogoBVL())
    seen = []

    class Recording(ScoreboardSpatial):
        def _predict_spatial(self, earray, stim):
            # A pulse train has cathodic phases in it; modulation amplitudes
            # never do. The sign says which one arrived:
            seen.append(float(stim.data.min()))
            return super()._predict_spatial(earray, stim)

    both = Model(spatial=Recording(implant, xrange=(-12, 12), yrange=(-8, 8),
                                   step=1),
                 temporal=FadingTemporal(tau=100)).build()
    both.predict_percept(LogoBVL())
    npt.assert_array_less(seen[-1], 0)
    # Stripping the modulation view happens on a stand-in, so the prepared
    # stimulus the caller holds keeps its schedule:
    npt.assert_equal(delivered._has_spatial_view, True)


def test_deactivating_an_encoded_electrode_keeps_the_schedule():
    implant = ArgusII()
    logo = LogoBVL()
    before = implant.prepare_stim(logo)._spatial_view()
    with _no_schedule_expansion():
        implant.deactivate(['A1', 'B2'])
        stim = implant.prepare_stim(logo)
        after = stim._spatial_view()
        npt.assert_equal(stim._has_spatial_view, True)
        npt.assert_equal(len(stim.electrodes), 58)
    keep = [i for i, e in enumerate(before.electrodes)
            if str(e) not in ('A1', 'B2')]
    npt.assert_array_equal(after.data, before.data[keep])
    # The waveform is still there to be had, and matches too:
    npt.assert_equal(stim.data.shape[0], 58)
