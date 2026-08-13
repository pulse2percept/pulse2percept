import copy
import multiprocessing
import warnings

import numpy as np
import pytest
import numpy.testing as npt
from matplotlib.axes import Subplot
import time

from pulse2percept.implants import ArgusI
from pulse2percept.stimuli import AmplitudeEncoder, Stimulus, VideoStimulus
from pulse2percept.percepts import Percept
from pulse2percept.models import (BaseModel, FadingTemporal, Model,
                                  NotBuiltError, ScoreboardSpatial,
                                  SpatialModel, TemporalModel)
from pulse2percept.utils import FreezeError, frame_interval
from pulse2percept.topography import Grid2D, Watson2014Map


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

    def get_default_params(self):
        params = super(ValidSpatialModel, self).get_default_params()
        params.update({'vfmap': Watson2014Map()})
        return params

    def _predict_spatial(self, earray, stim):
        if not self.is_built:
            raise NotBuiltError
        n_time = 1 if stim.time is None else stim.time.size
        return np.zeros((self.grid.x.size, n_time), dtype=np.float32)


def test_SpatialModel():
    # Build grid:
    model = ValidSpatialModel()
    npt.assert_equal(model.grid, None)
    npt.assert_equal(model.is_built, False)
    model.build()
    npt.assert_equal(model.is_built, True)
    npt.assert_equal(isinstance(model.grid, Grid2D), True)
    npt.assert_equal(isinstance(model.grid.ret.x, np.ndarray), True)

    # Can overwrite default values:
    model = ValidSpatialModel(xystep=1.234)
    npt.assert_almost_equal(model.xystep, 1.234)
    model.build(xystep=2.345)
    npt.assert_almost_equal(model.xystep, 2.345)

    # Cannot add more attributes:
    with pytest.raises(AttributeError):
        ValidSpatialModel(newparam=1)
    with pytest.raises(FreezeError):
        model.newparam = 1

    # Returns Percept object of proper size:
    npt.assert_equal(model.predict_percept(ArgusI()), None)
    for stim in [np.ones(16), np.zeros(16), {'A1': 2}, np.ones((16, 2))]:
        implant = ArgusI(stim=stim)
        percept = model.predict_percept(implant)
        npt.assert_equal(isinstance(percept, Percept), True)
        n_time = 1 if implant.stim.time is None else len(implant.stim.time)
        npt.assert_equal(percept.shape, (model.grid.x.shape[0],
                                         model.grid.x.shape[1],
                                         n_time))
        npt.assert_almost_equal(percept.data, 0)

    # Invalid calls:
    with pytest.raises(ValueError):
        # stim.time==None but requesting t_percept != None
        implant.stim = np.ones(16)
        model.predict_percept(implant, t_percept=[0, 1, 2])
    with pytest.raises(NotBuiltError):
        # must call build first
        model = ValidSpatialModel()
        model.predict_percept(ArgusI())
    with pytest.raises(TypeError):
        # must pass an implant
        ValidSpatialModel().build().predict_percept(Stimulus(3))


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

    model = RecordingSpatialModel(xystep=2).build()
    # Amplitudes chosen so that sorting the frames by value shuffles them with
    # respect to time (sorted: 1, 2, 3 -> frames 1, 2, 0):
    implant = ArgusI(stim={'A1': [3, 1, 2]})
    with warnings.catch_warnings():
        # A shuffled time axis makes Stimulus warn:
        warnings.simplefilter('error', UserWarning)
        percept = model.predict_percept(implant)

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

    model = CountingSpatialModel(xystep=2).build()
    # Four time points, but only two distinct frames:
    implant = ArgusI(stim={'A1': [2, 5, 2, 5]})
    percept = model.predict_percept(implant)
    # _predict_spatial was called once, on the two unique frames only:
    npt.assert_equal(n_calls, [2])
    npt.assert_almost_equal(percept.time, [0, 1, 2, 3])
    for idx, amp in enumerate([2, 5, 2, 5]):
        npt.assert_almost_equal(percept.data[..., idx], amp)


@pytest.mark.parametrize('param, value', [('engine', 'serial'),
                                          ('scheduler', 'dask')])
def test_SpatialModel_removed_params(param, value):
    # `engine` chose the Cython vs pure-Python axon-growth path and
    # `scheduler` drove the joblib/dask backends. Both were deprecated in
    # 0.9.1 and removed in 0.10.0, so they are now unknown parameters:
    with pytest.raises(AttributeError):
        ValidSpatialModel(**{param: value})
    with pytest.raises(AttributeError):
        ValidSpatialModel().set_params(**{param: value})


@pytest.mark.parametrize('param, value', [('engine', 'serial'),
                                          ('scheduler', 'dask')])
def test_Model_removed_params(param, value):
    # A Model built from instances never reaches BaseModel.__init__, so this
    # path needs checking separately:
    with pytest.raises(AttributeError):
        Model(spatial=ValidSpatialModel(), **{param: value})
    with pytest.raises(AttributeError):
        Model(spatial=ValidSpatialModel()).set_params({param: value})


def test_eq_SpatialModel():
    valid = ValidSpatialModel()

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
    differing_model = ValidSpatialModel(xrange=(-10, 10))
    npt.assert_equal(valid != differing_model, True)

def test_deepcopy_SpatialModel():
    original = ValidSpatialModel()
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
    model = ValidSpatialModel()
    model.build()
    # Simulated area might be larger than that:
    model = ValidSpatialModel(xrange=(-20.5, 20.5), yrange=(-16.1, 16.1))
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
        if not self.is_built:
            raise NotBuiltError
        return np.zeros((stim.data.shape[0], len(t_percept)), dtype=np.float32)


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
    npt.assert_equal(model.predict_percept(ArgusI().stim), None)
    model.dt = 1
    for stim in [np.ones((16, 3)), np.zeros((16, 3)),
                 {'A1': [1, 2]}, np.ones((16, 2))]:
        implant = ArgusI(stim=stim)
        percept = model.predict_percept(implant.stim)
        # By default, percept is output every 20ms. If stimulus is too short,
        # output at t=[0, 20]. This is mentioned in the docs - for really short
        # stimuli, users should specify the desired time points manually.
        n_time = 1 if implant.stim.time is None else 2
        npt.assert_equal(percept.shape, (implant.stim.shape[0], 1, n_time))
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
    with pytest.raises(NotBuiltError):
        # Must call build first:
        ValidTemporalModel().predict_percept(Stimulus(3))
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
    differing_model = ValidSpatialModel(xrange=(-10, 10))
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


@pytest.mark.parametrize('cls', [ValidSpatialModel, ValidTemporalModel])
def test_n_jobs_aliases_n_threads(cls):
    # `n_jobs` and `n_threads` are two names for the OpenMP thread count, and
    # must never disagree:
    model = cls()
    npt.assert_equal(model.n_jobs, model.n_threads)
    # Setting either name moves both, whether in the constructor...
    model = cls(n_jobs=3)
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
    npt.assert_equal(cls().n_threads, multiprocessing.cpu_count())
    # None and -1 both mean "every core", following scikit-learn:
    npt.assert_equal(cls(n_jobs=None).n_threads, multiprocessing.cpu_count())
    npt.assert_equal(cls(n_jobs=-1).n_threads, multiprocessing.cpu_count())
    # Nonsense is rejected rather than silently ignored:
    for bad in (0, -2, 2.5, 'many'):
        with pytest.raises(ValueError):
            cls(n_jobs=bad)
    # It is an alias, not a deprecation -- it must not warn:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cls(n_jobs=4)


def test_Model_n_jobs_aliases_n_threads():
    # Through a Model, n_jobs has to reach both sub-models:
    model = Model(spatial=ValidSpatialModel(), temporal=ValidTemporalModel(),
                  n_jobs=3)
    npt.assert_equal(model.spatial.n_threads, 3)
    npt.assert_equal(model.temporal.n_threads, 3)
    model.n_jobs = 6
    npt.assert_equal(model.spatial.n_threads, 6)
    npt.assert_equal(model.temporal.n_threads, 6)


def test_Model():
    # A None Model:
    model = Model()
    npt.assert_equal(model.has_space, False)
    npt.assert_equal(model.has_time, False)
    npt.assert_equal(str(model), "Model(spatial=None, temporal=None)")

    # Cannot add attributes outside the constructor:
    with pytest.raises(AttributeError):
        model.a
    with pytest.raises(FreezeError):
        model.a = 1

    # Wrong model type:
    with pytest.raises(TypeError):
        Model(spatial=ValidTemporalModel())
    with pytest.raises(TypeError):
        Model(temporal=ValidSpatialModel())

    # SpatialModel, but no TemporalModel:
    model = Model(spatial=ValidSpatialModel())
    npt.assert_equal(model.has_space, True)
    npt.assert_equal(model.has_time, False)
    npt.assert_almost_equal(model.xystep, 0.25)
    npt.assert_almost_equal(model.spatial.xystep, 0.25)
    model.xystep = 2
    npt.assert_almost_equal(model.xystep, 2)
    npt.assert_almost_equal(model.spatial.xystep, 2)
    # Cannot add more attributes:
    with pytest.raises(AttributeError):
        model.a
    with pytest.raises(FreezeError):
        model.a = 1

    # TemporalModel, but no SpatialModel:
    model = Model(temporal=ValidTemporalModel())
    npt.assert_equal(model.has_space, False)
    npt.assert_equal(model.has_time, True)
    npt.assert_almost_equal(model.dt, 5e-3)
    npt.assert_almost_equal(model.temporal.dt, 5e-3)
    model.dt = 1
    npt.assert_almost_equal(model.dt, 1)
    npt.assert_almost_equal(model.temporal.dt, 1)
    # Cannot add more attributes:
    with pytest.raises(AttributeError):
        model.a
    with pytest.raises(FreezeError):
        model.a = 1

    # SpatialModel and TemporalModel:
    model = Model(spatial=ValidSpatialModel(), temporal=ValidTemporalModel())
    npt.assert_equal(model.has_space, True)
    npt.assert_equal(model.has_time, True)
    npt.assert_almost_equal(model.xystep, 0.25)
    npt.assert_almost_equal(model.spatial.xystep, 0.25)
    npt.assert_almost_equal(model.dt, 5e-3)
    npt.assert_almost_equal(model.temporal.dt, 5e-3)
    # Setting a new spatial parameter:
    model.xystep = 2
    npt.assert_almost_equal(model.xystep, 2)
    npt.assert_almost_equal(model.spatial.xystep, 2)
    # Setting a new temporal parameter:
    model.dt = 1
    npt.assert_almost_equal(model.dt, 1)
    npt.assert_almost_equal(model.temporal.dt, 1)
    # Setting a parameter that's part of both spatial/temporal:
    npt.assert_equal(model.thresh_percept, {'spatial': 0, 'temporal': 0})
    model.thresh_percept = 1.234
    npt.assert_almost_equal(model.spatial.thresh_percept, 1.234)
    npt.assert_almost_equal(model.temporal.thresh_percept, 1.234)
    # Cannot add more attributes:
    with pytest.raises(AttributeError):
        model.a
    with pytest.raises(FreezeError):
        model.a = 1


def test_Model_set_params():
    # SpatialModel, but no TemporalModel:
    model = Model(spatial=ValidSpatialModel())
    model.set_params({'xystep': 2.33})
    npt.assert_almost_equal(model.xystep, 2.33)
    npt.assert_almost_equal(model.spatial.xystep, 2.33)

    # TemporalModel, but no SpatialModel:
    model = Model(temporal=ValidTemporalModel())
    model.set_params({'dt': 2.33})
    npt.assert_almost_equal(model.dt, 2.33)
    npt.assert_almost_equal(model.temporal.dt, 2.33)

    # SpatialModel and TemporalModel:
    model = Model(spatial=ValidSpatialModel(), temporal=ValidTemporalModel())
    # Setting both using the convenience function:
    model.set_params({'xystep': 5, 'dt': 2.33})
    npt.assert_almost_equal(model.xystep, 5)
    npt.assert_almost_equal(model.spatial.xystep, 5)
    npt.assert_equal(hasattr(model.temporal, 'xystep'), False)
    npt.assert_almost_equal(model.dt, 2.33)
    npt.assert_almost_equal(model.temporal.dt, 2.33)
    npt.assert_equal(hasattr(model.spatial, 'dt'), False)


def test_Model_build():
    # A None model:
    model = Model()
    # Nothing to build, so `is_built` is always True (we want to be able to
    # call `predict_percept`):
    npt.assert_equal(model.is_built, True)
    model.build()
    npt.assert_equal(model.is_built, True)

    # SpatialModel, but no TemporalModel:
    model = Model(spatial=ValidSpatialModel())
    npt.assert_equal(model.is_built, False)
    model.build()
    npt.assert_equal(model.is_built, True)

    # TemporalModel, but no SpatialModel:
    model = Model(temporal=ValidTemporalModel())
    npt.assert_equal(model.is_built, False)
    model.build()
    npt.assert_equal(model.is_built, True)

    # SpatialModel and TemporalModel:
    model = Model(spatial=ValidSpatialModel(), temporal=ValidTemporalModel())
    npt.assert_equal(model.is_built, False)
    model.build()
    npt.assert_equal(model.is_built, True)


def test_Model_predict_percept():
    # A None Model has nothing to build, nothing to perceive:
    model = Model()
    npt.assert_equal(model.predict_percept(ArgusI()), None)
    npt.assert_equal(model.predict_percept(ArgusI(stim={'A1': 1})), None)
    npt.assert_equal(model.predict_percept(ArgusI(stim={'A1': 1}),
                                           t_percept=[0, 1]), None)

    # Just the spatial model:
    model = Model(spatial=ValidSpatialModel()).build()
    npt.assert_equal(model.predict_percept(ArgusI()), None)
    # Just the temporal model:
    model = Model(temporal=ValidTemporalModel()).build()
    npt.assert_equal(model.predict_percept(ArgusI()), None)
    # Both spatial and temporal:

    # Invalid calls:
    model = Model(spatial=ValidSpatialModel(), temporal=ValidTemporalModel())
    with pytest.raises(NotBuiltError):
        # Must call build first:
        model.predict_percept(ArgusI())
    model.build()
    with pytest.raises(ValueError):
        # Cannot request t_percepts that are not multiples of dt:
        model.predict_percept(ArgusI(stim={'A1': np.ones(16)}),
                              t_percept=[0.1, 0.11])
    with pytest.raises(ValueError):
        # Has temporal model but stim.time is None:
        ValidTemporalModel().predict_percept(Stimulus(3))
    with pytest.raises(ValueError):
        # stim.time==None but requesting t_percept != None
        model.predict_percept(ArgusI(stim=np.ones(16)),
                              t_percept=[0, 1, 2])
    with pytest.raises(TypeError):
        # Must pass an implant:
        model.predict_percept(Stimulus(3))


@pytest.mark.parametrize('fps', [29.97, 30, 24])
def test_Model_predict_percept_frame_clock(fps):
    # A stimulus that came out of an encoder knows the frame rate of the video
    # behind it, and that is the rate worth reporting a percept at: one percept
    # frame per video frame. The pulse train's own time points are far finer
    # and carry no extra picture, and the hardcoded 20 ms default has nothing
    # to do with the source.
    implant = ArgusI()
    vid = VideoStimulus(np.random.rand(4, 4, 6), metadata={'fps': fps})
    implant.stim = AmplitudeEncoder(implant, amp_range=(0, 50),
                                    freq=60).encode(vid)
    model = Model(temporal=ValidTemporalModel()).build()
    percept = model.predict_percept(implant)
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
    both = Model(spatial=ScoreboardSpatial(xrange=(-2, 2), yrange=(-2, 2),
                                           xystep=1),
                 temporal=FadingTemporal()).build()
    npt.assert_equal(both.predict_percept(implant).data.shape[-1], 6)
    # An explicit `t_percept` still wins:
    npt.assert_equal(
        model.predict_percept(implant, t_percept=[0, 1, 2]).data.shape[-1], 3)
    # ... and a stimulus that did not come from an encoder keeps the 20 ms
    # default it always had:
    implant.stim = Stimulus(np.ones((16, 2)), time=[0, 100])
    npt.assert_almost_equal(model.predict_percept(implant).time,
                            np.arange(0, 101, 20))


def test_Model_predict_percept_correctly_parallelizes():
    # setup and time spatial model with 1 thread
    one_thread_spatial = Model(spatial=ValidSpatialModel(n_threads=1)).build()
    start_time_one_thread_spatial = time.perf_counter()
    one_thread_spatial.predict_percept(ArgusI())
    one_thread_spatial_predict_time = time.perf_counter() - start_time_one_thread_spatial

    # setup and time spatial model with 2 threads
    two_thread_spatial = Model(spatial=ValidSpatialModel(n_threads=2)).build()
    start_time_two_thread_spatial = time.perf_counter()
    two_thread_spatial.predict_percept(ArgusI())
    two_threaded_spatial_predict_time = time.perf_counter() - start_time_two_thread_spatial

    # we expect roughly a linear decrease in time as thread count increases
    npt.assert_almost_equal(actual=two_threaded_spatial_predict_time, desired=one_thread_spatial_predict_time / 2, decimal=1e-5)

    # setup and time temporal model with 1 thread
    one_thread_temporal = Model(temporal=ValidTemporalModel(n_threads=1)).build()
    start_time_one_thread_temporal = time.perf_counter()
    one_thread_temporal.predict_percept(ArgusI())
    one_thread_temporal_predict_time = time.perf_counter() - start_time_one_thread_temporal

    # setup and time temporal model with 2 threads
    two_thread_temporal = Model(temporal=ValidTemporalModel(n_threads=2)).build()
    start_time_two_thread_temporal = time.perf_counter()
    two_thread_temporal.predict_percept(ArgusI())
    two_thread_temporal_predict_time = time.perf_counter() - start_time_two_thread_temporal

    # we expect roughly a linear decrease in time as thread count increases
    npt.assert_almost_equal(actual=two_thread_temporal_predict_time, desired=one_thread_temporal_predict_time / 2, decimal=1e-5)


class ScalingSpatialModel(ValidSpatialModel):
    """Spatial model whose brightness equals the stimulus amplitude.

    `find_threshold` bisects on amplitude, so a model with a known,
    monotonic amplitude-to-brightness mapping gives a predictable threshold.
    """

    def _predict_spatial(self, earray, stim):
        if not self.is_built:
            raise NotBuiltError
        n_time = 1 if stim.time is None else stim.time.size
        out = np.zeros((self.grid.x.size, n_time), dtype=np.float32)
        out[:] = np.abs(stim.data).max()
        return out


class ScalingTemporalModel(ValidTemporalModel):
    """Temporal model whose brightness equals the stimulus amplitude."""

    def _predict_temporal(self, stim, t_percept):
        if not self.is_built:
            raise NotBuiltError
        out = np.zeros((stim.data.shape[0], len(t_percept)), dtype=np.float32)
        out[:] = np.abs(stim.data).max()
        return out


def test_SpatialModel_find_threshold():
    model = ScalingSpatialModel(xystep=5).build()
    implant = ArgusI(stim={'A1': 1})

    # Brightness equals amplitude, so the threshold is the target brightness:
    npt.assert_almost_equal(model.find_threshold(implant, 20), 20, decimal=0)
    npt.assert_almost_equal(model.find_threshold(implant, 55), 55, decimal=0)

    # `implant` must be a ProsthesisSystem:
    with pytest.raises(TypeError):
        model.find_threshold(Stimulus({'A1': 1}), 20)


def test_TemporalModel_find_threshold():
    model = ScalingTemporalModel().build()
    stim = Stimulus({'A1': 1})

    npt.assert_almost_equal(model.find_threshold(stim, 20), 20, decimal=0)

    # `stim` must be a Stimulus:
    with pytest.raises(TypeError):
        model.find_threshold(ArgusI(stim={'A1': 1}), 20)


def test_Model_find_threshold():
    model = Model(spatial=ScalingSpatialModel(xystep=5)).build()
    implant = ArgusI(stim={'A1': 1})

    npt.assert_almost_equal(model.find_threshold(implant, 20), 20, decimal=0)

    # `implant` must be a ProsthesisSystem:
    with pytest.raises(TypeError):
        model.find_threshold(Stimulus({'A1': 1}), 20)


def test_Model_deepcopy_memo():
    model = Model(spatial=ValidSpatialModel())

    # Called directly, without a memo dict:
    copied = model.__deepcopy__()
    npt.assert_equal(isinstance(copied, Model), True)
    npt.assert_equal(id(copied) != id(model), True)

    # An object already in the memo is returned as-is, not re-copied:
    sentinel = 'already copied'
    npt.assert_equal(model.__deepcopy__({id(model): sentinel}), sentinel)

    # Shared references are copied once, not duplicated:
    shared = ValidSpatialModel()
    pair = copy.deepcopy({'a': shared, 'b': shared})
    npt.assert_equal(pair['a'] is pair['b'], True)


def test_SpatialModel_deepcopy_memo():
    model = ValidSpatialModel()
    npt.assert_equal(model.__deepcopy__() == model, True)
    sentinel = 'already copied'
    npt.assert_equal(model.__deepcopy__({id(model): sentinel}), sentinel)


def test_Model_eq_and_hash():
    model = Model(spatial=ValidSpatialModel())

    # Identity short-circuit:
    npt.assert_equal(model == model, True)
    # Different type:
    npt.assert_equal(model == 'not a model', False)
    npt.assert_equal(model == ValidSpatialModel(), False)

    # Hashable, so models can go in sets/dicts:
    npt.assert_equal(isinstance(hash(model), int), True)
    npt.assert_equal(len({model, model}), 1)


def test_Model_pprint_params():
    # Covers both the spatial and the temporal branch of _pprint_params:
    both = Model(spatial=ValidSpatialModel(), temporal=ValidTemporalModel())
    params = both._pprint_params()
    npt.assert_equal('spatial' in params, True)
    npt.assert_equal('temporal' in params, True)
    # Parameters of the sub-models are pulled up:
    npt.assert_equal('xystep' in params, True)
    npt.assert_equal('dt' in params, True)
    npt.assert_equal(isinstance(str(both), str), True)


def test_Model_deepcopy_preserves_submodels_and_params():
    # A plain Model takes spatial/temporal as constructor arguments and does
    # not recreate them, so they must survive the copy:
    model = Model(spatial=ValidSpatialModel(), temporal=ValidTemporalModel())
    copied = copy.deepcopy(model)
    npt.assert_equal(isinstance(copied.spatial, ValidSpatialModel), True)
    npt.assert_equal(isinstance(copied.temporal, ValidTemporalModel), True)
    npt.assert_equal(copied == model, True)
    # ... as real copies, not as shared references:
    npt.assert_equal(id(copied.spatial) != id(model.spatial), True)
    npt.assert_equal(id(copied.temporal) != id(model.temporal), True)

    # Model parameters are forwarded to the sub-models, so they live in
    # `spatial`/`temporal` rather than in `Model.__dict__`. Rebuilding the
    # copy from the constructor alone would silently reset them to defaults:
    model = Model(spatial=ValidSpatialModel(xystep=5, thresh_percept=0.5))
    copied = copy.deepcopy(model)
    npt.assert_almost_equal(copied.xystep, 5)
    npt.assert_almost_equal(copied.thresh_percept, 0.5)
    npt.assert_equal(copied == model, True)

    # The copy is independent of the original:
    copied.xystep = 3
    npt.assert_almost_equal(model.xystep, 5)

    # A built model stays built:
    built = Model(spatial=ValidSpatialModel(xystep=5)).build()
    npt.assert_equal(copy.deepcopy(built).is_built, True)
