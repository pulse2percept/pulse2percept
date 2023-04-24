import copy

import numpy as np
import pytest
import numpy.testing as npt
from matplotlib.axes import Subplot
import time

from pulse2percept.implants import ArgusI
from pulse2percept.stimuli import Stimulus
from pulse2percept.percepts import Percept
from pulse2percept.models import (BaseModel, Model, NotBuiltError,
                                  SpatialModel, TemporalModel)
from pulse2percept.utils import FreezeError
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
