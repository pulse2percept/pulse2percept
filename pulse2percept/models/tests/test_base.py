import numpy as np
import pytest
import numpy.testing as npt
from matplotlib.axes import Subplot

from pulse2percept.implants import ArgusI
from pulse2percept.stimuli import Stimulus
from pulse2percept.percepts import Percept
from pulse2percept.models import (BaseModel, Model, NotBuiltError,
                                  SpatialModel, TemporalModel)
from pulse2percept.utils import FreezeError, Grid2D


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

    def dva2ret(self, dva):
        return dva

    def ret2dva(self, ret):
        return ret

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
    npt.assert_equal(isinstance(model.grid.xret, np.ndarray), True)

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
