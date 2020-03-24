import numpy as np
import pytest
import numpy.testing as npt

from pulse2percept import models
from pulse2percept import stimuli
from pulse2percept import implants


class ValidBaseModel(models.BaseModel):
    """A class that implements all abstract methods of BaseModel"""

    __slots__ = ('_private', 'valid')

    def __init__(self, **kwargs):
        super(ValidBaseModel, self).__init__(**kwargs)
        # You can only add attributes that are listed in slots:
        self._private = 0

    def _get_default_params(self):
        params = super(ValidBaseModel, self)._get_default_params()
        params.update({'valid': 1})
        return params

    def get_tissue_coords(self, xdva, ydva):
        return 290 * xdva, 290 * ydva

    def _predict_pixel_percept(self, xydva, img_stim, t=None):
        return 0

    def set_is_built(self):
        # This is not allowed outside constructor or ``build``:
        self._is_built = True


def test_BaseModel___init__():
    # Smoke test: Use default params:
    model = ValidBaseModel()
    npt.assert_almost_equal(model.valid, 1)
    # Set new value for existing param:
    model.valid = 2
    npt.assert_almost_equal(model.valid, 2)
    # Slots:
    npt.assert_equal(hasattr(model, '__slots__'), True)
    npt.assert_equal(hasattr(model, '__dict__'), False)
    # However, creating new params is not allowed:
    with pytest.raises(AttributeError):
        model.newparam = 0
    # Passing parameters that are not in slots is not allowed:
    with pytest.raises(AttributeError):
        ValidBaseModel(newparam=0)
    # Technically, nobody stops you from calling model.key = value on other
    # variables, such as private variables (given you know they exist):
    model._private = 2
    npt.assert_almost_equal(model._private, 2)


def test_BaseModel__pprint_params():
    # We can overwrite default param values if they are in ``_pprint_params``:
    model = ValidBaseModel(engine='serial')
    for key, value in model._pprint_params().items():
        if key in ('xrange', 'yrange', 'grid_type'):
            continue
        npt.assert_equal(getattr(model, key), value)
        setattr(model, key, 1234)
        npt.assert_equal(getattr(model, key), 1234)

        newmodel = ValidBaseModel(**{key: 1234})
        npt.assert_equal(getattr(newmodel, key), 1234)


def test_BaseModel_build():
    # Model must be built first thing
    model = ValidBaseModel(engine='serial')
    npt.assert_equal(model._is_built, False)
    model.build()
    npt.assert_equal(model._is_built, True)

    # Params passed to ``build`` must take effect:
    model = ValidBaseModel(engine='serial')
    model_params = model._pprint_params()
    for key, value in model_params.items():
        if isinstance(value, (int, float)):
            set_param = {key: 0.1234}
        elif isinstance(value, (list, set, tuple, np.ndarray)):
            set_param = {key: np.array([0, 0])}
        else:
            continue
        # Passing `set_param` during ``build`` must overwrite the earlier
        # value:
        model.build(**set_param)
        npt.assert_equal(getattr(model, key), set_param[key])


def test_BaseModel__is_built():
    # You cannot set `_is_built` outside the constructor or ``build``:
    model = ValidBaseModel(engine='serial')
    npt.assert_equal(model._is_built, False)
    with pytest.raises(AttributeError):
        model._is_built = True
    with pytest.raises(AttributeError):
        model.__is_built = True
    # After calling build, the flag should be set to True:
    model.build()
    npt.assert_equal(model._is_built, True)
    # You cannot set the flag in a new method you added, it has to be in
    # ``build``:
    with pytest.raises(AttributeError):
        model.set_is_built()


def test_BaseModel_predict_percept():
    img_stim = np.zeros(60)
    model = ValidBaseModel(engine='serial', xystep=5, xrange=(-30, 30),
                           yrange=(-20, 20))
    # Model must be built first:
    with pytest.raises(models.NotBuiltError):
        model.predict_percept(implants.ArgusII())

    # But then must pass through ``predict_percept`` just fine
    model.build()
    percept = model.predict_percept(implants.ArgusII(stim=img_stim))
    npt.assert_equal(percept.shape, (9, 13))
    npt.assert_almost_equal(percept, 0)

    # Requires ProsthesisSystem object:
    with pytest.raises(TypeError):
        model.predict_percept(img_stim)

    # None in, None out:
    npt.assert_equal(model.predict_percept(implants.ArgusII(stim=None)), None)

    # `img_stim` must have right size:
    for shape in [(2, 60), (59,), (2, 3, 4)]:
        with pytest.raises(ValueError):
            model.predict_percept(implants.ArgusII(stim=np.zeros(shape)))

    # Single-pixel percept:
    model = ValidBaseModel(engine='serial', xrange=(0.45, 0.45), yrange=(0, 0))
    model.build()
    percept = model.predict_percept(implants.ArgusII(stim=np.zeros(60)))
    npt.assert_equal(percept.shape, (1, 1))
    npt.assert_almost_equal(percept, 0)
