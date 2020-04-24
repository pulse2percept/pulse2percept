"""`BaseModel`"""
import sys
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np

from ..implants import ProsthesisSystem
from ..stimuli import Stimulus
from ..utils import PrettyPrint, Frozen, FreezeError, GridXY, parfor, Data


class Percept(Data):

    def __init__(self, data, space=None, time=None, metadata=None):
        x = None
        y = None
        if space is not None:
            if not isinstance(space, GridXY):
                raise TypeError("'space' must be a GridXY object, not "
                                "%s." % type(space))
            x = space._xflat
            y = space._yflat
        if time is not None:
            time = np.array([time]).flatten()
        self._internal = {
            'data': data,
            'axes': [('y', y), ('x', x), ('t', time)],
            'metadata': metadata
        }
        # def f(a1, a2):
        #     # https://stackoverflow.com/a/26410051
        #     return (((a1 - a2[:,:,np.newaxis])).prod(axis=1)<=0).any(axis=0)


class NotBuiltError(ValueError, AttributeError):
    """Exception class used to raise if model is used before building

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


# TODO RENAME, THIS CAN BE BASEMODEL
class BuildModel(Frozen, PrettyPrint, metaclass=ABCMeta):
    """Abstract base class for all models

    Provides the following functionality:

    *  Pretty-print class attributes (via ``_pprint_params`` and
       ``PrettyPrint``)
    *  Build a model (via ``build``) and flip the ``is_built`` switch
    *  User-settable parameters must be listed in ``get_default_params``
    *  New class attributes can only be added in the constructor
       (enforced via ``Frozen`` and ``FreezeError``).

    """

    def __init__(self, **params):
        """BuildModel constructor

        Parameters
        ----------
        **params : optional keyword arguments
            All keyword arguments must be listed in ``get_default_params``
        """
        # Set all default arguments:
        defaults = self.get_default_params()
        for key, val in defaults.items():
            setattr(self, key, val)
        # Then overwrite any arguments also given in `params`:
        for key, val in params.items():
            if key in defaults:
                setattr(self, key, val)
            else:
                err_str = ("'%s' is not a valid model parameter. Choose "
                           "from: %s." % (key, ', '.join(defaults.keys())))
                raise AttributeError(err_str)
        # This flag will be flipped once the ``build`` method was called
        self._is_built = False

    @abstractmethod
    def get_default_params(self):
        """Return a dict of user-settable model parameters"""
        raise NotImplementedError

    def _pprint_params(self):
        """Return a dict of class attributes to display when pretty-printing"""
        return {key: getattr(self, key)
                for key, _ in self.get_default_params().items()}

    def build(self, **build_params):
        """Build the model

        Every model must have a ```build`` method, which is meant to perform
        all expensive one-time calculations. You must call ``build`` before
        calling ``predict_percept``.

        You can override ``build`` in your own model (for a good example, see
        the AxonMapModel). You will want to make sure that:

        - all ``build_params`` take effect,
        - the flag ``_is_built`` is set before returning,
        - the method returns ``self``.
        """
        # Set additional parameters (they must be mentioned in the constructor
        # and/or in ``get_default_params``. Trying to add new class attributes
        # outside of that will cause a ``FreezeError``):
        for key, val in build_params.items():
            setattr(self, key, val)
        self.is_built = True
        return self

    @property
    def is_built(self):
        """A flag indicating whether the model has been built"""
        return self._is_built

    @is_built.setter
    def is_built(self, val):
        """This flag can only be set in the constructor or ``build``"""
        # getframe(0) is '_is_built', getframe(1) is 'set_attr'.
        # getframe(2) is the one we are looking for, and has to be either the
        # construct or ``build``:
        f_caller = sys._getframe(2).f_code.co_name
        if f_caller in ["__init__", "build"]:
            self._is_built = val
        else:
            err_s = ("The attribute `is_built` can only be set in the "
                     "constructor or in ``build``, not in ``%s``." % f_caller)
            raise AttributeError(err_s)


class SpatialModel(BuildModel, metaclass=ABCMeta):

    def __init__(self, **params):
        super().__init__(**params)
        self.grid = None

    def get_default_params(self):
        """Get a dictionary of default values for all model parameters"""
        params = {
            # We will be simulating a patch of the visual field (xrange/yrange
            # in degrees of visual angle), at a given spatial resolution (step
            # size):
            'xrange': (-20, 20),  # dva
            'yrange': (-15, 15),  # dva
            'xystep': 0.25,  # dva
            'grid_type': 'rectangular',
            # Below threshold, percept has brightness zero:
            'thresh_percept': 0,
            # JobLib or Dask can be used to parallelize computations:
            'engine': 'serial',
            'scheduler': 'threading',
            'n_jobs': 1,
            # True: print status messages, 0: silent
            'verbose': True
        }
        return params

    @abstractmethod
    def dva2ret(self, xdva):
        """Convert degrees of visual angle (dva) into retinal coords (um)"""
        raise NotImplementedError

    @abstractmethod
    def ret2dva(self, xret):
        """Convert retinal corods (um) to degrees of visual angle (dva)"""
        raise NotImplementedError

    def build(self, **build_params):
        # Set additional parameters (they must be mentioned in the constructor;
        # you can't add new class attributes outside of that):
        for key, val in build_params.items():
            setattr(self, key, val)
        # Build the spatial grid:
        self.grid = GridXY(self.xrange, self.yrange, step=self.xystep,
                           grid_type=self.grid_type)
        self.grid.xret = self.dva2ret(self.grid.x)
        self.grid.yret = self.dva2ret(self.grid.y)
        self.is_built = True
        return self

    @abstractmethod
    def _predict_spatial(self, earray, stim):
        """Called from ``predict_percept`` after error checking

        The problem is: if you instantiate SpatialModel, you want to be able to
        run it at any t. For example, you have a really long pulse train but
        you only care about current spread at a particular time point.

        This is in conflict with a Model() instance, where you need to calculate
        current spread at all stimulus times for the temporal model.

        MUST RETURN: SPACE x TIME
        """
        raise NotImplementedError

    def predict_percept(self, implant, t_percept=None):
        """Predict the spatial response

        Parameters
        ----------
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept : float or list of floats, optional, default: None
            The time points at which to output a percept (seconds).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        percept : :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``implant.stim`` is None.

        .. note ::

            Do not override this method if you are writing your own model.
            It performs error checks and reshaping of the stimulus container,
            before calling out to ``_predict_spatial``.
            Customize ``_predict_spatial`` instead.

        """
        print("predict spatial:")
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(("'implant' must be a ProsthesisSystem object, "
                             "not %s.") % type(implant))
        if implant.stim is None:
            # Nothing to see here:
            return None
        if implant.stim.time is None and t_percept is not None:
            raise ValueError("Cannot calculate spatial response at times "
                             "t_percept=%s, because stimulus does not "
                             "have a time component." % t_percept)
        if t_percept is None:
            t_percept = implant.stim.time
        # Make sure we don't change the user's Stimulus object:
        stim = deepcopy(implant.stim)
        # Make sure to operate on the compressed stim:
        print("- stim before", stim.shape)
        if not stim.is_compressed:
            stim.compress()
        print("- stim after", stim.shape)
        n_time = 1 if t_percept is None else np.array([t_percept]).size
        if stim.data.size == 0:
            # Stimulus was compressed to zero:
            resp = np.zeros((self.grid.x.size, n_time), dtype=np.float32)
        else:
            # Calculate the Stimulus at requested time points:
            if t_percept is not None:
                stim = Stimulus(stim[:, t_percept].reshape((-1, n_time)),
                                electrodes=stim.electrodes, time=t_percept)
            resp = self._predict_spatial(implant.earray, stim)
        print(resp.shape)
        return Percept(resp.reshape(list(self.grid.x.shape) + [-1]),
                       space=self.grid, time=t_percept)


class TemporalModel(BuildModel, metaclass=ABCMeta):

    def get_default_params(self):
        params = {
            # Simulation time step:
            'dt': 5e-6,
            # Below threshold, percept has brightness zero:
            'thresh_percept': 0,
            # True: print status messages, 0: silent
            'verbose': True
        }
        return params

    @abstractmethod
    def _predict_temporal(self, stim, t):
        """Called from ``predict_percept`` after error checking

        MUST RETURN: NxT
        we need to know the time points for outputting and in case there's a
        temporal model up next
        """
        raise NotImplementedError

    def predict_percept(self, stim, t_percept=None):
        """Predict the temporal response

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` or
               :py:class:`~pulse2percept.models.Percept`
            Either a Stimulus or a Percept object. The temporal model will be
            applied to each spatial location in the stimulus/percept.
        t_percept : float or list of floats, optional, default: None
            The time points at which to output a percept (seconds).
            If None, the time axis of the stimulus/percept is used.

        Returns
        -------
        percept : :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``stim`` is None.

        .. note ::

            Do not override this method if you are writing your own model.
            It performs error checks and reshaping of the stimulus container,
            before calling out to ``_predict_temporal``.
            Customize ``_predict_temporal`` instead.

        """
        print("predict temporal:")
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if stim is None:
            # Nothing to see here:
            return None
        if isinstance(stim, Percept):
            # Percept has shape (Y, X, T), needs to be (XY, T):
            _stim = stim.data.reshape((-1, stim.shape[-1]))
            _space = [len(stim.y), len(stim.x)]
            _time = stim.t
        elif isinstance(stim, Stimulus):
            # Make sure we don't change the user's Stimulus object:
            _stim = deepcopy(stim)
            _space = [len(stim.electrodes)]
            # Make sure to operate on the compressed stim:
            if not _stim.is_compressed:
                _stim.compress()
            _time = stim.time
        else:
            raise TypeError(("'stim' must be a Stimulus or Percept object, "
                             "not %s.") % type(stim))

        if _time is None and t_percept is not None:
            raise ValueError("Cannot calculate temporal response at times "
                             "t_percept=%s, because stimulus/percept does not "
                             "have a time component." % t_percept)
        if t_percept is None:
            # If no time vector is given, output at model time step (and make
            # sure to include the last time point). We always start at zero:
            t_percept = np.arange(0, _time[-1] + self.dt / 2.0, self.dt)
        if stim.data.size == 0:
            # Stimulus was compressed to zero:
            resp = np.zeros([_space] + [t_percept.size], dtype=np.float32)
        else:
            # Calculate the Stimulus at requested time points:
            resp = self._predict_temporal(stim, t_stim, t_percept)
        return Percept(resp.reshape(_space + [t_percept.size]),
                       space=self.grid, time=t_percept)

        # A Stimulus could be compressed to zero:
        if stim.data.size == 0:
            resp = np.zeros()
        print("predict temporal:")
        print("- stim:", stim)
        print("- time:", t)
        assert isinstance(stim, Stimulus) or isinstance(stim, Data)
        # Make sure we don't change the user's Stimulus object:
        _stim = deepcopy(stim)
        # Make sure to operate on the compressed stim:
        print("- stim before", _stim.shape)
        if isinstance(_stim, Stimulus) and not _stim.is_compressed:
            _stim.compress()
        print("- stim after", _stim.shape)

        if t is None:
            if len(_stim.time) == 1 and _stim.time[0] is None:
                # FIXME
                t = np.array([0], dtype=np.float32)
            else:
                # If no time vector given, output at model time step (make sure to
                # include the last time point):
                t = np.arange(_stim.time[0],
                              _stim.time[-1] + self.dt / 2.0,
                              self.dt)
        else:
            t = np.array([t]).flatten()
        print("- t after:", t)
        # A Stimulus could be compressed to zero:
        print("- compressed stim:", stim)
        if stim.data.size == 0:
            # TODO: Percept object
            print("stim.data.size==0")
            resp = np.zeros((np.prod(self.grid.x.shape), t.size),
                            dtype=np.float32)
        else:
            resp = self._predict_temporal(_stim, t)
        assert resp.ndim == 2
        assert resp.shape[1] == t.size
        return Data(resp, axes=[('space', None), ('time', t)])


class Model(PrettyPrint):
    """Model

    Parameters
    ----------
    spatial: : py: class: `~pulse2percept.models.SpatialModel` or None
        blah
    temporal: : py: class: `~pulse2percept.models.TemporalModel` or None
        blah
    **params:
        Additional keyword arguments(e.g., ``verbose=True``) to be passed to
        either the spatial model, the temporal model, or both.

    """

    def __init__(self, spatial=None, temporal=None, **params):
        """A Model provides the glue between a spatial and / or temporal model"""
        # Set the spatial model:
        if spatial is not None and not isinstance(spatial, SpatialModel):
            raise TypeError("'spatial' must be a SpatialModel, not "
                            "%s." % type(spatial))
        self.spatial = spatial
        # Set the temporal model:
        if temporal is not None and not isinstance(temporal, TemporalModel):
            raise TypeError("'temporal' must be a TemporalModel, not "
                            "%s." % type(temporal))
        self.temporal = temporal
        # Use user-specified parameter values instead of defaults:
        self.set_params(params)

    def __getattr__(self, attr):
        """Called when the default attr access fails with an AttributeError

        This method is called when the user tries to access an attribute(e.g.,
        ``model.a``), but ``a`` could not be found(either because it is part
        of the spatial / temporal model or because it doesn't exist).

        Returns
        -------
        attr: any
            Checks both spatial and temporal models and:

            *  returns the attribute if found.
            * if the attribute exists in both spatial / temporal model, returns
               a dictionary ``{'spatial': attr, 'temporal': attr}``.
            * if the attribtue is not found, raises an AttributeError.

        """
        if sys._getframe(2).f_code.co_name == '__init__':
            # We can set new class attributes in the constructor. Reaching this
            # point means the default attribute access failed - most likely
            # because we are trying to create a variable. In this case, simply
            # raise an exception:
            raise AttributeError("%s not found" % attr)
        # Outside the constructor, we need to check the spatial/temporal model:
        try:
            spatial = self.spatial.__getattribute__(attr)
        except AttributeError:
            spatial = None
        try:
            temporal = self.temporal.__getattribute__(attr)
        except AttributeError:
            temporal = None
        if spatial is None and temporal is None:
            raise AttributeError("%s has no attribute "
                                 "'%s'." % (self.__class__.__name__,
                                            attr))
        if spatial is None:
            return temporal
        if temporal is None:
            return spatial
        return {'spatial': spatial, 'temporal': temporal}

    def __setattr__(self, name, value):
        """Called when an attribute is set

        This method is called when a new attribute is set(e.g.,
        ``model.a=2``). This is allowed in the constructor, but will raise a
        ``FreezeError`` elsewhere.

        ``model.a = X`` can be used as a shorthand to set ``model.spatial.a``
        and / or ``model.temporal.a``.

        """
        if sys._getframe(1).f_code.co_name == '__init__':
            # Allow setting new attributes in the constructor:
            if isinstance(sys._getframe(1).f_locals['self'], self.__class__):
                super().__setattr__(name, value)
                return
        # Outside the constructor, we cannot add new attributes (FreezeError).
        # But, we have to check whether the attribute is part of the spatial
        # model, the temporal model, or both:
        found = False
        try:
            self.spatial.__setattr__(name, value)
            print("SET spatial", name, value)
            found = True
        except (AttributeError, FreezeError):
            pass
        try:
            self.temporal.__setattr__(name, value)
            print("SET temporal", name, value)
            found = True
        except (AttributeError, FreezeError):
            pass
        if not found:
            err_str = ("'%s' not found. You cannot add attributes to %s "
                       "outside the constructor." % (name,
                                                     self.__class__.__name__))
            raise FreezeError(err_str)

    def _pprint_params(self):
        """Return a dictionary of parameters to pretty - print"""
        params = {'spatial': self.spatial, 'temporal': self.temporal}
        # Also display the parameters from the spatial/temporal model:
        if self.has_space:
            params.update(self.spatial._pprint_params())
        if self.has_time:
            params.update(self.temporal._pprint_params())
        return params

    def set_params(self, params):
        """Set model parameters

        This is a convenience function to set parameters that might be part of
        the spatial model, the temporal model, or both.

        Alternatively, you can set the parameter directly, e.g.
        ``model.spatial.verbose = True``.

        .. note::

            If a parameter exists in both spatial and temporal models(e.g.,
            ``verbose``), both models will be updated.

        Parameters
        ----------
        params: dict
            A dictionary of parameters to set.
        """
        for key, val in params.items():
            setattr(self, key, val)

    def build(self, **build_params):
        """Builds the model"""
        self.set_params(build_params)
        if self.has_space:
            self.spatial.build()
        if self.has_time:
            self.temporal.build()
        return self

    def predict_percept(self, implant, t=None):
        """Predict a percept

        Parameters
        ----------
        implant: : py: class: `~pulse2percept.implants.ProsthesisSystem`
            Stimulus can be passed via
            : py: meth: `~pulse2percept.implants.ProsthesisSystem.stim`.
        t: float or list of floats
            The time points at which to output a percept(seconds).

        Returns
        -------
        percept: np.ndarray
            A < X x Y x T > matrix that contains the predicted brightness values
            at the specified(X, Y) spatial locations and times T.
        """
        print("")
        print("predict_percept:")
        print("- implant:", implant)
        print("- time:", t)
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError("'implant' must be a ProsthesisSystem object, not "
                            "%s." % type(implant))
        if implant.stim is None:
            # Nothing to see here:
            return None

        # Calculate the spatial response at all time points where the stimulus
        # changes:
        if self.has_space:
            resp = self.spatial.predict_percept(implant, t_percept=t)
        else:
            resp = implant.stim

        if self.has_time:
            # Problem: spatial comes first, must agree on format.
            # Could be no spatial, in which case it should just be the stimulus
            # (so stim and spatial should have the same format)
            resp = self.temporal.predict_percept(resp, t_percept=t)

        # TODO: Percept object
        return resp
        # return resp.data.reshape(list(self.spatial.grid.x.shape) + [-1])

    @property
    def has_space(self):
        """Returns True if the model has a spatial component"""
        return self.spatial is not None

    @property
    def has_time(self):
        """Returns True if the model has a temporal component"""
        return self.temporal is not None

    @property
    def is_built(self):
        """Returns True if the ``build`` model has been called"""
        _is_built = True
        if self.has_space:
            _is_built &= self.spatial.is_built
        if self.has_time:
            _is_built &= self.temporal.is_built
        return _is_built
