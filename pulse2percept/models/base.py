"""`BaseModel`, `Model`, `NotBuiltError`, `Percept`, `SpatialModel`,
   `TemporalModel`"""
import sys
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np

from ..implants import ProsthesisSystem
from ..stimuli import Stimulus
from ..percepts import Percept
from ..utils import PrettyPrint, Frozen, FreezeError, Grid2D, bisect
from ..utils.constants import ZORDER


class NotBuiltError(ValueError, AttributeError):
    """Exception class used to raise if model is used before building

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class BaseModel(Frozen, PrettyPrint, metaclass=ABCMeta):
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
        """BaseModel constructor

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

    def set_params(self, **params):
        """Set the parameters of this model"""
        for key, value in params.items():
            setattr(self, key, value)

    def _pprint_params(self):
        """Return a dict of class attributes to display when pretty-printing"""
        return {key: getattr(self, key)
                for key, _ in self.get_default_params().items()}

    def _build(self):
        """Customize the building process by implementing this method"""
        pass

    def build(self, **build_params):
        """Build the model

        Every model must have a ```build`` method, which is meant to perform
        all expensive one-time calculations. You must call ``build`` before
        calling ``predict_percept``.

        .. important::

            Don't override this method if you are building your own model.
            Customize ``_build`` instead.

        Parameters
        ----------
        build_params : additional parameters to set
            You can overwrite parameters that are listed in
            ``get_default_params``. Trying to add new class attributes outside
            of that will cause a ``FreezeError``.
            Example: ``model.build(param1=val)``

        """
        for key, val in build_params.items():
            setattr(self, key, val)
        self._build()
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


class SpatialModel(BaseModel, metaclass=ABCMeta):
    """Abstract base class for all spatial models

    Provides basic functionality for all spatial models:

    *  ``build``: builds the spatial grid used to calculate the percept.
       You can add your own ``_build`` method (note the underscore) that
       performs additional expensive one-time calculations.
    *  ``predict_percept``: predicts the percepts based on an implant/stimulus.
       You can add your own ``_predict_spatial`` method to customize this step.
       A user must call ``build`` before calling ``predict_percept``.

    To create your own spatial model, you must subclass ``SpatialModel`` and
    provide implementations for its three abstract methods:

    *  ``dva2ret``: a means to convert from degrees of visual angle (dva) to
       retinal coordinates (microns).
    *  ``ret2dva``: a means to convert from retinal coordinates to dva.
    *  ``_predict_spatial``: a method that accepts an ElectrodeArray as well as
       a Stimulus and computes the brightness at all spatial coordinates of
       ``self.grid``, returned as a 2D NumPy array (space x time).

    In addition, you can customize the following:

    *  ``__init__``: the constructor can be used to define additional
       parameters (note that you cannot add parameters on-the-fly)
    *  ``get_default_params``: all settable model parameters must be listed by
       this method
    *  ``_build`` (optional): a way to add one-time computations to the build
       process

    .. versionadded:: 0.6

    .. note ::

        You will not be able to add more parameters outside the constructor;
        e.g., ``model.newparam = 1`` will lead to a ``FreezeError``.

    .. seealso ::

        *  `Basic Concepts > Computational Models > Building your own model
           <topics-models-building-your-own>`
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.grid = None

    def get_default_params(self):
        """Return a dictionary of default values for all model parameters"""
        params = {
            # We will be simulating a patch of the visual field (xrange/yrange
            # in degrees of visual angle), at a given spatial resolution (step
            # size):
            'xrange': (-15, 15),  # dva
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
        """Build the model

        Performs expensive one-time calculations, such as building the spatial
        grid used to predict a percept. You must call ``build`` before
        calling ``predict_percept``.

        .. important::

            Don't override this method if you are building your own model.
            Customize ``_build`` instead.

        Parameters
        ----------
        build_params: additional parameters to set
            You can overwrite parameters that are listed in
            ``get_default_params``. Trying to add new class attributes outside
            of that will cause a ``FreezeError``.
            Example: ``model.build(param1=val)``

        """
        for key, val in build_params.items():
            setattr(self, key, val)
        # Build the spatial grid:
        self.grid = Grid2D(self.xrange, self.yrange, step=self.xystep,
                           grid_type=self.grid_type)
        self.grid.xret = self.dva2ret(self.grid.x)
        self.grid.yret = self.dva2ret(self.grid.y)
        self._build()
        self.is_built = True
        return self

    @abstractmethod
    def _predict_spatial(self, earray, stim):
        """Customized spatial response

        Called by the user from ``predict_percept`` after error checking.

        Parameters
        ----------
        earray: :py:class:`~pulse2percept.implants.ElectrodeArray`
            A valid electrode array.
        stim : :py:meth:`~pulse2percept.stimuli.Stimulus`
            A valid stimulus with a 2D data container (n_electrodes, n_time).

        Returns
        -------
        percept: np.ndarray
            A 2D NumPy array that has the same dimensions as the input stimulus
            (n_electrodes, n_time).
        """
        raise NotImplementedError

    def predict_percept(self, implant, t_percept=None):
        """Predict the spatial response

        .. important::

            Don't override this method if you are creating your own model.
            Customize ``_predict_spatial`` instead.

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``implant.stim`` is None.

        """
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
        # Make sure we don't change the user's Stimulus object:
        stim = deepcopy(implant.stim)
        # Make sure to operate on the compressed stim:
        if not stim.is_compressed:
            stim.compress()
        if t_percept is None:
            t_percept = stim.time
        n_time = 1 if t_percept is None else np.array([t_percept]).size
        if stim.data.size == 0:
            # Stimulus was compressed to zero:
            resp = np.zeros((self.grid.x.size, n_time), dtype=np.float32)
        else:
            # Calculate the Stimulus at requested time points:
            if t_percept is not None:
                # Save electrode parameters
                stim = Stimulus(stim[:, t_percept].reshape((-1, n_time)),
                                electrodes=stim.electrodes, time=t_percept,
                                metadata=stim.metadata)
            resp = self._predict_spatial(implant.earray, stim)
        return Percept(resp.reshape(list(self.grid.x.shape) + [-1]),
                       space=self.grid, time=t_percept,
                       metadata={'stim': stim})

    def find_threshold(self, implant, bright_th, amp_range=(0, 999), amp_tol=1,
                       bright_tol=0.1, max_iter=100):
        """Find the threshold current for a certain stimulus

        Estimates ``amp_th`` such that the output of
        ``model.predict_percept(stim(amp_th))`` is approximately ``bright_th``.

        Parameters
        ----------
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
            The implant and its stimulus to use. Stimulus amplitude will be
            up and down regulated until ``amp_th`` is found.
        bright_th : float
            Model output (brightness) that's considered "at threshold".
        amp_range : (amp_lo, amp_hi), optional
            Range of amplitudes to search (uA).
        amp_tol : float, optional
            Search will stop if candidate range of amplitudes is within
            ``amp_tol``
        bright_tol : float, optional
            Search will stop if model brightness is within ``bright_tol`` of
            ``bright_th``
        max_iter : int, optional
            Search will stop after ``max_iter`` iterations

        Returns
        -------
        amp_th : float
            Threshold current (uA), estimated so that the output of
            ``model.predict_percept(stim(amp_th))`` is within ``bright_tol`` of
            ``bright_th``.
        """
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError("'implant' must be a ProsthesisSystem, not "
                            "%s." % type(stim))

        def inner_predict(amp, fnc_predict, implant):
            _implant = deepcopy(implant)
            scale = amp / implant.stim.data.max()
            _implant.stim = Stimulus(scale * implant.stim.data,
                                     electrodes=implant.stim.electrodes,
                                     time=implant.stim.time)
            return fnc_predict(_implant).data.max()

        return bisect(bright_th, inner_predict,
                      args=[self.predict_percept, implant],
                      x_lo=amp_range[0], x_hi=amp_range[1], x_tol=amp_tol,
                      y_tol=bright_tol, max_iter=max_iter)

    def plot(self, use_dva=False, autoscale=True, ax=None):
        """Plot the model

        Parameters
        ----------
        use_dva : bool, optional
            Uses degrees of visual angle (dva) if True, else retinal
            coordinates (microns)
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot to fit the implant
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot
        """
        if use_dva:
            ax = self.grid.plot(autoscale=autoscale, ax=ax,
                                zorder=ZORDER['background'])
            ax.set_xlabel('x (dva)')
            ax.set_ylabel('y (dva)')
        else:
            ax = self.grid.plot(transform=self.dva2ret, autoscale=autoscale,
                                ax=ax, zorder=ZORDER['background'] + 1)
            ax.set_xlabel('x (microns)')
            ax.set_ylabel('y (microns)')
        return ax


class TemporalModel(BaseModel, metaclass=ABCMeta):
    """Abstract base class for all temporal models

    Provides basic functionality for all temporal models:

    *  ``build``: builds the model in order to calculate the percept.
       You can add your own ``_build`` method (note the underscore) that
       performs additional expensive one-time calculations.
    *  ``predict_percept``: predicts the percepts based on an implant/stimulus.
       You can add your own ``_predict_temporal`` method to customize this
       step. A user must call ``build`` before calling ``predict_percept``.

    To create your own temporal model, you must subclass ``SpatialModel`` and
    provide implementations for its three abstract methods:

    *  ``dva2ret``: a means to convert from degrees of visual angle (dva) to
       retinal coordinates (microns).
    *  ``ret2dva``: a means to convert from retinal coordinates to dva.
    *  ``_predict_temporal``: a method that accepts either a
       :py:class:`~pulse2percept.stimuli.Stimulus` or a
       :py:class:`~pulse2percept.percepts.Percept` object and a list of time
       points at which to calculate the resulting percept, returned as a 2D
       NumPy array (space x time).

    In addition, you can customize the following:

    *  ``__init__``: the constructor can be used to define additional
       parameters (note that you cannot add parameters on-the-fly)
    *  ``get_default_params``: all settable model parameters must be listed by
       this method
    *  ``_build`` (optional): a way to add one-time computations to the build
       process

    .. versionadded:: 0.6

    .. note ::

        You will not be able to add more parameters outside the constructor;
        e.g., ``model.newparam = 1`` will lead to a ``FreezeError``.

    .. seealso ::

        *  `Basic Concepts > Computational Models > Building your own model
           <topics-models-building-your-own>`
    """

    def get_default_params(self):
        """Return a dictionary of default values for all model parameters"""
        params = {
            # Simulation time step (ms):
            'dt': 0.005,
            # Below threshold, percept has brightness zero:
            'thresh_percept': 0,
            # True: print status messages, False: silent
            'verbose': True
        }
        return params

    @abstractmethod
    def _predict_temporal(self, stim, t_percept):
        """Customized temporal response

        Called by the user from ``predict_percept`` after error checking.

        Parameters
        ----------
        stim : :py:meth:`~pulse2percept.stimuli.Stimulus`
            A valid stimulus with a 2D data container (n_electrodes, n_time).
        t_percept : list of floats
            The time points at which to output a percept (ms).

        Returns
        -------
        percept: np.ndarray
            A 2D NumPy array (space x time) that specifies the percept at each
            spatial location and time step.
        """
        raise NotImplementedError

    def predict_percept(self, stim, t_percept=None):
        """Predict the temporal response

        .. important ::

            Don't override this method if you are creating your own model.
            Customize ``_predict_temporal`` instead.

        Parameters
        ----------
        stim: : py: class: `~pulse2percept.stimuli.Stimulus` or
               : py: class: `~pulse2percept.models.Percept`
            Either a Stimulus or a Percept object. The temporal model will be
            applied to each spatial location in the stimulus/percept.
        t_percept : float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, the percept will be output once very 20 ms (50 Hz frame
            rate).

            .. note ::

                If your stimulus is shorter than 20 ms, you should specify
                the desired time points manually.

        Returns
        -------
        percept : :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``stim`` is None.

        Notes
        -----
        *  If a list of time points is provided for ``t_percept``, the values
           will automatically be sorted.

        """
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if stim is None:
            # Nothing to see here:
            return None
        if not isinstance(stim, (Stimulus, Percept)):
            raise TypeError(("'stim' must be a Stimulus or Percept object, "
                             "not %s.") % type(stim))
        if stim.time is None:
            raise ValueError("Cannot calculate temporal response, because "
                             "stimulus/percept does not have a time "
                             "component.")
        # Make sure we don't change the user's Stimulus/Percept object:
        _stim = deepcopy(stim)
        if isinstance(stim, Stimulus):
            # Make sure to operate on the compressed stim:
            if not _stim.is_compressed:
                _stim.compress()
            _space = [len(stim.electrodes), 1]
        elif isinstance(stim, Percept):
            _space = [len(stim.ydva), len(stim.xdva)]
        _time = stim.time

        if t_percept is None:
            # If no time vector is given, output at 50 Hz frame rate. We always
            # start at zero and include the last time point:
            t_percept = np.arange(0, np.maximum(20, _time[-1]) + 1, 20)
        # We need to make sure the requested `t_percept` are sorted and
        # multiples of `dt`:
        t_percept = np.sort([t_percept]).flatten()
        remainder = np.mod(t_percept, self.dt) / self.dt
        atol = 1e-3
        within_atol = (remainder < atol) | (np.abs(1 - remainder) < atol)
        if not np.all(within_atol):
            raise ValueError("t=%s are not multiples of dt=%.2e." %
                             (t_percept[np.logical_not(within_atol)],
                              self.dt))
        if _stim.data.size == 0:
            # Stimulus was compressed to zero:
            resp = np.zeros(_space + [t_percept.size], dtype=np.float32)
        else:
            # Calculate the Stimulus at requested time points:
            resp = self._predict_temporal(_stim, t_percept)
        return Percept(resp.reshape(_space + [t_percept.size]),
                       space=None, time=t_percept,
                       metadata={'stim': stim})

    def find_threshold(self, stim, bright_th, amp_range=(0, 999), amp_tol=1,
                       bright_tol=0.1, max_iter=100, t_percept=None):
        """Find the threshold current for a certain stimulus

        Estimates ``amp_th`` such that the output of
        ``model.predict_percept(stim(amp_th))`` is approximately ``bright_th``.

        Parameters
        ----------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus`
            The stimulus to use. Stimulus amplitude will be up and down
            regulated until ``amp_th`` is found.
        bright_th : float
            Model output (brightness) that's considered "at threshold".
        amp_range : (amp_lo, amp_hi), optional
            Range of amplitudes to search (uA).
        amp_tol : float, optional
            Search will stop if candidate range of amplitudes is within
            ``amp_tol``
        bright_tol : float, optional
            Search will stop if model brightness is within ``bright_tol`` of
            ``bright_th``
        max_iter : int, optional
            Search will stop after ``max_iter`` iterations
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        amp_th : float
            Threshold current (uA), estimated so that the output of
            ``model.predict_percept(stim(amp_th))`` is within ``bright_tol`` of
            ``bright_th``.
        """
        if not isinstance(stim, Stimulus):
            raise TypeError("'stim' must be a Stimulus, not %s." % type(stim))

        def inner_predict(amp, fnc_predict, stim, **kwargs):
            _stim = Stimulus(amp * stim.data / stim.data.max(),
                             electrodes=stim.electrodes, time=stim.time)
            return fnc_predict(_stim, **kwargs).data.max()

        return bisect(bright_th, inner_predict,
                      args=[self.predict_percept, stim],
                      kwargs={'t_percept': t_percept},
                      x_lo=amp_range[0], x_hi=amp_range[1], x_tol=amp_tol,
                      y_tol=bright_tol, max_iter=max_iter)


class Model(PrettyPrint):
    """Computational model

    To build your own model, you can mix and match spatial and temporal models
    at will.

    For example, to create a model that combines the scoreboard model described
    in [Beyeler2019]_ with the temporal model cascade described in
    [Nanduri2012]_, use the following:

    .. code-block :: python

        model = Model(spatial=ScoreboardSpatial(),
                      temporal=Nanduri2012Temporal())

    .. seealso ::

        *  `Basic Concepts > Computational Models > Building your own model
           <topics-models-building-your-own>`

    .. versionadded:: 0.6

    Parameters
    ----------
    spatial: :py:class:`~pulse2percept.models.SpatialModel` or None
        blah
    temporal: :py:class:`~pulse2percept.models.TemporalModel` or None
        blah
    **params:
        Additional keyword arguments(e.g., ``verbose=True``) to be passed to
        either the spatial model, the temporal model, or both.

    """

    def __init__(self, spatial=None, temporal=None, **params):
        # Set the spatial model:
        if spatial is not None and not isinstance(spatial, SpatialModel):
            if issubclass(spatial, SpatialModel):
                # User should have passed an instance, not a class:
                spatial = spatial()
            else:
                raise TypeError("'spatial' must be a SpatialModel instance, "
                                "not %s." % type(spatial))
        self.spatial = spatial
        # Set the temporal model:
        if temporal is not None and not isinstance(temporal, TemporalModel):
            if issubclass(temporal, TemporalModel):
                # User should have passed an instance, not a class:
                temporal = temporal()
            else:
                raise TypeError("'temporal' must be a TemporalModel instance, "
                                "not %s." % type(temporal))
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
            *  if the attribute exists in both spatial / temporal model,
               returns a dictionary ``{'spatial': attr, 'temporal': attr}``.
            *  if the attribtue is not found, raises an AttributeError.

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
            found = True
        except (AttributeError, FreezeError):
            pass
        try:
            self.temporal.__setattr__(name, value)
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
        """Build the model

        Performs expensive one-time calculations, such as building the spatial
        grid used to predict a percept.

        Parameters
        ----------
        build_params: additional parameters to set
            You can overwrite parameters that are listed in
            ``get_default_params``. Trying to add new class attributes outside
            of that will cause a ``FreezeError``.
            Example: ``model.build(param1=val)``

        Returns
        -------
        self

        """
        self.set_params(build_params)
        if self.has_space:
            self.spatial.build()
        if self.has_time:
            self.temporal.build()
        return self

    def predict_percept(self, implant, t_percept=None):
        """Predict a percept

        .. important ::

            You must call ``build`` before calling ``predict_percept``.

        Parameters
        ----------
        implant: :py:class:`~pulse2percept.implants.ProsthesisSystem`
            A valid prosthesis system. A stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        percept: :py:class:`~pulse2percept.models.Percept`
            A Percept object whose ``data`` container has dimensions Y x X x T.
            Will return None if ``implant.stim`` is None.
        """
        if not self.is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError("'implant' must be a ProsthesisSystem object, not "
                            "%s." % type(implant))
        if implant.stim is None or (not self.has_space and not self.has_time):
            # Nothing to see here:
            return None
        if implant.stim.time is None and t_percept is not None:
            raise ValueError("Cannot calculate temporal response at times "
                             "t_percept=%s, because stimulus/percept does not "
                             "have a time component." % t_percept)

        if self.has_space and self.has_time:
            # Need to calculate the spatial response at all stimulus points
            # (i.e., whenever the stimulus changes):
            resp = self.spatial.predict_percept(implant, t_percept=None)
            if implant.stim.time is not None:
                # Then pass that to the temporal model, which will output at
                # all `t_percept` time steps:
                resp = self.temporal.predict_percept(resp, t_percept=t_percept)
        elif self.has_space:
            resp = self.spatial.predict_percept(implant, t_percept=t_percept)
        elif self.has_time:
            resp = self.temporal.predict_percept(implant.stim,
                                                 t_percept=t_percept)
        return resp

    def find_threshold(self, implant, bright_th, amp_range=(0, 999), amp_tol=1,
                       bright_tol=0.1, max_iter=100, t_percept=None):
        """Find the threshold current for a certain stimulus

        Estimates ``amp_th`` such that the output of
        ``model.predict_percept(stim(amp_th))`` is approximately ``bright_th``.

        Parameters
        ----------
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
            The implant and its stimulus to use. Stimulus amplitude will be
            up and down regulated until ``amp_th`` is found.
        bright_th : float
            Model output (brightness) that's considered "at threshold".
        amp_range : (amp_lo, amp_hi), optional
            Range of amplitudes to search (uA).
        amp_tol : float, optional
            Search will stop if candidate range of amplitudes is within
            ``amp_tol``
        bright_tol : float, optional
            Search will stop if model brightness is within ``bright_tol`` of
            ``bright_th``
        max_iter : int, optional
            Search will stop after ``max_iter`` iterations
        t_percept: float or list of floats, optional
            The time points at which to output a percept (ms).
            If None, ``implant.stim.time`` is used.

        Returns
        -------
        amp_th : float
            Threshold current (uA), estimated so that the output of
            ``model.predict_percept(stim(amp_th))`` is within ``bright_tol`` of
            ``bright_th``.
        """
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError("'implant' must be a ProsthesisSystem, not "
                            "%s." % type(stim))

        def inner_predict(amp, fnc_predict, implant, **kwargs):
            _implant = deepcopy(implant)
            scale = amp / implant.stim.data.max()
            _implant.stim = Stimulus(scale * implant.stim.data,
                                     electrodes=implant.stim.electrodes,
                                     time=implant.stim.time)
            return fnc_predict(_implant, **kwargs).data.max()

        return bisect(bright_th, inner_predict,
                      args=[self.predict_percept, implant],
                      kwargs={'t_percept': t_percept},
                      x_lo=amp_range[0], x_hi=amp_range[1], x_tol=amp_tol,
                      y_tol=bright_tol, max_iter=max_iter)

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
