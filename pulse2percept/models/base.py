import sys
import abc
import numpy as np

from ..implants import ProsthesisSystem
from ..utils import Frozen, PrettyPrint, GridXY, parfor


class NotBuiltError(ValueError, AttributeError):
    """Exception class used to raise if model is used before building

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class BaseModel(Frozen, PrettyPrint, metaclass=abc.ABCMeta):
    """Base model

    The BaseModel class defines which methods and attributes a model must
    have.

    You can set individual model parameters by passing them as keyword
    arguments (e.g., ``MyModel(engine='joblib')``). Note that these parameters
    must be listed in ``get_params``. If no kwargs are passed, all model
    parameters will be initialized with default values.

    .. note::
        You can add more parameters to your model by subclassing
        `_get_default_params`.

    To write your own model, create a class that inherits from `BaseModel`.
    To make the model complete (and compile), you will also need to fill in
    all methods marked with `@abc.abstractmethod` below. These include
    :py:meth:`BaseModel.build` and :py:meth:`BaseModel.predict_percept`.

    .. note::
        Have a look at the `ScoreboardModel` or `AxonMapModel` classes
        to get an idea of how to write a complete model.

    Parameters
    ----------
    xrange : (xmin, xmax)
    yrange : (ymin, ymax)
    xystep : double
    grid_type : 'rectangular'
    thresh_percept : double
    engine : {'joblib', 'dask', 'serial', 'cython'}
    scheduler : {'threading', 'multiprocessing'}
    n_jobs : int
    verbose : bool
    """

    def __init__(self, **kwargs):
        # First, set all default arguments:
        defaults = self._get_default_params()
        for key, val in defaults.items():
            setattr(self, key, val)
        # Then overwrite any arguments also given in `kwargs`:
        for key, val in kwargs.items():
            if key in defaults:
                setattr(self, key, val)
            else:
                err_str = ("'%s' is not a valid model parameter. Choose "
                           "from: %s." % (key, ', '.join(defaults.keys())))
                raise AttributeError(err_str)
        # Retinal grid:
        self.grid = GridXY(self.xrange, self.yrange, step=self.xystep,
                           grid_type=self.grid_type)
        # This flag will be flipped once the ``build`` method was called
        self.__is_built = False

    def _get_default_params(self):
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
            'engine': 'joblib',
            'scheduler': 'threading',
            'n_jobs': -1,
            # True: print status messages, 0: silent
            'verbose': True
        }
        return params

    def get_params(self):
        """Get a dictionary of all model parameters (don't override!)"""
        return {key: getattr(self, key)
                for key, _ in self._get_default_params().items()}

    @property
    def _is_built(self):
        """A flag indicating whether the model has been built"""
        return self.__is_built

    @_is_built.setter
    def _is_built(self, val):
        """This flag can only be set in the constructor or ``build``"""
        # getframe(0) is '_is_built', getframe(1) is 'set_attr'.
        # getframe(2) is the one we are looking for, and has to be either the
        # construct or ``build``:
        f_caller = sys._getframe(2).f_code.co_name
        if f_caller in ["__init__", "build"]:
            self.__is_built = val
        else:
            err_s = ("The attribute `_is_built` can only be set in the "
                     "constructor or in ``build``, not in ``%s``." % f_caller)
            raise AttributeError(err_s)

    def build(self, **build_params):
        """Build the model

        Every model must have a ```build`` method, which is meant to perform
        all expensive one-time calculations. You must call ``build`` before
        calling ``predict_percept``.

        You can override ``build`` in your own model (for a good example, see
        the AxonMapModel). You will want to make sure that:

        - all `build_params` take effect,
        - the flag `_is_built` is set,
        - the method returns `self`.

        One way to do this is to call the BaseModel's ``build`` method from
        within your own model:

        .. code-block:: python

            class MyModel(BaseModel):

                def build(self, **build_params):
                    super(MyModel, self).build(self, **build_params)
                    # Add your own code here...

        Parameters
        ----------
        **build_params : Additional build parameters
            Additional build parameters passed as keyword arguments (e.g.,
            ``model.build(engine='joblib')``). Note that these must be listed
            in ``get_params``; i.e., you can't add any new parameters outside
            the constructor.
        """
        # Set additional parameters (they must be mentioned in the constructor;
        # you can't add new class attributes outside of that):
        for key, val in build_params.items():
            setattr(self, key, val)
        # This flag indicates that the ``build`` method has been called. It has
        # to be set to True for other methods, such as ``predict_percept``, to
        # work:
        self._is_built = True
        return self

    @abc.abstractmethod
    def get_tissue_coords(self, xdva, ydva):
        """Convert dva into tissue coordinates"""
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_pixel_percept(self, xygrid, implant, t=None):
        """Calculate the percept at pixel location (xdva,ydva)

        Parameters
        ----------
        xygrid : tuple

        """
        raise NotImplementedError

    def predict_percept(self, implant, t=None, n_frames=None, fps=20):
        """Predict a percept

        Parameters
        ----------
        implant : `ProsthesisSystem`
            Stimulus can be passed via
            :py:meth:`~pulse2percept.implants.ProsthesisSystem.stim`.
        fps : int, double
            Frames per second at which the percept should be rendered.
        n_frames : int
            If None, will simulate for the duration of the stimulus plus one
            frame (rounding up).
        """
        if not self._is_built:
            raise NotBuiltError("Yout must call ``build`` first.")
        if not isinstance(implant, ProsthesisSystem):
            raise TypeError(("'implant' must be a ProsthesisSystem object, "
                             "not %s.") % type(implant))
        if implant.stim is None:
            # Nothing to see here:
            return None

        if implant.stim.time is None:
            # The stimulus does not have a time dimension: In this case, we
            # only need to run the spatial model:
            bright = parfor(self._predict_pixel_percept,
                            enumerate(self.grid),
                            func_args=[implant],
                            func_kwargs={'t': None},
                            engine=self.engine, scheduler=self.scheduler,
                            n_jobs=self.n_jobs,
                            out_shape=self.grid.shape)
            return np.where(bright > self.thresh_percept, bright, 0)

        # Stimulus has time, so we need both spatial + temporal model:
        if not self.has_time:
            raise ValueError("Model does not have a temporal part")
        if t is not None:
            t_percept = np.asarray(t)
        else:
            fps = np.double(fps)
            if fps <= 0:
                raise ValueError("fps must be nonnegative, not %f." % fps)
            if n_frames is None:
                if implant.stim.time.max() < 1.0 / fps:
                    # Simulate until the end of the stimulus:
                    t_end = implant.stim.time.max()
                    n_frames = 2
                else:
                    # We need to for the duration of the stimulus, rounding up:
                    n_frames = max(2, np.ceil(implant.stim.time.max() * fps))
                    t_end = n_frames / fps
            else:
                n_frames = int(n_frames)
                if n_frames <= 0:
                    raise ValueError("n_frames must be nonnegative, not "
                                     "%d" % n_frames)
                t_end = n_frames / fps
            t_percept = np.linspace(0, t_end, num=n_frames)
        # Simulate:
        self.reset_state()
        t_sim = 0
        percept = []
        cache_stim = None
        print('dt:', self.dt)
        print('t_percept:', t_percept)
        for tp in t_percept:
            # Step the temporal model from `t` to `t_percept`:
            # print('tp:', tp)
            while cache_stim is None or t_sim < tp:
                # Last time step might be smaller, if `t_percept` is not
                # divisible by `self.dt`:
                dt_sim = min(self.dt, tp - t_sim)
                # Calculate current map at time `t_sim`:
                stim_at_t = implant.stim.interp(time=t_sim)
                need_cmap = False
                if cache_stim is None:
                    need_cmap = True
                else:
                    if not np.allclose(stim_at_t.data, cache_stim.data):
                        need_cmap = True
                need_cmap = True
                if need_cmap:
                    # print('-', t_sim, 'calc cmap')
                    # print('-', stim_at_t)
                    # print('-', cache_stim)
                    cmap = parfor(self._predict_pixel_percept,
                                  enumerate(self.grid),
                                  func_args=[implant],
                                  func_kwargs={'t': t_sim},
                                  engine=self.engine,
                                  scheduler=self.scheduler,
                                  n_jobs=self.n_jobs)
                cache_stim = stim_at_t
                # Step the temporal model:
                # print('-', t_sim, 'step temp')
                frame = parfor(self._step_temporal_model,
                               enumerate(cmap),
                               func_args=[dt_sim],
                               engine=self.engine,
                               scheduler=self.scheduler,
                               n_jobs=self.n_jobs,
                               out_shape=self.grid.shape)
                # Add `frame` to `percept` output:
                frame = np.where(frame > self.thresh_percept, frame, 0)
                percept.append(frame)
                # Increase the time counter:
                t_sim += dt_sim
        return percept
