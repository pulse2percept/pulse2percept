"""`BaseModel`"""
import sys
import abc

from ..implants import ProsthesisSystem
from ..utils import PrettyPrint, GridXY, parfor


class NotBuiltError(ValueError, AttributeError):
    """Exception class used to raise if model is used before building

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class BaseModel(PrettyPrint, metaclass=abc.ABCMeta):
    """Base model

    The BaseModel class defines which methods and attributes a model must
    have. You can create your own model by adding a class that derives from
    BaseModel:

    .. code-block:: python

        class MyModel(BaseModel):

    The constructor is the only place where you can add new variables
    (i.e., class attributes). The signature of your own constructor should
    look like this:

    .. code-block:: python

        def __init__(self, **kwargs):

    meaning that all arguments are passed as keyword arguments. Also, make
    sure to call the BaseModel constructor first thing. So a complete
    example of a constructor could look like this:

    .. code-block:: python

        class MyModel(BaseModel):

            def __init__(self, **kwargs):
                super(MyModel, self).__init__(self, **kwargs)
                self.newvar = 0

    This is the only place where you can add new class attributes. Trying
    to set ``self.someothervar`` outside the constructor will raise an
    AttributeError. Of course, you can always set ``self.newvar = None`` in
    the constructor to make sure the variable exists, and then assign a new
    value in other class methods.

    .. note::

       Please note: If ``self.newvar`` already exists in the BaseModel class,
       the last line of the above code snippet would overwrite it.

    To make the model complete (and compile), you will also need to fill in
    all methods marked with ``@abc.abstractmethod`` below. These include
    :py:meth:`~pulse2percept.models.BaseModel.build` and
    :py:meth:`~pulse2percept.models.BaseModel.predict_percept`.
    Have a look at the ScoreboardModel or AxonMapModel classes below to get an
    idea of how to write a complete model.

    """
    __slots__ = ('xrange', 'yrange', 'xystep', 'grid', 'grid_type',
                 'thresh_percept', 'engine', 'scheduler', 'n_jobs', 'verbose',
                 '__is_built')

    def __init__(self, **kwargs):
        """Constructor

        Parameters
        ----------
        **kwargs: keyword arguments
            You can set individual model parameters by passing them as keyword
            arguments (e.g., ``MyModel(engine='joblib')``). Note that these
            parameters must be listed in ``get_params``. If no kwargs are
            passed, all model parameters will be initialized with default
            values. You can add more parameters to your model by subclassing
            ``_get_default_params``.
        """
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

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
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
        f_caller = sys._getframe(1).f_code.co_name
        if f_caller in ["__init__", "build"]:
            self.__is_built = val
        else:
            print(sys._getframe(0).f_code.co_name)
            print(sys._getframe(1).f_code.co_name)
            print(sys._getframe(2).f_code.co_name)
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

        - all ``build_params`` take effect,
        - the flag ``_is_built`` is set,
        - the method returns ``self``.

        One way to do this is to call the BaseModel's ``build`` method from
        within your own model:

            class MyModel(BaseModel):

                def build(self, \*\*build_params):
                    super(MyModel, self).build(self, \*\*build_params)
                    # Add your own code here...

        Parameters
        ----------
        \*\*build_params : Additional build parameters
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

    def predict_percept(self, implant, t=None):
        """Predict a percept

        Parameters
        ----------
        implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
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
        if implant.stim.time is not None:
            # The stimulus has a time dimension
            raise NotImplementedError

        # The stimulus does not have a time dimesnion. In this case, we
        # only need to run the spatial model:
        return parfor(self._predict_pixel_percept,
                      enumerate(self.grid),
                      func_args=[implant],
                      func_kwargs={'t': t},
                      engine=self.engine, scheduler=self.scheduler,
                      n_jobs=self.n_jobs,
                      out_shape=self.grid.shape)
