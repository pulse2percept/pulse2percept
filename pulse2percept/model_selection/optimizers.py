"""
`BaseOptimizer`, `GridSearchOptimizer`, `FunctionMinimizer`,
`NotFittedError`, `ParticleSwarmOptimizer`
"""
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.optimize import minimize

from ..utils import Frozen, PrettyPrint

try:
    from sklearn.base import clone as clone_estimator
    from sklearn.model_selection import ParameterGrid
    has_sklearn = True
except ImportError:
    has_sklearn = False

try:
    import pyswarm
    has_pyswarm = True
except ImportError:
    has_pyswarm = False


class NotFittedError(ValueError, AttributeError):
    """Exception class used to raise if optimizer is used before fitting

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class BaseOptimizer(Frozen, PrettyPrint, metaclass=ABCMeta):
    """BaseOptimizer

    .. versionadded:: 0.7

    Parameters
    ----------
    **params : optional keyword arguments
        All keyword arguments must be listed in ``get_default_params``

    """

    def __init__(self, estimator, search_params, **params):
        if not has_sklearn:
            raise ImportError("You do not have scikit-learn installed. "
                              "You can install it via $ pip install sklearn.")
        if not hasattr(estimator, 'greater_is_better'):
            raise ValueError(("%s must have an attribute "
                              "'greater_is_better'" % estimator))
        # Set all default arguments:
        defaults = self.get_default_params()
        for key, val in defaults.items():
            setattr(self, key, val)
        # Then overwrite any arguments also given in `params`:
        self.estimator = estimator
        if search_params is None:
            search_params = {}
        self.search_params = search_params
        for key, val in params.items():
            if key in defaults:
                setattr(self, key, val)
            else:
                err_str = ("'%s' is not a valid parameter. Choose from: "
                           "%s." % (key, ', '.join(defaults.keys())))
                raise AttributeError(err_str)
        # This flag will be flipped once the ``fit`` method was called
        self._is_fitted = False
        # Successful optimization will populate these fields:
        self._best_params = None
        self._best_score = None

    def get_default_params(self):
        """Return a dict of user-settable model parameters"""
        return {'estimator': None, 'search_params': {}, 'verbose': False}

    def _pprint_params(self):
        """Return a dict of class attributes to display when pretty-printing"""
        return {key: getattr(self, key)
                for key, _ in self.get_default_params().items()}

    @abstractmethod
    def _optimize(self, X, y, fit_params=None):
        raise NotImplementedError

    def fit(self, X, y=None, fit_params=None):
        """Performs the search"""
        self._optimize(X, y, fit_params=fit_params)
        if self.verbose:
            print('Best score: %f, Best params: %f' % (self._best_score,
                                                       self._best_params))
        self.estimator.set_params(**self._best_params)
        if fit_params is None:
            fit_params = {}
        self.estimator.fit(X, y=y, **fit_params)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise NotFittedError("Yout must call ``fit`` first.")
        return self.estimator.predict(X)

    def score(self, X, y, sample_weight=None):
        """Scoring function"""
        return self.estimator.score(X, y, sample_weight=None)

    @property
    def is_fitted(self):
        """A flag indicating whether the model has been fitted"""
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, val):
        """This flag can only be set in the constructor or ``fit``"""
        # getframe(0) is '_is_fitted', getframe(1) is 'set_attr'.
        # getframe(2) is the one we are looking for, and has to be either the
        # construct or ``fit``:
        f_caller = sys._getframe(2).f_code.co_name
        if f_caller in ["__init__", "fit"]:
            self._is_fitted = val
        else:
            err_s = ("The attribute `is_fitted` can only be set in the "
                     "constructor or in ``fit``, not in ``%s``." % f_caller)
            raise AttributeError(err_s)


class GridSearchOptimizer(BaseOptimizer):
    """Performs a grid search

    .. versionadded:: 0.7

    Parameters
    ----------
    estimator :
        A scikit-learn estimator. Make sure it has an attribute called
        `greater_is_better` that is set to True if greater values of
        ``estimator.score`` mean that the score is better (else False).
    search_params : sklearn.model_selection.ParameterGrid
        Grid of parameters with a discrete number of values for each.
        Can be generated from a dictionary:
        ParameterGrid({'param1': np.linspace(lb, ub, num=11)}).
    verbose : bool, optional, default: True
        Flag whether to print more stuff

    """

    def _optimize(self, X, y, fit_params=None):
        if fit_params is None:
            fit_params = {}
        best_params = {}
        best_loss = np.inf
        for params in ParameterGrid(self.search_params):
            estimator = clone_estimator(self.estimator)
            estimator.set_params(**params)
            estimator.fit(X, y=y, **fit_params)
            loss = estimator.score(X, y)
            loss = -loss if estimator.greater_is_better else loss
            if loss < best_loss:
                best_loss = loss
                best_params = params
        self._best_score = best_loss
        self._best_params = best_params


class FunctionMinimizer(BaseOptimizer):
    """Loss function minimization

    This class uses SciPy's `minimize
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
    to find the ``search_params`` that optimizes an ``estimator``'s
    score.

    The ``estimator`` should have an attribute called
    ``greater_is_better``.

    .. versionadded:: 0.7

    Parameters
    ----------
    estimator :
        A scikit-learn estimator. Make sure its scoring function has
        greater equals better.
    search_params : dict of tupels (lower bound, upper bound)
        Search parameters
    search_params_init : dict of floats, optional
        Initial values of all search parameters. If None, initialize to
        midpoint between lower and upper bounds
    method : str, optional
        Solving method to use (e.g., 'Nelder-Mead', 'Powell', 'L-BFGS-B')
    tol: float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options: dict, optional
        A dictionary of solver-specific options.
    verbose : bool, optional
        Flag whether to print progress

    """

    def get_default_params(self):
        params = super(FunctionMinimizer, self).get_default_params()
        params.update({'search_params_init': None,
                       'method': 'L-BFGS-B',
                       'tol': None,
                       'options': None})
        return params

    def _calc_error(self, search_vals, X, y, fit_params=None):
        """Calculates the estimator's error

        The error is calculated using the estimator's scoring function (assumes
        a true scoring function, i.e. greater == better).
        """
        # Need to pair these values with the names of the search params
        # to build a dict
        search_params = {}
        for k, v in zip(list(self.search_params.keys()), search_vals):
            search_params[k] = v

        # Clone the estimator to make sure we have a clean slate
        if fit_params is None:
            fit_params = {}
        estimator = clone_estimator(self.estimator)
        estimator.set_params(**search_params)
        estimator.fit(X, y=y, **fit_params)

        # Loss function: if `greater_is_better`, the estimator's ``score``
        # method is a true scoring function => invert to get an error function
        loss = estimator.score(X, y)
        loss = -loss if estimator.greater_is_better else loss
        return loss

    def _optimize(self, X, y, fit_params=None):
        # (lower, upper) bounds for every parameter
        bounds = [v for v in self.search_params.values()]
        if self.search_params_init is None:
            init = [b_lo + (b_hi - b_lo) / 2 for (b_lo, b_hi) in bounds]
        else:
            init = [v for v in self.search_params_init.values()]
        res = minimize(self._calc_error, init, args=(X, y, fit_params),
                       bounds=bounds, tol=self.tol, options=self.options)
        if not res['success']:
            print('Optimization unsucessful:')
            print(res)

        # Pair values of best params with their names to build a dict
        best_params = {}
        for k, v in zip(list(self.search_params.keys()), res['x']):
            best_params[k] = v
        self._best_score = res['fun']
        self._best_params = best_params


class ParticleSwarmOptimizer(BaseOptimizer):
    """Performs particle swarm optimization

    .. versionadded:: 0.7

    Parameters
    ----------
    estimator :
        A scikit-learn estimator. Make sure it has an attribute called
        `greater_is_better` that is set to True if greater values of
        ``estimator.score`` mean that the score is better (else False).
    search_params : dict of tupels (lower bound, upper bound)
        Search parameters
    swarm_size : int, optional, default: 10 * number of search params
        The number of particles in the swarm.
    max_iter : int, optional, default: 100
        Maximum number of iterations for the swarm to search.
    min_func : float, optional, default: 0.01
        The minimum change of swarm's best objective value before the
        search terminates.
    min_step : float, optional, default: 0.01
        The minimum step size of swarm's best objective value before
        the search terminates.
    verbose : bool, optional, default: True
        Flag whether to print more stuff

    """

    def __init__(self, estimator, search_params, **params):
        if not has_pyswarm:
            raise ImportError("You do not have pyswarm installed. "
                              "You can install it via $ pip install pyswarm.")
        super(ParticleSwarmOptimizer, self).__init__(estimator, search_params,
                                                     **params)
        if self.swarm_size is None:
            self.swarm_size = 10 * len(search_params)

    def get_default_params(self):
        params = super(ParticleSwarmOptimizer, self).get_default_params()
        params.update({'swarm_size': None,
                       'max_iter': 100,
                       'min_func': 0.01,
                       'min_step': 0.01})
        return params

    def _swarm_error(self, search_vals, X, y, fit_params=None):
        """Calculates the particle swarm error

        The error is calculated using the estimator's scoring function (assumes
        a true scoring function, i.e. greater == better).
        """
        # pyswarm provides values for all search parameters in a list:
        # Need to pair these values with the names of the search params
        # to build a dict
        search_params = {}
        for k, v in zip(list(self.search_params.keys()), search_vals):
            search_params[k] = v

        # Clone the estimator to make sure we have a clean slate
        if fit_params is None:
            fit_params = {}
        estimator = clone_estimator(self.estimator)
        estimator.set_params(**search_params)
        estimator.fit(X, y=y, **fit_params)

        # Loss function: if `greater_is_better`, the estimator's ``score``
        # method is a true scoring function => invert to get an error function
        loss = estimator.score(X, y)
        loss = -loss if estimator.greater_is_better else loss
        return loss

    def _optimize(self, X, y, fit_params=None):
        # Run particle swarm optimization
        lb = [v[0] for v in self.search_params.values()]
        ub = [v[1] for v in self.search_params.values()]
        best_vals, best_err = pyswarm.pso(self._swarm_error, lb, ub,
                                          swarmsize=self.swarm_size,
                                          maxiter=self.max_iter,
                                          minfunc=self.min_func,
                                          minstep=self.min_step,
                                          debug=self.verbose,
                                          args=[X, y],
                                          kwargs={'fit_params': fit_params})

        # Pair values of best params with their names to build a dict
        best_params = {}
        for k, v in zip(list(self.search_params.keys()), best_vals):
            best_params[k] = v
        self._best_score = best_err
        self._best_params = best_params
