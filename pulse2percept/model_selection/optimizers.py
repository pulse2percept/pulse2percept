import numpy as np
from scipy.optimize import minimize

try:
    from sklearn.base import BaseEstimator, clone as clone_estimator
    from sklearn.utils.validation import check_is_fitted
    has_sklearn = True
except ImportError:
    has_sklearn = False

try:
    import pyswarm
    has_pyswarm = True
except ImportError:
    has_pyswarm = False


class FunctionMinimizer(BaseEstimator):

    def __init__(self, estimator, search_params, search_params_init=None,
                 method='L-BFGS-B', max_iter=50, print_iter=1, min_step=1e-5,
                 verbose=True):
        """Performs function minimization

        Parameters
        ----------
        estimator :
            A scikit-learn estimator. Make sure its scoring function has
            greater equals better.
        search_params : dict of tupels (lower bound, upper bound)
            Search parameters
        search_params_init : dict of floats, optional, default: None
            Initial values of all search parameters. If None, initialize to
            midpoint between lower and upper bounds
        method : str, optional, default: 'L-BFGS-B'
            Solving method to use (e.g., 'Nelder-Mead', 'Powell', 'L-BFGS-B')
        max_iter : int, optional, default: 100
            Maximum number of iterations for the swarm to search.
        print_iter : int, optional, default: 10
            Print status message every x iterations
        min_step : float, optional, default: 0.1
            Minimum gradient change before termination.
        verbose : bool, optional, default: True
            Flag whether to print more stuff
        """
        self.estimator = estimator
        if not hasattr(estimator, 'greater_is_better'):
            raise ValueError(("%s must have an attribute "
                              "'greater_is_better'" % estimator))
        self.search_params = search_params
        if search_params_init is None:
            search_params_init = {}
            for k, v in self.search_params.items():
                search_params_init[k] = (v[1] - v[0]) / 2.0
        self.search_params_init = search_params_init
        self.method = method
        self.max_iter = max_iter
        self.print_iter = print_iter
        self.min_step = min_step
        self.verbose = verbose

    def calc_error(self, search_vals, X, y, fit_params=None):
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
        if np.mod(self.iter, self.print_iter) == 0:
            print("Iter %d: Loss=%f, %s" % (
                self.iter, loss, ', '.join(['%s: %f' % (k, v)
                                            for k, v
                                            in search_params.items()])))
        self.iter += 1
        return loss

    def fit(self, X, y, fit_params=None):
        """Runs the optimizer"""
        self.iter = 0
        # (lower, upper) bounds for every parameter
        bounds = [v for v in self.search_params.values()]
        init = [v for v in self.search_params_init.values()]
        options = {'maxfun': self.max_iter, 'gtol': self.min_step, 'eps': 100}
        res = minimize(self.calc_error, init, args=(X, y, fit_params),
                       bounds=bounds, options=options)
        if not res['success']:
            print('Optimization unsucessful:')
            print(res)

        # Pair values of best params with their names to build a dict
        self.best_params_ = {}
        for k, v in zip(list(self.search_params.keys()), res['x']):
            self.best_params_[k] = v
        self.best_train_score_ = res['fun']
        print('Best err:', res['fun'], 'Best params:', self.best_params_)

        # Fit the class attribute with best params
        if fit_params is None:
            fit_params = {}
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y=y, **fit_params)

    def predict(self, X):
        msg = "Estimator, %(name)s, must be fitted before predicting."
        check_is_fitted(self, "best_params_", msg=msg)
        return self.estimator.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y, sample_weight=None)


class GridSearchOptimizer(BaseEstimator):

    def __init__(self, estimator, search_params, verbose=True):
        """Performs a grid search

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
        self.estimator = estimator
        if not hasattr(estimator, 'greater_is_better'):
            raise ValueError(("%s must have an attribute "
                              "'greater_is_better'" % estimator))
        self.search_params = search_params
        self.verbose = verbose

    def fit(self, X, y, fit_params=None):
        if fit_params is None:
            fit_params = {}
        best_params = {}
        best_loss = np.inf
        for params in self.search_params:
            estimator = clone_estimator(self.estimator)
            estimator.set_params(**params)
            estimator.fit(X, y=y, **fit_params)
            loss = estimator.score(X, y)
            loss = -loss if estimator.greater_is_better else loss
            if loss < best_loss:
                best_loss = loss
                best_params = params
        self.best_params_ = best_params
        print('Best err:', best_loss, 'Best params:', self.best_params_)

        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y=y, **fit_params)
        return self

    def predict(self, X):
        msg = "Estimator, %(name)s, must be fitted before predicting."
        check_is_fitted(self, "best_params_", msg=msg)
        return self.estimator.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y, sample_weight=None)


class ParticleSwarmOptimizer(BaseEstimator):

    def __init__(self, estimator, search_params, swarm_size=None, max_iter=50,
                 min_func=0.01, min_step=0.01, verbose=True):
        """Performs particle swarm optimization

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
        if not has_pyswarm:
            raise ImportError("You do not have pyswarm installed. "
                              "You can install it via $ pip install pyswarm.")
        if swarm_size is None:
            swarm_size = 10 * len(search_params)
        self.estimator = estimator
        if not hasattr(estimator, 'greater_is_better'):
            raise ValueError(("%s must have an attribute "
                              "'greater_is_better'" % estimator))
        self.search_params = search_params
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.min_func = min_func
        self.min_step = min_step
        self.verbose = verbose

    def swarm_error(self, search_vals, X, y, fit_params=None):
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

    def fit(self, X, y, fit_params=None):
        # Run particle swarm optimization
        lb = [v[0] for v in self.search_params.values()]
        ub = [v[1] for v in self.search_params.values()]
        best_vals, best_err = pyswarm.pso(
            self.swarm_error, lb, ub, swarmsize=self.swarm_size,
            maxiter=self.max_iter, minfunc=self.min_func,
            minstep=self.min_step, debug=self.verbose, args=[X, y],
            kwargs={'fit_params': fit_params}
        )

        # Pair values of best params with their names to build a dict
        self.best_params_ = {}
        for k, v in zip(list(self.search_params.keys()), best_vals):
            self.best_params_[k] = v
        self.best_train_score_ = best_err
        print('Best err:', best_err, 'Best params:', self.best_params_)

        # Fit the class attribute with best params
        if fit_params is None:
            fit_params = {}
        self.estimator.set_params(**self.best_params_)
        self.estimator.fit(X, y=y, **fit_params)

    def predict(self, X):
        msg = "Estimator, %(name)s, must be fitted before predicting."
        check_is_fitted(self, "best_params_", msg=msg)
        return self.estimator.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y, sample_weight=None)
