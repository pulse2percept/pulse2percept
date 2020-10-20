import numpy.testing as npt
import pytest

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
# from sklearn.model_selection import ParameterGrid


from pulse2percept.model_selection import (FunctionMinimizer, NotFittedError,
                                           GridSearchOptimizer,
                                           ParticleSwarmOptimizer)


def generate_dummy_data():
    X = pd.DataFrame()
    X['subject'] = pd.Series(['S1', 'S1', 'S2', 'S2', 'S3', 'S3'])
    X['feature1'] = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    X['feature2'] = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = pd.DataFrame()
    y['subject'] = pd.Series(['S1', 'S1', 'S2', 'S2', 'S3', 'S3'],
                             index=X.index)
    y['target'] = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                            index=X.index)
    y['image'] = pd.Series([np.random.rand(10, 10)] * 6)
    y['area'] = pd.Series([1, 2, 3, 4, 5, 6])
    return X, y


class DummyPredictor(BaseEstimator):

    def __init__(self, dummy_var=1):
        self.dummy_var = dummy_var

    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X):
        return X['feature1']

    def score(self, X, y, sample_weight=None):
        return np.sum((y['target'] - self.dummy_var * X['feature1']) ** 2)


def test_FunctionMinimizer():
    # DummyPredictor always predicts 'feature1'.
    # The best `dummy_var` value is 1.
    X, y = generate_dummy_data()
    fmin = FunctionMinimizer(DummyPredictor(), {'dummy_var': (0.5, 2.5)},
                             has_loss_function=True)
    with pytest.raises(NotFittedError):
        fmin.predict(X)
    fmin.fit(X, y)
    npt.assert_almost_equal(fmin.estimator.dummy_var, 1.0)
    npt.assert_almost_equal(fmin.score(X, y), 0.0)


def test_GridSearchOptimizer():
    # DummyPredictor always predicts 'feature1'.
    # The best `dummy_var` value is 1.
    X, y = generate_dummy_data()
    search_params = {'dummy_var': np.linspace(0.5, 2, num=4)}
    fmin = GridSearchOptimizer(DummyPredictor(), search_params,
                               has_loss_function=True)
    # ParameterGrid(search_params))
    with pytest.raises(NotFittedError):
        fmin.predict(X)
    fmin.fit(X, y)
    npt.assert_almost_equal(fmin.estimator.dummy_var, 1.0)
    npt.assert_almost_equal(fmin.score(X, y), 0.0)


def test_ParticleSwarmOptimizer():
    # DummyPredictor always predicts 'feature1'.
    # The best `dummy_var` value is 1.
    X, y = generate_dummy_data()
    fmin = ParticleSwarmOptimizer(DummyPredictor(),
                                  {'dummy_var': (0.5, 2.5)},
                                  min_func=1e-6, min_step=1e-6,
                                  has_loss_function=True)
    with pytest.raises(NotFittedError):
        fmin.predict(X)
    # Test both {} and None:
    fmin.fit(X, y, fit_params=None)
    fmin.fit(X, y, fit_params={})
    npt.assert_almost_equal(fmin.estimator.dummy_var, 1.0, decimal=2)
    npt.assert_almost_equal(fmin.score(X, y), 0.0, decimal=2)
