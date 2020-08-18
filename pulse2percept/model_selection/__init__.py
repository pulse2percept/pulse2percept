"""Model selection

.. autosummary::
    :toctree: _api

    base
    predictors
    optimizers

"""
from .base import cross_val_predict
from .optimizers import (FunctionMinimizer, GridSearchOptimizer,
                         ParticleSwarmOptimizer)

__all__ = [
    'cross_val_predict',
    'FunctionMinimizer',
    'GridSearchOptimizer',
    'ParticleSwarmOptimizer'
]
