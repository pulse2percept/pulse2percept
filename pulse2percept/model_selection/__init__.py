"""Model selection

.. autosummary::
    :toctree: _api

    optimizers

"""
from .optimizers import (BaseOptimizer, FunctionMinimizer, GridSearchOptimizer,
                         NotFittedError, ParticleSwarmOptimizer)

__all__ = [
    'BaseOptimizer',
    'FunctionMinimizer',
    'GridSearchOptimizer',
    'NotFittedError',
    'ParticleSwarmOptimizer'
]
