"""

pulse2percept is organized into the following subpackages:

.. autosummary::
    :toctree: _api

    implants
    stimuli
    models
    percepts
    datasets
    io
    viz
    utils
"""
import logging
from .version import __version__


# Disable Jupyter Notebook handlers
# https://github.com/ipython/ipython/issues/8282
logging.getLogger().handlers = []

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Set up root logger for debug file
formatstr = '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=formatstr,
                    filename='debug.log',
                    filemode='w')

from . import datasets
from . import implants
# Avoid showing avconv/avprob error:
logger.setLevel(logging.ERROR)
from . import io
logger.setLevel(logging.INFO)
from . import models
from . import percepts
from . import stimuli
from . import viz

__all__ = [
    'datasets',
    'implants',
    'io',
    'models',
    'percepts',
    'stimuli',
    'utils',
    'viz'
]
