"""

pulse2percept is organized into the following subpackages:

.. autosummary::
    :toctree: _api

    implants
    stimuli
    models
    percepts
    datasets
    model_selection
    viz
    utils
    topography
"""
import matplotlib as mpl
from os import environ
from sys import platform
# Use TkAgg on OS X:
# https://stackoverflow.com/a/32082076
# https://stackoverflow.com/a/21789908
if platform == "darwin":
    mpl.use('TkAgg')
else:
    # Use Agg if there's no display:
    # https://stackoverflow.com/a/40931739
    if 'inline' not in mpl.get_backend():
        if environ.get('DISPLAY', '') == '':
            mpl.use('Agg')

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
from . import models
from . import model_selection
from . import percepts
from . import stimuli
from . import viz

__all__ = [
    'datasets',
    'implants',
    'models',
    'model_selection',
    'percepts',
    'stimuli',
    'topography',
    'utils',
    'viz'
]
