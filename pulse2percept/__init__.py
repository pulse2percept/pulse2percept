"""

pulse2percept is organized into the following subpackages:

.. autosummary::
    :toctree: _api

    implants
    stimuli
    models
    percepts
    datasets
    viz
    utils
"""
# This import is necessary to ensure consistency of the generated images across
# platforms, and for the tests to run on Travis:
# https://stackoverflow.com/questions/35403127/testing-matplotlib-based-plots-in-travis-ci
# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python
# http://www.davidketcheson.info/2015/01/13/using_matplotlib_image_comparison.html
from sys import platform
import matplotlib as mpl
import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
# if platform == "darwin":  # OS X
#     mpl.use('TkAgg')

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
from . import percepts
from . import stimuli
from . import viz

__all__ = [
    'datasets',
    'implants',
    'models',
    'percepts',
    'stimuli',
    'utils',
    'viz'
]
