from __future__ import absolute_import, division, print_function
from .version import __version__  # noqa
from .api import *
from . import retina
from . import implants
from . import stimuli
from . import files
from . import utils

import logging


# Disable Jupyter Notebook handlers
# https://github.com/ipython/ipython/issues/8282
logging.getLogger().handlers = []

# Set up root logger
formatstr = '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=formatstr,
                    filename='debug.log',
                    filemode='w')
logging.getLogger(__name__).info('Welcome to pulse2percept')

# Add streaming to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(formatstr))
logging.getLogger(__name__).addHandler(console)
