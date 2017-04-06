from __future__ import absolute_import, division, print_function
from .version import __version__  # noqa
import logging

# Disable Jupyter Notebook handlers
# https://github.com/ipython/ipython/issues/8282
logging.getLogger().handlers = []

# Set up root logger for debug file
formatstr = '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.DEBUG,
                    format=formatstr,
                    filename='debug.log',
                    filemode='w')

# Add streaming to console: Temporarily set level to ERROR so the imports
# don't trigger 'deprecated' warnings
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
console.setFormatter(logging.Formatter(formatstr))
logging.getLogger(__name__).addHandler(console)

from .api import *
from . import retina
from . import implants
from . import stimuli
from . import files
from . import utils

# Reset the logging level to INFO
console.setLevel(logging.INFO)
logging.getLogger(__name__).info('Welcome to pulse2percept')
