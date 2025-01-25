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
import logging
from importlib.metadata import version, PackageNotFoundError

# Use TkAgg on macOS, Agg elsewhere if no display:
if platform == "darwin":
    mpl.use("TkAgg")
else:
    if "inline" not in mpl.get_backend():
        if environ.get("DISPLAY", "") == "":
            mpl.use("Agg")

# Fetch version from pyproject.toml
try:
    __version__ = version("pulse2percept")
except PackageNotFoundError:
    __version__ = "unknown"

# Disable Jupyter Notebook handlers
# https://github.com/ipython/ipython/issues/8282
logging.getLogger().handlers = []

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Set up root logger for debug file
formatstr = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.DEBUG, format=formatstr, filename="debug.log", filemode="w"
)

from . import datasets
from . import implants
from . import models
from . import model_selection
from . import percepts
from . import stimuli
from . import utils
from . import viz

__all__ = [
    "datasets",
    "implants",
    "models",
    "model_selection",
    "percepts",
    "stimuli",
    "topography",
    "utils",
    "viz",
]
