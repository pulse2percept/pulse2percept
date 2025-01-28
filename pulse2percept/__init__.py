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

# Lazy import submodules to avoid circular imports
def _lazy_import(submodule_name):
    import importlib
    return importlib.import_module(f"pulse2percept.{submodule_name}")

datasets = _lazy_import("datasets")
implants = _lazy_import("implants")
models = _lazy_import("models")
model_selection = _lazy_import("model_selection")
percepts = _lazy_import("percepts")
stimuli = _lazy_import("stimuli")
topography = _lazy_import("topography")
utils = _lazy_import("utils")
viz = _lazy_import("viz")

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
