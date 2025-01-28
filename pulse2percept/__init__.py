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

# Lazy import mechanism for user-friendly API
def __getattr__(name):
    if name in __all__:
        try:
            # Use __import__ for lazy import
            module = __import__(f"{__name__}.{name}", fromlist=[""])
            globals()[name] = module
            return module
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Failed to import submodule '{name}' in module '{__name__}'."
                f" Check if the module is installed correctly."
            ) from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
