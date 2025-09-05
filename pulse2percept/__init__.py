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
    topography
"""
from __future__ import annotations

import logging
from os import environ
from sys import platform
from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Matplotlib backend hygiene
import matplotlib as mpl
if platform == "darwin":
    mpl.use("TkAgg")
else:
    if "inline" not in mpl.get_backend():
        if environ.get("DISPLAY", "") == "":
            mpl.use("Agg")

# Package version
try:
    __version__ = _pkg_version("pulse2percept")
except PackageNotFoundError:
    __version__ = "unknown"

# Disable Jupyter Notebook handlers (https://github.com/ipython/ipython/issues/8282)
logging.getLogger().handlers = []
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

_format = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.DEBUG, format=_format, filename="debug.log", filemode="w")

# Lazy import of submodules to avoid importing C extensions at top-level
_SUBMODULES = {
    "datasets",
    "implants",
    "models",
    "percepts",
    "stimuli",
    "utils",
    "viz",
    "topography",
}

__all__ = sorted(list(_SUBMODULES))

def __getattr__(name: str):
    # Lazy-load known submodules on first access
    if name in _SUBMODULES:
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod  # cache for subsequent access
        return mod
    raise AttributeError(name)

def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES))
