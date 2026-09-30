"""
pulse2percept is organized into the following subpackages:

.. autosummary::
    :toctree: _api

    implants
    stimuli
    models
    percepts
    datasets
    plotting
    utils
    topography
    units
    vision
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

# Libraries should not configure logging (see the logging docs). A NullHandler
# silences the "no handler" fallback without touching the root logger. Call
# ``set_debug_logging`` to write a debug file.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def set_debug_logging(fname="debug.log", level=logging.DEBUG, filemode="w"):
    """Write pulse2percept's log messages to a file

    Importing pulse2percept does not configure logging. This function adds a
    file handler to the ``pulse2percept`` logger only (not the root logger),
    so messages from other libraries are not captured.

    Parameters
    ----------
    fname : str, optional
        File to write the log to.
    level : int, optional
        Logging level, e.g. ``logging.DEBUG`` or ``logging.INFO``.
    filemode : str, optional
        ``'w'`` to start a fresh log, ``'a'`` to append to an existing one.

    Returns
    -------
    handler : :py:class:`logging.FileHandler`
        The handler that was installed. Pass it to
        ``logging.getLogger('pulse2percept').removeHandler`` to undo.

    Examples
    --------
    >>> import pulse2percept as p2p
    >>> handler = p2p.set_debug_logging()  # doctest: +SKIP

    """
    handler = logging.FileHandler(fname, mode=filemode)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler


from . import datasets
from . import implants
from . import models
from . import percepts
from . import plotting
from . import stimuli
from . import topography
from . import units
from . import utils
from . import vision
# Deprecated; re-exports from `plotting`:
from . import viz

__all__ = [
    "datasets",
    "implants",
    "models",
    "percepts",
    "plotting",
    "set_debug_logging",
    "stimuli",
    "topography",
    "units",
    "utils",
    "vision",
    "viz",
]
