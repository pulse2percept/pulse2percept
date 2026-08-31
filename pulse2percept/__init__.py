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

# A library must not configure logging on behalf of the application that
# imports it: handlers, levels and destinations are the application's to
# choose. Attaching a NullHandler to our own logger silences the "no handler
# could be found" fallback without touching the root logger, which is what the
# logging documentation prescribes for libraries. Call ``set_debug_logging``
# to opt in to the debug file that used to be configured on import.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def set_debug_logging(fname="debug.log", level=logging.DEBUG, filemode="w"):
    """Write pulse2percept's log messages to a file

    Importing pulse2percept does not configure logging: which messages are
    emitted, and where they go, is the application's decision. Call this to
    opt in to a file-based log of pulse2percept's own messages.

    Note that this configures the ``pulse2percept`` logger only, not the root
    logger, so it will not capture messages from other libraries.

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
