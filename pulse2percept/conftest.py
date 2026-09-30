"""Fixtures shared across the pulse2percept test suite.

Lives inside the package so that ``pytest --pyargs pulse2percept`` finds it
from any working directory; the root ``conftest.py`` is only found from inside
the repository.
"""
import os

import numpy as np
import pytest

from pulse2percept.stimuli import VideoStimulus

#: Frame count and rate of a short camera clip. 29.97 fps (33.367 ms per
#: frame) is incommensurate with the 6 Hz Argus II pulse rate.
CAMERA_N_FRAMES, CAMERA_FPS = 94, 29.97


@pytest.fixture
def camera_video():
    """A drifting grating as a short grayscale camera clip"""
    rows, cols = 60, 80
    x = np.linspace(0, 4 * np.pi, cols)[np.newaxis, :, np.newaxis]
    # Vertical ramp, so that different rows have different gray levels:
    y = np.linspace(0.5, 1, rows)[:, np.newaxis, np.newaxis]
    phase = 2 * np.pi * np.arange(CAMERA_N_FRAMES) / CAMERA_N_FRAMES
    return VideoStimulus(y * (0.5 + 0.5 * np.sin(x - phase)),
                         metadata={'fps': CAMERA_FPS})


@pytest.fixture(scope='module')
def axon_cache_in_tmp(tmp_path_factory):
    """Keep the axon-map cache out of the working directory.

    ``AxonMapSpatial`` pickles axon bundles to ``axon_pickle``, which defaults
    to the relative path ``axons.pickle``. Without this fixture, tests write
    the cache to the current directory and may reuse a stale cache from an
    earlier run.

    Module-scoped, so tests in one module share a cache for speed.

    Apply it to a whole test module with::

        pytestmark = pytest.mark.usefixtures('axon_cache_in_tmp')
    """
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('axon_cache'))
    try:
        yield
    finally:
        os.chdir(previous)
