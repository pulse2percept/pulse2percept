"""Fixtures shared across the pulse2percept test suite.

This lives inside the package rather than at the repository root so that it is
found no matter where pytest is invoked from: the root ``conftest.py`` is only
picked up when the working directory is inside the repository, which is not
guaranteed for ``pytest --pyargs pulse2percept``.
"""
import os

import numpy as np
import pytest

from pulse2percept.stimuli import VideoStimulus

#: Frame count and rate of a short camera clip, kept as the numbers several
#: tests reason about: 29.97 fps is 33.365 ms per frame, which is
#: incommensurate with the 6 Hz pulse rate Argus II runs at.
CAMERA_N_FRAMES, CAMERA_FPS = 94, 29.97


@pytest.fixture
def camera_video():
    """A drifting grating standing in for a short grayscale camera clip"""
    rows, cols = 60, 80
    x = np.linspace(0, 4 * np.pi, cols)[np.newaxis, :, np.newaxis]
    # A vertical ramp, so that sampling the frame at different rows -- which
    # is what an implant does -- reads different gray levels:
    y = np.linspace(0.5, 1, rows)[:, np.newaxis, np.newaxis]
    phase = 2 * np.pi * np.arange(CAMERA_N_FRAMES) / CAMERA_N_FRAMES
    return VideoStimulus(y * (0.5 + 0.5 * np.sin(x - phase)),
                         metadata={'fps': CAMERA_FPS})


@pytest.fixture(scope='module')
def axon_cache_in_tmp(tmp_path_factory):
    """Keep the axon-map cache out of the working directory.

    ``AxonMapSpatial`` pickles its grown axon bundles to ``axon_pickle``,
    which defaults to the *relative* path ``axons.pickle``. Without this
    fixture, any test run that builds an axon map drops that file into
    whatever directory pytest happened to be started from, and silently reuses
    whatever cache an earlier, unrelated run left behind -- which can mask a
    change in the axon-growing code.

    Module-scoped rather than function-scoped on purpose: tests in the same
    module still share one cache, which is what keeps them fast, but the cache
    cannot outlive the run or escape into the repository.

    Apply it to a whole test module with::

        pytestmark = pytest.mark.usefixtures('axon_cache_in_tmp')
    """
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp('axon_cache'))
    try:
        yield
    finally:
        os.chdir(previous)
