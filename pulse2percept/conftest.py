"""Fixtures shared across the pulse2percept test suite.

This lives inside the package rather than at the repository root so that it is
found no matter where pytest is invoked from: the root ``conftest.py`` is only
picked up when the working directory is inside the repository, which is not
guaranteed for ``pytest --pyargs pulse2percept``.
"""
import os

import pytest


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
