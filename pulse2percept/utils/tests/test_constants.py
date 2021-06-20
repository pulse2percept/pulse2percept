import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.utils.constants import DT, MIN_AMP, ZORDER


def test_zorder():
    npt.assert_equal(ZORDER['foreground'] > ZORDER['background'], True)
    npt.assert_equal(ZORDER['annotate'] > ZORDER['foreground'], True)

# These are silly, but hey - completeness


def test_min_amp():
    npt.assert_equal(MIN_AMP < 1e-4, True)


def test_dt():
    npt.assert_equal(DT < 1e-2, True)
