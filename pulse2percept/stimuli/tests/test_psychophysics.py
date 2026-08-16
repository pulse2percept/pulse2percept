import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.stimuli import GratingStimulus, BarStimulus
from pulse2percept.units import (DimensionMismatchError, dimensionless,
                                 ms, uA)
from pulse2percept.units import s as sec


def test_GratingStimulus():
    shape = (5, 5)
    grating = GratingStimulus(shape, spatial_freq=0.25)
    npt.assert_equal(grating.shape, (np.prod(shape), 51))
    npt.assert_equal(grating.vid_shape, (shape[0], shape[1], 51))
    npt.assert_almost_equal(grating.data.min(), 0)
    npt.assert_almost_equal(grating.data.max(), 1)

    # Drifting to the left/right:
    nx = 3
    for direction in [0, 180]:
        # A grating with 1-px white bar, drifting 1 column per frame:
        grating = GratingStimulus((nx, nx), direction=direction,
                                  spatial_freq=1.0 / nx,
                                  temporal_freq=1.0 / nx, time=np.arange(nx))
        data = grating.data.reshape(grating.vid_shape)
        for i in range(nx):
            if direction == 0:
                npt.assert_almost_equal(data[:, i, (i + 1) % nx], 1)
            else:
                npt.assert_almost_equal(data[:, nx - i - 1, i], 1)

    # Contrast vs. mask:
    for mask in ['circle', None]:
        # Mask will have value 0.5, so contrast still defines the min/max:
        grating = GratingStimulus(shape, spatial_freq=0.25, contrast=0.45,
                                  mask=mask)
        npt.assert_almost_equal(grating.data.max() - grating.data.min(), 0.45)
        npt.assert_almost_equal(grating.data.min(), 0.275)
        npt.assert_almost_equal(grating.data.max(), 0.725)

    # Masks:
    for mask in ['circle', 'gauss']:
        grating = GratingStimulus(shape, mask=mask)
        npt.assert_almost_equal(grating.data[:2, :].ravel(), 0.5, decimal=2)
        npt.assert_almost_equal(grating.data[3:5, :].ravel(), 0.5, decimal=2)
        npt.assert_almost_equal(grating.data[-2:, :].ravel(), 0.5, decimal=2)


def test_BarStimulus():
    shape = (15, 15)
    bar = BarStimulus(shape)
    npt.assert_equal(bar.shape, (np.prod(shape), 51))
    npt.assert_equal(bar.vid_shape, (shape[0], shape[1], 51))
    npt.assert_almost_equal(bar.data.max(), 1)

    # Contrast vs. mask:
    for mask in ['circle', 'gauss', None]:
        # Mask will have value 0.5, so contrast still defines the min/max:
        bar = BarStimulus(shape, contrast=0.45, mask=mask)
        npt.assert_almost_equal(bar.data.max() - bar.data.min(), 0.45,
                                decimal=2)
        npt.assert_almost_equal(bar.data.min(), 0.275, decimal=2)
        npt.assert_almost_equal(bar.data.max(), 0.725, decimal=2)

    # Masks:
    for mask in ['circle', 'gauss']:
        bar = BarStimulus(shape, mask=mask)
        npt.assert_almost_equal(bar.data[:2, :].ravel(), 0.5, decimal=2)
        npt.assert_almost_equal(bar.data[3:5, :].ravel(), 0.5, decimal=2)
        npt.assert_almost_equal(bar.data[-2:, :].ravel(), 0.5, decimal=2)


def test_psychophysics_time_units():
    # `time` is a duration, so it may be given as one:
    for cls, kwargs in [(GratingStimulus, {}), (BarStimulus, {})]:
        bare = cls((4, 4), time=100, **kwargs)
        unitful = cls((4, 4), time=0.1 * sec, **kwargs)
        npt.assert_array_equal(bare.data, unitful.data)
        npt.assert_array_equal(bare.time, unitful.time)
        # An explicit list of time points works too:
        listed = cls((4, 4), time=[0, 20, 40] * ms, **kwargs)
        npt.assert_almost_equal(listed.time, [0, 20, 40])
        # These are visual stimuli: their pixels are gray levels, but their
        # time is still physical.
        npt.assert_equal(unitful.unit, dimensionless)
        npt.assert_equal(unitful.time_unit, ms)
        with pytest.raises(DimensionMismatchError):
            cls((4, 4), time=5 * uA, **kwargs)
