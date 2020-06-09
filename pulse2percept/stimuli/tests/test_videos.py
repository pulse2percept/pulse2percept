import os
import numpy as np
import numpy.testing as npt
import pytest
from imageio import mimwrite

from pulse2percept.stimuli import VideoStimulus


def test_VideoStimulus():
    # Create a dummy video:
    fname = 'test.mp4'
    shape = (10, 32, 48)
    ndarray = np.random.rand(*shape)
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    stim = VideoStimulus(fname)
    npt.assert_equal(stim.shape, (np.prod(shape[1:]), shape[0]))
    npt.assert_almost_equal(stim.data,
                            ndarray.reshape((shape[0], -1)).transpose(),
                            decimal=1)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_size'], (shape[2], shape[1]))
    npt.assert_equal(stim.time, np.arange(shape[0]))
    npt.assert_equal(stim.electrodes, np.arange(np.prod(shape[1:])))
    os.remove(fname)

    # Resize the video:
    ndarray = np.ones(shape)
    mimwrite(fname, (255 * ndarray).astype(np.uint8), fps=1)
    resize = (16, 32)
    stim = VideoStimulus(fname, resize=resize)
    npt.assert_equal(stim.shape, (np.prod(resize), shape[0]))
    npt.assert_almost_equal(stim.data,
                            np.ones((np.prod(resize), shape[0])), decimal=1)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_size'], (shape[2], shape[1]))
    npt.assert_equal(stim.time, np.arange(shape[0]))
    npt.assert_equal(stim.electrodes, np.arange(np.prod(resize)))
    os.remove(fname)
