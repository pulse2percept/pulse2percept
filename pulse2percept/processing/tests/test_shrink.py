import os
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.stimuli import ImageStimulus, VideoStimulus, BostonTrain
from pulse2percept.processing import shrinked_single_image, shrinked_image, shrinked_video, shrinked_stim, _spatial_temporal_saliency


def test_shrinked_single_image():
    image = np.zeros((20, 20))
    for i in range(1, 8):
        for j in range(1, 8):
            image[i, j] = 1
    for i in range(10, 15):
        for j in range(14, 19):
            image[i, j] = 1
    shrinked = shrinked_single_image(image, wid=10, hei=10, L=1)
    npt.assert_equal(shrinked[3, 3], 1.0)
    npt.assert_equal(shrinked[4, 4], 0.0)
    npt.assert_almost_equal(shrinked[5, 5], 0.35745859)
    npt.assert_equal(shrinked[6, 6], 1.0)


def test_shrinked_image():
    image = np.zeros((15, 15))
    for i in range(3, 8):
        for j in range(4, 9):
            image[i, j] = 1
    for i in range(9, 13):
        for j in range(11, 15):
            image[i, j] = 1
    second_frame = np.zeros((15, 15))
    for i in range(4, 9):
        for j in range(3, 8):
            second_frame[i, j] = 1
    for i in range(11, 15):
        for j in range(10, 14):
            second_frame[i, j] = 1
    shrinked = shrinked_image(image, second_frame, wid=6, hei=6, L=1, num=5)
    npt.assert_equal(shrinked[3, 4], 1.0)
    npt.assert_almost_equal(shrinked[5, 6], 0.2399997)
    npt.assert_almost_equal(shrinked[6, 7], 1.0)
    npt.assert_almost_equal(shrinked[6, 6], 0.3999997)


def test_shrinked_video():
    video = np.zeros((3, 15, 15))
    for i in range(4, 9):
        for j in range(3, 8):
            video[0, i, j] = 1
    for i in range(11, 15):
        for j in range(10, 14):
            video[0, i, j] = 1
    for i in range(3, 8):
        for j in range(4, 9):
            video[1, i, j] = 1
    for i in range(9, 13):
        for j in range(11, 15):
            video[1, i, j] = 1
    for i in range(2, 7):
        for j in range(4, 10):
            video[2, i, j] = 1
    for i in range(10, 14):
        for j in range(11, 15):
            video[2, i, j] = 1
    shrinked = shrinked_video(video, wid=3, hei=3, L=2, num=5)
    npt.assert_almost_equal(shrinked[0, 3, 4], 0.79999995)
    npt.assert_almost_equal(shrinked[0, 5, 6], 0.40000009)
    npt.assert_almost_equal(shrinked[0, 6, 7], 0.0)
    npt.assert_almost_equal(shrinked[0, 6, 6], 0.40000009)
    npt.assert_almost_equal(shrinked[1, 3, 4], 1.0)
    npt.assert_almost_equal(shrinked[1, 5, 6], 1.0)
    npt.assert_almost_equal(shrinked[1, 6, 7], 0.08000006)
    npt.assert_almost_equal(shrinked[1, 6, 6], 0.40000009)


def test_shrinked_image_Stimulus():
    image = np.zeros((20, 20))
    for i in range(1, 8):
        for j in range(1, 8):
            image[i, j] = 1
    for i in range(10, 15):
        for j in range(14, 19):
            image[i, j] = 1
    stim = ImageStimulus(image)
    shrinked = shrinked_stim(stim, wid=10, hei=10, L=1)
    shrinked_data = shrinked.data.reshape(shrinked.img_shape)
    npt.assert_equal(shrinked_data[3, 3], 1.0)
    npt.assert_equal(shrinked_data[4, 4], 0.0)
    npt.assert_almost_equal(shrinked_data[5, 5], 0.35745859)
    npt.assert_equal(shrinked_data[6, 6], 1.0)


def test_shrinked_video_stimulus():
    video = np.zeros((15, 15, 3, 3))
    for i in range(4, 9):
        for j in range(3, 8):
            video[i, j, :, 0] = 1
    for i in range(11, 15):
        for j in range(10, 14):
            video[i, j, :, 0] = 1
    for i in range(3, 8):
        for j in range(4, 9):
            video[i, j, :, 1] = 1
    for i in range(9, 13):
        for j in range(11, 15):
            video[i, j, :, 2] = 1
    for i in range(2, 7):
        for j in range(4, 10):
            video[i, j, :, 2] = 1
    for i in range(10, 14):
        for j in range(11, 15):
            video[i, j, :, 2] = 1
    stim = VideoStimulus(video)
    shrinked = shrinked_stim(stim, wid=3, hei=3, L=2, num=5)
    shrinked_data = shrinked.data.reshape(shrinked.vid_shape)
    npt.assert_almost_equal(shrinked_data[3, 4, 0], 0.79999995)
    npt.assert_almost_equal(shrinked_data[5, 6, 0], 0.40000009)
    npt.assert_almost_equal(shrinked_data[6, 7, 0], 0.0)
    npt.assert_almost_equal(shrinked_data[6, 6, 0], 0.40000009)
    npt.assert_almost_equal(shrinked_data[3, 4, 1], 1.0)
    npt.assert_almost_equal(shrinked_data[5, 6, 1], 1.0)
    npt.assert_almost_equal(shrinked_data[6, 7, 1], 0.08000006)
    npt.assert_almost_equal(shrinked_data[6, 6, 1], 0.40000009)


def test_spatial_temporal_saliency():
    stim = BostonTrain(as_gray=True)
    video = stim.data.reshape(stim.vid_shape).transpose(2, 0, 1)[
        0:2, 40:140, 100:200]
    saliency = _spatial_temporal_saliency(video[0], video[1])
    npt.assert_almost_equal(saliency[30, 40], 0.0434458)
    npt.assert_almost_equal(saliency[40, 60], 0.1113361)
    npt.assert_almost_equal(saliency[20, 80], 0.0873726)
