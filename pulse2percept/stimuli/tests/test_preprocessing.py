import os
import numpy as np
import numpy.testing as npt
import pytest

from pulse2percept.stimuli import ImageStimulus, VideoStimulus, BostonTrain
from pulse2percept.stimuli._preprocessing import single_image_retargeting, image_retargeting, video_retargeting, stim_retargeting, _spatial_temporal_saliency
from pulse2percept.stimuli.preprocessing import (center_image, shift_image, scale_image,
                                                 trim_image)


def test_shift_image():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, :] = 255
    # Top row:
    top = shift_image(img, 0, -2)
    npt.assert_almost_equal(top[0, :], 255)
    npt.assert_almost_equal(top[1:, :], 0)
    # Bottom row:
    bottom = shift_image(img, 0, 2)
    npt.assert_almost_equal(bottom[:4, :], 0)
    npt.assert_almost_equal(bottom[4, :], 255)
    # Bottom right pixel:
    bottom = shift_image(img, 4, 2)
    npt.assert_almost_equal(bottom[4, 4], 255)
    npt.assert_almost_equal(bottom[:4, :], 0)
    npt.assert_almost_equal(bottom[:, :4], 0)
    with pytest.raises(ValueError):
        shift_image(np.ones(10), 1, 1)
    with pytest.raises(ValueError):
        shift_image(np.ones((3, 4, 5, 6)), 1, 1)


def test_center_image():
    # Create a horizontal bar:
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, :] = 255
    # Center phosphene:
    npt.assert_almost_equal(img, center_image(img), decimal=3)
    npt.assert_almost_equal(img, center_image(shift_image(img, 0, 2)))
    with pytest.raises(ValueError):
        center_image(np.ones(10))
    with pytest.raises(ValueError):
        center_image(np.ones((3, 4, 5, 6)))


def test_scale_image():
    # Create a horizontal bar:
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, :] = 255
    # Scale phosphene:
    npt.assert_almost_equal(img, scale_image(img, 1))
    npt.assert_almost_equal(scale_image(img, 0.1).ravel()[12], 255)
    npt.assert_almost_equal(scale_image(img, 0.1).ravel()[:12], 0)
    npt.assert_almost_equal(scale_image(img, 0.1).ravel()[13:], 0)
    with pytest.raises(ValueError):
        scale_image(np.ones(10), 1)
    with pytest.raises(ValueError):
        scale_image(np.ones((3, 4, 5, 6)), 1)
    with pytest.raises(ValueError):
        scale_image(img, 0)


@pytest.mark.parametrize('n_channels', (1, 3))
def test_trim_image(n_channels):
    img = np.squeeze(np.zeros((13, 29, n_channels), dtype=float))
    shape = img.shape
    img[1:-1, 1:-1, ...] = 0.1
    img[2:-2, 2:-2, ...] = 0.2
    npt.assert_equal(trim_image(img).shape[:2], (shape[0] - 2, shape[1] - 2))
    npt.assert_equal(trim_image(img, tol=0.05).shape[:2],
                     (shape[0] - 2, shape[1] - 2))
    npt.assert_equal(trim_image(img, tol=0.1).shape[:2],
                     (shape[0] - 4, shape[1] - 4))
    npt.assert_equal(trim_image(img, tol=0.2).shape[:2], (1, 0))
    npt.assert_equal(trim_image(img, tol=0.1).shape[:2],
                     trim_image(trim_image(img), tol=0.1).shape[:2])
    trimmed, rows, cols = trim_image(img, return_coords=True)
    npt.assert_almost_equal(trimmed, trim_image(img))
    npt.assert_equal(rows, (1, 12))
    npt.assert_equal(cols, (1, 28))

    with pytest.raises(ValueError):
        trim_image(np.ones(10))
    with pytest.raises(ValueError):
        trim_image(np.ones((3, 4, 5, 6)))
    with pytest.raises(ValueError):
        trim_image(img, tol=-1)


def test_single_image_retargeting():
    image = np.zeros((20, 20))
    for i in range(1, 8):
        for j in range(1, 8):
            image[i, j] = 1
    for i in range(10, 15):
        for j in range(14, 19):
            image[i, j] = 1
    shrinked = single_image_retargeting(image, wid=10, hei=10, L=1)
    npt.assert_equal(shrinked[3, 3], 1.0)
    npt.assert_equal(shrinked[4, 4], 0.0)
    npt.assert_almost_equal(shrinked[5, 5], 0.35745859)
    npt.assert_equal(shrinked[6, 6], 1.0)


def test_image_retargeting():
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
    shrinked = image_retargeting(image, second_frame, wid=6, hei=6, L=1, num=5)
    npt.assert_equal(shrinked[3, 4], 1.0)
    npt.assert_almost_equal(shrinked[5, 6], 0.2399997)
    npt.assert_almost_equal(shrinked[6, 7], 1.0)
    npt.assert_almost_equal(shrinked[6, 6], 0.3999997)


def test_video_retargeting():
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
    shrinked = video_retargeting(video, wid=3, hei=3, L=2, num=5)
    npt.assert_almost_equal(shrinked[0, 3, 4], 0.79999995)
    npt.assert_almost_equal(shrinked[0, 5, 6], 0.40000009)
    npt.assert_almost_equal(shrinked[0, 6, 7], 0.0)
    npt.assert_almost_equal(shrinked[0, 6, 6], 0.40000009)
    npt.assert_almost_equal(shrinked[1, 3, 4], 1.0)
    npt.assert_almost_equal(shrinked[1, 5, 6], 1.0)
    npt.assert_almost_equal(shrinked[1, 6, 7], 0.08000006)
    npt.assert_almost_equal(shrinked[1, 6, 6], 0.40000009)


def test_shrink_image_Stimulus():
    image = np.zeros((20, 20))
    for i in range(1, 8):
        for j in range(1, 8):
            image[i, j] = 1
    for i in range(10, 15):
        for j in range(14, 19):
            image[i, j] = 1
    stim = ImageStimulus(image)
    shrinked = stim_retargeting(stim, wid=10, hei=10, L=1)
    shrinked_data = shrinked.data.reshape(shrinked.img_shape)
    npt.assert_equal(shrinked_data[3, 3], 1.0)
    npt.assert_equal(shrinked_data[4, 4], 0.0)
    npt.assert_almost_equal(shrinked_data[5, 5], 0.35745859)
    npt.assert_equal(shrinked_data[6, 6], 1.0)


def test_video_retargeting_stimulus():
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
    shrinked = stim_retargeting(stim, wid=3, hei=3, L=2, num=5)
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
