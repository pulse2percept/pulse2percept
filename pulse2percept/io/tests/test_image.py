import numpy as np
import numpy.testing as npt
import pytest
from unittest import mock
from imp import reload

from pulse2percept.io import image
from pulse2percept import implants


def test_image2stim():
    reload(image)
    # Range of values
    amp_min = 2
    amp_max = 15

    # Create a standard Argus I array
    implant = implants.ArgusI()

    # Create a small image with 1 pixel per electrode
    img = np.zeros((4, 4))

    # An all-zero image should give us a really boring stimulation protocol
    pulses = image.image2stim(img, implant, valrange=[amp_min, amp_max])
    for pt in pulses:
        npt.assert_equal(pt.data.max(), amp_min)

    # Now put some structure in the image
    img[1, 1] = img[1, 2] = img[2, 1] = img[2, 2] = 0.75

    expected_max = [amp_max, 0.75 * (amp_max - amp_min) + amp_min]
    for max_contrast, val_max in zip([True, False], expected_max):
        pt = image.image2stim(img, implant, coding='amplitude',
                              max_contrast=max_contrast,
                              valrange=[amp_min, amp_max])

        # Make sure we have one pulse train per electrode
        npt.assert_equal(len(pt), implant.n_electrodes)

        # Make sure the brightest electrode has `amp_max`
        npt.assert_almost_equal(np.max([p.data.max() for p in pt]), val_max)

        # Make sure the dimmest electrode has `amp_min` as max amplitude
        npt.assert_almost_equal(np.min([np.abs(p.data).max() for p in pt]),
                                amp_min)

    # Invalid implant
    with pytest.raises(TypeError):
        image.image2stim("rainbow_cat.jpg", np.zeros(10))
    with pytest.raises(TypeError):
        e_array = implants.ElectrodeArray([])
        image.image2stim("rainbow_cat.jpg", e_array)

    # Invalid image
    with pytest.raises(IOError):
        image.image2stim("rainbow_cat.jpg", implants.ArgusI())

    # Smoke-test RGB
    image.image2stim(np.zeros((10, 10, 3)), implants.ArgusI())

    # Smoke-test invert
    image.image2stim(np.zeros((10, 10, 3)), implants.ArgusI(), invert=True)

    # Smoke-test normalize
    image.image2stim(np.ones((10, 10, 3)) * 2, implants.ArgusI(), invert=True)

    # Smoke-test frequency coding
    image.image2stim(np.zeros((10, 10, 3)), implants.ArgusI(),
                     coding='frequency')

    # Invalid coding
    with pytest.raises(ValueError):
        image.image2stim(np.zeros((10, 10)), implants.ArgusI(), coding='n/a')

    # Trigger an import error
    with mock.patch.dict("sys.modules", {"skimage": {}, "skimage.io": {}}):
        with pytest.raises(ImportError):
            reload(image)
            image.image2stim(img, implant)
