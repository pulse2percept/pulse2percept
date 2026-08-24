import os
import numpy as np
import numpy.testing as npt
import pytest

from skimage.color import rgb2gray
from skimage.io import imsave
from skimage.transform import resize as img_resize

from pulse2percept.stimuli import (AmplitudeEncoder, ImageStimulus, LogoBVL,
                                   LogoUCSB, SnellenChart)
from pulse2percept.units import DimensionMismatchError, dva, ms


def create_dummy_img(fname, shape, mode, gray=1.0, return_data=False):
    if mode == 'ones':
        ndarray = np.ones(shape) * gray
    elif mode == 'zeros':
        ndarray = np.zeros(shape)
    elif mode == 'rand':
        ndarray = np.random.rand(*shape) * gray
    elif mode == 'custom':
        ndarray = shape
    imsave(fname, (255 * ndarray).astype(np.uint8))
    if return_data:
        return ndarray


def test_ImageStimulus():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37, 4)
    ndarray = create_dummy_img(fname, shape, 'rand', return_data=True)

    # Make sure ImageStimulus loaded is identical to dummy image:
    stim = ImageStimulus(fname)
    npt.assert_equal(stim.shape, (np.prod(shape), 1))
    npt.assert_almost_equal(stim.data, ndarray.reshape((-1, 1)), decimal=2)
    npt.assert_equal(stim.metadata['source'], fname)
    npt.assert_equal(stim.metadata['source_shape'], shape)
    npt.assert_equal(stim.time, None)
    # Every pixel is named after its place in the image: a letter for the row,
    # a number for the column, and a suffix for the color channel:
    npt.assert_equal(len(stim.electrodes), np.prod(shape))
    npt.assert_equal(stim.electrodes[0], 'A1_R')
    npt.assert_equal(stim.electrodes[3], 'A1_A')
    npt.assert_equal(stim.electrodes[-1], 'Y37_A')
    # ... and the name maps back onto that same pixel:
    npt.assert_equal(stim.electrodes.index('C12_G'),
                     np.ravel_multi_index((2, 11, 1), shape))
    os.remove(fname)


def test_ImageStimulus_invert():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37)
    gray = 1 / 255.0
    create_dummy_img(fname, shape, 'ones', gray=gray)
    # Gray levels are between 0 and 1, and can be inverted:
    stim = ImageStimulus(fname)
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_almost_equal(stim.invert().data, 1 - gray)
    # Inverting does not change the original object:
    npt.assert_almost_equal(stim.data, gray)
    os.remove(fname)


def test_ImageStimulus_rgb2gray():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37, 3)
    gray = 1 / 255.0
    create_dummy_img(fname, shape, 'ones', gray=gray)
    # Gray levels are between 0 and 1, and can be inverted:
    stim_rgb = ImageStimulus(fname)
    stim_gray = stim_rgb.rgb2gray()
    npt.assert_almost_equal(stim_gray.data, gray)
    npt.assert_equal(stim_gray.img_shape, shape[:2])
    # Original stim unchanged:
    npt.assert_equal(stim_rgb.img_shape, shape)
    os.remove(fname)


def test_ImageStimulus_resize():
    fname = 'test.png'
    shape = (25, 37, 3)
    gray = 129 / 255.0
    create_dummy_img(fname, shape, 'ones', gray=gray)
    # Gray levels are between 0 and 1, and can be inverted:
    stim = ImageStimulus(fname)
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_equal(stim.resize((13, -1)).img_shape, (13, 19, 3))
    # Resize with one dimension -1:
    npt.assert_equal(stim.resize((-1, 24)).img_shape, (16, 24, 3))
    with pytest.raises(ValueError):
        stim.resize((-1, -1))
    os.remove(fname)


def test_ImageStimulus_resize_kwargs():
    """Keyword arguments reach scikit-image (Issue #501)"""
    # A white square on black. Nearest-neighbor interpolation keeps the image
    # binary; the default (bilinear, with anti-aliasing on the way down) does
    # not, which is what makes the two distinguishable:
    ndarray = np.zeros((8, 8), dtype=np.float32)
    ndarray[2:6, 2:6] = 1
    stim = ImageStimulus(ndarray)
    nearest = stim.resize((4, 4), order=0, anti_aliasing=False)
    npt.assert_equal(np.isin(nearest.data, [0, 1]).all(), True)
    npt.assert_equal(np.isin(stim.resize((4, 4)).data, [0, 1]).all(), False)
    # An unknown keyword argument is scikit-image's to reject, not ours:
    with pytest.raises(TypeError):
        stim.resize((4, 4), not_a_skimage_kwarg=0)


def test_ImageStimulus_apply():
    ndarray = np.random.rand(8, 12).astype(np.float32)
    stim = ImageStimulus(ndarray)
    # A shape-preserving function keeps every pixel's name:
    halved = stim.apply(lambda x: 0.5 * x)
    npt.assert_almost_equal(halved.data, 0.5 * stim.data)
    npt.assert_equal(halved.img_shape, (8, 12))
    npt.assert_equal(np.asarray(halved.electrodes),
                     np.asarray(stim.electrodes))
    # A function that changes the resolution is allowed, and the result is
    # named after its own pixel grid (Issue #500):
    resized = stim.apply(img_resize, (4, 6))
    npt.assert_equal(resized.img_shape, (4, 6))
    npt.assert_equal(resized.shape, (24, 1))
    npt.assert_equal(resized.electrodes[0], 'A1')
    npt.assert_equal(resized.electrodes[-1], 'D6')
    # Positional and keyword arguments both make it through:
    npt.assert_equal(stim.apply(img_resize, (4, 6), order=0).img_shape, (4, 6))
    npt.assert_equal(stim.apply(img_resize, output_shape=(4, 6)).img_shape,
                     (4, 6))
    # Names can be given explicitly, whether or not the shape changed:
    named = stim.apply(img_resize, (2, 2), electrodes=['a', 'b', 'c', 'd'])
    npt.assert_equal(list(named.electrodes), ['a', 'b', 'c', 'd'])
    with pytest.raises(ValueError):
        stim.apply(img_resize, (2, 2), electrodes=['a', 'b'])
    # Dropping the color channels changes the pixel count too:
    rgb = ImageStimulus(np.random.rand(8, 12, 3).astype(np.float32))
    npt.assert_equal(rgb.apply(rgb2gray).img_shape, (8, 12))


def test_ImageStimulus_crop():
    # test img with color channels
    fname = 'test.png'
    shape = (30, 50, 3)
    gray = create_dummy_img(fname, shape, 'rand')
    stim = ImageStimulus(fname)
    stim_cropped = stim.crop(idx_rect=[5, 10, 25, 40])
    npt.assert_equal(stim_cropped.img_shape, (20, 30, 3))
    npt.assert_equal(stim_cropped.data.reshape(stim_cropped.img_shape)[3, 7],
                     stim.data.reshape(stim.img_shape)[8, 17])
    npt.assert_equal(stim_cropped.data.reshape(stim_cropped.img_shape)[10, 28],
                     stim.data.reshape(stim.img_shape)[15, 38])
    npt.assert_equal(stim.electrodes.reshape(30, 50, 3)[8, 17, 0],
                     stim_cropped.electrodes.reshape(20, 30, 3)[3, 7, 0])
    npt.assert_equal(stim.electrodes.reshape(30, 50, 3)[15, 38, 2],
                     stim_cropped.electrodes.reshape(20, 30, 3)[10, 28, 2])

    # test img with no color channels
    fname_bw = 'test_bw.png'
    shape_bw = (30, 50)
    gray_bw = create_dummy_img(fname_bw, shape_bw, 'rand')
    stim_bw = ImageStimulus(fname_bw)
    stim_cropped_bw = stim_bw.crop(idx_rect=[5, 10, 25, 40])
    npt.assert_equal(stim_cropped_bw.img_shape, (20, 30))
    npt.assert_equal(stim_cropped_bw.data.reshape(stim_cropped_bw.img_shape)[3, 7],
                     stim_bw.data.reshape(stim_bw.img_shape)[8, 17])
    npt.assert_equal(stim_cropped_bw.data.reshape(stim_cropped_bw.img_shape)[10, 28],
                     stim_bw.data.reshape(stim_bw.img_shape)[15, 38])
    npt.assert_equal(stim_bw.electrodes.reshape(30, 50)[8, 17],
                     stim_cropped_bw.electrodes.reshape(20, 30)[3, 7])
    npt.assert_equal(stim_bw.electrodes.reshape(30, 50)[15, 38],
                     stim_cropped_bw.electrodes.reshape(20, 30)[10, 28])

    stim_cropped2 = stim.crop(left=10, right=8, top=6, bottom=7)
    npt.assert_equal(stim_cropped2.img_shape, (17, 32, 3))
    npt.assert_equal(stim_cropped2.data.reshape(stim_cropped2.img_shape)[3, 7],
                     stim.data.reshape(stim.img_shape)[9, 17])
    npt.assert_equal(stim_cropped2.data.reshape(stim_cropped2.img_shape)[10, 28],
                     stim.data.reshape(stim.img_shape)[16, 38])

    #"crop-indices and crop-width (left, right, up, down) cannot exist at the same time"
    with pytest.raises(ValueError):
        stim.crop(idx_rect=[5, 10, 25, 40], left=10)
    with pytest.raises(ValueError):
        stim.crop([5, 10, 25, 40], right=8)
    with pytest.raises(ValueError):
        stim.crop([5, 10, 25, 40], top=6)
    with pytest.raises(ValueError):
        stim.crop([5, 10, 25, 40], bottom=7)
    # "crop-width(left, right, up, down) cannot be negative"
    with pytest.raises(ValueError):
        stim.crop(left=-1)
    with pytest.raises(ValueError):
        stim.crop(right=-1)
    with pytest.raises(ValueError):
        stim.crop(top=-1)
    with pytest.raises(ValueError):
        stim.crop(bottom=-1)
    # "crop-width should be smaller than the shape of the image"
    with pytest.raises(ValueError):
        stim.crop(left=32, right=20)
    with pytest.raises(ValueError):
        stim.crop(top=12, bottom=18)
    # "crop-indices must be on the image"
    with pytest.raises(ValueError):
        stim.crop([-1, 10, 25, 40])
    with pytest.raises(ValueError):
        stim.crop([5, -1, 25, 40])
    with pytest.raises(ValueError):
        stim.crop([5, 10, 31, 40])
    with pytest.raises(ValueError):
        stim.crop([5, 10, 25, 51])
    # "crop-indices is invalid. It should be [y1,x1,y2,x2], where (y1,x1) is upperleft and (y2,x2) is bottom-right"
    with pytest.raises(ValueError):
        stim.crop([5, 10, 4, 40])
    with pytest.raises(ValueError):
        stim.crop([5, 10, 25, 9])
    
    os.remove(fname)
    os.remove(fname_bw)


def test_ImageStimulus_trim():
    shape = (13, 29)
    ndarray = np.zeros(shape)
    ndarray[1:-1, 1:-1] = 0.1
    ndarray[2:-2, 2:-2] = 0.2
    stim = ImageStimulus(ndarray)
    npt.assert_equal(stim.trim().img_shape, (shape[0] - 2, shape[1] - 2))
    npt.assert_equal(stim.trim(tol=0.05).img_shape,
                     (shape[0] - 2, shape[1] - 2))
    npt.assert_equal(stim.trim(tol=0.1).img_shape,
                     (shape[0] - 4, shape[1] - 4))
    npt.assert_equal(stim.trim(tol=0.2).img_shape, (1, 0))
    npt.assert_equal(stim.trim(tol=0.1).img_shape,
                     stim.trim().trim(tol=0.1).img_shape)


def test_ImageStimulus_threshold():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37, 3)
    gray = 129 / 255.0
    create_dummy_img(fname, shape, 'ones', gray=gray)
    # Gray levels are between 0 and 1, and can be inverted:
    stim = ImageStimulus(fname, as_gray=True)
    stim_th = stim.threshold(0.5)
    npt.assert_almost_equal(stim.data, gray)
    npt.assert_equal(stim.img_shape, shape[:2])
    os.remove(fname)


def test_ImageStimulus_rotate():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    stim = ImageStimulus(fname)
    # Vertical line:
    vert = stim.rotate(90, mode='reflect')
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 0], 0)
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 1], 0)
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 2], 1)
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 3], 0)
    npt.assert_almost_equal(vert.data.reshape(stim.img_shape)[:, 4], 0)
    # Diagonal, bottom-left to top-right:
    diag = stim.rotate(45, mode='reflect')
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[0, 4], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[2, 2], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[4, 0], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[0, 0], 0)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[4, 4], 0)
    # Diagonal, top-left to bottom-right:
    diag = stim.rotate(-45, mode='reflect')
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[0, 0], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[2, 2], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[4, 4], 1)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[0, 4], 0)
    npt.assert_almost_equal(diag.data.reshape(stim.img_shape)[4, 0], 0)
    os.remove(fname)


def test_ImageStimulus_rotate_kwargs():
    """Keyword arguments reach scikit-image (Issue #501)"""
    ndarray = np.zeros((5, 5), dtype=np.float32)
    ndarray[2, :] = 1
    stim = ImageStimulus(ndarray)
    # Nearest-neighbor interpolation keeps the bar binary, bilinear does not:
    npt.assert_equal(np.isin(stim.rotate(45, order=0).data, [0, 1]).all(), True)
    npt.assert_equal(np.isin(stim.rotate(45, order=1).data, [0, 1]).all(),
                     False)
    # 'cval' fills the corners the rotation leaves empty:
    corners = ([0, 0, 4, 4], [0, 4, 0, 4])
    npt.assert_almost_equal(
        stim.rotate(45, order=0, cval=0.3).data.reshape(5, 5)[corners], 0.3)
    npt.assert_almost_equal(
        stim.rotate(45, order=0).data.reshape(5, 5)[corners], 0)
    # 'resize' grows the canvas, so the result is named after its own grid
    # rather than inheriting 25 names it has no room for:
    grown = stim.rotate(45, resize=True)
    npt.assert_equal(grown.img_shape, (7, 7))
    npt.assert_equal(grown.shape, (49, 1))
    npt.assert_equal(grown.electrodes[-1], 'G7')
    # Rotating in place keeps every pixel's name:
    same = stim.rotate(45)
    npt.assert_equal(same.img_shape, (5, 5))
    npt.assert_equal(np.asarray(same.electrodes), np.asarray(stim.electrodes))
    # Names can be given explicitly:
    npt.assert_equal(len(stim.rotate(45, electrodes=np.arange(25)).electrodes),
                     25)


def test_ImageStimulus_shift():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    stim = ImageStimulus(fname)
    # Top row:
    top = stim.shift(0, -2)
    npt.assert_almost_equal(top.data.reshape(stim.img_shape)[0, :], 1)
    npt.assert_almost_equal(top.data.reshape(stim.img_shape)[1:, :], 0)
    # Bottom row:
    bottom = stim.shift(0, 2)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[:4, :], 0)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[4, :], 1)
    # Bottom right pixel:
    bottom = stim.shift(4, 2)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[4, 4], 1)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[:4, :], 0)
    npt.assert_almost_equal(bottom.data.reshape(stim.img_shape)[:, :4], 0)
    os.remove(fname)


def test_ImageStimulus_center():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    # Center phosphene:
    stim = ImageStimulus(fname)
    npt.assert_almost_equal(stim.data, stim.center().data)
    npt.assert_almost_equal(stim.data, stim.shift(0, 2).center().data)
    # 'loc' places the CoM somewhere other than the image center:
    top = stim.center(loc=(2, 0))
    npt.assert_almost_equal(top.data.reshape(shape)[0, :], 1)
    npt.assert_almost_equal(top.data.reshape(shape)[1:, :], 0)
    npt.assert_almost_equal(stim.center(loc=(2, 2)).data, stim.data)
    os.remove(fname)


def test_ImageStimulus_scale():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    # Scale phosphene:
    stim = ImageStimulus(fname)
    npt.assert_almost_equal(stim.data, stim.scale(1).data)
    npt.assert_almost_equal(stim.scale(0.1)[12], 1)
    npt.assert_almost_equal(stim.scale(0.1)[:12], 0)
    npt.assert_almost_equal(stim.scale(0.1)[13:], 0)
    with pytest.raises(ValueError):
        stim.scale(0)
    os.remove(fname)


def test_ImageStimulus_filter():
    # Create a dummy image:
    fname = 'test.png'
    shape = (25, 37)
    create_dummy_img(fname, shape, 'rand')
    stim = ImageStimulus(fname)

    for filt in ['sobel', 'scharr', 'canny', 'median']:
        filt_stim = stim.filter(filt)
        npt.assert_equal(filt_stim.shape, stim.shape)
        npt.assert_equal(filt_stim.img_shape, stim.img_shape)
        npt.assert_equal(np.asarray(filt_stim.electrodes),
                         np.asarray(stim.electrodes))
        npt.assert_equal(filt_stim.time, None)

    # Invalid filter name:
    with pytest.raises(TypeError):
        stim.filter({'invalid'})
    with pytest.raises(ValueError):
        stim.filter('invalid')

    os.remove(fname)


def test_ImageStimulus_encode():
    # An image is a single frame lasting 500 ms:
    stim = ImageStimulus(np.linspace(0, 1, 20).reshape((4, 5)))
    enc = stim.encode()
    npt.assert_almost_equal(enc.time[-1], 500)
    npt.assert_equal(enc.shape[0], stim.shape[0])
    # Gray levels map onto the amplitude range absolutely, so the darkest and
    # brightest pixels of this ramp land on its two ends:
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1), 50 * stim.data[:, 0],
                            decimal=4)

    # Amplitude encoding in custom range:
    enc = stim.encode(amp_range=(2, 43))
    npt.assert_almost_equal(enc.time[-1], 500)
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1).min(), 2, decimal=4)
    npt.assert_almost_equal(np.abs(enc.data).max(axis=1).max(), 43, decimal=4)

    # `encode` is a shorthand for AmplitudeEncoder, and forwards to it:
    npt.assert_almost_equal(stim.encode().data,
                            AmplitudeEncoder().encode(stim).data)
    with pytest.raises(TypeError):
        stim.encode(pulse={'invalid': 1})
    with pytest.raises(ValueError):
        stim.encode(pulse=LogoUCSB())


def test_ImageStimulus_plot():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    stim = ImageStimulus(fname)
    ax = stim.plot()
    npt.assert_equal(ax.axis(), (-0.5, 4.5, 4.5, -0.5))
    os.remove(fname)


def test_ImageStimulus_save():
    # Create a horizontal bar:
    fname = 'test.png'
    shape = (5, 5)
    ndarray = np.zeros(shape, dtype=np.uint8)
    ndarray[2, :] = 255
    imsave(fname, ndarray)
    stim = ImageStimulus(fname)
    fname2 = 'test2.png'
    stim.save(fname2)
    npt.assert_almost_equal(stim.data, ImageStimulus(fname2).data)
    os.remove(fname)
    os.remove(fname2)

    # Test that TIFF retains scaling between saving and loading
    fname3 = 'test.tif'
    shape = (5,5)
    ndarray = np.random.rand(*shape).astype('float32')
    imsave(fname3, ndarray)
    stim2 = ImageStimulus(fname3)
    fname4 = 'test2.tif'
    stim2.save(fname4)
    npt.assert_almost_equal(stim2.data, ImageStimulus(fname4).data)
    os.remove(fname3)
    os.remove(fname4)


@pytest.mark.parametrize('show_annotations', (True, False))
def test_SnellenChart(show_annotations):
    width = 840 if show_annotations else 444
    snellen = SnellenChart(show_annotations=show_annotations)
    npt.assert_equal(snellen.img_shape, (1348, width))
    npt.assert_equal(snellen.time, None)
    npt.assert_almost_equal(snellen.data.max(), 1)
    npt.assert_almost_equal(snellen.data.min(), 0)

    snellen = SnellenChart(row=1, show_annotations=show_annotations)
    npt.assert_equal(snellen.img_shape, (255, width))

    with pytest.raises(ValueError):
        SnellenChart(row=0)
    with pytest.raises(ValueError):
        SnellenChart(row=12)
    with pytest.raises(ValueError):
        SnellenChart(row=[1, 3])


def test_LogoBVL():
    logo = LogoBVL()
    npt.assert_equal(logo.img_shape, (576, 720, 4))
    npt.assert_equal(logo.time, None)
    npt.assert_almost_equal(logo.data.min(), 0)
    npt.assert_almost_equal(logo.data.max(), 1)


def test_LogoUCSB():
    logo = LogoUCSB()
    npt.assert_equal(logo.img_shape, (324, 727))
    npt.assert_equal(logo.time, None)
    npt.assert_almost_equal(logo.data.min(), 0)
    npt.assert_almost_equal(logo.data.max(), 1)


def test_ImageStimulus_rgb2gray_matches_skimage():
    """The fused RGBA blend must agree with skimage's two-step conversion.

    ``rgb2gray`` blends the alpha channel against black itself rather than
    calling ``rgba2rgb``, to avoid building a full-resolution intermediate.
    This pins the arithmetic to what skimage would have produced.
    """
    from skimage.color import rgba2rgb, rgb2gray as sk_rgb2gray

    rng = np.random.default_rng(0)
    rgba = rng.random((37, 53, 4)).astype(np.float32)
    got = ImageStimulus(rgba).rgb2gray().data
    want = sk_rgb2gray(rgba2rgb(rgba, background=(0, 0, 0)))
    npt.assert_array_equal(got.ravel(), want.ravel().astype(got.dtype))

    # Three channels take the other branch and must be untouched by the above:
    rgb = rng.random((37, 53, 3)).astype(np.float32)
    npt.assert_array_equal(ImageStimulus(rgb).rgb2gray().data.ravel(),
                           sk_rgb2gray(rgb).ravel())

    # A fully transparent image blends to black; a fully opaque one is
    # unaffected by the alpha channel:
    opaque = np.concatenate((rgb, np.ones((37, 53, 1), dtype=np.float32)),
                            axis=-1)
    npt.assert_allclose(ImageStimulus(opaque).rgb2gray().data,
                        ImageStimulus(rgb).rgb2gray().data, rtol=1e-6)
    clear = np.concatenate((rgb, np.zeros((37, 53, 1), dtype=np.float32)),
                           axis=-1)
    npt.assert_equal(np.all(ImageStimulus(clear).rgb2gray().data == 0), True)


def test_ImageStimulus_invert_preserves_alpha():
    """Inverting must leave the alpha channel alone and not touch the source"""
    rng = np.random.default_rng(1)
    rgba = rng.random((11, 13, 4)).astype(np.float32)
    stim = ImageStimulus(rgba)
    before = stim.data.copy()
    inverted = stim.invert().data.reshape(11, 13, 4)
    npt.assert_allclose(inverted[..., :3], 1.0 - rgba[..., :3], rtol=1e-6)
    npt.assert_array_equal(inverted[..., 3], rgba[..., 3])
    # The original is untouched:
    npt.assert_array_equal(stim.data, before)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.uint8])
def test_ImageStimulus_owns_its_data(dtype):
    # `img_as_float32` hands back an already-float32 image unchanged, so that
    # is the dtype where an image could end up sharing the caller's buffer:
    arr = (np.linspace(0, 1, 24).reshape((4, 6)) if dtype != np.uint8
           else np.arange(24, dtype=np.uint8).reshape((4, 6)))
    arr = np.ascontiguousarray(arr, dtype=dtype)
    stim = ImageStimulus(arr)
    before = stim.data.copy()
    arr[...] = 0
    npt.assert_array_equal(stim.data, before)
    npt.assert_equal(np.shares_memory(arr, stim.data), False)
    npt.assert_equal(stim.data.flags.writeable, False)
    # Freezing what the stimulus took must not reach back into what the
    # caller kept:
    npt.assert_equal(arr.flags.writeable, True)


def test_ImageStimulus_does_not_alias_another_stimulus():
    first = ImageStimulus(np.linspace(0, 1, 24, dtype=np.float32)
                          .reshape((4, 6)))
    second = ImageStimulus(first)
    npt.assert_equal(np.shares_memory(first.data, second.data), False)


def test_ImageStimulus_fov_none_by_default():
    stim = ImageStimulus(np.zeros((4, 6)))
    npt.assert_equal(stim.fov, None)
    # Pixels without a FOV have no angular size, and asking for one is an
    # error rather than a silent default:
    with pytest.raises(ValueError):
        stim.pixel_to_dva(0, 0)
    with pytest.raises(ValueError):
        stim.dva_to_pixel(0, 0)


def test_ImageStimulus_fov_scalar_infers_aspect_ratio():
    # A scalar is the horizontal FOV; square angular pixels fix the vertical
    # one:
    npt.assert_almost_equal(ImageStimulus(np.zeros((4, 8)), fov=16).fov,
                            (16, 8))
    npt.assert_almost_equal(ImageStimulus(np.zeros((8, 4)), fov=16).fov,
                            (16, 32))


def test_ImageStimulus_fov_tuple():
    npt.assert_almost_equal(ImageStimulus(np.zeros((4, 8)), fov=(20, 5)).fov,
                            (20, 5))


def test_ImageStimulus_fov_units():
    npt.assert_almost_equal(ImageStimulus(np.zeros((4, 8)), fov=16 * dva).fov,
                            (16, 8))
    npt.assert_almost_equal(
        ImageStimulus(np.zeros((4, 8)), fov=(20, 5) * dva).fov, (20, 5))
    with pytest.raises(DimensionMismatchError):
        ImageStimulus(np.zeros((4, 8)), fov=16 * ms)


@pytest.mark.parametrize('fov', [0, -3, np.inf, np.nan, (10, 0), (10, -1)])
def test_ImageStimulus_fov_must_be_positive_and_finite(fov):
    with pytest.raises(ValueError):
        ImageStimulus(np.zeros((4, 8)), fov=fov)


def test_ImageStimulus_fov_rejects_wrong_length():
    with pytest.raises(ValueError):
        ImageStimulus(np.zeros((4, 8)), fov=(10, 20, 30))


@pytest.mark.parametrize('shape', [(4, 8), (5, 7)])
def test_ImageStimulus_pixel_to_dva_centers(shape):
    # The FOV is the outer extent, so the outermost pixel *centers* sit half a
    # pixel inside it. Holds for even and odd pixel counts alike:
    n_rows, n_cols = shape
    stim = ImageStimulus(np.zeros(shape), fov=(n_cols, n_rows))
    x, y = stim.pixel_to_dva([0, n_cols - 1], [0, n_rows - 1])
    npt.assert_almost_equal(x, [-n_cols / 2 + 0.5, n_cols / 2 - 0.5])
    # Row 0 is the top of the image and therefore the largest y:
    npt.assert_almost_equal(y, [n_rows / 2 - 0.5, -n_rows / 2 + 0.5])
    # The image is centered on the origin:
    npt.assert_almost_equal(stim.pixel_to_dva((n_cols - 1) / 2,
                                              (n_rows - 1) / 2), (0, 0))


def test_ImageStimulus_pixel_dva_roundtrip():
    stim = ImageStimulus(np.zeros((5, 9)), fov=(30, 20))
    col, row = np.meshgrid(np.arange(9), np.arange(5))
    npt.assert_almost_equal(stim.dva_to_pixel(*stim.pixel_to_dva(col, row)),
                            (col, row))
    x, y = np.array([-11.3, 0.0, 7.5]), np.array([2.2, -6.0, 0.0])
    npt.assert_almost_equal(stim.pixel_to_dva(*stim.dva_to_pixel(x, y)),
                            (x, y))


def test_ImageStimulus_fov_survives_transforms():
    stim = ImageStimulus(np.random.rand(4, 8, 3), fov=(16, 8))
    # A resize keeps the extent and resamples the pixels:
    npt.assert_almost_equal(stim.resize((8, 16)).fov, (16, 8))
    npt.assert_almost_equal(stim.rgb2gray().fov, (16, 8))
    npt.assert_almost_equal(stim.invert().fov, (16, 8))
    npt.assert_almost_equal(stim.rgb2gray().threshold(0.5).fov, (16, 8))
    npt.assert_almost_equal(stim.rgb2gray().center().fov, (16, 8))
    npt.assert_almost_equal(stim.rotate(30).fov, (16, 8))
    # Constructing one image from another carries the FOV over:
    npt.assert_almost_equal(ImageStimulus(stim).fov, (16, 8))
    # ... unless the caller overrides it:
    npt.assert_almost_equal(ImageStimulus(stim, fov=(4, 2)).fov, (4, 2))


def test_ImageStimulus_crop_updates_fov():
    stim = ImageStimulus(np.random.rand(4, 8), fov=(16, 8))
    # A crop keeps the angular pixel size (2 dva here) and drops pixels:
    npt.assert_almost_equal(stim.crop(left=2).fov, (12, 8))
    npt.assert_almost_equal(stim.crop(top=1, bottom=1).fov, (16, 4))
    npt.assert_almost_equal(stim.crop(idx_rect=(1, 2, 3, 6)).fov, (8, 4))


def test_ImageStimulus_apply_drops_fov_on_reshape():
    stim = ImageStimulus(np.random.rand(4, 8), fov=(16, 8))
    npt.assert_almost_equal(stim.apply(lambda x: x * 0.5).fov, (16, 8))
    # `apply` cannot know what an arbitrary reshape did to the geometry:
    npt.assert_equal(stim.apply(lambda x: x[:2, :4]).fov, None)


def test_ImageStimulus_fov_of_builtin_images():
    npt.assert_almost_equal(LogoUCSB(resize=(8, 16), fov=32).fov, (32, 16))
    npt.assert_equal(LogoUCSB(resize=(8, 16)).fov, None)
