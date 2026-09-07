import warnings

import numpy as np
import numpy.testing as npt
import pytest

import pulse2percept as p2p
from pulse2percept.stimuli import (ImageStimulus, LogoBVL, LogoUCSB,
                                   VideoStimulus, samples)
from pulse2percept.units import deg, dva
from pulse2percept.vision import Scene


def _legacy(cls, **kwargs):
    """Instantiate a legacy sample class without its deprecation warning"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return cls(**kwargs)


@pytest.mark.parametrize('loader,legacy', [
    (samples.logo_bvl, LogoBVL),
    (samples.logo_ucsb, LogoUCSB),
])
def test_samples_match_legacy(loader, legacy):
    new = loader()
    old = _legacy(legacy)
    # Loaders return plain stimuli, not a sample type of their own:
    npt.assert_equal(type(new), ImageStimulus)
    npt.assert_equal(new.img_shape, old.img_shape)
    npt.assert_equal(new.time, None)
    npt.assert_almost_equal(new.data, old.data)
    npt.assert_almost_equal(new.data.min(), 0)
    npt.assert_almost_equal(new.data.max(), 1)


def test_samples_image_options():
    logo = samples.logo_bvl(resize=(32, 32), as_gray=True)
    npt.assert_equal(logo.img_shape, (32, 32))
    npt.assert_almost_equal(logo.data, _legacy(LogoBVL, resize=(32, 32),
                                               as_gray=True).data)
    ucsb = samples.logo_ucsb(metadata={'foo': 'bar'})
    npt.assert_equal(ucsb.metadata['foo'], 'bar')


@pytest.mark.parametrize('legacy,alt', [
    (LogoBVL, 'samples.logo_bvl'),
    (LogoUCSB, 'samples.logo_ucsb'),
])
def test_samples_legacy_classes_deprecated(legacy, alt):
    with pytest.warns(DeprecationWarning) as record:
        legacy()
    msg = str(record[0].message)
    npt.assert_equal(alt in msg, True)
    npt.assert_equal('0.12.0' in msg, True)


def test_samples_namespace():
    # The module is reachable, but its loaders are not promoted to the
    # top-level namespace:
    npt.assert_equal(p2p.stimuli.samples is samples, True)
    for name in samples.__all__:
        npt.assert_equal(hasattr(p2p.stimuli, name), False)
    # The two video samples were removed along with their assets:
    for gone in ('BostonTrain', 'GirlPool'):
        npt.assert_equal(hasattr(p2p.stimuli, gone), False)
    for gone in ('boston_train', 'girl_pool'):
        npt.assert_equal(hasattr(samples, gone), False)


#: Properties of the packaged clip, as decoded (not as the container claims:
#: the MP4 header advertises more frames than the file actually holds).
_BUNNY_SHAPE, _BUNNY_FRAMES, _BUNNY_FPS = (359, 640), 115, 24.0


def test_big_buck_bunny():
    video = samples.big_buck_bunny()
    # A loader returns a plain stimulus, not a sample type of its own:
    npt.assert_equal(type(video), VideoStimulus)
    npt.assert_equal(video.vid_shape, _BUNNY_SHAPE + (3, _BUNNY_FRAMES))
    npt.assert_equal(video.time.size, _BUNNY_FRAMES)
    npt.assert_almost_equal(video.metadata['fps'], _BUNNY_FPS)
    npt.assert_almost_equal(np.diff(video.time), 1000.0 / _BUNNY_FPS)
    npt.assert_equal(np.all(np.isfinite(video.data)), True)
    npt.assert_equal(video.data.min() >= 0, True)
    npt.assert_equal(video.data.max() <= 1, True)


def test_big_buck_bunny_options():
    gray = samples.big_buck_bunny(as_gray=True, resize=(30, 40))
    npt.assert_equal(gray.vid_shape, (30, 40, _BUNNY_FRAMES))
    npt.assert_equal(np.all(np.isfinite(gray.data)), True)
    rgb = samples.big_buck_bunny(resize=(30, 40))
    npt.assert_equal(rgb.vid_shape, (30, 40, 3, _BUNNY_FRAMES))


def test_big_buck_bunny_metadata():
    video = samples.big_buck_bunny(resize=(8, 8))
    npt.assert_equal(video.metadata['title'], 'Big Buck Bunny')
    npt.assert_equal(video.metadata['creator'], 'Blender Foundation')
    npt.assert_equal(video.metadata['license'], 'CC BY 3.0')
    # User metadata merges the usual way, and wins over the defaults:
    user = samples.big_buck_bunny(resize=(8, 8),
                                  metadata={'foo': 'bar', 'title': 'clip'})
    npt.assert_equal(user.metadata['foo'], 'bar')
    npt.assert_equal(user.metadata['title'], 'clip')
    npt.assert_equal(user.metadata['license'], 'CC BY 3.0')


def test_big_buck_bunny_not_top_level():
    npt.assert_equal(hasattr(p2p.stimuli, 'big_buck_bunny'), False)


def _ink(scene):
    """Visual-field coordinates of the C's pixels, as ``(x, y)`` arrays"""
    img = scene.source.data.reshape(scene.source.img_shape)
    rows, cols = np.where(img < 0.5)
    return scene.pixel_to_dva(cols, rows)


def test_landolt_c():
    scene = samples.landolt_c(gap=0.5 * dva, fov=15 * dva, shape=(512, 512))
    npt.assert_equal(isinstance(scene, Scene), True)
    npt.assert_equal(scene.shape, (512, 512))
    npt.assert_equal(scene.fov, (15.0, 15.0))
    npt.assert_equal(np.unique(scene.source.data).tolist(), [0.0, 1.0])
    meta = scene.source.metadata
    npt.assert_almost_equal(meta['gap'], 0.5)
    npt.assert_equal(meta['position'], (0.0, 0.0))
    npt.assert_almost_equal(meta['orientation'], 0.0)
    npt.assert_equal(meta['polarity'], 'dark')
    # A scalar fov is the horizontal one; the vertical follows from `shape`:
    npt.assert_equal(samples.landolt_c(fov=10, shape=(256, 512)).fov,
                     (10.0, 5.0))


def test_landolt_c_geometry():
    gap, fov, shape = 1.0, 20.0, (512, 512)
    # One pixel of slack in each direction, since the extent is measured
    # between the centers of the outermost lit pixels:
    tol = 2 * fov / shape[0]
    scene = samples.landolt_c(gap=gap, fov=fov, shape=shape)
    x, y = _ink(scene)
    # Outer diameter is 5 gaps, in both directions:
    npt.assert_almost_equal(x.max() - x.min(), 5 * gap, decimal=1)
    npt.assert_almost_equal(y.max() - y.min(), 5 * gap, decimal=1)
    npt.assert_array_less(abs(x.max() + x.min()), tol)
    npt.assert_array_less(abs(y.max() + y.min()), tol)
    # The stroke is one gap wide: with the opening pointing right, the column
    # through the center of the C crosses ink from -2.5 to -1.5 gaps:
    left = x[(abs(y) < tol) & (x < 0)]
    npt.assert_almost_equal(left.max() - left.min(), gap, decimal=1)
    # ... and the opening is one gap wide, measured across the gap direction:
    npt.assert_equal(np.any((x > 0) & (abs(y) < gap / 2)), False)
    # Ink resumes right at the slot edge, so the opening is exactly a gap wide:
    edge = abs(y[(x > 1.5 * gap) & (x < 2.5 * gap)]).min()
    npt.assert_almost_equal(edge, gap / 2, decimal=1)


@pytest.mark.parametrize('orientation,direction', [
    (0, (1, 0)), (90, (0, 1)), (180, (-1, 0)), (270, (0, -1)),
    # An arbitrary angle, and one wrapped past a full turn:
    (45, (np.sqrt(0.5), np.sqrt(0.5))), (360 + 90, (0, 1)),
])
def test_landolt_c_orientation(orientation, direction):
    gap = 1.0
    scene = samples.landolt_c(gap=gap, orientation=orientation * deg, fov=20,
                              shape=(512, 512))
    img = scene.source.data.reshape(scene.source.img_shape)
    rows, cols = np.where(img > 0.5)
    x, y = scene.pixel_to_dva(cols, rows)
    # The background pixels that fall inside the annulus are the opening, and
    # they sit on the side the orientation points to:
    radius = np.hypot(x, y)
    inside = (radius > 1.5 * gap) & (radius < 2.5 * gap)
    npt.assert_almost_equal([x[inside].mean() / radius[inside].mean(),
                             y[inside].mean() / radius[inside].mean()],
                            direction, decimal=2)


def test_landolt_c_position():
    gap, position = 0.5, (5.0, -3.0)
    scene = samples.landolt_c(gap=gap, position=position * dva, fov=20 * dva,
                              shape=(512, 512))
    x, y = _ink(scene)
    npt.assert_almost_equal([(x.min() + x.max()) / 2,
                             (y.min() + y.max()) / 2], position, decimal=1)
    # Eccentricity changes, angular size does not:
    npt.assert_almost_equal(x.max() - x.min(), 5 * gap, decimal=1)
    npt.assert_equal(scene.source.metadata['position'], position)


def test_landolt_c_polarity():
    kwargs = dict(gap=1, orientation=30 * deg, fov=12, shape=(128, 128))
    dark = samples.landolt_c(polarity='dark', **kwargs).source.data
    light = samples.landolt_c(polarity='light', **kwargs).source.data
    npt.assert_almost_equal(light, 1.0 - dark)


@pytest.mark.parametrize('kwargs,msg', [
    (dict(gap=0), "'gap'"),
    (dict(gap=-1), "'gap'"),
    (dict(gap=np.inf), "'gap'"),
    (dict(position=(0, 0, 0)), "'position'"),
    (dict(position=(np.nan, 0)), "'position'"),
    (dict(orientation=np.nan), "'orientation'"),
    (dict(polarity='inverted'), "'polarity'"),
    (dict(shape=(0, 10)), "'shape'"),
    (dict(shape=(10, 10, 10)), "'shape'"),
    (dict(shape=(10.5, 10)), "'shape'"),
    (dict(fov=0), "'fov'"),
    # The whole outer circle has to fit: 5 * gap = 5 dva across, centered 8 dva
    # out in a 10 dva field:
    (dict(gap=1, position=(8, 0), fov=10), 'half-FOV'),
    (dict(gap=1, position=(0, -8), fov=10), 'half-FOV'),
    # 0.1 dva across a 10-degree, 128-pixel frame is 1.3 pixels:
    (dict(gap=0.1, fov=10, shape=(128, 128)), 'resolve the opening'),
])
def test_landolt_c_invalid(kwargs, msg):
    with pytest.raises(ValueError) as excinfo:
        samples.landolt_c(**kwargs)
    npt.assert_equal(msg in str(excinfo.value), True)


@pytest.mark.parametrize('loader,shape', [
    (samples.bvl_cake, (495, 435, 3)),
    (samples.ucsb_surf, (476, 845, 3)),
])
def test_samples_photos(loader, shape):
    stim = loader()
    # A loader returns a plain stimulus, not a sample type of its own:
    npt.assert_equal(type(stim), ImageStimulus)
    npt.assert_equal(stim.img_shape, shape)
    npt.assert_equal(stim.time, None)
    npt.assert_equal(np.all(np.isfinite(stim.data)), True)
    npt.assert_equal(stim.data.min() >= 0, True)
    npt.assert_equal(stim.data.max() <= 1, True)
    npt.assert_equal(loader(as_gray=True).img_shape, shape[:2])
    npt.assert_equal(loader(resize=(16, 24)).img_shape, (16, 24, 3))
    npt.assert_equal(loader(metadata={'foo': 'bar'}).metadata['foo'], 'bar')


def test_samples_photo_metadata():
    cake = samples.bvl_cake(resize=(8, 8))
    npt.assert_equal(cake.metadata['title'], 'Bionic Vision Lab cake')
    npt.assert_equal(cake.metadata['license'], 'BSD-3-Clause')
    surf = samples.ucsb_surf(resize=(8, 8))
    npt.assert_equal(surf.metadata['title'], 'UCSB surf')
    # Provenance lives under 'credit': ImageStimulus overwrites 'source' with
    # the local file name.
    npt.assert_equal(surf.metadata['credit'],
                     'Courtesy of the National Library of Medicine')
    # User metadata merges the usual way, and wins over the defaults:
    user = samples.ucsb_surf(resize=(8, 8), metadata={'title': 'frame'})
    npt.assert_equal(user.metadata['title'], 'frame')
    npt.assert_equal(user.metadata['credit'],
                     'Courtesy of the National Library of Medicine')


def test_samples_photos_not_top_level():
    for name in ('bvl_cake', 'ucsb_surf'):
        npt.assert_equal(hasattr(samples, name), True)
        npt.assert_equal(hasattr(p2p.stimuli, name), False)
