import warnings

import numpy as np
import numpy.testing as npt
import pytest

import pulse2percept as p2p
from pulse2percept.stimuli import (ImageStimulus, LogoBVL, LogoUCSB,
                                   SnellenChart, VideoStimulus, samples)


def _legacy(cls, **kwargs):
    """Instantiate a legacy sample class without its deprecation warning"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return cls(**kwargs)


@pytest.mark.parametrize('loader,legacy', [
    (samples.logo_bvl, LogoBVL),
    (samples.logo_ucsb, LogoUCSB),
    (samples.snellen_chart, SnellenChart),
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
    (SnellenChart, 'samples.snellen_chart'),
])
def test_samples_legacy_classes_deprecated(legacy, alt):
    with pytest.warns(DeprecationWarning) as record:
        legacy()
    msg = str(record[0].message)
    npt.assert_equal(alt in msg, True)
    npt.assert_equal('0.12.0' in msg, True)


@pytest.mark.parametrize('show_annotations', (True, False))
@pytest.mark.parametrize('row', [None] + list(range(1, 12)))
def test_snellen_chart(row, show_annotations):
    """Every row/annotation combination matches the deprecated class"""
    kwargs = dict(row=row, show_annotations=show_annotations)
    new = samples.snellen_chart(**kwargs)
    npt.assert_equal(type(new), ImageStimulus)
    npt.assert_equal(new.img_shape, _legacy(SnellenChart, **kwargs).img_shape)
    npt.assert_almost_equal(new.data, _legacy(SnellenChart, **kwargs).data)
    npt.assert_equal(new.time, None)
    # Annotations are the right-hand columns, so cropping them narrows the
    # chart without changing its height:
    npt.assert_equal(new.img_shape[1], 840 if show_annotations else 444)


@pytest.mark.parametrize('row', [0, 12, -1, -11, True, 1.5, [1, 3], 'first'])
def test_snellen_chart_invalid_row(row):
    with pytest.raises(ValueError) as excinfo:
        samples.snellen_chart(row=row)
    npt.assert_equal('"row"' in str(excinfo.value), True)


def test_samples_namespace():
    # The module is reachable, but its loaders are not promoted to the
    # top-level namespace:
    npt.assert_equal(p2p.stimuli.samples is samples, True)
    for name in samples.__all__:
        npt.assert_equal(hasattr(p2p.stimuli, name), False)
    # `samples` publishes the loaders only: the deprecated classes stay
    # importable for `pulse2percept.stimuli`, but are not new public API here.
    for legacy in ('LogoBVL', 'LogoUCSB', 'SnellenChart'):
        npt.assert_equal(legacy in samples.__all__, False)
        npt.assert_equal(hasattr(p2p.stimuli, legacy), True)
    # The two video samples were removed along with their assets:
    for gone in ('BostonTrain', 'GirlPool'):
        npt.assert_equal(hasattr(p2p.stimuli, gone), False)
    # The procedural optotypes are generated, not bundled, so they live in
    # `psychophysics` now:
    for gone in ('boston_train', 'girl_pool', 'landolt_c', 'tumbling_e'):
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


#: Properties of the two packaged NLM clips, as decoded. Both sources are
#: variable-frame-rate, so the reader pads them to a constant rate: the frame
#: counts here are the padded ones, not the number of distinct frames.
_NLM_CLIPS = [
    (samples.ucsb_flyover, 53, 24.32),
    (samples.ucsb_pedestrians, 45, 24.52),
]


@pytest.mark.parametrize('loader,n_frames,fps', _NLM_CLIPS)
def test_samples_nlm_clips(loader, n_frames, fps):
    video = loader()
    # A loader returns a plain stimulus, not a sample type of its own:
    npt.assert_equal(type(video), VideoStimulus)
    npt.assert_equal(video.vid_shape, (346, 640, 3, n_frames))
    npt.assert_equal(video.time.size, n_frames)
    npt.assert_almost_equal(video.metadata['fps'], fps)
    npt.assert_almost_equal(np.diff(video.time), 1000.0 / fps, decimal=3)
    npt.assert_equal(np.all(np.isfinite(video.data)), True)
    npt.assert_equal(video.data.min() >= 0, True)
    npt.assert_equal(video.data.max() <= 1, True)
    npt.assert_equal(loader(as_gray=True, resize=(30, 40)).vid_shape,
                     (30, 40, n_frames))
    npt.assert_equal(loader(resize=(30, 40)).vid_shape, (30, 40, 3, n_frames))


@pytest.mark.parametrize('loader,title', [
    (samples.ucsb_flyover, 'UCSB flyover'),
    (samples.ucsb_pedestrians, 'UCSB pedestrians'),
])
def test_samples_nlm_clip_metadata(loader, title):
    video = loader(resize=(8, 8))
    npt.assert_equal(video.metadata['title'], title)
    npt.assert_equal(video.metadata['credit'],
                     'Courtesy of the National Library of Medicine')
    npt.assert_equal(video.metadata['license'],
                     'Public domain (U.S. government work)')
    # User metadata merges the usual way, and wins over the defaults:
    user = loader(resize=(8, 8), metadata={'foo': 'bar', 'title': 'clip'})
    npt.assert_equal(user.metadata['foo'], 'bar')
    npt.assert_equal(user.metadata['title'], 'clip')
    npt.assert_equal(user.metadata['credit'],
                     'Courtesy of the National Library of Medicine')


def test_samples_nlm_clips_not_top_level():
    for name in ('ucsb_flyover', 'ucsb_pedestrians'):
        npt.assert_equal(hasattr(samples, name), True)
        npt.assert_equal(hasattr(p2p.stimuli, name), False)


@pytest.mark.parametrize('loader,shape', [
    (samples.bvl_cake, (495, 435, 3)),
    (samples.cajal_retina, (745, 500, 3)),
    (samples.ucsb_bike, (600, 900, 3)),
    (samples.ucsb_surf, (476, 845, 3)),
    (samples.zebrafish_retina, (544, 760, 3)),
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
    cajal = samples.cajal_retina(resize=(8, 8))
    npt.assert_equal(cajal.metadata['creator'], 'Santiago Ramon y Cajal')
    npt.assert_equal(cajal.metadata['license'], 'Public domain')
    zebra = samples.zebrafish_retina(resize=(8, 8))
    npt.assert_equal(zebra.metadata['title'],
                     'Sunrise in the eye: zebrafish retina')
    npt.assert_equal(zebra.metadata['credit'], 'Wellcome Collection')
    npt.assert_equal(zebra.metadata['license'], 'CC BY 4.0')


def test_samples_photos_not_top_level():
    for name in ('bvl_cake', 'cajal_retina', 'ucsb_bike', 'ucsb_surf',
                 'zebrafish_retina'):
        npt.assert_equal(hasattr(samples, name), True)
        npt.assert_equal(hasattr(p2p.stimuli, name), False)
