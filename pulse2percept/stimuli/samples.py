"""Bundled sample images and videos.

Loaders return ordinary :py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus` objects::

    from pulse2percept.stimuli import samples
    logo = samples.logo_bvl()

:py:class:`~pulse2percept.stimuli.LogoBVL` and
:py:class:`~pulse2percept.stimuli.LogoUCSB` are deprecated in favor of
:py:func:`logo_bvl` and :py:func:`logo_ucsb` and will be removed in v0.12.

.. versionadded:: 0.11.0
"""
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np

from .base import ImageStimulus, VideoStimulus
from ..utils.deprecation import deprecated

__all__ = [
    'big_buck_bunny',
    'bvl_cake',
    'cajal_retina',
    'logo_bvl',
    'logo_ucsb',
    'ucsb_bike',
    'ucsb_flyover',
    'ucsb_pedestrians',
    'ucsb_surf',
    'zebrafish_retina',
]


def _sample_path(filename):
    """Return the absolute path of a bundled sample asset"""
    return join(dirname(__file__), 'data', 'samples', filename)


def _plot_frames(video, n_frames=5):
    """Plot evenly spaced frames of a video side by side (used by the docs)"""
    height, width = video.vid_shape[:2]
    fig, axes = plt.subplots(1, n_frames,
                             figsize=(2.4 * n_frames, 2.4 * height / width))
    idx = np.linspace(0, video.data.shape[1] - 1, n_frames).round().astype(int)
    cmap = 'gray' if len(video.vid_shape) == 3 else None
    for ax, t in zip(axes, idx):
        ax.imshow(video.data[:, t].reshape(video.vid_shape[:-1]), cmap=cmap,
                  vmin=0)
        ax.set_title(f'{video.time[t]:.0f} ms')
        ax.axis('off')
    fig.tight_layout()


#: CC BY 3.0 attribution, stored in the stimulus metadata. See
#: ``data/samples/README.rst``.
_BIG_BUCK_BUNNY_CREDIT = {
    'title': 'Big Buck Bunny',
    'creator': 'Blender Foundation',
    'license': 'CC BY 3.0',
}


def big_buck_bunny(resize=None, electrodes=None, metadata=None,
                   as_gray=False):
    """Big Buck Bunny video clip

    Load a 115-frame, 24 fps excerpt of *Big Buck Bunny* (359x640 RGB).

    The clip is © 2008 Blender Foundation and licensed CC BY 3.0; attribution
    is included in ``metadata``. It has no intrinsic field of view.

    .. note::
        At full resolution the stimulus contains 689,280 electrodes. Use
        ``resize`` and/or ``as_gray`` before passing it to most models.

    .. plot::

        from pulse2percept.stimuli import samples
        samples._plot_frames(samples.big_buck_bunny(resize=(180, 320)))

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of each video
        frame.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) frame.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary. Keys given
        here override the attribution defaults above.

    as_gray : bool, optional
        Flag whether to convert the video to grayscale.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.VideoStimulus`

    Examples
    --------
    >>> from pulse2percept.stimuli import samples
    >>> video = samples.big_buck_bunny(resize=(60, 80))
    >>> video.vid_shape
    (60, 80, 3, 115)

    """
    meta = dict(_BIG_BUCK_BUNNY_CREDIT)
    if isinstance(metadata, dict):
        meta.update(metadata)
    elif metadata is not None:
        meta['user'] = metadata
    return VideoStimulus(_sample_path('big-buck-bunny.mp4'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=meta, compress=False)


#: The photograph is released under pulse2percept's own BSD 3-Clause license.
#: See ``data/samples/README.rst``.
_BVL_CAKE_CREDIT = {
    'title': 'Bionic Vision Lab cake',
    'license': 'BSD-3-Clause',
}


def bvl_cake(resize=None, electrodes=None, metadata=None, as_gray=False):
    """Bionic Vision Lab cake photograph

    Load a 495x435 RGB photograph of a Bionic Vision Lab cake.

    The photograph is distributed under the BSD 3-Clause license; attribution
    is included in ``metadata``.

    .. plot::

        from pulse2percept.stimuli import samples
        samples.bvl_cake().plot().axis('off')

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary. Keys given
        here override the attribution defaults above.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus`

    Examples
    --------
    >>> from pulse2percept.stimuli import samples
    >>> samples.bvl_cake().img_shape
    (495, 435, 3)

    """
    meta = dict(_BVL_CAKE_CREDIT)
    if isinstance(metadata, dict):
        meta.update(metadata)
    elif metadata is not None:
        meta['user'] = metadata
    return ImageStimulus(_sample_path('bvl-cake.jpg'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=meta, compress=False)


#: Wikimedia Commons designates the drawing public domain under its
#: ``PD-old-70`` tag; Cajal died in 1934. See ``data/samples/README.rst``.
_CAJAL_RETINA_CREDIT = {
    'title': 'Cajal retina drawing',
    'creator': 'Santiago Ramon y Cajal',
    'credit': 'Wikimedia Commons',
    'license': 'Public domain',
}


def cajal_retina(resize=None, electrodes=None, metadata=None, as_gray=False):
    """Cajal's drawing of the retina

    Load a 745x500 RGB scan of Cajal's drawing of the retina.

    Wikimedia Commons designates the drawing public domain. Attribution is
    included in ``metadata``; see ``data/samples/README.rst`` for provenance.

    .. plot::

        from pulse2percept.stimuli import samples
        samples.cajal_retina().plot().axis('off')

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary. Keys given
        here override the attribution defaults above.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus`

    Examples
    --------
    >>> from pulse2percept.stimuli import samples
    >>> samples.cajal_retina().img_shape
    (745, 500, 3)

    """
    meta = dict(_CAJAL_RETINA_CREDIT)
    if isinstance(metadata, dict):
        meta.update(metadata)
    elif metadata is not None:
        meta['user'] = metadata
    return ImageStimulus(_sample_path('cajal-retina.jpg'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=meta, compress=False)


def logo_bvl(resize=None, electrodes=None, metadata=None, as_gray=False):
    """Bionic Vision Lab (BVL) logo

    Load the 576x720x4 Bionic Vision Lab (BVL) logo.

    .. plot::

        from pulse2percept.stimuli import samples
        samples.logo_bvl().plot().axis('off')

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale. The alpha channel of
        the RGBA source is blended with the color black.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus`

    """
    return ImageStimulus(_sample_path('bionic-vision-lab.png'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=metadata, compress=False)


def logo_ucsb(resize=None, electrodes=None, metadata=None):
    """UCSB logo

    Load a 324x727 white-on-black logo of the University of California, Santa
    Barbara.

    .. plot::

        from pulse2percept.stimuli import samples
        samples.logo_ucsb().plot().axis('off')

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus`
        A grayscale image stimulus.

    """
    return ImageStimulus(_sample_path('ucsb.png'), resize=resize, as_gray=True,
                         electrodes=electrodes, metadata=metadata,
                         compress=False)


#: The photograph is released under pulse2percept's own BSD 3-Clause license.
#: See ``data/samples/README.rst``.
_UCSB_BIKE_CREDIT = {
    'title': 'UCSB bike path',
    'license': 'BSD-3-Clause',
}


def ucsb_bike(resize=None, electrodes=None, metadata=None, as_gray=False):
    """UCSB bike path photograph

    Load a 600x900x3 RGB photograph of a cyclist, a pedestrian, a crosswalk,
    and a stop sign on the UCSB campus.

    The photograph is distributed under the BSD 3-Clause license;
    attribution is included in ``metadata``.

    .. plot::

        from pulse2percept.stimuli import samples
        samples.ucsb_bike().plot().axis('off')

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary. Keys given
        here override the attribution defaults above.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus`

    Examples
    --------
    >>> from pulse2percept.stimuli import samples
    >>> samples.ucsb_bike().img_shape
    (600, 900, 3)

    """
    meta = dict(_UCSB_BIKE_CREDIT)
    if isinstance(metadata, dict):
        meta.update(metadata)
    elif metadata is not None:
        meta['user'] = metadata
    return ImageStimulus(_sample_path('ucsb-bike.jpg'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=meta, compress=False)


#: Credit for assets from the NLM video *Towards a Smart Bionic Eye*.
#: Stored under ``credit``, because ``metadata['source']`` is overwritten
#: with the local file name. See ``data/samples/README.rst``.
_NLM_CREDIT = {
    'credit': 'Courtesy of the National Library of Medicine',
    'license': 'Public domain (U.S. government work)',
}

_UCSB_FLYOVER_CREDIT = dict(_NLM_CREDIT, title='UCSB flyover')


def ucsb_flyover(resize=None, electrodes=None, metadata=None, as_gray=False):
    """UCSB aerial flyover clip

    Load a 346x640x3 RGB aerial shot sweeping over the UCSB campus and
    shoreline, 53 frames at 24.32 fps (2.18 s of video).

    The clip is cut from *Towards a Smart Bionic Eye*, produced by the
    National Library of Medicine / National Institutes of Health. It is a
    U.S. government work in the public domain (not BSD-licensed); attribution
    is included in ``metadata``.

    The clip has no intrinsic field of view; use a
    :py:class:`~pulse2percept.vision.Scene` to set its visual-field extent.

    .. note::
       The variable-frame-rate source is resampled to a constant rate:
       about 10 of the 53 frames repeat the frame before them.

    .. plot::

        from pulse2percept.stimuli import samples
        samples._plot_frames(samples.ucsb_flyover(resize=(173, 320)))

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of each video
        frame.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) frame.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary. Keys given
        here override the attribution defaults above.

    as_gray : bool, optional
        Flag whether to convert the video to grayscale.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.VideoStimulus`

    Examples
    --------
    >>> from pulse2percept.stimuli import samples
    >>> samples.ucsb_flyover(resize=(60, 80)).vid_shape
    (60, 80, 3, 53)

    """
    meta = dict(_UCSB_FLYOVER_CREDIT)
    if isinstance(metadata, dict):
        meta.update(metadata)
    elif metadata is not None:
        meta['user'] = metadata
    return VideoStimulus(_sample_path('ucsb-flyover.mp4'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=meta, compress=False)


_UCSB_PEDESTRIANS_CREDIT = dict(_NLM_CREDIT, title='UCSB pedestrians')


def ucsb_pedestrians(resize=None, electrodes=None, metadata=None,
                     as_gray=False):
    """UCSB pedestrians clip

    Load a 346x640x3 RGB shot of pedestrians on the UCSB campus, 45 frames at
    24.52 fps (1.84 s of video).

    The clip is cut from *Towards a Smart Bionic Eye*, produced by the
    National Library of Medicine / National Institutes of Health. It is a
    U.S. government work in the public domain (not BSD-licensed); attribution
    is included in ``metadata``.

    The clip has no intrinsic field of view; use a
    :py:class:`~pulse2percept.vision.Scene` to set its visual-field extent.

    .. note::
       The variable-frame-rate source is resampled to a constant rate:
       about 9 of the 45 frames repeat the frame before them.

    .. plot::

        from pulse2percept.stimuli import samples
        samples._plot_frames(samples.ucsb_pedestrians(resize=(173, 320)))

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of each video
        frame.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) frame.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary. Keys given
        here override the attribution defaults above.

    as_gray : bool, optional
        Flag whether to convert the video to grayscale.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.VideoStimulus`

    Examples
    --------
    >>> from pulse2percept.stimuli import samples
    >>> samples.ucsb_pedestrians(resize=(60, 80)).vid_shape
    (60, 80, 3, 45)

    """
    meta = dict(_UCSB_PEDESTRIANS_CREDIT)
    if isinstance(metadata, dict):
        meta.update(metadata)
    elif metadata is not None:
        meta['user'] = metadata
    return VideoStimulus(_sample_path('ucsb-pedestrians.mp4'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=meta, compress=False)


_UCSB_SURF_CREDIT = dict(_NLM_CREDIT, title='UCSB surf')


def ucsb_surf(resize=None, electrodes=None, metadata=None, as_gray=False):
    """UCSB coastline video frame

    Load a 476x845x3 RGB frame of the UCSB coastline, as a naturalistic image
    stimulus.

    The frame comes from *Towards a Smart Bionic Eye*, produced by the
    National Library of Medicine / National Institutes of Health. It is a
    U.S. government work in the public domain (not BSD-licensed); attribution
    is included in ``metadata``.

    .. plot::

        from pulse2percept.stimuli import samples
        samples.ucsb_surf().plot().axis('off')

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary. Keys given
        here override the attribution defaults above.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus`

    Examples
    --------
    >>> from pulse2percept.stimuli import samples
    >>> samples.ucsb_surf().img_shape
    (476, 845, 3)

    """
    meta = dict(_UCSB_SURF_CREDIT)
    if isinstance(metadata, dict):
        meta.update(metadata)
    elif metadata is not None:
        meta['user'] = metadata
    return ImageStimulus(_sample_path('ucsb-surf.jpg'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=meta, compress=False)


#: CC BY 4.0 attribution, stored in the stimulus metadata. See
#: ``data/samples/README.rst``.
_ZEBRAFISH_RETINA_CREDIT = {
    'title': 'Sunrise in the eye: zebrafish retina',
    'creator': 'Dr Kara Cerveny & Dr Steve Wilson',
    'credit': 'Wellcome Collection',
    'license': 'CC BY 4.0',
}


def zebrafish_retina(resize=None, electrodes=None, metadata=None,
                     as_gray=False):
    """Zebrafish retina micrograph

    Load a 544x760x3 RGB fluorescence micrograph of a zebrafish retina
    ("Sunrise in the eye"), as a false-color image stimulus.

    The micrograph is by Dr Kara Cerveny and Dr Steve Wilson (Wellcome
    Collection), licensed CC BY 4.0; attribution is included in
    ``metadata``.

    .. plot::

        from pulse2percept.stimuli import samples
        samples.zebrafish_retina().plot().axis('off')

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary. Keys given
        here override the attribution defaults above.

    as_gray : bool, optional
        Flag whether to convert the image to grayscale.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus`

    Examples
    --------
    >>> from pulse2percept.stimuli import samples
    >>> samples.zebrafish_retina().img_shape
    (544, 760, 3)

    """
    meta = dict(_ZEBRAFISH_RETINA_CREDIT)
    if isinstance(metadata, dict):
        meta.update(metadata)
    elif metadata is not None:
        meta['user'] = metadata
    return ImageStimulus(_sample_path('zebrafish-retina.jpg'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=meta, compress=False)


@deprecated(alt_func='pulse2percept.stimuli.samples.logo_bvl',
            deprecated_version='0.11.0', removed_version='0.12.0')
class LogoBVL(ImageStimulus):
    """Bionic Vision Lab (BVL) logo

    Load the 576x720x4 Bionic Vision Lab (BVL) logo.

    .. versionadded:: 0.7

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    """
    __slots__ = ()

    def __init__(self, resize=None, electrodes=None, metadata=None,
                 as_gray=False):
        super().__init__(_sample_path('bionic-vision-lab.png'), resize=resize,
                         as_gray=as_gray, electrodes=electrodes,
                         metadata=metadata, compress=False)


@deprecated(alt_func='pulse2percept.stimuli.samples.logo_ucsb',
            deprecated_version='0.11.0', removed_version='0.12.0')
class LogoUCSB(ImageStimulus):
    """UCSB logo

    Load a 324x727 white-on-black logo of the University of California, Santa
    Barbara.

    .. versionadded:: 0.7

    Parameters
    ----------
    resize : ``(height, width)`` or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. By default,
        pixels are named by row letter, column number, and color-channel
        suffix (e.g. 'A1', 'C12', 'A1_R').

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    """
    __slots__ = ()

    def __init__(self, resize=None, electrodes=None, metadata=None):
        super().__init__(_sample_path('ucsb.png'), resize=resize,
                         as_gray=True, electrodes=electrodes,
                         metadata=metadata, compress=False)
