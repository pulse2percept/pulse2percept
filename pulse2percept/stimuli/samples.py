""":py:func:`~pulse2percept.stimuli.samples.big_buck_bunny`,
:py:func:`~pulse2percept.stimuli.samples.bvl_cake`,
:py:func:`~pulse2percept.stimuli.samples.cajal_retina`,
:py:func:`~pulse2percept.stimuli.samples.logo_bvl`,
:py:func:`~pulse2percept.stimuli.samples.logo_ucsb`,
:py:func:`~pulse2percept.stimuli.samples.snellen_chart`,
:py:func:`~pulse2percept.stimuli.samples.ucsb_bike`,
:py:func:`~pulse2percept.stimuli.samples.ucsb_flyover`,
:py:func:`~pulse2percept.stimuli.samples.ucsb_pedestrians`,
:py:func:`~pulse2percept.stimuli.samples.ucsb_surf`,
:py:func:`~pulse2percept.stimuli.samples.zebrafish_retina`

Sample stimuli bundled with pulse2percept, for demos, docs, and tests.

The loaders return ordinary
:py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus` objects; they are not
stimulus types of their own. Access them through the module rather than the
top-level namespace::

    from pulse2percept.stimuli import samples
    logo = samples.logo_bvl()

:py:class:`~pulse2percept.stimuli.LogoBVL`,
:py:class:`~pulse2percept.stimuli.LogoUCSB` and
:py:class:`~pulse2percept.stimuli.SnellenChart` are the deprecated
predecessors of :py:func:`logo_bvl`, :py:func:`logo_ucsb` and
:py:func:`snellen_chart`. They load the same assets and are kept unchanged
until they are removed in v0.12.0.

.. versionadded:: 0.11.0
"""
from os.path import dirname, join

import numpy as np
from skimage.io import imread

from .base import ImageStimulus, VideoStimulus
from ..utils.deprecation import deprecated

__all__ = [
    'big_buck_bunny',
    'bvl_cake',
    'cajal_retina',
    'logo_bvl',
    'logo_ucsb',
    'snellen_chart',
    'ucsb_bike',
    'ucsb_flyover',
    'ucsb_pedestrians',
    'ucsb_surf',
    'zebrafish_retina',
]


def _sample_path(filename):
    """Return the absolute path of a bundled sample asset"""
    return join(dirname(__file__), 'data', 'samples', filename)


#: Start and stop row of each of the chart's eleven lines, in source pixels.
#: Line 1 is 20/200, line 2 is 20/100, and so on.
_SNELLEN_ROWS = (
    [5, 260],  # line 1
    [310, 450],  # line 2
    [505, 600],
    [645, 715],
    [755, 810],
    [840, 883],
    [965, 1003],
    [1057, 1088],
    [1170, 1193],
    [1243, 1263],
    [1317, 1335]  # line 11
)


def _snellen_source(show_annotations, row):
    """Return the Snellen chart, cropped as requested

    A path when the whole chart is wanted, and a decoded array when a crop
    is, since cropping has to happen before the pixels become a stimulus.
    """
    source = join(dirname(__file__), 'data', 'snellen.png')
    if row is None and show_annotations is not False:
        return source
    source = imread(source)
    if show_annotations is False:
        # Crop the line numbers and acuity annotations:
        source = source[:, :444]
    if row is not None:
        # Only these eleven lines exist, and they are 1-indexed. A bool or a
        # negative index would otherwise select some other line silently.
        valid = (isinstance(row, (int, np.integer)) and
                 not isinstance(row, bool) and
                 1 <= row <= len(_SNELLEN_ROWS))
        if not valid:
            raise ValueError(f'Invalid value for "row": {row!r}. Choose an '
                             f'int between 1 and {len(_SNELLEN_ROWS)}, where '
                             f'row 1 is 20/200.')
        first, last = _SNELLEN_ROWS[row - 1]
        source = source[first:last]
    return source


#: The clip is CC BY 3.0, not BSD like the rest of pulse2percept, so its
#: attribution travels with the stimulus. See ``data/samples/README.rst``.
_BIG_BUCK_BUNNY_CREDIT = {
    'title': 'Big Buck Bunny',
    'creator': 'Blender Foundation',
    'license': 'CC BY 3.0',
}


def big_buck_bunny(resize=None, electrodes=None, metadata=None,
                   as_gray=False):
    """Big Buck Bunny video clip

    Load a 359x640x3 RGB excerpt of *Big Buck Bunny*, 115 frames at 24 fps
    (4.79 s of video), as a naturalistic video stimulus. ``time`` holds frame
    onsets, so it ends at 4750 ms, when the last frame comes up.

    The clip is copyright 2008 Blender Foundation
    (`bigbuckbunny.org <https://www.bigbuckbunny.org>`_) and is distributed
    under the Creative Commons Attribution 3.0 license, not under
    pulse2percept's BSD license; ``metadata`` carries the attribution.

    The clip has no intrinsic field of view; wrap it in a
    :py:class:`~pulse2percept.vision.Scene` to say how much of the visual
    field it covers.

    .. note::
       At full resolution this is 689,280 electrodes x 115 time points
       (~317 MB). Pass ``resize`` and/or ``as_gray`` before feeding it to a
       model.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of each video
        frame.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the frame: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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

    Load a 495x435x3 RGB photograph of a cake decorated with the Bionic Vision
    Lab logo, as a naturalistic image stimulus.

    The photograph is made available by its copyright holder under the same
    BSD 3-Clause license as pulse2percept; ``metadata`` carries a short form
    of that.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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

    Load a 745x500x3 RGB scan of Santiago Ramon y Cajal's drawing of the
    layered structure of the retina, as a high-contrast line-art image
    stimulus.

    Wikimedia Commons designates the drawing public domain, so it is not
    covered by pulse2percept's BSD license; ``metadata`` carries the
    attribution and ``data/samples/README.rst`` the terms and their limits.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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


def snellen_chart(resize=None, show_annotations=True, row=None,
                  electrodes=None, metadata=None):
    """Snellen chart

    Load the 1348x840 Snellen chart commonly used to measure visual acuity,
    as a grayscale image stimulus.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    show_annotations : bool, optional
        If False, crop the line numbers and acuity annotations on the right,
        leaving the letters alone.

    row : int or None, optional
        Select a single row (1 to 11) of the chart. Row 1 corresponds to
        20/200, row 2 to 20/100, and so on. Anything outside that range
        raises.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.ImageStimulus`

    Examples
    --------
    >>> from pulse2percept.stimuli import samples
    >>> samples.snellen_chart(row=1).img_shape
    (255, 840)

    """
    return ImageStimulus(_snellen_source(show_annotations, row),
                         resize=resize, as_gray=True, electrodes=electrodes,
                         metadata=metadata, compress=False)


#: The photograph is released under pulse2percept's own BSD 3-Clause license.
#: See ``data/samples/README.rst``.
_UCSB_BIKE_CREDIT = {
    'title': 'UCSB bike path',
    'license': 'BSD-3-Clause',
}


def ucsb_bike(resize=None, electrodes=None, metadata=None, as_gray=False):
    """UCSB bike path photograph

    Load a 600x900x3 RGB photograph of a cyclist, a pedestrian, a crosswalk,
    and a stop sign on the UCSB campus, as a naturalistic image stimulus of an
    everyday mobility scene.

    The photograph is made available by its copyright holder under the same
    BSD 3-Clause license as pulse2percept; ``metadata`` carries a short form
    of that.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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


#: Shared by every asset cut from the NLM video *Towards a Smart Bionic Eye*.
#: ``ImageStimulus`` and ``VideoStimulus`` overwrite ``metadata['source']``
#: with the local file name, so provenance goes under ``credit``. See
#: ``data/samples/README.rst``.
_NLM_CREDIT = {
    'credit': 'Courtesy of the National Library of Medicine',
    'license': 'Public domain (U.S. government work)',
}

_UCSB_FLYOVER_CREDIT = dict(_NLM_CREDIT, title='UCSB flyover')


def ucsb_flyover(resize=None, electrodes=None, metadata=None, as_gray=False):
    """UCSB aerial flyover clip

    Load a 346x640x3 RGB aerial shot sweeping over the UCSB campus and
    shoreline, 53 frames at 24.32 fps (2.18 s of video). ``time`` holds frame
    onsets, so it ends at 2138 ms, when the last frame comes up.

    The clip is cut from *Towards a Smart Bionic Eye*, produced by the
    National Library of Medicine / National Institutes of Health. As a U.S.
    government work it is in the public domain in the United States, and is
    therefore not covered by pulse2percept's BSD license; ``metadata`` carries
    the requested attribution.

    The clip has no intrinsic field of view; wrap it in a
    :py:class:`~pulse2percept.vision.Scene` to say how much of the visual
    field it covers.

    .. note::
       The source is variable-frame-rate, so the reader resamples it to a
       constant rate: about 10 of the 53 frames repeat the frame before them.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of each video
        frame.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the frame: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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
    24.52 fps (1.84 s of video). ``time`` holds frame onsets, so it ends at
    1794 ms, when the last frame comes up.

    The clip is cut from *Towards a Smart Bionic Eye*, produced by the
    National Library of Medicine / National Institutes of Health. As a U.S.
    government work it is in the public domain in the United States, and is
    therefore not covered by pulse2percept's BSD license; ``metadata`` carries
    the requested attribution.

    The clip has no intrinsic field of view; wrap it in a
    :py:class:`~pulse2percept.vision.Scene` to say how much of the visual
    field it covers.

    .. note::
       The source is variable-frame-rate, so the reader resamples it to a
       constant rate: about 9 of the 45 frames repeat the frame before them.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of each video
        frame.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the frame: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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
    National Library of Medicine / National Institutes of Health. As a U.S.
    government work it is in the public domain in the United States, and is
    therefore not covered by pulse2percept's BSD license; ``metadata`` carries
    the requested attribution.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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


#: The micrograph is CC BY 4.0, not BSD like the rest of pulse2percept, so its
#: attribution travels with the stimulus. See ``data/samples/README.rst``.
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

    The micrograph is by Dr Kara Cerveny and Dr Steve Wilson, held by the
    Wellcome Collection, and licensed CC BY 4.0 rather than under
    pulse2percept's BSD license; ``metadata`` carries the attribution.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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


@deprecated(alt_func='pulse2percept.stimuli.samples.snellen_chart',
            deprecated_version='0.11.0', removed_version='0.12.0')
class SnellenChart(ImageStimulus):
    """Snellen chart

    Load the 1348x840 Snellen chart commonly used to measure visual acuity.

    .. versionadded:: 0.7

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    show_annotations : {True, False}, optional
        If True, show the full Snellen chart including annotations of the rows
        and corresponding acuity measures.

    row : None, optional
        Select a single row (between 1 and 11) from the Snellen chart.
        For example, row 1 corresponds to 20/200, row 2 to 20/100.

    electrodes : int, string or list thereof; optional, default: None
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) image.

    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """
    __slots__ = ()

    def __init__(self, resize=None, show_annotations=True, row=None,
                 electrodes=None, metadata=None):
        super().__init__(_snellen_source(show_annotations, row),
                         resize=resize, as_gray=True, electrodes=electrodes,
                         metadata=metadata, compress=False)


@deprecated(alt_func='pulse2percept.stimuli.samples.logo_bvl',
            deprecated_version='0.11.0', removed_version='0.12.0')
class LogoBVL(ImageStimulus):
    """Bionic Vision Lab (BVL) logo

    Load the 576x720x4 Bionic Vision Lab (BVL) logo.

    .. versionadded:: 0.7

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the image
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

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
