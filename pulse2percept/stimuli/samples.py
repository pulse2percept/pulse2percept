""":py:func:`~pulse2percept.stimuli.samples.big_buck_bunny`,
   :py:func:`~pulse2percept.stimuli.samples.bvl_cake`,
   :py:func:`~pulse2percept.stimuli.samples.cajal_retina`,
   :py:func:`~pulse2percept.stimuli.samples.landolt_c`,
   :py:func:`~pulse2percept.stimuli.samples.logo_bvl`,
   :py:func:`~pulse2percept.stimuli.samples.logo_ucsb`,
   :py:func:`~pulse2percept.stimuli.samples.tumbling_e`,
   :py:func:`~pulse2percept.stimuli.samples.ucsb_bike`,
   :py:func:`~pulse2percept.stimuli.samples.ucsb_flyover`,
   :py:func:`~pulse2percept.stimuli.samples.ucsb_pedestrians`,
   :py:func:`~pulse2percept.stimuli.samples.ucsb_surf`,
   :py:func:`~pulse2percept.stimuli.samples.zebrafish_retina`

Sample stimuli bundled with pulse2percept, for demos, docs, and tests.

The loaders return ordinary
:py:class:`~pulse2percept.stimuli.ImageStimulus` or
:py:class:`~pulse2percept.stimuli.VideoStimulus` objects (or, for procedural
optotypes, a :py:class:`~pulse2percept.vision.Scene` wrapping one); they are
not stimulus types of their own. Access them through the module rather than
the top-level namespace::

    from pulse2percept.stimuli import samples
    logo = samples.logo_bvl()

.. versionadded:: 0.11.0
"""
from os.path import dirname, join

import numpy as np

from .images import ImageStimulus
from .videos import VideoStimulus
from ..units import as_value, deg, dva

__all__ = [
    'big_buck_bunny',
    'bvl_cake',
    'cajal_retina',
    'landolt_c',
    'logo_bvl',
    'logo_ucsb',
    'tumbling_e',
    'ucsb_bike',
    'ucsb_flyover',
    'ucsb_pedestrians',
    'ucsb_surf',
    'zebrafish_retina',
]

#: Landolt-C proportions, in multiples of the gap width. The stroke width is
#: half the difference, and therefore one gap wide as well.
_INNER_DIAMETER, _OUTER_DIAMETER = 3.0, 5.0

#: Tumbling-E proportions: the glyph spans five stroke widths either way, with
#: one-stroke bars separated by one-stroke gaps.
_E_EXTENT = 5.0

#: Fewest pixels across an optotype's critical feature that still rasterize it
_MIN_FEATURE_PX = 2


def _sample_path(filename):
    """Return the absolute path of a bundled sample asset"""
    return join(dirname(__file__), 'data', 'samples', filename)


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


#: Cajal died in 1934, so the drawing is out of copyright worldwide. See
#: ``data/samples/README.rst``.
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

    The drawing is in the public domain and is therefore not covered by
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


def _check_shape(shape):
    """Return ``shape`` as a positive integer ``(rows, cols)``"""
    shape = np.asarray(shape)
    if shape.shape != (2,) or not np.issubdtype(shape.dtype, np.integer):
        raise ValueError(f"'shape' must be a (rows, cols) pair of integers, "
                         f"not {shape.tolist()}.")
    if np.any(shape < 1):
        raise ValueError(f"'shape' must be positive, not {shape.tolist()}.")
    return int(shape[0]), int(shape[1])


def _optotype_grid(shape, fov):
    """Return ``(x, y, (width, height))`` for a procedural optotype

    ``x`` and ``y`` hold the pixel centers in visual-field coordinates,
    following the :py:class:`~pulse2percept.vision.Scene` convention: ``fov``
    is the outer extent of the frame, and row 0 holds the largest ``y``.
    """
    # Local import: `vision` imports `stimuli`, so this cannot be top-level.
    from ..vision.scene import _resolve_fov
    n_rows, n_cols = _check_shape(shape)
    width, height = _resolve_fov(fov, n_rows, n_cols)
    cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    x = (cols + 0.5) * (width / n_cols) - width / 2
    y = height / 2 - (rows + 0.5) * (height / n_rows)
    return x, y, (width, height)


def _check_raster(size, name, feature, fov, shape):
    """Raise unless ``size`` spans ``_MIN_FEATURE_PX`` pixels of the raster

    ``size`` is the angular size of parameter ``name``, which rasterizes as
    ``feature`` (the C's opening, the E's bars). Measured on the coarser of
    the two angular pixel sizes, so neither axis may under-resolve it.
    """
    px = max(fov[0] / shape[1], fov[1] / shape[0])
    if size / px < _MIN_FEATURE_PX:
        raise ValueError(
            f"A {name} of {size:g} dva is {size / px:.2g} pixels across at "
            f"fov={fov} dva and shape={shape}, which does not resolve the "
            f"{feature}. At least {_MIN_FEATURE_PX} pixels are required: "
            f"increase 'shape' or reduce 'fov'.")


def _landolt_mask(x, y, gap, position, orientation):
    """Boolean mask of the C: an annulus with a gap-wide slot cut out of it"""
    theta = np.deg2rad(orientation)
    # Coordinates relative to the optotype's center, then rotated so that the
    # opening always points along +u:
    dx, dy = x - position[0], y - position[1]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    radius = np.hypot(u, v)
    annulus = ((radius >= _INNER_DIAMETER / 2 * gap) &
               (radius <= _OUTER_DIAMETER / 2 * gap))
    # The opening is a slot of width `gap` measured across the gap direction,
    # which is what "gap size" means for a Landolt C:
    slot = (u > 0) & (np.abs(v) <= gap / 2)
    return annulus & ~slot


def landolt_c(gap=1, position=(0, 0), orientation=0, fov=10, polarity='dark',
              shape=(512, 512)):
    """Landolt C optotype

    Rasterize a Landolt C at a given angular size, eccentricity, and gap
    orientation, and place it in a :py:class:`~pulse2percept.vision.Scene`.

    The C follows the standard proportions, all expressed in multiples of the
    gap width ``gap``: stroke width ``gap``, inner diameter ``3 * gap``, outer
    diameter ``5 * gap``. ``gap`` is therefore the critical feature size,
    which is what an acuity task varies; ``position`` moves the optotype
    through the visual field without changing that size.

    The image is binary (gray levels 0 and 1), not antialiased.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    gap : float or Quantity, optional
        Angular width of the critical opening, in degrees of visual angle
        (e.g. ``0.5 * dva``).
    position : (x, y), optional
        Center of the optotype in visual-field coordinates, in dva. ``y``
        grows upwards.
    orientation : float or Quantity, optional
        Direction the opening points, in degrees counterclockwise from the
        positive x axis (e.g. ``90 * deg``): 0 right, 90 up, 180 left, 270
        down. Any finite angle is accepted.
    fov : float or (width, height), optional
        How much of the visual field the scene covers, in dva. A scalar is the
        horizontal extent, and the vertical one follows from ``shape``.
    polarity : {'dark', 'light'}, optional
        ``'dark'`` draws a black C on white, ``'light'`` a white C on black.
    shape : (rows, cols), optional
        Size of the rasterized frame, in pixels.

    Returns
    -------
    scene : :py:class:`~pulse2percept.vision.Scene`

    Examples
    --------
    A 0.5-degree gap pointing up, five degrees to the right of fixation:

    >>> from pulse2percept.stimuli import samples
    >>> from pulse2percept.units import deg, dva
    >>> scene = samples.landolt_c(gap=0.5 * dva, position=(5, 0) * dva,
    ...                           orientation=90 * deg, fov=15 * dva)
    >>> scene.fov
    (15.0, 15.0)

    """
    # Local import: `vision` imports `stimuli`, so this cannot be top-level.
    from ..vision.scene import Scene
    gap = float(as_value(gap, dva, 'gap'))
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError(f"'gap' is an angular width and must be finite and "
                         f"positive, not {gap}.")
    center = np.asarray(as_value(position, dva, 'position'), dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError(f"'position' must be a finite (x, y) pair in dva, "
                         f"not {position!r}.")
    orientation = float(as_value(orientation, deg, 'orientation'))
    if not np.isfinite(orientation):
        raise ValueError(f"'orientation' must be a finite angle in degrees, "
                         f"not {orientation}.")
    if polarity not in ('dark', 'light'):
        raise ValueError(f"'polarity' is either 'dark' (black C on white) or "
                         f"'light' (white C on black), not {polarity!r}.")
    x, y, (width, height) = _optotype_grid(shape, fov)

    # Cropping a C changes the task rather than the picture, so refuse it:
    radius = _OUTER_DIAMETER / 2 * gap
    for name, offset, extent in (('horizontally', center[0], width),
                                 ('vertically', center[1], height)):
        if abs(offset) + radius > extent / 2:
            raise ValueError(
                f"A Landolt C with gap={gap:g} dva at position="
                f"{center.tolist()} dva reaches {abs(offset) + radius:g} dva "
                f"{name} from fixation, past the {extent / 2:g} dva half-FOV. "
                f"Increase 'fov', or move the optotype closer to fixation.")
    # An opening narrower than a couple of pixels rasterizes as a closed ring,
    # i.e. as a different optotype:
    _check_raster(gap, 'gap', 'opening', (width, height), x.shape)
    mask = _landolt_mask(x, y, gap, center, orientation)
    ink, paper = (0.0, 1.0) if polarity == 'dark' else (1.0, 0.0)
    img = np.where(mask, ink, paper).astype(np.float32)
    metadata = {'sample': 'landolt_c', 'gap': gap,
                'position': (float(center[0]), float(center[1])),
                'orientation': orientation, 'polarity': polarity,
                'fov': (width, height)}
    return Scene(ImageStimulus(img, metadata=metadata), fov=(width, height))


def _tumbling_e_mask(x, y, stroke, position, orientation):
    """Boolean mask of the E: a spine column plus three full-width bars"""
    theta = np.deg2rad(orientation)
    # Coordinates relative to the optotype's center, then rotated so that the
    # bars always point along +u:
    dx, dy = x - position[0], y - position[1]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    half = _E_EXTENT / 2 * stroke
    within = (np.abs(u) <= half) & (np.abs(v) <= half)
    # The spine is the leftmost one-stroke column, spanning the full height:
    spine = u <= -(half - stroke)
    # Top, middle, and bottom bars, each one stroke thick and full width:
    bars = (np.abs(v) <= stroke / 2) | (np.abs(v) >= half - stroke)
    return within & (spine | bars)


def tumbling_e(stroke=1, position=(0, 0), orientation=0, fov=10,
               polarity='dark', shape=(512, 512)):
    """Tumbling E optotype

    Rasterize a Tumbling E at a given angular size, eccentricity, and
    orientation, and place it in a :py:class:`~pulse2percept.vision.Scene`.

    The E follows the standard 5x5 construction, all expressed in multiples of
    the stroke width ``stroke``: overall width and height ``5 * stroke``, bars
    and the gaps between them one ``stroke`` each. ``stroke`` is therefore the
    critical feature size, which is what an acuity task varies; ``position``
    moves the optotype through the visual field without changing that size.

    The four cardinal orientations are the conventional Tumbling-E task,
    although any finite angle is accepted here.

    .. note::
       The Tumbling E and the Landolt C
       (:py:func:`~pulse2percept.stimuli.samples.landolt_c`) are different
       optotypes measured with different tasks (bar direction vs. gap
       direction). Thresholds obtained with one are not numerically
       interchangeable with the other.

    The image is binary (gray levels 0 and 1), not antialiased.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    stroke : float or Quantity, optional
        Angular width of a bar, in degrees of visual angle (e.g.
        ``0.5 * dva``). The whole E is ``5 * stroke`` across.
    position : (x, y), optional
        Center of the optotype in visual-field coordinates, in dva. ``y``
        grows upwards.
    orientation : float or Quantity, optional
        Direction the bars point, in degrees counterclockwise from the
        positive x axis (e.g. ``90 * deg``): 0 right, 90 up, 180 left, 270
        down. Any finite angle is accepted.
    fov : float or (width, height), optional
        How much of the visual field the scene covers, in dva. A scalar is the
        horizontal extent, and the vertical one follows from ``shape``.
    polarity : {'dark', 'light'}, optional
        ``'dark'`` draws a black E on white, ``'light'`` a white E on black.
    shape : (rows, cols), optional
        Size of the rasterized frame, in pixels.

    Returns
    -------
    scene : :py:class:`~pulse2percept.vision.Scene`

    Examples
    --------
    A 0.5-degree stroke pointing up, five degrees to the right of fixation:

    >>> from pulse2percept.stimuli import samples
    >>> from pulse2percept.units import deg, dva
    >>> scene = samples.tumbling_e(stroke=0.5 * dva, position=(5, 0) * dva,
    ...                            orientation=90 * deg, fov=15 * dva)
    >>> scene.fov
    (15.0, 15.0)

    """
    # Local import: `vision` imports `stimuli`, so this cannot be top-level.
    from ..vision.scene import Scene
    stroke = float(as_value(stroke, dva, 'stroke'))
    if not np.isfinite(stroke) or stroke <= 0:
        raise ValueError(f"'stroke' is an angular width and must be finite "
                         f"and positive, not {stroke}.")
    center = np.asarray(as_value(position, dva, 'position'), dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError(f"'position' must be a finite (x, y) pair in dva, "
                         f"not {position!r}.")
    orientation = float(as_value(orientation, deg, 'orientation'))
    if not np.isfinite(orientation):
        raise ValueError(f"'orientation' must be a finite angle in degrees, "
                         f"not {orientation}.")
    if polarity not in ('dark', 'light'):
        raise ValueError(f"'polarity' is either 'dark' (black E on white) or "
                         f"'light' (white E on black), not {polarity!r}.")
    x, y, (width, height) = _optotype_grid(shape, fov)

    # Cropping an E changes the task rather than the picture, so refuse it.
    # The glyph is a square, so off-cardinal angles need the axis-aligned
    # extent of the rotated square, not its half-width:
    theta = np.deg2rad(orientation)
    half = _E_EXTENT / 2 * stroke
    extent = half * (abs(np.cos(theta)) + abs(np.sin(theta)))
    for name, offset, fov_size in (('horizontally', center[0], width),
                                   ('vertically', center[1], height)):
        if abs(offset) + extent > fov_size / 2:
            raise ValueError(
                f"A Tumbling E with stroke={stroke:g} dva at position="
                f"{center.tolist()} dva and orientation={orientation:g} deg "
                f"reaches {abs(offset) + extent:g} dva {name} from fixation, "
                f"past the {fov_size / 2:g} dva half-FOV. Increase 'fov', or "
                f"move the optotype closer to fixation.")
    # Bars narrower than a couple of pixels merge with their gaps, i.e. turn
    # the E into a filled square:
    _check_raster(stroke, 'stroke', 'bars', (width, height), x.shape)
    mask = _tumbling_e_mask(x, y, stroke, center, orientation)
    ink, paper = (0.0, 1.0) if polarity == 'dark' else (1.0, 0.0)
    img = np.where(mask, ink, paper).astype(np.float32)
    metadata = {'sample': 'tumbling_e', 'stroke': stroke,
                'position': (float(center[0]), float(center[1])),
                'orientation': orientation, 'polarity': polarity,
                'fov': (width, height)}
    return Scene(ImageStimulus(img, metadata=metadata), fov=(width, height))
