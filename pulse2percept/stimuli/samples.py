""":py:func:`~pulse2percept.stimuli.samples.logo_bvl`,
   :py:func:`~pulse2percept.stimuli.samples.logo_ucsb`,
   :py:func:`~pulse2percept.stimuli.samples.boston_train`,
   :py:func:`~pulse2percept.stimuli.samples.girl_pool`

Sample stimuli bundled with pulse2percept, for demos, docs, and tests.

The loaders return ordinary
:py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus` objects; they are not
stimulus types of their own. Access them through the module rather than the
top-level namespace::

    from pulse2percept.stimuli import samples
    logo = samples.logo_bvl()

.. versionadded:: 0.11.0
"""
from os.path import dirname, join

from .images import ImageStimulus
from .videos import VideoStimulus

__all__ = [
    'boston_train',
    'girl_pool',
    'logo_bvl',
    'logo_ucsb',
]


def _sample_path(filename):
    """Return the absolute path of a bundled sample asset"""
    return join(dirname(__file__), 'data', 'samples', filename)


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


def boston_train(resize=None, electrodes=None, as_gray=False, metadata=None):
    """Boston train sequence

    Load the Boston subway sequence, consisting of 94 frames of 240x426x3
    pixels each.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the video
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) video frame.

    as_gray : bool, optional
        Flag whether to convert the video to grayscale.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.VideoStimulus`

    """
    return VideoStimulus(_sample_path('boston-train.mp4'), format="MP4",
                         resize=resize, as_gray=as_gray,
                         electrodes=electrodes, metadata=metadata,
                         compress=False)


def girl_pool(resize=None, electrodes=None, as_gray=False, metadata=None):
    """A girl jumping into a swimming pool

    Load the "girl jumping in a pool" sequence, consisting of 91 frames of
    240x426x3 pixels each.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    resize : (height, width) or None, optional
        A tuple specifying the desired height and the width of the video
        stimulus.

    electrodes : int, string or list thereof; optional
        Optionally, you can provide your own electrode names. If none are
        given, each pixel is named after its place in the image: a letter for
        the row, a number for the column, and a suffix for the color channel
        (e.g. 'A1', 'C12', 'A1_R'). See
        :py:class:`~pulse2percept.stimuli.ElectrodeNames`.

        .. note::
           The number of electrode names provided must match the number of
           pixels in the (resized) video frame.

    as_gray : bool, optional
        Flag whether to convert the video to grayscale.

    metadata : dict, optional
        Additional stimulus metadata can be stored in a dictionary.

    Returns
    -------
    stim : :py:class:`~pulse2percept.stimuli.VideoStimulus`

    """
    return VideoStimulus(_sample_path('girl-pool.mp4'), format="MP4",
                         resize=resize, as_gray=as_gray,
                         electrodes=electrodes, metadata=metadata,
                         compress=False)
