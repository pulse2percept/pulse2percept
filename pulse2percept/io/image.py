"""`image2stim`"""
import numpy as np
import logging
# Rather than trying to import these all over, try once and then remember
# by setting a flag.
try:
    import skimage.io as sio
    import skimage.transform as sit
    import skimage.color as sic
    has_skimage = True
except (ImportError, AttributeError):
    # Might also raise "dict object has no attribute 'io'"
    has_skimage = False

from ..implants import ArgusI, ArgusII
from ..stimuli import PulseTrain
from ..utils import deprecated


@deprecated(deprecated_version=0.6)
def image2stim(img, implant, coding='amplitude', valrange=[0, 50],
               max_contrast=False, const_val=20, invert=False,
               tsample=0.005 / 1000, dur=0.5, pulsedur=0.5 / 1000.,
               interphasedur=0.5 / 1000., pulsetype='cathodicfirst'):
    """Converts an image into a series of pulse trains

    This function creates an input stimulus from an RGB or grayscale image.
    The image is down-sampled to fit the spatial layout of the implant
    (currently supported are ArgusI and ArgusII arrays).
    Requires Scikit-Image.

    Parameters
    ----------
    img : str|array_like
        An input image, either a valid filename (string) or a numpy array
        (row x col x channels).
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        An ElectrodeArray object that describes the implant.
    coding : {'amplitude', 'frequency'}, optional
        A string describing the coding scheme:

        * 'amplitude': Image intensity is linearly converted to a current
                       amplitude between ``valrange[0]`` and ``valrange[1]``.
                       Frequency is held constant at ``const_freq``.
        * 'frequency': Image intensity is linearly converted to a pulse
                       frequency between ``valrange[0]`` and ``valrange[1]``.
                       Amplitude is held constant at ``const_amp``.

        Default: 'amplitude'
    valrange : list, optional
        Range of stimulation values to be used (If ``coding`` is 'amplitude',
        specifies min and max current; if ``coding`` is 'frequency', specifies
        min and max frequency).
        Default: [0, 50]
    max_contrast : bool, optional
        Flag wether to maximize image contrast (True) or not (False).
        Default: False
    const_val : float, optional
        For frequency coding: The constant amplitude value to be used for all
        pulse trains. For amplitude coding: The constant frequency value to
        be used for all pulse trains.
        Default: 20
    invert : bool, optional
        Flag whether to invert the grayscale values of the image (True) or
        not (False).
        Default: False
    tsample : float, optional
        Sampling time step (seconds). Default: 0.005 / 1000 seconds.
    dur : float, optional
        Stimulus duration (seconds). Default: 0.5 seconds.
    pulsedur : float, optional
        Duration of single (positive or negative) pulse phase in seconds.
    interphasedur : float, optional
        Duration of inter-phase interval (between positive and negative
        pulse) in seconds.
    pulsetype : {'cathodicfirst', 'anodicfirst'}, optional
        A cathodic-first pulse has the negative phase first, whereas an
        anodic-first pulse has the positive phase first.

    Returns
    -------
    pulses : list
        A list of p2p.stimuli.PulseTrain objects, one for each electrode in
        the implant.

    """
    if not has_skimage:
        # We don't want to repeatedly import Scikit-Image. This would (e.g.)
        # unnecessarily slow down `video2pulsetrain`.
        raise ImportError("You do not have scikit-image installed. "
                          "You can install it via $ pip install scikit-image.")

    # Make sure range of values is valid
    assert len(valrange) == 2 and valrange[1] > valrange[0]

    isargus1 = isinstance(implant, ArgusI)
    isargus2 = isinstance(implant, ArgusII)
    if not isargus1 and not isargus2:
        raise TypeError("For now, implant must be of type ArgusI or ArgusII.")

    if isinstance(img, str):
        # Load image from filename
        img_orig = sio.imread(img, as_grey=True).astype(np.float32)
        logging.getLogger(__name__).info("Loaded file '%s'." % img)
    else:
        if img.ndim == 2:
            # Grayscale
            img_orig = img.astype(np.float32)
        else:
            # Assume RGB, convert to grayscale
            assert img.shape[-1] == 3
            img_orig = sic.rgb2gray(np.array(img)).astype(np.float32)

    # Make sure all pixels are between 0 and 1
    if img_orig.max() > 1.0:
        img_orig /= 255.0

    # Let Scikit-Image do the resampling: Downscale image to fit array
    # Use mode 'reflect' for np.pad: Pads with the reflection of the vector
    # mirrored on the first and last values of the vector along each axis.
    if isargus1:
        img_stim = sit.resize(img_orig, (4, 4), mode='reflect')
    elif isargus2:
        img_stim = sit.resize(img_orig, (6, 10), mode='reflect')

    # If specified, invert the mapping of grayscale values:
    if invert:
        img_stim = 1.0 - img_stim

    # If specified, maximize the contrast in the image:
    if max_contrast:
        img_stim -= img_stim.min()
        if img_stim.max() > 0:
            img_stim /= img_stim.max()

    # With all pixels between 0 and 1, now scale to valrange
    assert np.all(img_stim >= 0.0) and np.all(img_stim <= 1.0)
    img_stim = img_stim * np.diff(valrange) + valrange[0]
    assert np.all(img_stim >= valrange[0]) and np.all(img_stim <= valrange[1])

    stim = []
    for _, px in np.ndenumerate(img_stim):
        if coding == 'amplitude':
            amp = px
            freq = const_val
        elif coding == 'frequency':
            amp = const_val
            freq = px
        else:
            e_s = "Acceptable values for `coding` are 'amplitude' or"
            e_s += "'frequency'."
            raise ValueError(e_s)

        pt = PulseTrain(tsample, freq=freq, amp=amp, dur=dur,
                        pulse_dur=pulsedur,
                        interphase_dur=interphasedur,
                        pulsetype=pulsetype)
        stim.append(pt)

    return stim
