import numpy as np
import logging

from .image import image2stim
from ..stimuli import TimeSeries

# Rather than trying to import these all over, try once and then remember
# by setting a flag.
try:
    import skimage
    import skimage.transform as sit
    import skimage.color as sic
    has_skimage = True
except (ImportError, AttributeError):
    # Might also raise AttributeError: dict object has no attribute 'transform'
    has_skimage = False
try:
    import skvideo
    import skvideo.io as svio
    skvideo._HAS_AVCONV
    skvideo._HAS_FFMPEG
    has_skvideo = True
except (ImportError, AttributeError):
    has_skvideo = False


def _set_skvideo_path(ffmpeg_path=None, libav_path=None):
    """Sets the path to the FFMPEG and/or LibAV libraries.

    If scikit-video complains that either ffmpeg or libav cannot be found,
    you can set the path to the executables directly. On Unix, these binaries
    usually live in /usr/bin. On Windows, point to the directory that contains
    your ffmpeg.exe.

    Parameters
    ----------
    ffmpeg_path : str, optional, default: system's default path
        Path to ffmpeg library.
    libav_path : str, optional, default: system's default path
        Path to libav library.

    """
    if libav_path is not None:
        skvideo.setLibAVPath(libav_path)
    if ffmpeg_path is not None:
        skvideo.setFFmpegPath(ffmpeg_path)


def load_video_metadata(filename, ffmpeg_path=None, libav_path=None):
    """Returns a video files metadata

    This function loads the  metadata of a video file, which is returned
    as a dict. Among the available information are the width and height
    of the video, the aspect ratio, the frame rate, the bit rate, and the
    duration.

    Parameters
    ----------
    filename : str
        Video file name
    ffmpeg_path : str, optional, default: system's default path
        Path to ffmpeg library.
    libav_path : str, optional, default: system's default path
        Path to libav library.

    Returns
    -------
    metadata : dict
        A dictionary containing all metadata.
    """
    if not has_skvideo:
        raise ImportError("You do not have scikit-video installed. "
                          "You can install it via $ pip install sk-video.")

    # Set the path if necessary
    _set_skvideo_path(ffmpeg_path, libav_path)

    metadata = skvideo.io.ffprobe(filename)
    if not metadata:
        raise OSError('File %s could not be found.' % filename)
    return metadata['video']


def load_video_framerate(filename, ffmpeg_path=None, libav_path=None):
    """Returns the video frame rate

    This function returns the frame rate of the video, as given by its
    metadata field '@r_frame_rate'.

    Parameters
    ----------
    filename : str
        Video file name
    ffmpeg_path : str, optional, default: system's default path
        Path to ffmpeg library.
    libav_path : str, optional, default: system's default path
        Path to libav library.

    Returns
    -------
    fps : float
        Video frame rate (frames per second).
    """
    if not has_skvideo:
        raise ImportError("You do not have scikit-video installed. "
                          "You can install it via $ pip install sk-video.")

    # Set the path if necessary
    _set_skvideo_path(ffmpeg_path, libav_path)

    # Load all metadata
    metadata = load_video_metadata(filename)
    if '@r_frame_rate' not in metadata:
        raise AttributeError('Meta data does not contain field @r_frame_rate.')

    # Parse frame rate entry: It's a rational expression encoded as a string.
    # Hence, split by '/' and to the divison by hand.
    str_fps = metadata['@r_frame_rate'].split('/')
    return float(str_fps[0]) / float(str_fps[1])


def load_video(filename, as_timeseries=True, as_gray=False, ffmpeg_path=None,
               libav_path=None):
    """Loads a video from file.

    This function loads a video from file with the help of Scikit-Video, and
    returns the data either as a NumPy array (if `as_timeseries` is False)
    or as a ``TimeSeries`` object (if `as_timeseries` is True).

    Parameters
    ----------
    filename : str
        Video file name
    as_timeseries: bool, optional, default: True
        If True, returns the data as a ``TimeSeries`` object.
    as_gray : bool, optional, default: False
        If True, loads only the luminance channel of the video.
    ffmpeg_path : str, optional, default: system's default path
        Path to ffmpeg library.
    libav_path : str, optional, default: system's default path
        Path to libav library.

    Returns
    -------
    video : ndarray | TimeSeries
        If `as_timeseries` is False, returns video data according to the
        Scikit-Video standard; that is, an ndarray of dimension (T, M, N, C),
        (T, M, N), (M, N, C), or (M, N), where T is the number of frames,
        M is the height, N is the width, and C is the number of channels (will
        be either 1 for grayscale or 3 for RGB).

        If `as_timeseries` is True, returns video data as a TimeSeries object
        of dimension (M, N, C), (M, N, T), (M, N, C), or (M, N).
        The sampling rate corresponds to 1 / frame rate.

    Examples
    --------
    Load a video as a ``TimeSeries`` object:

    >>> from skvideo import datasets
    >>> video = load_video(datasets.bikes())
    >>> video.tsample
    0.04
    >>> video.shape
    (272, 640, 3, 250)

    Load a video as a NumPy ndarray:

    >>> from skvideo import datasets
    >>> video = load_video(datasets.bikes(), as_timeseries=False)
    >>> video.shape
    (250, 272, 640, 3)

    Load a video as a NumPy ndarray and convert to grayscale:

    >>> from skvideo import datasets
    >>> video = load_video(datasets.bikes(), as_timeseries=False, as_gray=True)
    >>> video.shape
    (250, 272, 640, 1)

    """
    if not has_skvideo:
        raise ImportError("You do not have scikit-video installed. "
                          "You can install it via $ pip install sk-video.")

    # Set the path if necessary
    _set_skvideo_path(ffmpeg_path, libav_path)

    if skvideo._HAS_FFMPEG:
        backend = 'ffmpeg'
    else:
        backend = 'libav'
    video = svio.vread(filename, as_grey=as_gray, backend=backend)
    logging.getLogger(__name__).info("Loaded video from file '%s'." % filename)
    d_s = "Loaded video has shape (T, M, N, C) = " + str(video.shape) 

    if as_timeseries:
        # TimeSeries has the time as the last dimensions: re-order dimensions,
        # then squeeze out singleton dimensions
        axes = np.roll(range(video.ndim), -1)
        video = np.squeeze(np.transpose(video, axes=axes))
        fps = load_video_framerate(filename)
        d_s = "Reshaped video to shape (M, N, C, T) = " + str(video.shape) 
        return TimeSeries(1.0 / fps, video)
    else:
        # Return as ndarray
        return video


def load_video_generator(filename, ffmpeg_path=None, libav_path=None):
    """Returns a generator that can load a video from file frame-by-frame.

    This function returns a generator `reader` that can load a video from a
    file frame-by-frame. Every call to `reader.nextFrame()` will return a
    single frame of the video as a NumPy array with dimensions (M, N) or
    (M, N, C), where M is the height, N is the width, and C is the number of
    channels (will be either 1 for grayscale or 3 for RGB).

    Parameters
    ----------
    filename : str
        Video file name
    ffmpeg_path : str, optional, default: system's default path
        Path to ffmpeg library.
    libav_path : str, optional, default: system's default path
        Path to libav library.

    Returns
    -------
    reader : skvideo.io.FFmpegReader | skvideo.io.LibAVReader
        A Scikit-Video reader object

    Examples
    --------
    >>> from skvideo import datasets
    >>> reader = load_video_generator(datasets.bikes())
    >>> for frame in reader.nextFrame():
    ...    pass

    """
    if not has_skvideo:
        raise ImportError("You do not have scikit-video installed. "
                          "You can install it via $ pip install sk-video.")

    # Set the path if necessary
    _set_skvideo_path(ffmpeg_path, libav_path)

    # Try loading
    if skvideo._HAS_FFMPEG:
        reader = svio.FFmpegReader(filename)
    elif skvideo._HAS_AVCONV:
        reader = svio.LibAVReader(filename)
    else:
        raise ImportError("You have neither ffmpeg nor libav (which comes "
                          "with avprobe) installed.")

    logging.getLogger(__name__).info("Loaded video from file '%s'." % filename)
    return reader


def save_video(data, filename, width=None, height=None, fps=30,
               ffmpeg_path=None, libav_path=None):
    """Saves a video to file.

    This function stores a NumPy ndarray to file using Scikit-Video.

    Parameters
    ----------
    data : ndarray | TimeSeries
        Video data as a NumPy ndarray must have dimension (T, M, N, C),
        (T, M, N), (M, N, C), or (M, N), where T is the number of frames,
        M is the height, N is the width, and C is the number of channels (must
        be either 1 for grayscale or 3 for RGB).

        Video data as a TimeSeries object must have dimension (M, N, C, T) or
        (M, N, T). The sampling step will be used as the video's frame rate.
    filename : str
        Video file name.
    width : int, optional
        Desired width of the movie.
        Default: Automatically determined based on `height` (without changing
        the aspect ratio). If `height` is not given, the percept's original
        width is used.
    height : int, optional
        Desired height of the movie.
        Default: Automatically determined based on `width` (without changing
        the aspect ratio). If `width` is not given, the percept's original
        height is used.
    fps : int, optional, default: 30
        Desired frame rate of the video (frames per second).
    ffmpeg_path : str, optional, default: system's default path
        Path to ffmpeg library.
    libav_path : str, optional, default: system's default path
        Path to libav library.

    Notes
    -----
    To distinguish between 3-D inputs of shape (T, M, N) and (M, N, C), the
    last dimension is checked: If it is small (<= 4), it is most likely a
    color channel - hence the input is interpreted as (M, N, C).
    Else it is interpreted as (T, M, N).

    """
    if not has_skvideo:
        raise ImportError("You do not have scikit-video installed. "
                          "You can install it via $ pip install sk-video.")
    if not has_skimage:
        raise ImportError("You do not have scikit-image installed. "
                          "You can install it via $ pip install scikit-image.")

    is_ndarray = is_timeseries = False
    if isinstance(data, np.ndarray):
        is_ndarray = True
    elif isinstance(data, TimeSeries):
        is_timeseries = True
    else:
        raise TypeError('Data to be saved must be either a NumPy ndarray '
                        'or a ``TimeSeries`` object')

    # Set the path if necessary, then choose backend
    _set_skvideo_path(ffmpeg_path, libav_path)
    if skvideo._HAS_FFMPEG:
        backend = 'ffmpeg'
    else:
        backend = 'libav'

    if is_ndarray:
        # Use Scikit-Video utility to interpret dimensions and convert to
        # (T, M, N, C)
        data = skvideo.utils.vshape(data)
        oldheight = data.shape[1]
        oldwidth = data.shape[2]
        length = data.shape[0]
    elif is_timeseries:
        # Resample the percept to the given frame rate
        new_tsample = 1.0 / float(fps)
        d_s = 'old: tsample=%f' % data.tsample
        d_s += ', shape=' + str(data.shape) 
        data = data.resample(new_tsample)
        d_s = 'new: tsample=%f' % new_tsample
        d_s += ', shape=' + str(data.shape)
        oldheight = data.shape[0]
        oldwidth = data.shape[1]
        length = data.shape[-1]

    # Calculate the desired height and width
    if not height and not width:
        height = oldheight
        width = oldwidth
    elif height and not width:
        width = int(height * 1.0 / oldheight * oldwidth)
    elif width and not height:
        height = int(width * 1.0 / oldwidth * oldheight)
    d_s = "Video scaled to (M, N, T) = (%d, %d, %d)." % (height, width, length) 

    # Reshape and scale the data
    savedata = np.zeros((length, height, width, 3), dtype=np.float32)
    for i in range(length):
        if is_ndarray:
            frame = skimage.img_as_float(data[i, ...])
        elif is_timeseries:
            frame = data.data[..., i] / float(data.data.max())
            frame = skimage.img_as_float(frame)

        # resize wants the data to be between 0 and 1
        frame = sic.gray2rgb(sit.resize(frame, (height, width),
                                        mode='reflect'))
        savedata[i, ...] = frame * 255.0

    # Set the input frame rate and frame size
    inputdict = {}
    inputdict['-r'] = str(int(fps))
    inputdict['-s'] = '%dx%d' % (width, height)

    # Set the output frame rate
    outputdict = {}
    outputdict['-r'] = str(int(fps))

    # Save data to file
    svio.vwrite(filename, savedata, inputdict=inputdict, outputdict=outputdict,
                backend=backend)
    logging.getLogger(__name__).info("Saved video to file '%s'." % filename)


def save_video_sidebyside(videofile, percept, savefile, fps=30,
                          ffmpeg_path=None, libav_path=None):
    """Saves both an input video and the percept to file, side-by-side.

    This function creates a new video from an input video file and a
    ``TimeSeries`` object, assuming they correspond to model
    input and model output, and plots them side-by-side.
    Both input video and percept are resampled according to `fps`.
    The percept is resized to match the height of the input video.

    Parameters
    ----------
    videofile : str
        File name of input video.
    percept : TimeSeries
        A TimeSeries object with dimension (M, N, C, T) or (M, N, T), where
        T is the number of frames, M is the height, N is the width, and C is
        the number of channels.
    savefile : str
        File name of output video.
    fps : int, optional, default: 30
        Desired frame rate of output video.
    ffmpeg_path : str, optional, default: system's default path
        Path to ffmpeg library.
    libav_path : str, optional, default: system's default path
        Path to libav library.

    """
    if not has_skvideo:
        raise ImportError("You do not have scikit-video installed. "
                          "You can install it via $ pip install sk-video.")
    if not has_skimage:
        raise ImportError("You do not have scikit-image installed. "
                          "You can install it via $ pip install scikit-image.")

    if not isinstance(percept, TimeSeries):
        raise TypeError("`percept` must be of type TimeSeries.")

    # Set the path if necessary
    _set_skvideo_path(ffmpeg_path, libav_path)

    # Load video from file
    video = load_video(videofile, as_timeseries=True, ffmpeg_path=ffmpeg_path,
                       libav_path=libav_path)

    # Re-sample time series to new frame rate
    new_tsample = 1.0 / float(fps)
    video = video.resample(new_tsample)
    percept = percept.resample(new_tsample)

    # After re-sampling, both video and percept should have about the same
    # length (up to some rounding error)
    combined_len = np.minimum(video.shape[-1], percept.shape[-1])
    if np.abs(video.shape[-1] - percept.shape[-1]) > 5:
        raise ValueError('Could not resample percept, len(video)='
                         '%d != len(percept)=%d.' % (video.shape[-1],
                                                     percept.shape[-1]))

    # Up-scale percept to fit video dimensions
    pheight = video.shape[0]
    pwidth = int(percept.shape[1] / float(percept.shape[0]) * video.shape[0])

    # Show the two side-by-side
    combined = np.zeros((combined_len, pheight, video.shape[1] + pwidth, 3))
    combined = combined.astype(np.float32)
    for i in range(combined_len):
        vframe = skimage.img_as_float(video.data[..., i])
        pframe = percept.data[..., i] / percept.data.max()
        pframe = skimage.img_as_float(pframe)
        pframe = sic.gray2rgb(sit.resize(pframe, (pheight, pwidth),
                                         mode='reflect'))
        combined[i, ...] = np.concatenate((vframe, pframe), axis=1)

    save_video(combined, savefile, fps=fps, ffmpeg_path=ffmpeg_path,
               libav_path=libav_path)


def video2stim(filename, implant, framerate=20, coding='amplitude',
               valrange=[0, 50], max_contrast=False, const_val=20,
               invert=False, tsample=0.005 / 1000, pulsedur=0.5 / 1000,
               interphasedur=0.5 / 1000, pulsetype='cathodicfirst',
               ffmpeg_path=None, libav_path=None):
    """Converts a video into a series of pulse trains

    This function creates an input stimulus from a video.
    Every frame of the video is passed to `image2pulsetrain`, where it is
    down-sampled to fit the spatial layout of the implant (currently supported
    are ArgusI and ArgusII arrays).
    In this mapping, rows of the image correspond to rows in the implant
    (top row, Argus I: A1 B1 C1 D1, Argus II: A1 A2 ... A10).

    Requires Scikit-Image and Scikit-Video.

    Parameters
    ----------
    img : str|array_like
        An input image, either a valid filename (string) or a numpy array
        (row x col x channels).
    implant : ProsthesisSystem
        An ElectrodeArray object that describes the implant.
    coding : {'amplitude', 'frequency'}, optional
        A string describing the coding scheme:
        - 'amplitude': Image intensity is linearly converted to a current
                       amplitude between `valrange[0]` and `valrange[1]`.
                       Frequency is held constant at `const_freq`.
        - 'frequency': Image intensity is linearly converted to a pulse
                       frequency between `valrange[0]` and `valrange[1]`.
                       Amplitude is held constant at `const_amp`.
        Default: 'amplitude'
    valrange : list, optional
        Range of stimulation values to be used (If `coding` is 'amplitude',
        specifies min and max current; if `coding` is 'frequency', specifies
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
        A list of PulseTrain objects, one for each electrode in
        the implant.

    """

    # Load generator to read video frame-by-frame
    reader = load_video_generator(filename, ffmpeg_path, libav_path)

    # Temporarily increase logger level to suppress info messages
    current_level = logging.getLogger(__name__).getEffectiveLevel()
    logging.getLogger(__name__).setLevel(logging.WARN)

    # Convert the desired framerate to a duration (seconds)
    dur = 1.0 / framerate

    # Read one frame at a time, and append to previous frames
    video = []
    for img in reader.nextFrame():
        frame = image2stim(img, implant, coding=coding, valrange=valrange,
                           max_contrast=max_contrast, const_val=const_val,
                           invert=invert, tsample=tsample, dur=dur,
                           pulsedur=pulsedur, interphasedur=interphasedur,
                           pulsetype=pulsetype)
        if video:
            # List of pulse trains: Append new frame to each element
            [v.append(f) for v, f in zip(video, frame)]
        else:
            # Initialize with a list of pulse trains
            video = frame

    # Restore logger level
    logging.getLogger(__name__).setLevel(current_level)

    return video
