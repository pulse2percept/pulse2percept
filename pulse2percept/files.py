import numpy as np
import logging

from pulse2percept import utils

# Rather than trying to import these all over, try once and then remember
# by setting a flag.
try:
    import skimage
    import skimage.transform as sit
    import skimage.color as sic
    has_skimage = True
except (ImportError, AttributeError):
    # Might also raise "dict object has no attribute 'transform'"
    has_skimage = False

try:
    import skvideo
    import skvideo.io as svio
    skvideo._HAS_AVCONV
    skvideo._HAS_FFMPEG
    has_skvideo = True
except (ImportError, AttributeError):
    has_skvideo = False


def set_skvideo_path(ffmpeg_path=None, libav_path=None):
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
    set_skvideo_path(ffmpeg_path, libav_path)

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
    set_skvideo_path(ffmpeg_path, libav_path)

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
    or as a ``p2p.utils.TimeSeries`` object (if `as_timeseries` is True).

    Parameters
    ----------
    filename : str
        Video file name
    as_timeseries: bool, optional, default: True
        If True, returns the data as a ``p2p.utils.TimeSeries`` object.
    as_gray : bool, optional, default: False
        If True, loads only the luminance channel of the video.
    ffmpeg_path : str, optional, default: system's default path
        Path to ffmpeg library.
    libav_path : str, optional, default: system's default path
        Path to libav library.

    Returns
    -------
    video : ndarray | p2p2.utils.TimeSeries
        If `as_timeseries` is False, returns video data according to the
        Scikit-Video standard; that is, an ndarray of dimension (T, M, N, C),
        (T, M, N), (M, N, C), or (M, N), where T is the number of frames,
        M is the height, N is the width, and C is the number of channels (will
        be either 1 for grayscale or 3 for RGB).

        If `as_timeseries` is True, returns video data as a TimeSeries object
        of dimension (M, N, C), (M, N, T), (M, N, C), or (M, N).
        The sampling rate corresponds to 1 / frame rate.

    """
    if not has_skvideo:
        raise ImportError("You do not have scikit-video installed. "
                          "You can install it via $ pip install sk-video.")

    # Set the path if necessary
    set_skvideo_path(ffmpeg_path, libav_path)

    if skvideo._HAS_FFMPEG:
        backend = 'ffmpeg'
    else:
        backend = 'libav'
    video = svio.vread(filename, as_grey=as_gray, backend=backend)
    logging.getLogger(__name__).info("Loaded video from file '%s'." % filename)
    d_s = "Loaded video has shape (T, M, N, C) = " + str(video.shape)
    logging.getLogger(__name__).debug(d_s)

    if as_timeseries:
        # TimeSeries has the time as the last dimensions: re-order dimensions,
        # then squeeze out singleton dimensions
        axes = np.roll(range(video.ndim), -1)
        video = np.squeeze(np.transpose(video, axes=axes))
        fps = load_video_framerate(filename)
        d_s = "Reshaped video to shape (M, N, C, T) = " + str(video.shape)
        logging.getLogger(__name__).debug(d_s)
        return utils.TimeSeries(1.0 / fps, video)
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

    """
    if not has_skvideo:
        raise ImportError("You do not have scikit-video installed. "
                          "You can install it via $ pip install sk-video.")

    # Set the path if necessary
    set_skvideo_path(ffmpeg_path, libav_path)

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
    data : ndarray | p2p.utils.TimeSeries
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
    elif isinstance(data, utils.TimeSeries):
        is_timeseries = True
    else:
        raise TypeError('Data to be saved must be either a NumPy ndarray '
                        'or a ``p2p.utils.TimeSeries`` object')

    # Set the path if necessary, then choose backend
    set_skvideo_path(ffmpeg_path, libav_path)
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
        logging.getLogger(__name__).debug(d_s)
        data = data.resample(new_tsample)
        d_s = 'new: tsample=%f' % new_tsample
        d_s += ', shape=' + str(data.shape)
        logging.getLogger(__name__).debug(d_s)
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
    logging.getLogger(__name__).debug(d_s)

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
    ``p2p.utils.TimeSeries`` object, assuming they correspond to model
    input and model output, and plots them side-by-side.
    Both input video and percept are resampled according to `fps`.
    The percept is resized to match the height of the input video.

    Parameters
    ----------
    videofile : str
        File name of input video.
    percept : p2p.utils.TimeSeries
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

    if not isinstance(percept, utils.TimeSeries):
        raise TypeError("`percept` must be of type p2p.utils.TimeSeries.")

    # Set the path if necessary
    set_skvideo_path(ffmpeg_path, libav_path)

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


def find_files_like(datapath, pattern):
    """Finds files in a folder whose name matches a pattern

    This function looks for files in folder `datapath` that match a regular
    expression `pattern`.

    Parameters
    ----------
    datapath : str
        Path to search
    pattern : str
        A valid regular expression pattern

    Examples
    --------
    # Find all '.npz' files in parent dir
    >>> files = find_files_like('..', r'.*\.npz$')
    """
    # No need to import these at module level
    from os import listdir
    import re

    # Traverse file list and look for `pattern`
    filenames = []
    pattern = re.compile(pattern)
    for file in listdir(datapath):
        if pattern.search(file):
            filenames.append(file)

    return filenames
