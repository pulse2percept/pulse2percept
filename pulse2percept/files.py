import subprocess
import numpy as np
import scipy.io as sio
import os
import logging

from pulse2percept import utils

try:
    # Their init file is weird: Telling mock to disable the module will
    # not result in an ImportError... But, all of the functionality will
    # be missing... So check for some class attributes instead.
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
    ffmpeg_path : str, optional
        Path to ffmpeg library. If not given, use the system's default path.
    libav_path : str, optional
        Path to libav library. If not given, use the system's default path.

    """
    if libav_path is not None:
        skvideo.setLibAVPath(libav_path)
    if ffmpeg_path is not None:
        skvideo.setFFmpegPath(ffmpeg_path)


def load_video(filename, ffmpeg_path=None, libav_path=None):
    """Loads a video from file.

    This function loads a video from file and returns it as a NumPy ndarray
    with the help of the Scikit-Video.

    Parameters
    ----------
    filename : str
        Video file name
    ffmpeg_path : str, optional
        Path to ffmpeg library. If not given, use the system's default path.
    libav_path : str, optional
        Path to libav library. If not given, use the system's default path.

    Returns
    -------
    video : ndarray
        Video data as an ndarray of dimension (T, M, N, C), (T, M, N),
        (M, N, C), or (M, N), where T is the number of frames, M is the height,
        N is the width, and C is the number of channels.

    Examples
    --------
    >>> from skvideo import datasets
    >>> video = load_video(datasets.bikes())
    >>> video.shape
    (250, 272, 640, 3)

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
    video = svio.vread(filename, backend=backend)
    logging.getLogger(__name__).info("Loaded video from file '%s'." % filename)

    return video


def load_video_generator(filename, ffmpeg_path=None, libav_path=None):
    """Returns a generator that can load a video from file frame-by-frame.

    This function loads a video from file and returns it as a NumPy ndarray
    with the help of the Scikit-Video.

    Parameters
    ----------
    filename : str
        Video file name
    ffmpeg_path : str, optional
        Path to ffmpeg library. If not given, use the system's default path.
    libav_path : str, optional
        Path to libav library. If not given, use the system's default path.

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
    set_skvideo_path(ffmpeg_path, libav_path)

    # Try loading
    if skvideo._HAS_FFMPEG:
        reader = svio.FFmpegReader(filename)
    elif skvideo._HAS_AVCONV:
        reader = svio.LibAVReader(filename)
    else:
        raise ImportError("You have neither ffmpeg nor libav (which comes "
                          "with avprobe) installed.")

    return reader


def save_video(filename, data, ffmpeg_path=None, libav_path=None):
    """Saves a video to file.

    This function stores a NumPy ndarray to file using Scikit-Video.

    Parameters
    ----------
    filename : str
        Video file name.
    data : ndarray
        Video data as an ndarray of dimension (T, M, N, C), (T, M, N),
        (M, N, C), or (M, N), where T is the number of frames, M is the height,
        N is the width, and C is the number of channels.

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
    svio.vwrite(filename, data, backend=backend)
    logging.getLogger(__name__).info("Saved video to file '%s'." % filename)


def save_percept(filename, percept, max_contrast=True, ffmpeg_path=None,
                 libav_path=None):
    """Saves a percept to file.

    This function stores a percept (`TimeSeries`) to file using Scikit-Video.

    Parameters
    ----------
    filename : str
        Video file name.
    percept : p2p.utils.TimeSeries
        A `TimeSeries` object describing a percept.
    max_contrast : bool, optional
        Flag whether to maximize contrast in the video. If set to True, the
        highest brightness value in the movie will be mapped to 255, and the
        lowest to 0. Else, a maximum brightness value  of 50 in the movie is
        assumed.

    """
    if not has_skvideo:
        raise ImportError("You do not have scikit-video installed. "
                          "You can install it via $ pip install sk-video.")

    if not isinstance(percept, utils.TimeSeries):
        raise TypeError("`percept` must be of type p2p.utils.TimeSeries.")

    # Set the path if necessary
    set_skvideo_path(ffmpeg_path, libav_path)

    # Permute dimensions to fit Scikit-Video's format
    axes = np.arange(percept.data.ndim)
    data = np.transpose(percept.data, axes=np.roll(axes, 1))

    # Normalize range of brightness values
    if max_contrast:
        # Maximize contrast: 0 <= grayscale <= 255
        data -= data.min()
        data /= data.max() * 255.0
    else:
        # Assume brightness value 50 corresponds to 255 grayscale
        data = np.minimum(1.0, data / 50.0)
        data *= 255.0

    if skvideo._HAS_FFMPEG:
        backend = 'ffmpeg'
    else:
        backend = 'libav'
    svio.vwrite(filename, data.astype(np.uint8), backend=backend)
    logging.getLogger(__name__).info("Saved video to file '%s'." % filename)


@utils.deprecated('p2p.files.save_video')
def savemoviefiles(filestr, data, path='savedImages/'):
    """Saves a brightness movie to .npy, .mat, and .avi format

    This function is deprecated as of v0.2 and will be removed
    completely in v0.3.

    Parameters
    ----------
    filestr : str
        Name of the resulting files without file type (suffix .npy, .mat
        or .avi will be appended)
    data : array
        A 3-D NumPy array containing the data, such as the `data` field of
        a utils.TimeSeries object
    path : str, optional
        Path to directory where files should be saved.
        Default: savedImages/

    .. deprecated:: 0.2
    """
    np.save(path + filestr, data)  # save as npy
    sio.savemat(path + filestr + '.mat', dict(mov=data))  # save as matfile
    npy2movie(path + filestr + '.avi', data)  # save as avi


@utils.deprecated('p2p.files.save_video')
def npy2movie(filename, movie, rate=30):
    """Saves a NumPy array to .avi on Windows

    This function is deprecated as of v0.2.

    Creates avi files will work on a 64x Windows as well as a PC provided
    that the ffmpeg folder is included in the right location.

    Most uptodate version of ffmpeg can be found here:
    https://ffmpeg.zeranoe.com/builds/win64/static/

    Used instructions from here with modifications:
    http://adaptivesamples.com/how-to-install-ffmpeg-on-windows

    Parameters
    ----------
    filename : str
        File name of .avi movie to be produced
    movie : array
        A 3-D NumPy array containing the data, such as the `data` field of
        a utils.TimeSeries object
    rate : float, optional
        Frame rate. Default: 30 Hz

    .. deprecated:: 0.2
    """
    if os.name != 'nt':
        raise OSError("npy2movie only works on Windows.")

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("You do not have PIL installed.")

    cmdstring = ('ffmpeg.exe',
                 '-y',
                 '-r', '%d' % rate,
                 '-f', 'image2pipe',
                 '-vcodec', 'mjpeg',
                 '-i', 'pipe:',
                 '-vcodec', 'libxvid',
                 filename
                 )
    logging.getLogger(__name__).info(filename)
    p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

    for i in range(movie.shape[-1]):
        im = Image.fromarray(np.uint8(scale(movie[:, :, i], 0, 255)))
        p.stdin.write(im.tobytes('jpeg', 'L'))

    p.stdin.close()


@utils.deprecated('p2p.stimuli.image2pulsetrain')
def scale(inarray, newmin=0.0, newmax=1.0):
    """Scales an image such that its lowest value attains newmin and
    it's highest value attains newmax.

    This function is deprecated as of v0.2 and will be removed
    completely in v0.3.

    written by Ione Fine, based on code from Rick Anthony

    Parameters
    ----------
    inarray : array
        The input array
    newmin : float, optional
        The desired lower bound of values in `inarray`. Default: 0
    newmax : float, optional
        The desired upper bound of values in `inarray`. Default: 1

    .. deprecated:: 0.2
    """

    oldmin = inarray.min()
    oldmax = inarray.max()

    delta = (newmax - newmin) / (oldmax - oldmin)
    outarray = delta * (inarray - oldmin) + newmin
    return outarray


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
    >>> files = find_files_like('..', '.*\.npz$')
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
