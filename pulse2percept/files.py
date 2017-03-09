import subprocess
import numpy as np
import scipy.io as sio
import os
import logging

from pulse2percept import utils
from pulse2percept import electrode2currentmap as e2cm


@utils.deprecated
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


@utils.deprecated
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


@utils.deprecated('e2cm.image2pulsetrain')
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
