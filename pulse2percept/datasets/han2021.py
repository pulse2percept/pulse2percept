"""`fetch_han2021`"""
from os.path import join, isfile
import numpy as np

from .base import get_data_dir, fetch_url
from pulse2percept.stimuli import VideoStimulus

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    has_pandas = False

try:
    import h5py
    has_h5py = True
except ImportError:
    has_h5py = False


def fetch_han2021(videos=None, resize=None, as_gray=None, data_path=None,
                  download_if_missing=True):
    """Load the original videos of outdoor scenes from [Han2021]_

    Download the original videos or simulated prosthetic vision of outdoor scenes 
    described in [Han2021] from https://osf.io/pf2ja/ (303MB) to ``data_path``. 
    By default, all datasets are stored in '~/pulse2percept_data/', but a 
    different path can be specified.

    The size of the videos are 320*180. The number of frames of the videos are 125 to 126.

    .. versionadded:: 0.9

    Parameters
    videos: str | list of strings | None, optional
        The name of videos you wants to put into the data. By default, all
        subjects are selected.
        Available names:
        'sample1', 'sample2', 'sample3', 'sample4', 'stim1', 'stim2', 'stim3', 'stim4', 
        'stim5', 'stim6', 'stim7', 'stim8', 'stim9', 'stim10', 'stim11', 'stim12', 
        'stim13', 'stim14', 'stim15', 'stim16'
    resize : (height, width) or None, optional, default: None
        A tuple specifying the desired height and the width of each video frame.
        The original size is 320*180.
    as_gray : bool, optional
        Flag whether to convert the image to grayscale.
        A four-channel image is interpreted as RGBA (e.g., a PNG), and the
        alpha channel will be blended with the color black.
    data_path: string, optional
        Specify another download and cache folder for the dataset. By default
        all pulse2percept data is stored in '~/pulse2percept_data' subfolders.
    download_if_missing : optional
        If False, raise an IOError if the data is not locally available
        instead of trying to download it from the source site.

    Returns
    -------
    data: dict of VideoStimulus
        VideoStimulus of the original videos in [Han2021]_ is returned

    """
    if not has_h5py:
        raise ImportError("You do not have h5py installed. "
                          "You can install it via $ pip install h5py.")
    if not has_pandas:
        raise ImportError("You do not have pandas installed. "
                          "You can install it via $ pip install pandas.")
    # Create the local data directory if it doesn't already exist:
    data_path = get_data_dir(data_path)

    # Download the dataset if it doesn't already exist:
    file_path = join(data_path, 'han2021.zip')
    if not isfile(file_path):
        if download_if_missing:
            url = 'https://osf.io/pf2ja/download'
            checksum = 'e31a74a6ac9decfa8d8b9eccd0c71da868f8dfa9f0475a4caca82085307d67b1'
            fetch_url(url, file_path, remote_checksum=checksum)
        else:
            raise IOError(f"No local file {file_path} found")

    # Open the HDF5 file:
    hf = h5py.File(file_path, 'r')
    data = dict()
    if resize != None:
        size = resize
    else:
        size = (320, 180)
    if videos == None:
        videos = hf.keys()
        for key in videos:
            vid = np.asarray(hf[key])
            name = key[0:-4]
            metadata = {'plugin': 'ffmpeg',
                        'nframes': vid.shape[3],
                        'ffmpeg_version': '4.2.2 built with gcc 9.2.1 (GCC) 20200122',
                        'codec': 'h264',
                        'pix_fmt': 'yuv420p(tv',
                        'fps': 25.0,
                        'source_size': (960, 540),
                        'size': size,
                        'rotate': 0,
                        'duration': vid.shape[3]/25.0,
                        'source': key,
                        'source_shape': (540, 960, 3, vid.shape[3])}
            data[name] = VideoStimulus(
                vid, metadata=metadata, resize=resize, as_gray=as_gray)
    else:
        if type(videos) == str:
            videos = [videos]
        for name in videos:
            key = name+'.mp4'
            if key not in hf.keys():
                raise ValueError(
                    f"[Han2021]'s original videos do not include '{name}'"
                    f". Available names: 'sample1', 'sample2', 'sample3', 'sample4', "
                    f"'stim1', 'stim2', 'stim3', 'stim4', 'stim5', 'stim6', 'stim7', "
                    f"'stim8', 'stim9', 'stim10', 'stim11', 'stim12', 'stim13', "
                    f"'stim14', 'stim15', 'stim16'"
                )
            vid = np.asarray(hf[key])
            metadata = {'plugin': 'ffmpeg',
                        'nframes': vid.shape[3],
                        'ffmpeg_version': '4.2.2 built with gcc 9.2.1 (GCC) 20200122',
                        'codec': 'h264',
                        'pix_fmt': 'yuv420p(tv',
                        'fps': 25.0,
                        'source_size': (960, 540),
                        'size': size,
                        'rotate': 0,
                        'duration': vid.shape[3]/25.0,
                        'source': key,
                        'source_shape': (540, 960, 3, vid.shape[3])}

            data[name] = VideoStimulus(
                vid, metadata=metadata, resize=resize, as_gray=as_gray)
    hf.close()
    return data
