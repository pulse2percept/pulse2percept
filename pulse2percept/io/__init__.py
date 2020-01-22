"""I/O functions to load/store data as images and videos.

.. autosummary::
    :toctree: _api

    image
    video

"""
from .image import image2stim
from .video import (load_video_metadata, load_video_framerate, load_video,
                    load_video_generator, save_video, save_video_sidebyside,
                    video2stim)

__all__ = [
    'image2stim',
    'load_video',
    'load_video_framerate',
    'load_video_generator',
    'load_video_metadata',
    'save_video',
    'save_video_sidebyside'
]
