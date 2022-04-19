"""Processing techniques that can be used to process image or video before it becomes stimuli
.. autosummary::
    :toctree: _api

    shrink
..

"""

from .shrink import shrinked_image, shrinked_single_image, shrinked_video, shrinked_video_1d


__all__ = [
    'shrinked_image',
    'shrinked_single_image',
    'shrinked_video',
    'shrinked_video_1d'
]
