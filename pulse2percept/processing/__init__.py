"""Processing techniques that can be used to process image or video before it becomes stimuli
.. autosummary::
    :toctree: _api

    shrink
..

"""

from .shrink import shrinked_single_image, shrinked_image, shrinked_video, shrinked_stim, _spatial_temporal_saliency


__all__ = [
    'shrinked_single_image',
    'shrinked_image', 
    'shrinked_video',
    'shrinked_stim', 
    '_spatial_temporal_saliency'
]
