"""Various visualization functions.

.. autosummary::
    :toctree: _api

    axon_map

"""
# This import is necessary to ensure consistency of the generated images across
# platforms, and for the tests to run on Travis:
# https://stackoverflow.com/questions/35403127/testing-matplotlib-based-plots-in-travis-ci
# http://www.davidketcheson.info/2015/01/13/using_matplotlib_image_comparison.html
import matplotlib
matplotlib.use('agg')

from .axon_map import plot_axon_map, plot_implant_on_axon_map

__all__ = [
    'plot_axon_map',
    'plot_implant_on_axon_map'
]
