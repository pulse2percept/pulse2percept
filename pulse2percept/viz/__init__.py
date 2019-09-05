# This import is necessary to ensure consistency of the generated images across
# platforms, and for the tests to run on Travis:
# https://stackoverflow.com/questions/35403127/testing-matplotlib-based-plots-in-travis-ci
# http://www.davidketcheson.info/2015/01/13/using_matplotlib_image_comparison.html
import matplotlib
matplotlib.use('agg')

from .base import plot_fundus

__all__ = [
    'plot_fundus'
]
