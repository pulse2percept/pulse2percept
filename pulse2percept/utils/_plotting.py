"""Private Matplotlib helpers shared by plotting code"""
from matplotlib.ticker import FuncFormatter

from .constants import UM_PER_MM


def _um_to_mm_label(value, _pos=None):
    # Rounding hides locator float noise (e.g. 1e-13); +0.0 turns -0 into 0
    return f"{round(value / UM_PER_MM, 9) + 0.0:g}"


def set_mm_ticks(ax):
    """Label the x, y (and z, if 3D) ticks of ``ax`` in mm

    Data and limits stay in microns. Labels are computed from the current tick
    positions at draw time, so they stay correct after later limit changes.
    """
    axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, 'zaxis') else [])
    # One formatter per axis: formatters bind to the axis they are set on
    for axis in axes:
        axis.set_major_formatter(FuncFormatter(_um_to_mm_label))
