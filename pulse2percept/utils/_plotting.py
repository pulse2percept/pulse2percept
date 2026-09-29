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
    formatter = FuncFormatter(_um_to_mm_label)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    if hasattr(ax, 'zaxis'):
        ax.zaxis.set_major_formatter(formatter)
