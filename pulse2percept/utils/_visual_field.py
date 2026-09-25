"""Fovea-centered eccentricity rings and polar-angle meridians for plots

Display annotations only: nothing here touches data, grids, or sampling.
Coordinates are Cartesian visual-field dva; polar angle is geometric deg,
0 = +x, 90 = +y, counterclockwise.
"""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..units import as_value, deg, dva

# Mid-gray reads on both light scenes and dark percepts
GRID_COLOR = '0.5'

# `rings=True` doubles from here (dva): 1.25, 2.5, 5, 10, 20, 40, ...
_FIRST_RING = 1.25

# `meridians=True` spacing (deg)
_MERIDIAN_STEP = 45.0

# Polar angle (deg) of ring labels, between the 45 and 90 deg meridians
_LABEL_ANGLE = 65.0


def _label_angle(center, extent):
    """``_LABEL_ANGLE``, or the direction to the field's middle if
    ``center`` lies outside the field (where 65 deg would be off-screen)"""
    left, right, bottom, top = extent
    if left <= center[0] <= right and bottom <= center[1] <= top:
        return _LABEL_ANGLE
    return np.rad2deg(np.arctan2((bottom + top) / 2 - center[1],
                                 (left + right) / 2 - center[0]))


def _is_off(value):
    return value is None or value is False


def ring_radii(rings, r_max, r_min=0.0):
    """Eccentricities (dva) a ``rings`` argument selects

    True is the doubling sequence from ``_FIRST_RING``, a number is a
    spacing, and a sequence is the eccentricities themselves. Automatic rings
    are kept in ``(r_min, r_max]``; explicit ones are returned as given,
    sorted.
    """
    if _is_off(rings):
        return np.zeros(0)
    # 1e-9 keeps a ring that lands exactly on r_max
    if rings is True:
        n = 0
        if r_max >= _FIRST_RING:
            n = int(np.floor(np.log2(r_max / _FIRST_RING) + 1e-9)) + 1
        radii = _FIRST_RING * 2.0 ** np.arange(n)
        return radii[radii > r_min]
    rings = np.asarray(as_value(rings, dva, 'rings'), dtype=float)
    if rings.ndim == 0:
        step = float(rings)
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"'rings' is a spacing in degrees and must be "
                             f"finite and positive, not {step}.")
        radii = step * np.arange(1, int(max(r_max, 0) / step + 1e-9) + 1)
        return radii[radii > r_min]
    radii = np.sort(rings.ravel())
    if radii.size == 0 or not np.all(np.isfinite(radii)) or radii.min() <= 0:
        raise ValueError(f"'rings' must be finite positive eccentricities in "
                         f"degrees, not {rings.tolist()}.")
    return radii


def meridian_angles(meridians):
    """Polar angles (deg, in [0, 360)) a ``meridians`` argument selects

    True is every ``_MERIDIAN_STEP`` deg, a number is that spacing from 0,
    and a sequence is the angles themselves.
    """
    if _is_off(meridians):
        return np.zeros(0)
    if meridians is True:
        meridians = _MERIDIAN_STEP
    angles = np.asarray(as_value(meridians, deg, 'meridians'), dtype=float)
    if angles.ndim == 0:
        step = float(angles)
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"'meridians' is a polar-angle spacing in "
                             f"degrees and must be finite and positive, not "
                             f"{step}.")
        return step * np.arange(int(np.ceil(360 / step - 1e-9)))
    if angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError(f"'meridians' must be finite polar angles in "
                         f"degrees, not {angles.tolist()}.")
    return np.unique(np.mod(angles.ravel(), 360))


def visible_band(center, extent):
    """Eccentricity range ``(r_min, r_max)`` for automatic rings in a field

    ``extent`` is ``(left, right, bottom, top)`` in dva. With ``center``
    inside, rings stop at the nearest edge; otherwise they span the
    eccentricities the field covers.
    """
    cx, cy = center
    left, right, bottom, top = extent
    nearest_edge = min(cx - left, right - cx, cy - bottom, top - cy)
    if nearest_edge >= 0:
        return 0.0, nearest_edge
    near = np.hypot(cx - np.clip(cx, left, right),
                    cy - np.clip(cy, bottom, top))
    far = max(np.hypot(x - cx, y - cy)
              for x in (left, right) for y in (bottom, top))
    return float(near), float(far)


def _ray_span(center, angle, extent):
    """Distances along a ray from ``center`` where it enters and leaves
    ``extent``, or None if it misses"""
    t0, t1 = 0.0, np.inf
    rad = np.deg2rad(angle)
    left, right, bottom, top = extent
    for c, d, lo, hi in ((center[0], np.cos(rad), left, right),
                         (center[1], np.sin(rad), bottom, top)):
        # cos(90 deg) is 6e-17, not 0
        if abs(d) < 1e-12:
            if not lo <= c <= hi:
                return None
            continue
        a, b = (lo - c) / d, (hi - c) / d
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return (t0, t1) if t1 > t0 else None


def _identity(x, y):
    return x, y


def draw(ax, radii, angles, center, extent, to_axes=_identity,
         color=GRID_COLOR):
    """Draw rings and meridians about ``center``; returns the artists

    Meridians run from ``center`` to the edge of ``extent`` (dva).
    ``to_axes`` maps dva onto the axes' data coordinates.
    """
    cx, cy = center
    artists = []
    theta = np.linspace(0, 2 * np.pi, 181)
    label = np.deg2rad(_label_angle(center, extent))
    for radius in radii:
        artists += ax.plot(*to_axes(cx + radius * np.cos(theta),
                                    cy + radius * np.sin(theta)),
                           color=color, linestyle='--', linewidth=0.8,
                           alpha=0.9)
        # Screen-space alignment, so the label sits outside the ring even
        # when `to_axes` flips y:
        artists.append(ax.text(*to_axes(cx + radius * np.cos(label),
                                        cy + radius * np.sin(label)),
                               f'{radius:g}\N{DEGREE SIGN}', color=color,
                               fontsize=8, alpha=0.95, ha='left',
                               va='bottom', clip_on=True))
    for angle in angles:
        span = _ray_span(center, angle, extent)
        if span is None:
            continue
        t = np.asarray(span)
        rad = np.deg2rad(angle)
        artists += ax.plot(*to_axes(cx + t * np.cos(rad),
                                    cy + t * np.sin(rad)),
                           color=color, linestyle='-', linewidth=0.6,
                           alpha=0.6)
    return artists


def rasterize(shape, radii, angles, center, extent, to_pixel,
              color=GRID_COLOR, dpi=100.0):
    """The same grid as a transparent ``(rows, cols, 4)`` RGBA float array

    The HTML player's frame canvas covers ordinary Matplotlib artists, so
    animations need the grid as image data. ``to_pixel`` maps dva onto
    continuous pixel coordinates, (0, 0) at the top-left pixel center.
    """
    n_rows, n_cols = shape
    fig = Figure(figsize=(n_cols / dpi, n_rows / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.patch.set_alpha(0)
    ax.set_axis_off()
    # One axes unit per pixel, y running down, as `imshow` draws a frame:
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    draw(ax, radii, angles, center, extent, to_pixel, color=color)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32) / 255.0
