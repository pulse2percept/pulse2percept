"""Visual-field geometry for image and video stimuli (private)

An image or video that knows the field of view it subtends can be placed in the
visual field rather than merely in a pixel grid. The convention:

*  ``fov`` is the *outer* angular extent of the frame, ``(width, height)`` in
   degrees of visual angle, centered on the frame.
*  Pixel coordinates address pixel *centers*, which sit half an angular pixel
   inside that extent.
*  Row 0 is the top of the frame and therefore the largest ``y``.
"""
import numpy as np

from ..units import as_value, dva


def resolve_fov(fov, n_rows, n_cols):
    """Normalize a user-supplied ``fov`` to ``(width, height)`` in dva

    A scalar is the horizontal FOV; the vertical one follows from the frame's
    aspect ratio, which is the same as assuming square angular pixels.
    """
    if fov is None:
        return None
    fov = as_value(fov, dva, 'fov')
    fov = np.asarray(fov, dtype=float)
    if fov.ndim == 0:
        width = float(fov)
        height = width * n_rows / n_cols
    elif fov.shape == (2,):
        width, height = (float(f) for f in fov)
    else:
        raise ValueError(f'"fov" must be a scalar (horizontal FOV) or a '
                         f'(width, height) pair, not {fov.tolist()}.')
    for name, f in (('width', width), ('height', height)):
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f'"fov" {name} must be a finite positive number '
                             f'of degrees, not {f}.')
    return (width, height)


class HasFieldOfView:
    """Mixin giving a frame-shaped stimulus an optional field of view

    The host class owns the ``_fov`` slot and defines ``_frame_shape`` as the
    ``(rows, cols)`` of one frame.
    """
    __slots__ = ()

    @property
    def fov(self):
        """Field of view ``(width, height)`` in degrees of visual angle

        ``None`` if the stimulus has no visual-field geometry, in which case
        its pixels are just pixels.
        """
        return self._fov

    @property
    def _angular_pixel(self):
        """Angular size ``(width, height)`` of one pixel, in dva"""
        if self._fov is None:
            raise ValueError(
                f"This {type(self).__name__} has no field of view, so its "
                f"pixels have no angular size. Pass 'fov' to the constructor.")
        n_rows, n_cols = self._frame_shape
        return (self._fov[0] / n_cols, self._fov[1] / n_rows)

    def _fov_for_shape(self, shape):
        """FOV of a frame of ``shape`` pixels at this stimulus' pixel size

        The rule for operations that keep the angular pixel size and change the
        pixel count (a crop, a trim, a rotation that grows the canvas), as
        opposed to a resize, which keeps the extent and resamples the pixels.
        """
        if self._fov is None:
            return None
        dx, dy = self._angular_pixel
        return (shape[1] * dx, shape[0] * dy)

    def pixel_to_dva(self, col, row):
        """Visual-field coordinates of a pixel center

        Parameters
        ----------
        col, row : float or array_like
            Pixel coordinates, where ``(0, 0)`` is the center of the top-left
            pixel. Fractional values address points between pixel centers.

        Returns
        -------
        x, y : np.ndarray
            Visual-field coordinates in degrees of visual angle, relative to
            the center of the frame. ``y`` grows upwards, so row 0 has the
            largest ``y``.

        """
        dx, dy = self._angular_pixel
        col = np.asarray(col, dtype=float)
        row = np.asarray(row, dtype=float)
        x = (col + 0.5) * dx - self._fov[0] / 2
        y = self._fov[1] / 2 - (row + 0.5) * dy
        return x, y

    def dva_to_pixel(self, x, y):
        """Pixel coordinates of a point in the visual field

        The inverse of
        :py:meth:`~pulse2percept.stimuli.ImageStimulus.pixel_to_dva`.

        Parameters
        ----------
        x, y : float or array_like
            Visual-field coordinates in degrees of visual angle, relative to
            the center of the frame.

        Returns
        -------
        col, row : np.ndarray
            Continuous pixel coordinates, where ``(0, 0)`` is the center of the
            top-left pixel. They are not rounded and not clipped to the frame:
            a point outside the FOV maps outside the pixel grid.

        """
        dx, dy = self._angular_pixel
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = (x + self._fov[0] / 2) / dx - 0.5
        row = (self._fov[1] / 2 - y) / dy - 0.5
        return col, row
