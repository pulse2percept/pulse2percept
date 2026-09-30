""":py:class:`~pulse2percept.vision.Scotoma`"""
import numpy as np

from ..units import as_value, dva
from ..utils import PrettyPrint


class Scotoma(PrettyPrint):
    """A region of the visual field where native vision is lost

    A scotoma is eye-centered: defined in dva relative to the fovea, it does
    not move with gaze. Neither does an implant, so scotoma and implant keep
    their relative positions while the *scene* moves.

    A scotoma defines only how much vision is lost where. How lost vision is
    drawn (black, gray, inpainted) is set by
    :py:class:`~pulse2percept.vision.Scene`.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    mask : callable
        ``mask(x, y)`` returning the loss at eye-centered coordinates ``x``,
        ``y`` (dva). 0 is intact native vision, 1 is complete loss; values in
        between are partial loss (e.g., a measured or graded scotoma).
    name : str, optional
        Name used when printing.

    Examples
    --------
    A central geographic-atrophy scotoma 10 dva across:

    >>> from pulse2percept.vision import Scotoma
    >>> from pulse2percept.units import dva
    >>> scotoma = Scotoma.circle(5 * dva)
    >>> float(scotoma(0, 0)), float(scotoma(9, 0))
    (1.0, 0.0)

    """

    def __init__(self, mask, name=None):
        if not callable(mask):
            raise TypeError(f"'mask' must be callable, not {type(mask)}.")
        self.mask = mask
        self.name = name

    def _pprint_params(self):
        return {'name': self.name}

    def __call__(self, x, y):
        """Fraction of native vision lost at each point

        Parameters
        ----------
        x, y : float or array_like
            Eye-centered coordinates in dva, relative to the fovea. ``y``
            increases upward.

        Returns
        -------
        loss : np.ndarray
            Loss in [0, 1], broadcast to the shape of ``x`` and ``y``.

        """
        x = np.asarray(as_value(x, dva, 'x'), dtype=float)
        y = np.asarray(as_value(y, dva, 'y'), dtype=float)
        for name, coord in (('x', x), ('y', y)):
            # NaN compares false against every radius, so an elliptical mask
            # would report intact vision:
            if not np.all(np.isfinite(coord)):
                raise ValueError(f"'{name}' must be finite.")
        loss = np.broadcast_to(np.asarray(self.mask(x, y), dtype=float),
                               np.broadcast_shapes(x.shape, y.shape))
        if not np.all(np.isfinite(loss)):
            raise ValueError("A scotoma mask must return finite values.")
        if loss.min() < 0 or loss.max() > 1:
            raise ValueError(f"A scotoma mask returns the fraction of native "
                             f"vision lost and must stay in [0, 1], but this "
                             f"one returned values in "
                             f"[{loss.min():g}, {loss.max():g}].")
        return loss

    def mirror(self, name=None):
        """A copy reflected across the vertical meridian

        ``mirrored(x, y) == original(-x, y)`` in eye-centered coordinates.

        .. versionadded:: 0.11.0

        Parameters
        ----------
        name : str, optional
            Name for the mirrored scotoma (default: original name, marked
            as mirrored)

        Returns
        -------
        scotoma : :py:class:`~pulse2percept.vision.Scotoma`
            A new scotoma. The original is left unchanged.

        Examples
        --------
        Fellow eye of a bilateral loss, mirror-symmetric about the vertical
        meridian:

        >>> from pulse2percept.units import dva
        >>> from pulse2percept.vision import Scotoma
        >>> left_scotoma = Scotoma.circle(3 * dva, center=(6, 0) * dva)
        >>> right_scotoma = left_scotoma.mirror()
        >>> float(right_scotoma(-6, 0)), float(right_scotoma(6, 0))
        (1.0, 0.0)

        """
        mask = self.mask

        def mirrored(x, y):
            return mask(-x, y)

        if name is None and self.name is not None:
            name = f'mirror of {self.name}'
        return type(self)(mirrored, name=name)

    @classmethod
    def ellipse(cls, x_radius, y_radius, center=(0, 0), name=None):
        """An elliptical scotoma: complete loss inside, intact outside

        Parameters
        ----------
        x_radius, y_radius : float or Quantity
            Semi-axes of the ellipse, in dva.
        center : (x, y), optional
            Center relative to the fovea, in dva. Defaults to the fovea.
        name : str, optional
            Name used when printing.

        """
        x_radius = as_value(x_radius, dva, 'x_radius')
        y_radius = as_value(y_radius, dva, 'y_radius')
        for label, radius in (('x_radius', x_radius), ('y_radius', y_radius)):
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError(f"'{label}' must be a finite positive number "
                                 f"of degrees, not {radius}.")
        cx, cy = np.asarray(as_value(center, dva, 'center'), dtype=float)
        if not np.isfinite([cx, cy]).all():
            # A NaN center would put every point outside (intact vision):
            raise ValueError(f"'center' must be finite, not ({cx}, {cy}).")

        def mask(x, y):
            xr, yr = (x - cx) / x_radius, (y - cy) / y_radius
            return (xr ** 2 + yr ** 2 <= 1).astype(float)

        if name is None:
            name = f'ellipse({x_radius:g}, {y_radius:g}) at ({cx:g}, {cy:g})'
        return cls(mask, name=name)

    @classmethod
    def circle(cls, radius, center=(0, 0), name=None):
        """A circular scotoma: complete loss inside, intact outside

        Parameters
        ----------
        radius : float or Quantity
            Radius of the scotoma, in dva.
        center : (x, y), optional
            Center relative to the fovea, in dva. Defaults to the fovea.
        name : str, optional
            Name used when printing.

        """
        if name is None:
            radius_dva = as_value(radius, dva, 'radius')
            cx, cy = np.asarray(as_value(center, dva, 'center'), dtype=float)
            name = f'circle({radius_dva:g}) at ({cx:g}, {cy:g})'
        return cls.ellipse(radius, radius, center=center, name=name)
