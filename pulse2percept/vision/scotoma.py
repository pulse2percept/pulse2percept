""":py:class:`~pulse2percept.vision.Scotoma`"""
import numpy as np

from ..units import as_value, dva
from ..utils import PrettyPrint


class Scotoma(PrettyPrint):
    """A region of the visual field where native vision is lost

    A scotoma is eye-centered: it is defined in degrees of visual angle
    relative to the fovea, and it does not move when gaze does. Neither does an
    implant, which sits on the retina; the two hold their positions relative to
    each other while the *scene* moves past them.

    A scotoma says only how much vision is lost where. What lost vision looks
    like -- black, gray, blurred, filled in -- is a separate question, and one
    for whoever composes the final image.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    mask : callable
        ``mask(x, y)`` returning the loss at eye-centered visual-field
        coordinates ``x``, ``y`` (in dva). 0 is intact native vision and 1 is
        complete loss; anything in between is a partial defect, which is what
        leaves room for a measured or graded scotoma without another API.
    name : str, optional
        What to call this scotoma when it is printed.

    Examples
    --------
    A central geographic-atrophy scotoma 10 degrees across:

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
        """The fraction of native vision lost at each point

        Parameters
        ----------
        x, y : float or array_like
            Eye-centered visual-field coordinates in degrees of visual angle,
            relative to the fovea. ``y`` grows upwards.

        Returns
        -------
        loss : np.ndarray
            Loss in [0, 1], broadcast to the shape of ``x`` and ``y``.

        """
        x = np.asarray(as_value(x, dva, 'x'), dtype=float)
        y = np.asarray(as_value(y, dva, 'y'), dtype=float)
        for name, coord in (('x', x), ('y', y)):
            # A NaN coordinate compares false against every radius, so an
            # elliptical mask would report intact vision rather than raise:
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

    @classmethod
    def ellipse(cls, x_radius, y_radius, center=(0, 0), name=None):
        """An elliptical scotoma, lost inside and intact outside

        Parameters
        ----------
        x_radius, y_radius : float or Quantity
            Semi-axes of the ellipse, in degrees of visual angle.
        center : (x, y), optional
            Where the ellipse sits relative to the fovea, in dva. Defaults to
            the fovea itself.
        name : str, optional
            What to call this scotoma when it is printed.

        """
        x_radius = as_value(x_radius, dva, 'x_radius')
        y_radius = as_value(y_radius, dva, 'y_radius')
        for label, radius in (('x_radius', x_radius), ('y_radius', y_radius)):
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError(f"'{label}' must be a finite positive number "
                                 f"of degrees, not {radius}.")
        cx, cy = np.asarray(as_value(center, dva, 'center'), dtype=float)
        if not np.isfinite([cx, cy]).all():
            # Same trap as a NaN coordinate, and quieter: every point would
            # fall outside, leaving an entirely intact visual field:
            raise ValueError(f"'center' must be finite, not ({cx}, {cy}).")

        def mask(x, y):
            xr, yr = (x - cx) / x_radius, (y - cy) / y_radius
            return (xr ** 2 + yr ** 2 <= 1).astype(float)

        if name is None:
            name = f'ellipse({x_radius:g}, {y_radius:g}) at ({cx:g}, {cy:g})'
        return cls(mask, name=name)

    @classmethod
    def circle(cls, radius, center=(0, 0), name=None):
        """A circular scotoma, lost inside and intact outside

        Parameters
        ----------
        radius : float or Quantity
            Radius of the scotoma, in degrees of visual angle.
        center : (x, y), optional
            Where the circle sits relative to the fovea, in dva. Defaults to
            the fovea itself.
        name : str, optional
            What to call this scotoma when it is printed.

        """
        if name is None:
            radius_dva = as_value(radius, dva, 'radius')
            cx, cy = np.asarray(as_value(center, dva, 'center'), dtype=float)
            name = f'circle({radius_dva:g}) at ({cx:g}, {cy:g})'
        return cls.ellipse(radius, radius, center=center, name=name)
