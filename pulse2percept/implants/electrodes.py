"""`Electrode`, `PointSource`, `DiskElectrode`, `SquareElectrode`, `HexElectrode`"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon

import numpy as np
from abc import ABCMeta, abstractmethod
# Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working:
from collections.abc import Sequence

from ..utils import PrettyPrint


class Electrode(PrettyPrint, metaclass=ABCMeta):
    """Electrode

    Abstract base class for all electrodes.
    """
    __slots__ = ('x', 'y', 'z', 'plot_patch', 'plot_kwargs')

    def __init__(self, x, y, z):
        if isinstance(x, (Sequence, np.ndarray)):
            raise TypeError("x must be a scalar, not %s." % (type(x)))
        if isinstance(y, (Sequence, np.ndarray)):
            raise TypeError("y must be a scalar, not %s." % type(y))
        if isinstance(z, (Sequence, np.ndarray)):
            raise TypeError("z must be a scalar, not %s." % type(z))
        self.x = x
        self.y = y
        self.z = z
        # A matplotlib.patches object (e.g., Circle, Rectangle) that can be
        # used to plot the electrode:
        self.plot_patch = None
        # Any keyword arguments that should be passed to the call above:
        # (e.g., {'radius': 5}):
        self.plot_kwargs = {}

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'x': self.x, 'y': self.y, 'z': self.z}

    @abstractmethod
    def electric_potential(self, x, y, z, *args, **kwargs):
        raise NotImplementedError

    def plot(self, autoscale=False, ax=None):
        """Plot

        Parameters
        ----------
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None given, a new one will be created.

        Returns
        -------
        ax : ``matplotlib.axes.Axes``
            Returns the axis object of the plot

        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        if self.plot_patch is not None:
            if isinstance(self.plot_patch, list):
                # Special case: draw multiple objects
                for p, kw in zip(self.plot_patch, self.plot_kwargs):
                    ax.add_patch(p((self.x, self.y), zorder=10, **kw))
            else:
                # Regular use case: single object
                ax.add_patch(self.plot_patch((self.x, self.y), zorder=10,
                                             **self.plot_kwargs))
            # This is needed in MPL 3.0.X to set the axis limit correctly:
            ax.autoscale_view()
        if autoscale:
            ax.set_xlim(self.x - pad, self.x + pad)
            ax.set_ylim(self.y - pad, self.y + pad)
        return ax


class PointSource(Electrode):
    """Idealized current point source

    Parameters
    ----------
    x/y/z : double
        3D location of the point source

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ()

    def __init__(self, x, y, z):
        super(PointSource, self).__init__(x, y, z)
        self.plot_patch = Circle
        self.plot_kwargs = {'radius': 5, 'linewidth': 2,
                            'ec': (0.3, 0.3, 0.3, 1),
                            'fc': (0.8, 0.8, 0.8, 0.7)}

    def electric_potential(self, x, y, z, amp, sigma):
        """Calculate electric potential at (x, y, z)

        Parameters
        ----------
        x/y/z : double
            3D location at which to evaluate the electric potential
        amp : double
            amplitude of the constant current pulse
        sigma : double
            resistivity of the extracellular solution

        Returns
        -------
        pot : double
            The electric potential at (x, y, z)

        The electric potential :math:`V(r)` of a point source is given by:

        .. math::

            V(r) = \\frac{\\sigma I}{4 \\pi r},

        where :math:`\\sigma` is the resistivity of the extracellular solution
        (typically Ames medium, :math:`\\sigma = 110 \\Ohm cm`),
        :math:`I` is the amplitude of the constant current pulse,
        and :math:`r` is the distance from the stimulating electrode to the
        point at which the voltage is being computed.

        """
        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2)
        if np.isclose(r, 0):
            return sigma * amp
        return sigma * amp / (4.0 * np.pi * r)


class DiskElectrode(Electrode):
    """Circular disk electrode

    Parameters
    ----------
    x/y/z : double
        3D location that is the center of the disk electrode
    r : double
        Disk radius in the x,y plane

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('r',)

    def __init__(self, x, y, z, r):
        super(DiskElectrode, self).__init__(x, y, z)
        if isinstance(r, (Sequence, np.ndarray)):
            raise TypeError("Electrode radius must be a scalar.")
        if r <= 0:
            raise ValueError("Electrode radius must be > 0, not %f." % r)
        self.r = r
        self.plot_patch = Circle
        self.plot_kwargs = {'radius': r, 'linewidth': 2,
                            'ec': (0.3, 0.3, 0.3, 1),
                            'fc': (0.8, 0.8, 0.8, 0.7)}

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'r': self.r})
        return params

    def electric_potential(self, x, y, z, v0):
        """Calculate electric potential at (x, y, z)

        Parameters
        ----------
        x/y/z : double
            3D location at which to evaluate the electric potential
        v0 : double
            The quasi-static disk potential relative to a ground electrode at
            infinity

        Returns
        -------
        pot : double
            The electric potential at (x, y, z).


        The electric potential :math:`V(r,z)` of a disk electrode is given by
        [WileyWebster1982]_:

        .. math::

            V(r,z) = \\sin^{-1} \\bigg\\{ \\frac{2a}{\\sqrt{(r-a)^2 + z^2} + \\sqrt{(r+a)^2 + z^2}} \\bigg\\} \\times \\frac{2 V_0}{\\pi},

        for :math:`z \\neq 0`, where :math:`r` and :math:`z` are the radial
        and axial distances from the center of the disk, :math:`V_0` is the
        disk potential, :math:`\\sigma` is the medium conductivity,
        and :math:`a` is the disk radius.

        """
        radial_dist = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        axial_dist = z - self.z
        if np.isclose(axial_dist, 0):
            # Potential on the electrode surface (Eq. 9 in Wiley & Webster):
            if radial_dist > self.r:
                # Outside the electrode:
                return 2.0 * v0 / np.pi * np.arcsin(self.r / radial_dist)
            else:
                # On the electrode:
                return v0
        else:
            # Off the electrode surface (Eq. 10):
            numer = 2 * self.r
            denom = np.sqrt((radial_dist - self.r) ** 2 + axial_dist ** 2)
            denom += np.sqrt((radial_dist + self.r) ** 2 + axial_dist ** 2)
            return 2.0 * v0 / np.pi * np.arcsin(numer / denom)


class SquareElectrode(Electrode):
    """Square electrode

    .. versionadded:: 0.7

    Parameters
    ----------
    x/y/z : double
        3D location that is the center of the square electrode
    a : double
        Side length of the square

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('a')

    def __init__(self, x, y, z, a):
        super(SquareElectrode, self).__init__(x, y, z)
        if isinstance(a, (Sequence, np.ndarray)):
            raise TypeError("Side length must be a scalar.")
        if a <= 0:
            raise ValueError("Side length must be > 0, not %f." % a)
        self.a = a
        self.plot_patch = Rectangle
        self.plot_kwargs = {'width': a, 'height': a, 'angle': 0,
                            'linewidth': 2, 'ec': (0.3, 0.3, 0.3, 1),
                            'fc': (0.8, 0.8, 0.8, 0.7)}

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'a': self.a})
        return params

    def electric_potential(self, x, y, z, v0):
        raise NotImplementedError


class HexElectrode(Electrode):
    """Hexagonal electrode

    .. versionadded:: 0.7

    Parameters
    ----------
    x/y/z : double
        3D location that is the center of the hexagonal electrode
    a : double
        Length of line drawn from the center of the hexagon to the midpoint of
        one of its sides.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('a')

    def __init__(self, x, y, z, a):
        super(HexElectrode, self).__init__(x, y, z)
        if isinstance(a, (Sequence, np.ndarray)):
            raise TypeError("Apothem of the hexagon must be a scalar.")
        if a <= 0:
            raise ValueError("Apothem of the hexagon must be > 0, not "
                             "%f." % a)
        self.a = a
        self.plot_patch = RegularPolygon
        self.plot_kwargs = {'numVertices': 6, 'radius': a, 'alpha': 0.2,
                            'orientation': np.radians(30), 'fc': 'k',
                            'ec': 'k'}

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'a': self.a})
        return params

    def electric_potential(self, x, y, z, v0):
        raise NotImplementedError
