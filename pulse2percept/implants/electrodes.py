""":py:class:`~pulse2percept.implants.Electrode`, 
   :py:class:`~pulse2percept.implants.PointSource`, 
   :py:class:`~pulse2percept.implants.DiskElectrode`, 
   :py:class:`~pulse2percept.implants.SquareElectrode`,
   :py:class:`~pulse2percept.implants.HexElectrode`"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon

from math import isclose
import numpy as np
from abc import ABCMeta, abstractmethod
# Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working:
from collections.abc import Sequence

from ..units import Quantity, as_value, deg, um
from ..utils import PrettyPrint
from ..utils.constants import ZORDER


#: Matplotlib orientation (rad) for horizontal and vertical hex grids.
_HEX_MPL_ORIENTATION = {'horizontal': 0.0, 'vertical': np.radians(30)}

#: Circumradius-to-apothem ratio of a regular hexagon.
_HEX_CIRCUMRADIUS = 1.0 / np.cos(np.radians(30))


def _is_nonscalar(value):
    """Whether ``value`` is a sequence where a single number is expected

    The ``float`` shortcut keeps grid building off the ABC
    ``__instancecheck__`` path: a float is never a sequence, and every
    coordinate an :py:class:`~pulse2percept.implants.ElectrodeGrid` lays out
    is a ``np.float64``, which is a float.
    """
    return (not isinstance(value, float) and
            isinstance(value, (Sequence, np.ndarray)))


class Electrode(PrettyPrint, metaclass=ABCMeta):
    """Electrode

    Abstract base class for all electrodes.

    Parameters
    ----------
    x/y/z : double
        3D location of the electrode (um).
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the right visual field.
        Positive ``y`` values move the electrode into the left visual field.
        Positive ``z`` values move the electrode either into the cortex or
        into the vitreos humor.
    name : str, optional
        Electrode name
    activated : bool
        To deactivate, set to ``False``. Deactivated electrodes cannot receive
        stimuli.

    Notes
    -----
    *  Coordinates may be given as plain numbers of microns or as unitful
       quantities (e.g. ``1.2 * mm``), which are converted to microns. See
       :py:mod:`pulse2percept.units`. Electrodes always *store* plain numbers
       in microns: :py:attr:`x`, :py:attr:`y` and :py:attr:`z` are ordinary
       floats, and so is everything downstream of them.
    """
    __slots__ = ('x', 'y', 'z', 'name', 'activated', 'plot_patch',
                 'plot_kwargs', 'plot_deactivated_kwargs')

    #: The unit electrode coordinates are stored in. Electrodes hold plain
    #: numbers, which is what every kernel downstream of them expects; this
    #: says what those numbers mean.
    coordinate_unit = um

    def __init__(self, x, y, z, name=None, activated=True):
        # Normalized before the checks below rather than after, so that a
        # quantity wrapping an array (``np.arange(3) * um``) is refused for the
        # same reason a bare array is, instead of being stored as one:
        x = as_value(x, um, 'x')
        y = as_value(y, um, 'y')
        z = as_value(z, um, 'z')
        if _is_nonscalar(x):
            raise TypeError(f"x must be a scalar, not {type(x)}.")
        if _is_nonscalar(y):
            raise TypeError(f"y must be a scalar, not {type(y)}.")
        if _is_nonscalar(z):
            raise TypeError(f"z must be a scalar, not {type(z)}.")
        self.x = x
        self.y = y
        self.z = z
        self.name = name
        self.activated = activated
        # A matplotlib.patches object (e.g., Circle, Rectangle) that can be
        # used to plot the electrode:
        self.plot_patch = None
        # Any keyword arguments that should be passed to the call above:
        # (e.g., {'radius': 5}):
        self.plot_kwargs = {}
        self.plot_deactivated_kwargs = {}

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'x': self.x, 'y': self.y, 'z': self.z, 'name': self.name,
                'activated': self.activated}

    def coordinates(self, unit=None):
        """3D position of the electrode

        .. versionadded:: 0.10.0

        Parameters
        ----------
        unit : :py:class:`~pulse2percept.units.Unit`, optional
            Length unit to express the position in. If None, the position is
            returned as it is stored (microns).

        Returns
        -------
        coords : (3,) np.ndarray
            An ordinary NumPy array ``[x, y, z]``, never a
            :py:class:`~pulse2percept.units.Quantity`.

        Examples
        --------
        >>> from pulse2percept.implants import DiskElectrode
        >>> from pulse2percept.units import mm
        >>> DiskElectrode(1000, 0, 100, 200).coordinates(mm)
        array([1. , 0. , 0.1])

        """
        xyz = np.array([self.x, self.y, self.z], dtype=float)
        if unit is None:
            return xyz
        return Quantity(xyz, self.coordinate_unit).to_value(unit)

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
        kwargs = self.plot_kwargs
        if not self.activated:
            kwargs = self.plot_deactivated_kwargs
        if self.plot_patch is not None:
            if isinstance(self.plot_patch, list):
                # Special case: draw multiple objects
                for p, kw in zip(self.plot_patch, kwargs):
                    ax.add_patch(p((self.x, self.y),
                                   zorder=ZORDER['foreground'], **kw))
            else:
                # Regular use case: single object
                ax.add_patch(self.plot_patch((self.x, self.y),
                                             zorder=ZORDER['foreground'],
                                             **kwargs))
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
        3D location of the electrode.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
    name : str, optional
        Electrode name
    activated : bool
        To deactivate, set to ``False``. Deactivated electrodes cannot receive
        stimuli.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ()

    def __init__(self, x, y, z, name=None, activated=True):
        super(PointSource, self).__init__(x, y, z, name=name,
                                          activated=activated)
        self.plot_patch = Circle
        self.plot_kwargs = {'radius': 5, 'linewidth': 2,
                            'ec': (0.3, 0.3, 0.3, 1),
                            'fc': (1, 1, 1, 0.8)}
        self.plot_deactivated_kwargs = {'radius': 5, 'linewidth': 2,
                                        'ec': (0.6, 0.6, 0.6, 1),
                                        'fc': (1, 1, 1, 0.6)}

    def electric_potential(self, x, y, z, amp, sigma):
        """Calculate electric potential at (x, y, z)

        Parameters
        ----------
        x/y/z : double
            3D location (um) at which to evaluate the electric potential.
            May be given as a unitful quantity, e.g. ``0.2 * mm``.
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
        # ``amp`` and ``sigma`` are deliberately left alone: current density
        # and resistivity are dimensions p2p has not defined, and inventing
        # them here to check two arguments would be worse than not checking.
        x = as_value(x, um, 'x')
        y = as_value(y, um, 'y')
        z = as_value(z, um, 'z')
        r = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2)
        if isclose(r, 0):
            return sigma * amp
        return sigma * amp / (4.0 * np.pi * r)


class DiskElectrode(Electrode):
    """Circular disk electrode

    Parameters
    ----------
    x/y/z : double
        3D location of the electrode.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
    radius : double
        Disk radius (um) in the x,y plane
    name : str, optional
        Electrode name
    activated : bool
        To deactivate, set to ``False``. Deactivated electrodes cannot receive
        stimuli.

    Notes
    -----
    *  Lengths may be given as plain numbers of microns or as unitful
       quantities (e.g. ``DiskElectrode(1 * mm, 0, 0.1 * mm, 200 * um)``). See
       :py:mod:`pulse2percept.units`.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('radius',)

    def __init__(self, x, y, z, radius, name=None, activated=True):
        super(DiskElectrode, self).__init__(x, y, z, name, activated=activated)
        radius = as_value(radius, um, 'radius')
        if _is_nonscalar(radius):
            raise TypeError("Electrode radius must be a scalar.")
        if radius <= 0:
            raise ValueError(f"Electrode radius must be > 0, not {radius}.")
        self.radius = radius
        self.plot_patch = Circle
        self.plot_kwargs = {'radius': radius, 'linewidth': 2,
                            'ec': (0.3, 0.3, 0.3, 1),
                            'fc': (1, 1, 1, 0.8)}
        self.plot_deactivated_kwargs = {'radius': radius, 'linewidth': 2,
                                        'ec': (0.6, 0.6, 0.6, 1),
                                        'fc': (1, 1, 1, 0.6)}

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'radius': self.radius})
        return params

    def electric_potential(self, x, y, z, v0):
        """Calculate electric potential at (x, y, z)

        Parameters
        ----------
        x/y/z : double
            3D location (um) at which to evaluate the electric potential.
            May be given as a unitful quantity, e.g. ``0.2 * mm``.
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
        # Only the location is normalized here; ``v0`` is an electrical
        # quantity, not a geometric one:
        x = as_value(x, um, 'x')
        y = as_value(y, um, 'y')
        z = as_value(z, um, 'z')
        radial_dist = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        axial_dist = z - self.z
        if isclose(axial_dist, 0):
            # Potential on the electrode surface (Eq. 9 in Wiley & Webster):
            if radial_dist > self.radius:
                # Outside the electrode:
                return 2.0 * v0 / np.pi * np.arcsin(self.radius / radial_dist)
            else:
                # On the electrode:
                return v0
        else:
            # Off the electrode surface (Eq. 10):
            numer = 2 * self.radius
            denom = np.sqrt((radial_dist - self.radius) ** 2 +
                            axial_dist ** 2)
            denom += np.sqrt((radial_dist + self.radius) ** 2 +
                             axial_dist ** 2)
            return 2.0 * v0 / np.pi * np.arcsin(numer / denom)


class SquareElectrode(Electrode):
    """Square electrode

    .. versionadded:: 0.7

    Parameters
    ----------
    x/y/z : double
        3D location of the electrode.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
    side_length : double
        Side length (um) of the square
    name : str, optional
        Electrode name
    activated : bool
        To deactivate, set to ``False``. Deactivated electrodes cannot receive
        stimuli.

    Notes
    -----
    *  Lengths may be given as plain numbers of microns or as unitful
       quantities (e.g. ``50 * um``). See :py:mod:`pulse2percept.units`.

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('side_length',)

    def __init__(self, x, y, z, side_length, name=None, activated=True):
        super(SquareElectrode, self).__init__(x, y, z, name=name,
                                              activated=activated)
        side_length = as_value(side_length, um, 'side_length')
        if _is_nonscalar(side_length):
            raise TypeError("Side length must be a scalar.")
        if side_length <= 0:
            raise ValueError(f"Side length must be > 0, not {side_length}.")
        self.side_length = side_length
        self.plot_patch = Rectangle
        self.plot_kwargs = {'width': side_length, 'height': side_length,
                            'angle': 0, 'linewidth': 2,
                            'ec': (0.3, 0.3, 0.3, 1),
                            'fc': (1, 1, 1, 0.8)}
        self.plot_deactivated_kwargs = {'width': side_length,
                                        'height': side_length, 'angle': 0,
                                        'linewidth': 2,
                                        'ec': (0.6, 0.6, 0.6, 1),
                                        'fc': (1, 1, 1, 0.6)}

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'side_length': self.side_length})
        return params

    def electric_potential(self, x, y, z, v0):
        raise NotImplementedError


class HexElectrode(Electrode):
    """Hexagonal electrode
    
    .. versionadded:: 0.7
    
    Parameters
    ----------
    x/y/z : double
        3D location of the electrode.
        The coordinate system is centered over the fovea.
        Positive ``x`` values move the electrode into the nasal retina.
        Positive ``y`` values move the electrode into the superior retina.
        Positive ``z`` values move the electrode away from the retina into the
        vitreous humor (sometimes called electrode-retina distance).
    apothem : double
        Apothem (um) of the hexagon. The flat-to-flat width is
        ``2 * apothem``.
    name : str, optional
        Electrode name.
    activated : bool
        To deactivate, set to ``False``. Deactivated electrodes cannot receive
        stimuli.
    orientation : {'horizontal', 'vertical'}, optional
        Hexagon orientation. Defaults to ``'vertical'``.
    
        .. versionadded:: 0.11.0
    rot : double, optional
        Rotation angle (deg, positive counter-clockwise).
    
        .. versionadded:: 0.11.0
    
    Notes
    -----
    *  Lengths may be given as plain numbers of microns or as unitful quantities.
       See :py:mod:`pulse2percept.units`.
    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('apothem', 'orientation', 'rot')

    def __init__(self, x, y, z, apothem, name=None, activated=True,
                 orientation='vertical', rot=0):
        super(HexElectrode, self).__init__(x, y, z, name=name,
                                           activated=activated)
        apothem = as_value(apothem, um, 'apothem')
        if _is_nonscalar(apothem):
            raise TypeError("Apothem of the hexagon must be a scalar.")
        if apothem <= 0:
            raise ValueError(f"Apothem of the hexagon must be > 0, not "
                             f"{apothem}.")
        if orientation not in _HEX_MPL_ORIENTATION:
            raise ValueError(f"'orientation' must be one of "
                             f"{sorted(_HEX_MPL_ORIENTATION)}, not "
                             f"'{orientation}'.")
        self.apothem = apothem
        self.orientation = orientation
        self.rot = as_value(rot, deg, 'rot')
        self.plot_patch = RegularPolygon
        self.plot_kwargs = {**self._hex_patch_kwargs(), 'alpha': 0.2,
                            'ec': (0.3, 0.3, 0.3, 1),
                            'fc': (1, 1, 1, 0.8)}
        self.plot_deactivated_kwargs = {**self._hex_patch_kwargs(),
                                        'alpha': 0.2,
                                        'ec': (0.6, 0.6, 0.6, 1),
                                        'fc': (1, 1, 1, 0.6)}

    def _hex_patch_kwargs(self):
        """Return Matplotlib ``RegularPolygon`` geometry for this hexagon."""
        return {'numVertices': 6,
                'radius': self.apothem * _HEX_CIRCUMRADIUS,
                'orientation': (_HEX_MPL_ORIENTATION[self.orientation] +
                                np.radians(self.rot))}

    @property
    def width(self):
        """Flat-to-flat width (um) of the hexagon
        
        .. versionadded:: 0.11.0
        """
        return 2 * self.apothem

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'apothem': self.apothem,
                       'orientation': self.orientation, 'rot': self.rot})
        return params

    def electric_potential(self, x, y, z, v0):
        raise NotImplementedError
