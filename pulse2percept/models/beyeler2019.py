""":py:class:`~pulse2percept.models.AxonMapModel`,
   :py:class:`~pulse2percept.models.AxonMapSpatial` [Beyeler2019]_"""

import os
import numpy as np
import pickle
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..units import deg, dva, um
from ..utils import deprecated_alias
from ..utils.constants import UM_PER_MM, ZORDER
from ..topography import Watson2014Map
from ..implants import ElectrodeArray
from ..stimuli import Stimulus
from ..models import Model, SpatialModel
from .base import _blend_meridian, _warn_ignores_z, _warn_rho_vs_pitch
from ._beyeler2019 import (fast_scoreboard, fast_axon_map, fast_jansonius,
                           fast_find_closest_axon)        

# Log all warnings.warn() at the WARNING level:
import warnings


#: Layout of the payload in ``axon_pickle``. Bump this whenever the tuple
#: written by ``AxonMapSpatial._build``, or the parameter dict alongside it,
#: changes shape or meaning, so that a cache left over from an older version
#: is regrown instead of misread.
_AXON_CACHE_VERSION = 3


def _is_axon_cache(payload):
    """Whether an ``axon_pickle`` payload is one this version can read"""
    return (isinstance(payload, tuple) and len(payload) == 4 and
            payload[0] == _AXON_CACHE_VERSION)


def _flatten_bundles(bundles):
    """Concatenate a list of bundles, collapsing repeated references

    ``find_closest_axon`` returns one entry per grid point, but those entries
    are references into a much smaller set of distinct bundles -- a few
    hundred for several thousand grid points -- and pickling preserves that
    sharing. Collapsing them keeps the flattened array, and the arc lengths
    :py:meth:`AxonMapSpatial._calc_axon_sensitivity_flat` accumulates over
    it, proportional to the number of *distinct* bundles rather than to the
    number of grid points.

    Bundles that are equal but are separate objects are treated as distinct.
    That costs a little speed and changes nothing about the result.

    Parameters
    ----------
    bundles : list of Nx2 arrays
        One bundle per point on the grid.

    Returns
    -------
    flat : (M, 2) array
        The distinct bundles, concatenated, in their original dtype.
    boff : (n_distinct + 1,) array
        Offsets of each distinct bundle into ``flat``.
    bundle_id : (len(bundles),) array
        Which distinct bundle each entry of ``bundles`` refers to.
    """
    seen = {}
    distinct = []
    bundle_id = np.empty(len(bundles), dtype=np.intp)
    for pos, bundle in enumerate(bundles):
        idx = seen.get(id(bundle))
        if idx is None:
            idx = seen[id(bundle)] = len(distinct)
            distinct.append(bundle)
        bundle_id[pos] = idx
    lens = np.array([len(bundle) for bundle in distinct], dtype=np.intp)
    if np.any(lens == 0):
        raise ValueError("Every bundle must have at least one segment.")
    flat = np.ascontiguousarray(np.concatenate(distinct))
    return flat, np.concatenate(([0], np.cumsum(lens))), bundle_id


class ScoreboardSpatial(SpatialModel):
    """Scoreboard model of [Beyeler2019]_ (spatial module only)

    Implements the scoreboard model described in [Beyeler2019]_, where all
    percepts are Gaussian blobs.

    .. note ::

        Use this class if you want to combine the spatial model with a temporal
        model.
        Use :py:class:`~pulse2percept.models.ScoreboardModel` if you want a
        a standalone model.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        The device this model predicts percepts for. Required: a percept is
        what a particular implant produces, and ``predict_percept`` takes what
        is presented to that device.

        .. versionadded:: 0.11.0

    rho : double, optional
        Exponential decay constant describing phosphene size (microns).
    min_current_spread : float, optional
        An electrode is skipped at grid points where its Gaussian current
        spread has decayed below this fraction of its peak. The default
        (1e-8, about 6.1 ``rho`` away) drops the Gaussian *times* the 
        stimulus amplitude, summed over the skipped electrodes, so the error
        at a point is bounded by ``min_current_spread`` times the summed 
        amplitude across electrodes.
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : tuple, (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    step : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
        and y axes different step sizes.

        .. versionchanged:: 0.10.0

            Renamed from ``xystep``, which suggested that one step size
            applies to both axes. The old name still works, but is
            deprecated and will be removed in v0.11.0.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
        that provides retinotopic mappings.
        By default, :py:class:`~pulse2percept.topography.Watson2014Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    noise : float or int, optional
        Adds salt-and-pepper noise to each percept frame. An integer will be
        interpreted as the number of pixels to subject to noise in each frame.
        A float between 0 and 1 will be interpreted as a ratio of pixels to
        subject to noise in each frame.

    n_threads : int, optional
        Number of CPU threads to use during parallelization using OpenMP.
        Defaults to max number of user CPU cores.
    n_jobs : int, optional
        Alias for ``n_threads``; ``None`` or ``-1`` uses every core.

    .. important ::
        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.xrange = (-10, 10)``) un-builds the model, and the next
        ``predict_percept`` builds it again.
    """

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        base_params = super(ScoreboardSpatial, self).get_default_params()
        params = {'rho': 100, 'vfmap': Watson2014Map()}
        return {**base_params, **params}

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        return {**super().get_param_units(), 'rho': um}

    def _build(self):
        _warn_rho_vs_pitch(self)

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""
        _warn_ignores_z(self, earray)
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        x_el, y_el, _ = self._electrode_coords(earray, stim)
        return fast_scoreboard(self._stim_values(stim), x_el, y_el,
                               self.grid.ret.x.ravel(),
                               self.grid.ret.y.ravel(),
                               self.rho,
                               self.thresh_percept,
                               self._cutoff_r2(self.rho),
                               0, 0, # don't set current boundaries
                               self.n_threads)


class ScoreboardModel(Model):
    """Scoreboard model of [Beyeler2019]_ (standalone model)

    Implements the scoreboard model described in [Beyeler2019]_, where all
    percepts are Gaussian blobs.

    .. note ::

        Use this class if you want a standalone model.
        Use :py:class:`~pulse2percept.models.ScoreboardSpatial` if you want
        to combine the spatial model with a temporal model.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        The device this model predicts percepts for. Required: a percept is
        what a particular implant produces, and ``predict_percept`` takes what
        is presented to that device.

        .. versionadded:: 0.11.0

    rho : double, optional
        Exponential decay constant describing phosphene size (microns).
    min_current_spread : float, optional
        An electrode is skipped at grid points where its Gaussian current
        spread has decayed below this fraction of its peak. The default
        (1e-8, about 6.1 ``rho`` away) drops the Gaussian *times* the
        stimulus amplitude, summed over the skipped electrodes, so the error
        at a point is bounded by ``min_current_spread`` times the summed
        amplitude across electrodes.
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : tuple, (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    step : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
        and y axes different step sizes.

        .. versionchanged:: 0.10.0

            Renamed from ``xystep``, which suggested that one step size
            applies to both axes. The old name still works, but is
            deprecated and will be removed in v0.11.0.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
        object that provides retinotopic mappings.
        By default, :py:class:`~pulse2percept.topography.Watson2014Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    noise : float or int, optional
        Adds salt-and-pepper noise to each percept frame. An integer will be
        interpreted as the number of pixels to subject to noise in each frame.
        A float between 0 and 1 will be interpreted as a ratio of pixels to
        subject to noise in each frame.
    n_threads : int, optional
        Number of CPU threads to use during parallelization using OpenMP.
        Defaults to max number of user CPU cores.
    n_jobs : int, optional
        Alias for ``n_threads``; ``None`` or ``-1`` uses every core.

    .. important ::
        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.xrange = (-10, 10)``) un-builds the model, and the next
        ``predict_percept`` builds it again.

    """

    def __init__(self, **params):
        super(ScoreboardModel, self).__init__(spatial=ScoreboardSpatial(),
                                              temporal=None,
                                              **params)


class AxonMapSpatial(SpatialModel):
    """Axon map model of [Beyeler2019]_ (spatial module only)

    Implements the axon map model described in [Beyeler2019]_, where percepts
    are elongated along nerve fiber bundle trajectories of the retina.

    .. note: :

        Use this class if you want to combine the spatial model with a temporal
        model.
        Use: py: class: `~pulse2percept.models.AxonMapModel` if you want a
        a standalone model.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        The device this model predicts percepts for. Required: a percept is
        what a particular implant produces, and ``predict_percept`` takes what
        is presented to that device.

        .. versionadded:: 0.11.0

    lam : double, optional
        Exponential decay constant along the axon(microns).

        .. versionchanged:: 0.10.0

            Renamed from ``axlambda``, which reads poorly next to ``rho``. The
            old name still works, but is deprecated and will be removed in
            v0.11.0.
    rho : double, optional
        Exponential decay constant away from the axon(microns).
    min_current_spread : float, optional
        An electrode is skipped at axon segments where its Gaussian current
        spread has decayed below this fraction of its peak. The default
        (1e-8, about 6.1 ``rho`` away) drops the Gaussian *times* the stimulus
        amplitude, summed over the skipped electrodes, so the error at a point
        is bounded by ``min_current_spread`` times the summed amplitude across
        electrodes.
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    step : int or double or tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
        and y axes different step sizes.

        .. versionchanged:: 0.10.0

            Renamed from ``xystep``, which suggested that one step size
            applies to both axes. The old name still works, but is
            deprecated and will be removed in v0.11.0.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
        object that provides retinotopic mappings.
        By default, :py:class:`~pulse2percept.topography.Watson2014Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    noise : float or int, optional
        Adds salt-and-pepper noise to each percept frame. An integer will be
        interpreted as the number of pixels to subject to noise in each frame.
        A float between 0 and 1 will be interpreted as a ratio of pixels to
        subject to noise in each frame.
    loc_od, loc_od : (x,y), optional
        Location of the optic disc in degrees of visual angle. Note that the
        optic disc in a left eye will be corrected to have a negative x
        coordinate.
    n_axons : int, optional
        Number of axons to generate.
    axons_range : (min, max) of float or Quantity, optional
        The range of angles(in degrees) at which axons exit the optic disc.
        This corresponds to the range of $\\phi_0$ values used in
        [Jansonius2009]_.
    n_ax_segments : int, optional
        Number of segments an axon is made of.
    ax_segments_range : (min, max), optional
        Lower and upper bounds for the radial position values(polar coords)
        for each axon.
    min_ax_sensitivity : float, optional
        Axon segments whose contribution to brightness is smaller than this
        value will be pruned to improve computational efficiency. Set to a
        value between 0 and 1.
    meridian_blend : float, optional
        Gaussian standard deviation (dva) for smoothing across the horizontal
        meridian. Default: 1. Set to 0 to disable.

        .. versionadded:: 0.10.0
    axon_pickle : str, optional
        File name in which to store precomputed axon maps.
    ignore_pickle : bool, optional
        A flag whether to ignore the pickle file in future calls to
        ``model.build()``.
    n_threads : int, optional
        Number of CPU threads to use during parallelization using OpenMP.
        Defaults to max number of user CPU cores.
    n_jobs : int, optional
        Alias for ``n_threads``; ``None`` or ``-1`` uses every core.

    .. important ::
        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.lam = 100``) un-builds the model, and the next
        ``predict_percept`` builds it again.

    Notes
    -----
    *  The axon map is not very accurate when the upper bound of
       `ax_segments_range` is greater than 90 deg.
    """

    #: ``lam`` used to be called ``axlambda``. The old name still reads and
    #: writes ``lam``, with a ``DeprecationWarning``:
    axlambda = deprecated_alias('lam', deprecated_version='0.10.0',
                                removed_version='0.11.0')

    def __init__(self, **params):
        super(AxonMapSpatial, self).__init__(**params)
        self.axon_contrib = None
        self.axon_idx_start = None
        self.axon_idx_end = None
        self._built_eye = None

    @property
    def eye(self):
        """The eye the axon map is grown for, which is the implanted one

        .. versionchanged:: 0.11.0

            No longer a parameter of its own. An axon map describes the retina
            a particular device sits on, so the bound implant is what says
            which eye, and the two can no longer disagree.
        """
        self._require_implant()
        return self.implant.eye

    @property
    def is_built(self):
        """False again once the bound implant has changed eyes

        The one build-invalidating change the parameter machinery cannot see:
        ``eye`` is not a parameter, and the implant is the same object it was
        built with.
        """
        return super().is_built and self._built_eye == self.implant.eye

    def get_default_params(self):
        base_params = super(AxonMapSpatial, self).get_default_params()
        params = {
            'rho': 300,
            'lam': 500,
            # Set the (x,y) location of the optic disc:
            'loc_od': (15.5, 1.5),
            'n_axons': 1000,
            'axons_range': (-180, 180),
            # Number of sampling points along the radial axis (polar coords):
            'n_ax_segments': 500,
            # Lower and upper bounds for the radial position values (polar
            # coordinates):
            'ax_segments_range': (0, 50),
            # Axon segments whose contribution to brightness is smaller than
            # this value will be pruned:
            'min_ax_sensitivity': 1e-3,
            # Meridian blend width (dva); 0 disables:
            'meridian_blend': 1,
            # Precomputed axon maps stored in the following file:
            'axon_pickle': 'axons.pickle',
            # You can force a build by ignoring pickles:
            'ignore_pickle': False,
            # Use the Watson transform for dva <=> ret:
            'vfmap': Watson2014Map()
        }
        return {**base_params, **params}

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        # `axons_range` is a range of ordinary polar angles, not visual angle;
        # `ax_segments_range` is a radial position in the Jansonius model's own
        # coordinates, so it stays undeclared:
        return {**super().get_param_units(), 'rho': um, 'lam': um,
                'loc_od': dva, 'meridian_blend': dva, 'axons_range': deg}

    def _jansonius2009(self, phi0, beta_sup=-1.9, beta_inf=0.5, eye='RE'):
        """Grows a single axon bundle based on the model by Jansonius (2009)

        This function generates the trajectory of a single nerve fiber bundle
        based on the mathematical model described in [Jansonius2009]_.

        Parameters
        ----------
        phi0: float
            Angular position of the axon at its starting point(polar
            coordinates, degrees). Must be within[-180, 180].
        beta_sup: float, optional
            Scalar value for the superior retina(see Eq. 5, `\beta_s` in the
            paper).
        beta_inf: float, optional
            Scalar value for the inferior retina(see Eq. 6, `\beta_i` in the
            paper.)

        Returns
        -------
        ax_pos: Nx2 array
            Returns a two - dimensional array of axonal positions, where
            ax_pos[0, :] contains the(x, y) coordinates of the axon segment
            closest to the optic disc, and aubsequent row indices move the axon
            away from the optic disc. Number of rows is at most ``n_rho``, but
            might be smaller if the axon crosses the meridian.

        Notes
        -----
        The study did not include axons with phi0 in [-60, 60] deg.

        """
        # Check for the location of the optic disc:
        loc_od = self.loc_od
        if eye.upper() not in ['LE', 'RE']:
            e_s = f"Unknown eye string '{eye}': Choose from 'LE', 'RE'."
            raise ValueError(e_s)
        if eye.upper() == 'LE':
            # The Jansonius model doesn't know about left eyes: We invert the x
            # coordinate of the optic disc here, run the model, and then invert
            # all x coordinates of all axon fibers back.
            loc_od = (-loc_od[0], loc_od[1])
        if np.abs(phi0) > 180.0:
            raise ValueError('phi0 must be within [-180, 180].')
        if self.n_ax_segments < 1:
            raise ValueError('Number of radial sampling points must be >= 1.')
        if np.any(np.array(self.ax_segments_range) < 0):
            raise ValueError('ax_segments_range cannot be negative.')
        if self.ax_segments_range[0] > self.ax_segments_range[1]:
            raise ValueError('Lower bound on rho cannot be larger than the '
                             ' upper bound.')
        is_superior = phi0 > 0
        rho = np.linspace(*self.ax_segments_range, num=self.n_ax_segments,
                          dtype=np.float32)
        xprime, yprime = fast_jansonius(rho, phi0, beta_sup, beta_inf)
        # Find the array elements where the axon crosses the meridian:
        if is_superior:
            # Find elements in inferior retina
            idx = np.where(yprime < 0)[0]
        else:
            # Find elements in superior retina
            idx = np.where(yprime > 0)[0]
        if idx.size:
            # Keep only up to first occurrence
            xprime = xprime[:idx[0]]
            yprime = yprime[:idx[0]]
        # Adjust coordinate system, having fovea=[0, 0] instead of
        # `loc_od`=[0, 0]:
        xmodel = xprime + loc_od[0]
        ymodel = yprime
        if loc_od[0] > 0:
            # If x-coordinate of optic disc is positive, use Appendix A
            idx = xprime > -loc_od[0]
        else:
            # Else we need to flip the sign
            idx = xprime < -loc_od[0]
        ymodel[idx] = yprime[idx] + loc_od[1] * (xmodel[idx] / loc_od[0]) ** 2
        # In a left eye, need to flip back x coordinates:
        if eye.upper() == 'LE':
            xmodel *= -1
        # Return as Nx2 array:
        return np.vstack((xmodel, ymodel)).astype(np.float32).T

    def grow_axon_bundles(self, n_bundles=None, prune=True):
        """Grow a number of axon bundles

        This method generates the trajectory of a number of nerve fiber
        bundles based on the mathematical model described in [Beyeler2019]_,
        which is based on [Jansonius2009]_.

        Bundles originate at the optic nerve head with initial angle ``phi0``.
        The method generates ``n_bundles`` axon bundles whose ``phi0`` values
        are linearly sampled from ``self.axons_range`` (polar coords).
        Each axon will consist of ``self.n_ax_segments`` segments that span
        ``self.ax_segments_range`` distance from the optic nerve head (polar
        coords).

        Parameters
        ----------
        n_bundles : int, optional
            Number of axon bundles to generate. If None, ``self.n_axons`` is
            used
        prune : bool, optional
            If set to True, will remove axon segments that are outside the
            simulated area ``self.xrange``, ``self.yrange`` for the sake of
            computational efficiency.

        Returns
        -------
        bundles : list of Nx2 arrays
            A list of bundles, where every bundle is an Nx2 array consisting of
            the x,y coordinates of each axon segment (retinal coords, microns). 
            Note that each bundle will most likely have a different N

        """
        if n_bundles is None:
            n_bundles = self.n_axons
        # Build the Jansonius model: Grow a number of axon bundles in all dirs:
        phi = np.linspace(*self.axons_range, num=n_bundles)
        bundles = [self._jansonius2009(p, eye=self.eye) for p in phi]
        # Keep only non-zero sized bundles:
        bundles = list(filter(lambda x: len(x) > 0, bundles))
        if prune:
            # Remove axon bundles outside the simulated area:
            xmin, xmax = self.xrange
            ymin, ymax = self.yrange
            bundles = list(filter(lambda x: (np.max(x[:, 0]) >= xmin and
                                             np.min(x[:, 0]) <= xmax and
                                             np.max(x[:, 1]) >= ymin and
                                             np.min(x[:, 1]) <= ymax),
                                  bundles))
            # Keep only reasonably sized axon bundles:
            bundles = list(filter(lambda x: len(x) > 10, bundles))
        # Convert to um:
        bundles = [np.array(self.vfmap.dva_to_ret(b[:, 0], b[:, 1])).T
                   for b in bundles]
        return bundles

    def find_closest_axon(self, bundles, xret=None, yret=None,
                          return_index=False, return_segment=False):
        """Finds the closest axon segment for a point on the retina

        This function will search a number of nerve fiber bundles (``bundles``)
        and return the bundle that is closest to a particular point (or list of
        points) on the retinal surface (``xret``, ``yret``).

        Parameters
        ----------
        bundles : list of Nx2 arrays
            A list of bundles, where every bundle is an Nx2 array consisting of
            the x,y coordinates of each axon segment (retinal coords, microns).
            Note that each bundle will most likely have a different N
        xret, yret : scalar or list of scalars
            The x,y location on the retina (in microns, where the fovea is the
            origin) for which to find the closests axon.
        return_index : bool, optional
            If True, the function will also return the index into ``bundles``
            that represents the closest axon
        return_segment : bool, optional
            If True, the function will also return the row index, within the
            closest bundle, of the segment nearest the point. The search
            already determines this, so asking for it here saves
            :py:meth:`calc_axon_sensitivity` from working it out again.

        Returns
        -------
        axon : Nx2 array or list of Nx2 arrays
            For each point in (xret, yret), returns an Nx2 array that represents
            the closest axon to that point. Each row in the array contains the
            x,y retinal coordinates (microns) of a particular axon segment.
        idx_axon : scalar or list of scalars, optional
            If ``return_index`` is True, also returns the index in ``bundles``
            of the closest axon (or list of closest axons).
        idx_segment : scalar or list of scalars, optional
            If ``return_segment`` is True, also returns the row index of the
            closest segment within that axon.

        """
        if len(bundles) <= 0:
            raise ValueError("bundles must have length greater than zero")
        if xret is None:
            xret = self.grid.ret.x
        if yret is None:
            yret = self.grid.ret.y
        xret = np.asarray(xret, dtype=np.float32)
        yret = np.asarray(yret, dtype=np.float32)
        # Offsets of each bundle into the concatenation of all of them, which
        # is what the tree is built over. `searchsorted` on these turns a
        # segment's index in that flat array back into the bundle it came
        # from, so there is no need to materialize a bundle ID per segment:
        boff = np.concatenate(([0], np.cumsum([len(ax) for ax in bundles])))
        flat_bundles = np.concatenate(bundles)
        kdtree = cKDTree(flat_bundles, leafsize=60)
        # Create query list of xy pairs
        query = np.stack((xret.ravel(), yret.ravel()), axis=1)
        # Find index of closest segment with the model's thread budget:
        _, closest_seg = kdtree.query(query, workers=max(1, self.n_threads))

        # Look up the axon ID for every axon segment:
        closest_idx = (np.searchsorted(boff, closest_seg, side='right') -
                       1).astype(np.uint32)
        # ...and where within that bundle the closest segment sits:
        idx_segment = closest_seg - boff[closest_idx]
        if len(closest_idx) == 1:
            closest_idx = closest_idx[0]
            idx_segment = idx_segment[0]
            closest_axon = bundles[closest_idx]
        else:
            closest_axon = [bundles[n] for n in closest_idx]
        if return_index and return_segment:
            return closest_axon, closest_idx, idx_segment
        if return_segment:
            return closest_axon, idx_segment
        if return_index:
            return closest_axon, closest_idx
        return closest_axon

    def calc_axon_sensitivity(self, bundles):
        """Calculate the sensitivity of each axon segment to electrical current

        This function combines the x,y coordinates of each bundle segment with
        a sensitivity value that depends on the distance of the segment to the
        cell body and ``self.lam``.

        The number of ``bundles`` must equal the number of points on
        `self.grid``. The function will then assume that the i-th bundle passes
        through the i-th point on the grid. This is used to determine the bundle
        segment that is closest to the i-th point on the grid, and to cut off
        all segments that extend beyond the soma. This effectively transforms
        a *bundle* into an *axon*, where the first axon segment now corresponds
        with the i-th location of the grid.

        After that, each axon segment gets a sensitivity value that depends
        on the distance of the segment to the soma (with decay rate
        ``self.lam``). This is typically done during the build process, so
        that the only work left to do during run time is to multiply the
        sensitivity value with the current applied to each segment.

        Parameters
        ----------
        bundles : list of Nx2 arrays
            A list of bundles, where every bundle is an Nx2 array consisting of
            the x,y coordinates of each axon segment (retinal coords, microns).
            Note that each bundle will most likely have a different N

        Returns
        -------
        axon_contrib : list of Nx3 arrays
            A list with one entry per point on ``self.grid``. Each entry is a
            Nx3 array, where the first two columns contain the retinal
            coordinates of each axon segment (microns), and the third column
            contains the sensitivity of the segment to electrical current.
            The latter depends on ``self.lam``. Note that each axon will
            most likely have a different N, since segments whose sensitivity
            falls below ``min_ax_sensitivity`` are trimmed.

        """
        contrib, starts = self._calc_axon_sensitivity_flat(*_flatten_bundles(
            bundles))
        return [contrib[lo:hi] for lo, hi in zip(starts[:-1], starts[1:])]

    def _calc_axon_sensitivity_flat(self, flat, boff, bundle_id, seg=None):
        """Vectorized core of :py:meth:`calc_axon_sensitivity`

        Computes every axon at once rather than one grid point at a time. Two
        facts make that possible. An axon is a *contiguous* run of its
        bundle's segments, running back from the one nearest the soma: the
        distance walked from the soma only grows as you move away from it, so
        the segments that survive the ``min_ax_sensitivity`` trim are a
        prefix of that walk. And the arc length along a bundle is a property
        of the bundle, not of the grid point, so it can be accumulated once
        per bundle and reused by every grid point that picked it.

        Parameters
        ----------
        flat : (M, 2) array
            All distinct bundles, concatenated.
        boff : (n_bundles + 1,) array
            Offsets of each distinct bundle into ``flat``.
        bundle_id : (n_points,) array
            Index of the bundle belonging to each point on ``self.grid``.
        seg : (n_points,) array, optional
            Index into ``flat`` of the segment nearest each grid point. The
            nearest-neighbor search in :py:meth:`find_closest_axon` already
            knows this, so ``_build`` passes it through rather than pay for
            the search below a second time. Derived here when not given,
            which is the case for the public entry point.

        Returns
        -------
        contrib : (N, 3) array
            Every axon's segments, concatenated: x, y, sensitivity.
        starts : (n_points + 1,) array
            Offsets of each axon into ``contrib``.
        """
        lam = self.lam
        xyret = np.column_stack((self.grid.ret.x.ravel(),
                                 self.grid.ret.y.ravel()))
        blens = np.diff(boff)
        # Bundles arrive as float32. Accumulating arc length over hundreds of
        # segments in float32 loses more than the coordinates are worth, so
        # the geometry below runs in float64 and only the result is handed
        # back at the caller's precision:
        out_dtype = np.promote_types(flat.dtype, np.float32)
        flat = flat.astype(np.float64, copy=False)
        xyret = xyret.astype(np.float64, copy=False)

        # Arc length from the start of a bundle to each of its segments. Laid
        # out like `flat`, and reset to zero wherever a new bundle begins:
        step = np.hypot(*(flat[1:] - flat[:-1]).T)
        step[boff[1:-1] - 1] = 0.0  # never walk across a seam between bundles
        arc = np.empty(len(flat))
        arc[0] = 0.0
        np.cumsum(step, out=arc[1:])
        arc -= np.repeat(arc[boff[:-1]], blens)

        base = boff[bundle_id]
        if seg is None:
            # Distance from every grid point to every segment of *its*
            # bundle, as one flat array of (point, segment) pairs:
            lens = blens[bundle_id]
            pair_off = np.concatenate(([0], np.cumsum(lens)))
            within = np.arange(pair_off[-1]) - np.repeat(pair_off[:-1], lens)
            pairs = np.repeat(base, lens) + within
            d2 = ((flat[pairs, 0] - np.repeat(xyret[:, 0], lens)) ** 2 +
                  (flat[pairs, 1] - np.repeat(xyret[:, 1], lens)) ** 2)
            # The segment closest to the soma, resolving ties towards the
            # lower index the way ``np.argmin`` does:
            closest = np.repeat(np.minimum.reduceat(d2, pair_off[:-1]), lens)
            cand = np.where(d2 <= closest, within, np.iinfo(np.intp).max)
            seg = base + np.minimum.reduceat(cand, pair_off[:-1])
        else:
            seg = np.asarray(seg, dtype=np.intp)

        # Walking out from the soma, segment `k` of the bundle sits at
        # `d0 + (arc[seg] - arc[k])`. Only include segments closer than
        # `max_d2`; those are the ones whose sensitivity stays above
        # `min_ax_sensitivity`:
        d0 = np.hypot(flat[seg, 0] - xyret[:, 0], flat[seg, 1] - xyret[:, 1])
        max_d2 = -2.0 * lam ** 2 * np.log(self.min_ax_sensitivity)
        if max_d2 <= 0:
            # Not even the soma itself clears the bar, so every axon is empty:
            n_keep = np.zeros(len(bundle_id), dtype=np.intp)
        else:
            # That distance falls as `k` rises, so the segments to keep are a
            # contiguous run ending at `seg`, and its lower end is one
            # searchsorted away. `arc` only increases *within* a bundle, so
            # offset each bundle past the last to make the array monotone
            # overall and the whole lookup a single vectorized call:
            span = arc[boff[1:] - 1].max() + 1.0
            offset = bundle_id * span
            lo = np.searchsorted(arc + np.repeat(np.arange(len(blens)) * span,
                                                 blens),
                                 arc[seg] + d0 - np.sqrt(max_d2) + offset,
                                 side='right')
            n_keep = np.maximum(seg - np.maximum(lo, base) + 1, 0)

        starts = np.concatenate(([0], np.cumsum(n_keep)))
        # Segments run *back* from the one nearest the soma:
        gather = (np.repeat(seg, n_keep) -
                  (np.arange(starts[-1]) - np.repeat(starts[:-1], n_keep)))
        dist = np.repeat(d0 + arc[seg], n_keep) - arc[gather]
        contrib = np.empty((starts[-1], 3), dtype=out_dtype)
        contrib[:, :2] = flat[gather]
        contrib[:, 2] = np.exp(-dist ** 2 / (2.0 * lam ** 2))
        return contrib, starts

    def calc_bundle_tangent(self, xc, yc):
        """Calculates orientation of fiber bundle tangent at (xc, yc)

        Parameters
        ----------
        xc, yc: float
            (x, y) retinal location of point at which to calculate bundle 
            orientation in microns.

        Returns
        -------
        tangent : scalar
            An angle in radians
        """
        # Check for scalar:
        if isinstance(xc, (list, np.ndarray)):
            raise TypeError("xc must be a scalar")
        if isinstance(yc, (list, np.ndarray)):
            raise TypeError("yc must be a scalar")
        # Find the fiber bundle closest to (xc, yc):
        bundles = self.grow_axon_bundles()
        bundle = self.find_closest_axon(bundles, xret=xc, yret=yc)
        # For that bundle, find the bundle segment closest to (xc, yc):
        idx = np.argmin((bundle[:, 0] - xc) ** 2 + (bundle[:, 1] - yc) ** 2)
        # Calculate orientation from atan2(dy, dx):
        if idx == 0:
            # Bundle index 0: there's no index -1
            dx = bundle[1, :] - bundle[0, :]
        elif idx == bundle.shape[0] - 1:
            # Bundle index -1: there's no index len(bundle)
            dx = bundle[-1, :] - bundle[-2, :]
        else:
            # Else: Look at previous and subsequent segments:
            dx = (bundle[idx + 1, :] - bundle[idx - 1, :]) / 2
        dx[1] *= -1
        tangent = np.arctan2(*dx[::-1])
        # Confine to (-pi/2, pi/2):
        if tangent < np.deg2rad(-90):
            tangent += np.deg2rad(180)
        if tangent > np.deg2rad(90):
            tangent -= np.deg2rad(180)
        return tangent
    

    def calc_bundle_tangent_fast(self, xc, yc, bundles=None):
        """Calculates orientation of fiber bundle tangent at (xc, yc)
        This function supports multiple queries (xc and yc can be arrays), without
        requiring growing the axon bundles again for each point (like calc_bundle_tangent).
        It uses a ckdtree, which will be slower for single points, but significantly faster 
        for multiple points. 

        Parameters
        ----------
        xc, yc: array of floats
            (x, y) retinal location of point at which to calculate bundle 
            orientation in microns.

        Returns
        -------
        tangent : array of floats
            Angles in radians
        """

        if bundles is None:
            bundles = self.grow_axon_bundles()
        xc = np.asarray(xc, dtype=np.float32)
        yc = np.asarray(yc, dtype=np.float32)
        # For every axon segment, store the corresponding axon ID:
        axon_idx = [[idx] * len(ax) for idx, ax in enumerate(bundles)]
        axon_idx = [item for sublist in axon_idx for item in sublist]
        axon_idx = np.array(axon_idx, dtype=np.uint32)
        # Build a long list of all axon segments - their corresponding axon IDs
        # is given by `axon_idx` above:
        flat_bundles = np.concatenate(bundles)
        kdtree = cKDTree(flat_bundles, leafsize=60)
        # Create query list of xy pairs
        query = np.stack((xc.ravel(), yc.ravel()), axis=1)
        # Find index of closest segment
        _, closest_seg = kdtree.query(query)
        segs = axon_idx[closest_seg]
        prev_segs = axon_idx[np.where(closest_seg > 0, closest_seg, 1) - 1]
        next_segs = axon_idx[np.where(closest_seg < len(axon_idx)-2, closest_seg, len(axon_idx)-2) + 1]

        offset_l = np.where(prev_segs == segs, -1, 0)
        offset_r = np.where(next_segs == segs, 1, 0)
        dx = flat_bundles[np.minimum(closest_seg + offset_r, len(flat_bundles)-1)] - flat_bundles[np.maximum(closest_seg + offset_l, 0)]

        dx[:, 1] *= -1
        tangent = np.arctan2(dx[:, 1], dx[:, 0])

        # Confine to (-pi/2, pi/2):
        tangent = np.where(tangent < -np.pi/2, tangent+np.pi, tangent)
        tangent = np.where(tangent > np.pi/2, tangent - np.pi, tangent)
        return tangent.reshape(xc.shape)


    def _warn_placement(self):
        """Warn when the implant is not where nerve fiber bundles run"""
        placement = self.implant.placement
        if placement is None or placement == 'epiretinal':
            return
        warnings.warn(
            f"{type(self).__name__} predicts elongated percepts because an "
            f"epiretinal array stimulates passing nerve fiber bundles. This "
            f"implant is {placement}, where that mechanism does not apply, so "
            f"the streaks below are an artifact of the model rather than a "
            f"prediction about the device. A placement-appropriate "
            f"local-response scoreboard model is a safer phenomenological "
            f"starting point.")

    def _correct_loc_od(self):
        """Put the optic disc on the nasal side of whichever eye this is"""
        sign = -1 if self.eye == 'LE' else 1
        self.loc_od = (sign * np.abs(self.loc_od[0]), self.loc_od[1])

    def _build(self):
        if self.lam < 10:
            raise ValueError('"lam" < 10 is not supported by this model. '
                             'Consider using ScoreboardModel instead.')
        self._warn_placement()
        _warn_rho_vs_pitch(self)
        self._built_eye = self.implant.eye
        # In a left eye, the OD must have a negative x coordinate:
        self._correct_loc_od()
        # Check whether pickle file needs to be rebuilt:
        need_axons = False
        cached = None
        if self.ignore_pickle:
            need_axons = True
        else:
            # Check if math for Jansonius model has been done before:
            if os.path.isfile(self.axon_pickle):
                params, cached = pickle.load(open(self.axon_pickle, 'rb'))
                # A cache written by an older version stores something else
                # here. Rather than try to read it, grow the bundles again and
                # overwrite it; the file is derived data, so the only cost is
                # one slow build. Settle this *before* looking at `params`,
                # whose keys are versioned too -- an old cache names the grid
                # step `xystep`, and probing that here would warn the user
                # about a name they never used:
                if not _is_axon_cache(cached):
                    need_axons = True
                else:
                    for key, value in params.items():
                        if (not hasattr(self, key) or
                                not np.allclose(getattr(self, key), value)):
                            need_axons = True
                            break
            else:
                need_axons = True
        # Build the Jansonius model: Grow a number of axon bundles in all dirs:
        if need_axons:
            bundles = self.grow_axon_bundles()
            _, bundle_id, idx_segment = self.find_closest_axon(
                bundles, return_index=True, return_segment=True)
            bundle_id = np.atleast_1d(bundle_id).astype(np.intp)
            idx_segment = np.atleast_1d(idx_segment).astype(np.intp)
            # Grid points cluster onto a fraction of the bundles that were
            # grown. Dropping the rest keeps the cache small and the arc
            # lengths below proportional to what is actually used:
            used, bundle_id = np.unique(bundle_id, return_inverse=True)
            bundles = [bundles[idx] for idx in used]
            bundle_id = np.ravel(bundle_id).astype(np.intp)
        else:
            _, bundles, bundle_id, idx_segment = cached
        # Calculate axon contributions. A list of (differently shaped) NumPy
        # arrays cannot be accessed in parallel without the gil, so the axons
        # come back already concatenated into a really long Nx3 array, along
        # with the start and end indices of each slice:
        flat, boff, _ = _flatten_bundles(bundles)
        axon_contrib, starts = self._calc_axon_sensitivity_flat(
            flat, boff, bundle_id, seg=boff[bundle_id] + idx_segment)
        self.axon_contrib = np.ascontiguousarray(axon_contrib,
                                                 dtype=np.float32)
        self.axon_idx_start = starts[:-1]
        self.axon_idx_end = starts[1:]
        if need_axons:
            # Pickle axons along with all important parameters:
            params = {'loc_od': self.loc_od,
                      'n_axons': self.n_axons, 'axons_range': self.axons_range,
                      'xrange': self.xrange, 'yrange': self.yrange,
                      'step': self.step, 'n_ax_segments': self.n_ax_segments,
                      'ax_segments_range': self.ax_segments_range}
            pickle.dump((params, (_AXON_CACHE_VERSION, bundles, bundle_id,
                                  idx_segment)),
                        open(self.axon_pickle, 'wb'))

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at specific times ``t``"""
        _warn_ignores_z(self, earray)
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        x_el, y_el, _ = self._electrode_coords(earray, stim)
        return fast_axon_map(self._stim_values(stim), x_el, y_el,
                             self.axon_contrib,
                             self.axon_idx_start.astype(np.uint32),
                             self.axon_idx_end.astype(np.uint32),
                             self.rho,
                             self.thresh_percept,
                             self._cutoff_r2(self.rho),
                             self.n_threads)

    def _postprocess_spatial(self, resp):
        """Blend across the horizontal meridian"""
        blended = _blend_meridian(resp, self.grid, 'horizontal',
                                  self.meridian_blend)
        if blended is resp:
            # No blending asked for; leave the response bit-for-bit alone.
            return resp
        # Restore percept threshold after blending:
        blended[np.abs(blended) < self.thresh_percept] = 0
        return blended

    def plot(self, use_dva=False, style='hull', annotate=True, autoscale=True,
             ax=None, figsize=None):
        """Plot the axon map

        Parameters
        ----------
        use_dva : bool, optional
            Uses degrees of visual angle (dva) if True, else retinal
            coordinates (microns)
        style : {'hull', 'scatter', 'cell'}, optional
            Grid plotting style:

            * 'hull': Show the convex hull of the grid (that is, the outline of
              the smallest convex set that contains all grid points).
            * 'scatter': Scatter plot all grid points
            * 'cell': Show the outline of each grid cell as a polygon. Note that
              this can be costly for a high-resolution grid.
        annotate : bool, optional
            Flag whether to label the four retinal quadrants
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object
        figsize : (float, float), optional
            Desired (width, height) of the figure in inches

        """
        if ax is None:
            ax = plt.gca()
        if figsize is not None:
            ax.figure.set_size_inches(figsize)
        ax.set_facecolor('white')
        ax.set_aspect('equal')

        # In a left eye, the OD must have a negative x coordinate:
        self._correct_loc_od()

        # Grow axon bundles to be drawn:
        axon_bundles = self.grow_axon_bundles(n_bundles=100, prune=False)

        if use_dva:
            # Use degrees of visual angle (dva) as axis unit:
            units = 'degrees of visual angle'
            # Make sure we're filling the simulated area, rounded up/down,
            # but no smaller than (-18, 18):
            xmin = min(np.floor(self.xrange[0] / 3) * 3, -18)
            xmax = max(np.ceil(self.xrange[1] / 3) * 3, 18)
            ymin = min(np.floor(self.yrange[0] / 3) * 3, -18)
            ymax = max(np.ceil(self.yrange[1] / 3) * 3, 18)
            od_xy = self.loc_od
            od_w = 6.44
            od_h = 6.85
            # Convert axon bundles to dva:
            axon_bundles = [np.array(self.vfmap.ret_to_dva(bundle[:, 0],
                                                             bundle[:, 1])).T
                            for bundle in axon_bundles]
            labels = ['upper', 'lower', 'left', 'right']
        else:
            # Use retinal coordinates (microns) as axis unit.
            units = 'microns'
            # Make sure we're filling the simulated area, rounded up/down,
            # but no smaller than (-5000, 5000):
            # Rounded to whole millimeters, which is what the ticks below are
            # spaced by:
            xmin, ymin = self.vfmap.dva_to_ret(self.xrange[0], self.yrange[0])
            xmin = min(np.floor(xmin / UM_PER_MM) * UM_PER_MM, -5000)
            ymin = min(np.floor(ymin / UM_PER_MM) * UM_PER_MM, -5000)
            xmax, ymax = self.vfmap.dva_to_ret(self.xrange[1], self.yrange[1])
            xmax = max(np.ceil(xmax / UM_PER_MM) * UM_PER_MM, 5000)
            ymax = max(np.ceil(ymax / UM_PER_MM) * UM_PER_MM, 5000)
            od_xy = self.vfmap.dva_to_ret(*self.loc_od)
            od_w = 1770
            od_h = 1880
            if self.eye == 'RE':
                labels = ['superior', 'inferior', 'temporal', 'nasal']
            else:
                labels = ['superior', 'inferior', 'nasal', 'temporal']

        # Draw axon pathways:
        for bundle in axon_bundles:
            # Set segments outside the drawing window to NaN:
            x_idx = np.logical_or(bundle[:, 0] < xmin, bundle[:, 0] > xmax)
            bundle[x_idx, 0] = np.nan
            y_idx = np.logical_or(bundle[:, 1] < ymin, bundle[:, 1] > ymax)
            bundle[y_idx, 1] = np.nan
            ax.plot(bundle[:, 0], bundle[:, 1], c=(0.6, 0.6, 0.6),
                    linewidth=2, zorder=ZORDER['background'])
        # Show elliptic optic nerve head (width/height are averages from
        # the human retina literature):
        ax.add_patch(Ellipse(od_xy, width=od_w, height=od_h, alpha=1,
                             color='white', zorder=ZORDER['background'] + 1))
        # Show extent of simulated grid:
        if self.is_built:
            self.grid.plot(ax=ax, style=style, zorder=ZORDER['background'] + 2,
                           use_dva=use_dva)
        ax.set_xlabel(f'x ({units})')
        ax.set_ylabel(f'y ({units})')
        if autoscale:
            ax.axis((xmin, xmax, ymin, ymax))
        if annotate:
            ann = ax.inset_axes([0.05, 0.05, 0.2, 0.2],
                                zorder=ZORDER['annotate'])
            ann.annotate('', (0.5, 1), (0.5, 0),
                         arrowprops={'arrowstyle': '<->'})
            ann.annotate('', (1, 0.5), (0, 0.5),
                         arrowprops={'arrowstyle': '<->'})
            positions = [(0.5, 1), (0.5, 0), (0, 0.5), (1, 0.5)]
            valign = ['bottom', 'top', 'center', 'center']
            rots = [0, 0, 90, -90]
            for label, pos, va, rot in zip(labels, positions, valign, rots):
                ann.annotate(label, pos, ha='center', va=va, rotation=rot)
            ann.axis('off')
            ann.set_xticks([])
            ann.set_yticks([])
        return ax


class AxonMapModel(Model):
    """Axon map model of [Beyeler2019]_ (standalone model)

    Implements the axon map model described in [Beyeler2019]_, where percepts
    are elongated along nerve fiber bundle trajectories of the retina.

    .. note: :

        Use this class if you want a standalone model.
        Use: py: class: `~pulse2percept.models.AxonMapSpatial` if you want
        to combine the spatial model with a temporal model.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.ProsthesisSystem`
        The device this model predicts percepts for. Required: a percept is
        what a particular implant produces, and ``predict_percept`` takes what
        is presented to that device.

        .. versionadded:: 0.11.0

    lam : double, optional
        Exponential decay constant along the axon(microns).

        .. versionchanged:: 0.10.0

            Renamed from ``axlambda``, which reads poorly next to ``rho``. The
            old name still works, but is deprecated and will be removed in
            v0.11.0.
    rho : double, optional
        Exponential decay constant away from the axon(microns).
    min_current_spread : float, optional
        An electrode is skipped at axon segments where its Gaussian current
        spread has decayed below this fraction of its peak. The default
        (1e-8, about 6.1 ``rho`` away) drops the Gaussian *times* the
        stimulus amplitude, summed over the skipped electrodes, so the error
        at a point is bounded by ``min_current_spread`` times the summed
        amplitude across electrodes.
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    step : int or double or tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``step=0.5``. Pass a tuple to give the x
        and y axes different step sizes.

        .. versionchanged:: 0.10.0

            Renamed from ``xystep``, which suggested that one step size
            applies to both axes. The old name still works, but is
            deprecated and will be removed in v0.11.0.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.topography.VisualFieldMap`
        object that provides retinotopic mappings.
        By default, :py:class:`~pulse2percept.topography.Watson2014Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.
    noise : float or int, optional
        Adds salt-and-pepper noise to each percept frame. An integer will be
        interpreted as the number of pixels to subject to noise in each frame.
        A float between 0 and 1 will be interpreted as a ratio of pixels to
        subject to noise in each frame.
    loc_od, loc_od : (x,y), optional
        Location of the optic disc in degrees of visual angle. Note that the
        optic disc in a left eye will be corrected to have a negative x
        coordinate.
    n_axons : int, optional
        Number of axons to generate.
    axons_range : (min, max) of float or Quantity, optional
        The range of angles(in degrees) at which axons exit the optic disc.
        This corresponds to the range of $\\phi_0$ values used in
        [Jansonius2009]_.
    n_ax_segments : int, optional
        Number of segments an axon is made of.
    ax_segments_range : (min, max), optional
        Lower and upper bounds for the radial position values(polar coords)
        for each axon.
    min_ax_sensitivity : float, optional
        Axon segments whose contribution to brightness is smaller than this
        value will be pruned to improve computational efficiency. Set to a
        value between 0 and 1.
    meridian_blend : float, optional
        Gaussian standard deviation (dva) for smoothing across the horizontal
        meridian. Default: 1. Set to 0 to disable.

        .. versionadded:: 0.10.0
    axon_pickle : str, optional
        File name in which to store precomputed axon maps.
    ignore_pickle : bool, optional
        A flag whether to ignore the pickle file in future calls to
        ``model.build()``.
    n_threads : int, optional
        Number of CPU threads to use during parallelization using OpenMP.
        Defaults to max number of user CPU cores.
    n_jobs : int, optional
        Alias for ``n_threads``; ``None`` or ``-1`` uses every core.

    .. important ::
        Changing a model parameter outside the constructor (e.g., by directly
        setting ``model.lam = 100``) un-builds the model, and the next
        ``predict_percept`` builds it again.

    Notes
    -----
    *  The axon map is not very accurate when the upper bound of
       `ax_segments_range` is greater than 90 deg.
    """

    def __init__(self, **params):
        super(AxonMapModel, self).__init__(spatial=AxonMapSpatial(),
                                           temporal=None,
                                           **params)

