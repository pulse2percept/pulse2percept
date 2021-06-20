"""`AxonMapModel`, `AxonMapSpatial` [Beyeler2019]_"""

import os
import numpy as np
import pickle
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..utils import parfor, Watson2014Transform
from ..utils.constants import ZORDER
from ..implants import ProsthesisSystem, ElectrodeArray
from ..stimuli import Stimulus
from ..models import Model, SpatialModel
from ._beyeler2019 import (fast_scoreboard, fast_axon_map, fast_jansonius,
                           fast_find_closest_axon)

# Log all warnings.warn() at the WARNING level:
import warnings
import logging
logging.captureWarnings(True)


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
    rho : double, optional
        Exponential decay constant describing phosphene size (microns).
    x_range : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    y_range : tuple, (y_min, y_max)
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    xystep : int, double, tuple
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``x_range=(0, 1)`` and ``xystep=0.5``.
    grid_type : {'rectangular', 'hexagonal'}
        Whether to simulate points on a rectangular or hexagonal grid

    .. important ::

        If you change important model parameters outside the constructor (e.g.,
        by directly setting ``model.axlambda = 100``), you will have to call
        ``model.build()`` again for your changes to take effect.

    """

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        params = super(ScoreboardSpatial, self).get_default_params()
        params.update({'rho': 100})
        return params

    @staticmethod
    def dva2ret(xdva):
        """Convert degrees of visual angle (dva) into retinal coords (um)"""
        return Watson2014Transform.dva2ret(xdva)

    @staticmethod
    def ret2dva(xret):
        """Convert retinal corods (um) to degrees of visual angle (dva)"""
        return Watson2014Transform.ret2dva(xret)

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""
        if not np.allclose([e.z for e in earray.electrode_objects], 0):
            msg = ("Nonzero electrode-retina distances do not have any effect "
                   "on the model output.")
            warnings.warn(msg)
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        return fast_scoreboard(stim.data,
                               np.array([earray[e].x for e in stim.electrodes],
                                        dtype=np.float32),
                               np.array([earray[e].y for e in stim.electrodes],
                                        dtype=np.float32),
                               self.grid.xret.ravel(),
                               self.grid.yret.ravel(),
                               self.rho,
                               self.thresh_percept)


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
    rho : double, optional
        Exponential decay constant describing phosphene size (microns).
    x_range : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    y_range : tuple, (y_min, y_max)
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    xystep : int, double, tuple
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``x_range=(0, 1)`` and ``xystep=0.5``.
    grid_type : {'rectangular', 'hexagonal'}
        Whether to simulate points on a rectangular or hexagonal grid
    """

    def __init__(self, **params):
        super(ScoreboardModel, self).__init__(spatial=ScoreboardSpatial(),
                                              temporal=None, **params)


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
    axlambda: double, optional
        Exponential decay constant along the axon(microns).
    rho: double, optional
        Exponential decay constant away from the axon(microns).
    eye: {'RE', LE'}, optional
        Eye for which to generate the axon map.
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : tuple, (y_min, y_max)
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    xystep : int, double, tuple
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``x_range=(0, 1)`` and ``xystep=0.5``.
    grid_type : {'rectangular', 'hexagonal'}
        Whether to simulate points on a rectangular or hexagonal grid
    loc_od, loc_od: (x,y), optional
        Location of the optic disc in degrees of visual angle. Note that the
        optic disc in a left eye will be corrected to have a negative x
        coordinate.
    n_axons: int, optional
        Number of axons to generate.
    axons_range: (min, max), optional
        The range of angles(in degrees) at which axons exit the optic disc.
        This corresponds to the range of $\\phi_0$ values used in
        [Jansonius2009]_.
    n_ax_segments: int, optional
        Number of segments an axon is made of.
    ax_segments_range: (min, max), optional
        Lower and upper bounds for the radial position values(polar coords)
        for each axon.
    min_ax_sensitivity: float, optional
        Axon segments whose contribution to brightness is smaller than this
        value will be pruned to improve computational efficiency. Set to a
        value between 0 and 1.
    axon_pickle: str, optional
        File name in which to store precomputed axon maps.
    ignore_pickle: bool, optional
        A flag whether to ignore the pickle file in future calls to
        ``model.build()``.

    Notes
    -----
    *  The axon map is not very accurate when the upper bound of
       `ax_segments_range` is greater than 90 deg.
    """

    def __init__(self, **params):
        super(AxonMapSpatial, self).__init__(**params)
        self.axon_contrib = None
        self.axon_idx_start = None
        self.axon_idx_end = None

    def get_default_params(self):
        base_params = super(AxonMapSpatial, self).get_default_params()
        params = {
            # Left or right eye:
            'eye': 'RE',
            'rho': 200,
            'axlambda': 200,
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
            # Precomputed axon maps stored in the following file:
            'axon_pickle': 'axons.pickle',
            # You can force a build by ignoring pickles:
            'ignore_pickle': False,
        }
        params.update(base_params)
        return params

    @staticmethod
    def dva2ret(xdva):
        """Convert degrees of visual angle (dva) into retinal coords (um)

        The axon map model converts degrees of visual angle into a retinal 
        distance from the optic axis (um) using Eq. A5 in [Watson2014]_.

        """
        return Watson2014Transform.dva2ret(xdva)

    @staticmethod
    def ret2dva(xret):
        """Convert retinal corods (um) to degrees of visual angle (dva)

        The axon map model converts an eccentricity measurement on the retinal
        surface(in micrometers), measured from the optic axis, into degrees
        of visual angle using Eq. A6 in [Watson2014]_.

        """
        return Watson2014Transform.ret2dva(xret)

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
            e_s = "Unknown eye string '%s': Choose from 'LE', 'RE'." % eye
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
        if self.engine == 'cython':
            xprime, yprime = fast_jansonius(rho, phi0, beta_sup, beta_inf)
        else:
            if is_superior:
                # Axon is in superior retina, compute `b` (real number) from
                # Eq. 5:
                b = np.exp(beta_sup + 3.9 * np.tanh(-(phi0 - 121.0) / 14.0))
                # Equation 3, `c` a positive real number:
                c = 1.9 + 1.4 * np.tanh((phi0 - 121.0) / 14.0)
            else:
                # Axon is in inferior retina: compute `b` (real number) from
                # Eq. 6:
                b = -np.exp(beta_inf + 1.5 * np.tanh(-(-phi0 - 90.0) / 25.0))
                # Equation 4, `c` a positive real number:
                c = 1.0 + 0.5 * np.tanh((-phi0 - 90.0) / 25.0)

            # Spiral as a function of `rho`:
            phi = phi0 + b * (rho - rho.min()) ** c
            # Convert to Cartesian coordinates:
            xprime = rho * np.cos(np.deg2rad(phi))
            yprime = rho * np.sin(np.deg2rad(phi))
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
        engine = 'serial' if self.engine == 'cython' else self.engine
        bundles = parfor(self._jansonius2009, phi,
                         func_kwargs={'eye': self.eye},
                         engine=engine, n_jobs=self.n_jobs,
                         scheduler=self.scheduler)
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
        bundles = [self.dva2ret(b) for b in bundles]
        return bundles

    def find_closest_axon(self, bundles, xret=None, yret=None,
                          return_index=False):
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

        Returns
        -------
        axon : Nx2 array or list of Nx2 arrays
            For each point in (xret, yret), returns an Nx2 array that represents
            the closest axon to that point. Each row in the array contains the
            x,y retinal coordinates (microns) of a particular axon segment.
        idx_axon : scalar or list of scalars, optional
            If ``return_index`` is True, also returns the index in ``bundles``
            of the closest axon (or list of closest axons).

        """
        if len(bundles) <= 0:
            raise ValueError("bundles must have length greater than zero")
        if xret is None:
            xret = self.grid.xret
        if yret is None:
            yret = self.grid.yret
        xret = np.asarray(xret, dtype=np.float32)
        yret = np.asarray(yret, dtype=np.float32)
        # For every axon segment, store the corresponding axon ID:
        axon_idx = [[idx] * len(ax) for idx, ax in enumerate(bundles)]
        axon_idx = [item for sublist in axon_idx for item in sublist]
        axon_idx = np.array(axon_idx, dtype=np.uint32)
        # Build a long list of all axon segments - their corresponding axon IDs
        # is given by `axon_idx` above:
        flat_bundles = np.concatenate(bundles)
        kdtree = cKDTree(flat_bundles, leafsize=60)
        # Create query list of xy pairs
        query = np.stack((xret.ravel(), yret.ravel()), axis=1)
        # Find index of closest segment
        _, closest_seg = kdtree.query(query)

        # Look up the axon ID for every axon segment:
        closest_idx = axon_idx[closest_seg]
        if len(closest_idx) == 1:
            closest_idx = closest_idx[0]
            closest_axon = bundles[closest_idx]
        else:
            closest_axon = [bundles[n] for n in closest_idx]
        if return_index:
            return closest_axon, closest_idx
        return closest_axon

    def calc_axon_sensitivity(self, bundles):
        """Calculate the sensitivity of each axon segment to electrical current

        This function combines the x,y coordinates of each bundle segment with
        a sensitivity value that depends on the distance of the segment to the
        cell body and ``self.axlambda``.

        The number of ``bundles`` must equal the number of points on
        `self.grid``. The function will then assume that the i-th bundle passes
        through the i-th point on the grid. This is used to determine the bundle
        segment that is closest to the i-th point on the grid, and to cut off
        all segments that extend beyond the soma. This effectively transforms
        a *bundle* into an *axon*, where the first axon segment now corresponds
        with the i-th location of the grid.

        After that, each axon segment gets a sensitivity value that depends
        on the distance of the segment to the soma (with decay rate 
        ``self.axlambda``). This is typically done during the build process, so
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
            A list of axon segments and sensitivity values. Each entry in the
            list is a Nx3 array, where the first two columns contain the retinal
            coordinates of each axon segment (microns), and the third column
            contains the sensitivity of the segment to electrical current.
            The latter depends on ``self.axlambda``.

        """
        xyret = np.column_stack((self.grid.xret.ravel(),
                                 self.grid.yret.ravel()))
        # Only include axon segments that are < `max_d2` from the soma. These
        # axon segments will have `sensitivity` > `self.min_ax_sensitivity`:
        max_d2 = -2.0 * self.axlambda ** 2 * np.log(self.min_ax_sensitivity)
        axon_contrib = []
        for xy, bundle in zip(xyret, bundles):
            idx = np.argmin((bundle[:, 0] - xy[0]) ** 2 +
                            (bundle[:, 1] - xy[1]) ** 2)
            # Cut off the part of the fiber that goes beyond the soma:
            axon = np.flipud(bundle[0: idx + 1, :])
            # Add the exact location of the soma:
            axon = np.concatenate((xy.reshape((1, -1)), axon), axis=0)
            # For every axon segment, calculate distance from soma by
            # summing up the individual distances between neighboring axon
            # segments (by "walking along the axon"):
            d2 = np.cumsum(np.sqrt(np.diff(axon[:, 0], axis=0) ** 2 +
                                   np.diff(axon[:, 1], axis=0) ** 2)) ** 2
            idx_d2 = d2 < max_d2
            sensitivity = np.exp(-d2[idx_d2] / (2.0 * self.axlambda ** 2))
            idx_d2 = np.concatenate(([False], idx_d2))
            contrib = np.column_stack((axon[idx_d2, :], sensitivity))
            axon_contrib.append(contrib)
        return axon_contrib

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

    def _correct_loc_od(self):
        if self.eye.upper() == 'LE':
            # In a left eye, the optic disc must have a negative x coordinate:
            self.loc_od = (-np.abs(self.loc_od[0]), self.loc_od[1])
        elif self.eye.upper() == 'RE':
            # In a right eye, the optic disc must have a positive x coordinate:
            self.loc_od = (np.abs(self.loc_od[0]), self.loc_od[1])
        else:
            err_str = ("Eye should be either 'LE' or 'RE', not %s." % self.eye)
            raise ValueError(err_str)

    def _build(self):
        if self.axlambda < 10:
            raise ValueError('"axlambda" < 10 is not supported by this model. '
                             'Consider using ScoreboardModel instead.')
        # In a left eye, the OD must have a negative x coordinate:
        self._correct_loc_od()
        # Check whether pickle file needs to be rebuilt:
        need_axons = False
        if self.ignore_pickle:
            need_axons = True
        else:
            # Check if math for Jansonius model has been done before:
            if os.path.isfile(self.axon_pickle):
                params, axons = pickle.load(open(self.axon_pickle, 'rb'))
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
            axons = self.find_closest_axon(bundles)
        # Calculate axon contributions (depends on axlambda):
        # Axon contribution is a list of (differently shaped) NumPy arrays,
        # and a list cannot be accessed in parallel without the gil. Instead
        # we need to concatenate it into a really long Nx3 array, and pass the
        # start and end indices of each slice:
        axon_contrib = self.calc_axon_sensitivity(axons)
        self.axon_contrib = np.concatenate(axon_contrib).astype(np.float32)
        len_axons = [a.shape[0] for a in axon_contrib]
        self.axon_idx_end = np.cumsum(len_axons)
        self.axon_idx_start = self.axon_idx_end - np.array(len_axons)
        # Pickle axons along with all important parameters:
        params = {'loc_od': self.loc_od,
                  'n_axons': self.n_axons, 'axons_range': self.axons_range,
                  'xrange': self.xrange, 'yrange': self.yrange,
                  'xystep': self.xystep, 'n_ax_segments': self.n_ax_segments,
                  'ax_segments_range': self.ax_segments_range}
        pickle.dump((params, axons), open(self.axon_pickle, 'wb'))

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at specific times ``t``"""
        if not np.allclose([e.z for e in earray.electrode_objects], 0):
            msg = ("Nonzero electrode-retina distances do not have any effect "
                   "on the model output.")
            warnings.warn(msg)
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        return fast_axon_map(stim.data,
                             np.array([earray[e].x for e in stim.electrodes],
                                      dtype=np.float32),
                             np.array([earray[e].y for e in stim.electrodes],
                                      dtype=np.float32),
                             self.axon_contrib,
                             self.axon_idx_start.astype(np.uint32),
                             self.axon_idx_end.astype(np.uint32),
                             self.rho,
                             self.thresh_percept)

    def plot(self, use_dva=False, annotate=True, autoscale=True, ax=None):
        """Plot the axon map

        Parameters
        ----------
        use_dva : bool, optional
            Uses degrees of visual angle (dva) if True, else retinal
            coordinates (microns)
        annotate : bool, optional
            Flag whether to label the four retinal quadrants
        autoscale : bool, optional
            Whether to adjust the x,y limits of the plot
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            A Matplotlib axes object. If None, will either use the current axes
            (if exists) or create a new Axes object

        """
        if ax is None:
            ax = plt.gca()
        ax.set_facecolor('white')
        ax.set_aspect('equal')

        # In a left eye, the OD must have a negative x coordinate:
        self._correct_loc_od()

        # Grow axon bundles to be drawn:
        axon_bundles = self.grow_axon_bundles(n_bundles=100, prune=False)

        if use_dva:
            # Use degrees of visual angle (dva) as axis unit:
            units = 'degrees of visual angle'
            xmin, xmax, ymin, ymax = -18, 18, -18, 18
            od_xy = self.loc_od
            od_w = 6.44
            od_h = 6.85
            grid_transform = None
            # Flip y upside down for dva:
            axon_bundles = [self.ret2dva(bundle) * [1, -1]
                            for bundle in axon_bundles]
            labels = ['upper', 'lower', 'left', 'right']
        else:
            # Use retinal coordinates (microns) as axis unit.
            units = 'microns'
            xmin, xmax, ymin, ymax = -5000, 5000, -5000, 5000
            od_xy = self.dva2ret(self.loc_od)
            od_w = 1770
            od_h = 1880
            grid_transform = self.dva2ret
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
            self.grid.plot(ax=ax, transform=grid_transform,
                           zorder=ZORDER['background'] + 2)
        ax.set_xlabel('x (%s)' % units)
        ax.set_ylabel('y (%s)' % units)
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
    axlambda: double, optional
        Exponential decay constant along the axon(microns).
    rho: double, optional
        Exponential decay constant away from the axon(microns).
    eye: {'RE', LE'}, optional
        Eye for which to generate the axon map.
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : tuple, (y_min, y_max)
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    xystep : int, double, tuple
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``x_range=(0, 1)`` and ``xystep=0.5``.
    grid_type : {'rectangular', 'hexagonal'}
        Whether to simulate points on a rectangular or hexagonal grid
    loc_od, loc_od: (x,y), optional
        Location of the optic disc in degrees of visual angle. Note that the
        optic disc in a left eye will be corrected to have a negative x
        coordinate.
    n_axons: int, optional
        Number of axons to generate.
    axons_range: (min, max), optional
        The range of angles(in degrees) at which axons exit the optic disc.
        This corresponds to the range of $\\phi_0$ values used in
        [Jansonius2009]_.
    n_ax_segments: int, optional
        Number of segments an axon is made of.
    ax_segments_range: (min, max), optional
        Lower and upper bounds for the radial position values(polar coords)
        for each axon.
    min_ax_sensitivity: float, optional
        Axon segments whose contribution to brightness is smaller than this
        value will be pruned to improve computational efficiency. Set to a
        value between 0 and 1.
    axon_pickle: str, optional
        File name in which to store precomputed axon maps.
    ignore_pickle: bool, optional
        A flag whether to ignore the pickle file in future calls to
        ``model.build()``.

    Notes
    -----
    *  The axon map is not very accurate when the upper bound of
       `ax_segments_range` is greater than 90 deg.
    """

    def __init__(self, **params):
        super(AxonMapModel, self).__init__(spatial=AxonMapSpatial(),
                                           temporal=None, **params)

    def predict_percept(self, implant, t_percept=None):
        # Need to add an additional check before running the base method:
        if isinstance(implant, ProsthesisSystem):
            if implant.eye != self.spatial.eye:
                raise ValueError(("The implant is in %s but the model was "
                                  "built for %s.") % (implant.eye,
                                                      self.spatial.eye))
        return super(AxonMapModel, self).predict_percept(implant,
                                                         t_percept=t_percept)
