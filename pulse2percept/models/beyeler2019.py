"""`AxonMapModel`, `AxonMapSpatial` [Beyeler2019]_"""

import os
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..utils import parfor, Grid2D, Watson2014Transform
from ..implants import ProsthesisSystem, ElectrodeArray
from ..stimuli import Stimulus
from ..models import Model, SpatialModel
from ._beyeler2019 import (fast_scoreboard, fast_axon_map, fast_jansonius,
                           fast_find_closest_axon)


class ScoreboardSpatial(SpatialModel):

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
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        assert isinstance(earray, ElectrodeArray)
        assert isinstance(stim, Stimulus)
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

    def __init__(self, **params):
        super(ScoreboardModel, self).__init__(spatial=ScoreboardSpatial(),
                                              temporal=None, **params)


class AxonMapSpatial(SpatialModel):
    """Axon map model

    Implements the axon map model described in [Beyeler2019]_, where percepts
    are elongated along nerve fiber bundle trajectories of the retina.

    Parameters
    ----------
    axlambda : double
        Exponential decay constant along the axon (microns).
    rho : double
        Exponential decay constant away from the axon (microns).

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
            'rho': 100,
            'axlambda': 100,
            # Set the (x,y) location of the optic disc:
            'loc_od_x': 15.5,
            'loc_od_y': 1.5,
            'n_axons': 500,
            'axons_range': (-180, 180),
            # Number of sampling points along the radial axis (polar coords):
            'n_ax_segments': 500,
            # Lower and upper bounds for the radial position values(polar
            # coordinates):
            'ax_segments_range': (0, 50),
            # Precomputed axon maps stored in the following file:
            'axon_pickle': 'axons.pickle',
            # You can force a build by ignoring pickles:
            'ignore_pickle': False,
        }
        params.update(base_params)
        return params

    @staticmethod
    def dva2ret(xdva):
        """Convert degrees of visual angle (dva) into retinal coords (um)"""
        return Watson2014Transform.dva2ret(xdva)

    @staticmethod
    def ret2dva(xret):
        """Convert retinal corods (um) to degrees of visual angle (dva)"""
        return Watson2014Transform.ret2dva(xret)

    def _jansonius2009(self, phi0, beta_sup=-1.9, beta_inf=0.5, eye='RE'):
        """Grows a single axon bundle based on the model by Jansonius (2009)

        This function generates the trajectory of a single nerve fiber bundle
        based on the mathematical model described in [Beyeler2019]_.

        Parameters
        ----------
        phi0: float
            Angular position of the axon at its starting point(polar
            coordinates, degrees). Must be within[-180, 180].
        beta_sup: float, optional, default: -1.9
            Scalar value for the superior retina(see Eq. 5, `\beta_s` in the
            paper).
        beta_inf: float, optional, default: 0.5
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
        loc_od = (self.loc_od_x, self.loc_od_y)
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

    def grow_axon_bundles(self, n_bundles=None):
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
        # Remove axon bundles outside the simulated area:
        bundles = list(filter(lambda x: (np.max(x[:, 0]) >= self.xrange[0] and
                                         np.min(x[:, 0]) <= self.xrange[1] and
                                         np.max(x[:, 1]) >= self.yrange[0] and
                                         np.min(x[:, 1]) <= self.yrange[1]),
                              bundles))
        # Keep only reasonably sized axon bundles:
        bundles = list(filter(lambda x: len(x) > 10, bundles))
        # Convert to um:
        bundles = [self.dva2ret(b) for b in bundles]
        return bundles

    def find_closest_axon(self, bundles, xret=None, yret=None):
        """Finds the closest axon segment for every point (``xret``, ``yret``)
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
        # For every pixel on the grid, find the closest axon segment:
        if self.engine == 'cython':
            closest_seg = fast_find_closest_axon(flat_bundles,
                                                 xret.ravel(),
                                                 yret.ravel())
        else:
            closest_seg = [np.argmin((flat_bundles[:, 0] - x) ** 2 +
                                     (flat_bundles[:, 1] - y) ** 2)
                           for x, y in zip(xret.ravel(),
                                           yret.ravel())]
        # Look up the axon ID for every axon segment:
        closest_axon = axon_idx[closest_seg]
        return [bundles[n] for n in closest_axon]

    def calc_axon_contribution(self, axons):
        xyret = np.column_stack((self.grid.xret.ravel(),
                                 self.grid.yret.ravel()))
        axon_contrib = []
        for xy, bundle in zip(xyret, axons):
            idx = np.argmin((bundle[:, 0] - xy[0]) ** 2 +
                            (bundle[:, 1] - xy[1]) ** 2)
            # Cut off the part of the fiber that goes beyond the soma:
            axon = np.flipud(bundle[0: idx + 1, :])
            # Add the exact location of the soma:
            axon = np.insert(axon, 0, xy, axis=0)
            # For every axon segment, calculate distance from soma by
            # summing up the individual distances between neighboring axon
            # segments (by "walking along the axon"):
            d2 = np.cumsum(np.diff(axon[:, 0], axis=0) ** 2 +
                           np.diff(axon[:, 1], axis=0) ** 2)
            sensitivity = np.exp(-d2 / (2.0 * self.axlambda ** 2))
            contrib = np.column_stack((axon[1:, :], sensitivity))
            axon_contrib.append(contrib)
        return axon_contrib

    def calc_bundle_tangent(self, xc, yc):
        """Calculates orientation of fiber bundle tangent at (xc,yc)

        Parameters
        ----------
        xc, yc : float
            (x,y) location of point at which to calculate bundle orientation
            in microns.
        """
        # Check for scalar:
        if isinstance(xc, (list, np.ndarray)):
            raise TypeError("xc must be a scalar")
        if isinstance(yc, (list, np.ndarray)):
            raise TypeError("yc must be a scalar")
        # Find the fiber bundle closest to (xc, yc):
        bundles = self.grow_axon_bundles()
        bundle = self.find_closest_axon(bundles, xret=xc, yret=yc)[0]
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

    def _build(self):
        if self.eye == 'LE':
            if self.loc_od_x > 0:
                err_str = ("In a left eye, the x-coordinate of the optic"
                           "disc should be negative, not %f" % self.loc_od_x)
                raise ValueError(err_str)
        elif self.eye == 'RE':
            if self.loc_od_x < 0:
                err_str = ("In a right eye, the x-coordinate of the optic"
                           "disc should be positive, not %f" % self.loc_od_x)
                raise ValueError(err_str)
        else:
            err_str = ("Eye should be either 'LE' or 'RE', not %s." % self.eye)
            raise ValueError(err_str)
        need_axons = False
        # You can ignore pickle files and force a rebuild with this flag:
        if self.ignore_pickle:
            need_axons = True
        else:
            # Check if math for Jansonius model has been done before:
            if os.path.isfile(self.axon_pickle):
                params, axons = pickle.load(open(self.axon_pickle, 'rb'))
                for key, value in params.items():
                    if not np.allclose(getattr(self, key), value):
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
        axon_contrib = self.calc_axon_contribution(axons)
        self.axon_contrib = np.concatenate(axon_contrib).astype(np.float32)
        len_axons = [a.shape[0] for a in axon_contrib]
        self.axon_idx_end = np.cumsum(len_axons)
        self.axon_idx_start = self.axon_idx_end - np.array(len_axons)
        # Pickle axons along with all important parameters:
        params = {'loc_od_x': self.loc_od_x, 'loc_od_y': self.loc_od_y,
                  'n_axons': self.n_axons, 'axons_range': self.axons_range,
                  'xrange': self.xrange, 'yrange': self.yrange,
                  'xystep': self.xystep, 'n_ax_segments': self.n_ax_segments,
                  'ax_segments_range': self.ax_segments_range}
        pickle.dump((params, axons), open(self.axon_pickle, 'wb'))

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at specific times ``t``"""
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        assert isinstance(earray, ElectrodeArray)
        assert isinstance(stim, Stimulus)
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

    def plot(self, annotate=False, upside_down=False, ax=None, autoscale=True):
        if ax is None:
            ax = plt.gca()
        ax.set_facecolor('white')
        ax.set_aspect('equal')

        # Draw axon pathways:
        axon_bundles = self.grow_axon_bundles(n_bundles=100)
        for bundle in axon_bundles:
            ax.plot(bundle[:, 0], bundle[:, 1], c=(0.6, 0.6, 0.6), linewidth=2,
                    zorder=1)

        # Show elliptic optic nerve head (width/height are averages from the
        # human retina literature):
        ax.add_patch(Ellipse(self.dva2ret((self.loc_od_x, self.loc_od_y)),
                             width=1770, height=1880, alpha=1, color='white',
                             zorder=2))

        if autoscale:
            ax.axis([-5000, 5000, -4000, 4000])
        xmin, xmax, ymin, ymax = ax.axis()
        ax.set_xticks(np.linspace(xmin, xmax, num=5))
        ax.set_yticks(np.linspace(ymin, ymax, num=5))
        ax.set_xlabel('x (microns)')
        ax.set_ylabel('y (microns)')

        # Annotate the four retinal quadrants near the corners of the plot:
        # superior/inferior x temporal/nasal
        if annotate:
            if upside_down:
                topbottom = ['bottom', 'top']
            else:
                topbottom = ['top', 'bottom']
            if eye == 'RE':
                temporalnasal = ['temporal', 'nasal']
            else:
                temporalnasal = ['nasal', 'temporal']
            for yy, valign, si in zip([ymax, ymin], topbottom,
                                      ['superior', 'inferior']):
                for xx, halign, tn in zip([xmin, xmax], ['left', 'right'],
                                          temporalnasal):
                    ax.text(xx, yy, si + ' ' + tn,
                            color='black', fontsize=14,
                            horizontalalignment=halign,
                            verticalalignment=valign,
                            bbox={'boxstyle': 'square,pad=-0.1', 'ec': 'none',
                                  'fc': (1, 1, 1, 0.7)},
                            zorder=99)

        # Need to flip y axis to have upper half == upper visual field
        if upside_down:
            ax.invert_yaxis()

        return ax


class AxonMapModel(Model):

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
