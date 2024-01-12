"""`EllipsoidElectrode`, `LinearEdgeThread`, `NeuralinkThread`"""
from abc import ABCMeta, abstractmethod
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from ..ensemble import EnsembleImplant
from ..electrodes import Electrode
from ..electrode_arrays import ElectrodeArray
from ..base import ProsthesisSystem
from ...utils import parse_3d_orient


class EllipsoidElectrode(Electrode):
    
    __slots__ = ('rx', 'ry', 'rz', 'orient', 'plot_3d_kwargs', 
                 'rot', 'angles', 'direction')
    
    def __init__(self, x=0, y=0, z=0, rx=7, ry=7, rz=12, orient=np.array([0, 0, 1]), 
                 orient_mode='direction', name=None, activated=True):
        """Ellipsoid electrode
        
        Parameters
        ----------
        x, y, z : float
            Coordinates of the electrode.
        rx, ry, rz : float
            Radii of the ellipsoid along the x, y, and z axes.
        orient : np.ndarray with shape (3) or (3, 3)
            Orientation of the thread in 3D space. 
            orient defaults to positive z direction

            orient can be:
            - A length 3 vector specifying the direction that the 
              thread should extend in (if orient_mode == 'direction')
            - A list of 3 angles, (r_x, r_y, r_z), specifying the rotation 
              in degrees about each axis (starting with x). 
              (If orient_mode == 'angle')
            - 3D rotation matrix, specifying the direction that the thread 
              should extend in (i.e. a unit vector in the x direction will
              point in the direction after being rotated by this matrix)
        """
        super().__init__(x, y, z, name=name, activated=activated)
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.plot_patch = Ellipse
        self.plot_kwargs = {'width': rx, 'height': ry, 'angle': 0,
                            'linewidth': 2,
                            'ec': (0.3, 0.3, 0.3, 1),
                            'fc': (1, 1, 1, 0.8)}
        self.plot_deactivated_kwargs = {'width': rx, 'height': ry, 'angle': 0,
                                        'linewidth': 2,
                                        'ec': (0.6, 0.6, 0.6, 1),
                                        'fc': (1, 1, 1, 0.6)}
        self.plot_3d_kwargs = {'color': 'yellow', 'alpha': 0.95,
                               'rstride': 2, 'cstride': 2}
        
        self.rot, self.angles, self.direction = parse_3d_orient(orient, orient_mode)


    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'rx': self.rx, 'ry': self.ry, 'rz': self.rz, 'angles': self.angles})
        return params


    def electric_potential(self, x, y, z, v0):
        raise NotImplementedError
    

    def plot3D(self, ax=None, **kwargs):
        """Plot the electrode in 3D space
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, a new figure and axes will be created.
        """
        if ax is None:
            fig_kwargs = ['figsize']
            ax = plt.gca()
            if ax.name != '3d':
                plt.close()
                fig = plt.figure(**{k:v for k, v in kwargs.items() if k in fig_kwargs})
                ax = fig.add_subplot(111, projection='3d')
        else:
            if ax.name != '3d':
                raise ValueError('ax must be a 3D axis')
        
        # prepare for plotting in 3d
        npoints = 15 # resolution, less is faster
        thetas = np.linspace(0, 2 * np.pi, npoints)
        phis = np.linspace(0, np.pi, npoints)
        plotx = self.rx * np.outer(np.cos(thetas), np.sin(phis))
        ploty = self.ry * np.outer(np.sin(thetas), np.sin(phis))
        plotz = self.rz * np.outer(np.ones_like(thetas), np.cos(phis))
        stacked_points = np.stack([plotx, ploty, plotz], axis=-1).reshape(plotx.shape[0], plotx.shape[1], 3, 1)
        rotated = np.matmul(self.rot, stacked_points).reshape(plotx.shape[0], plotx.shape[1], 3)
        plotx = rotated[:, :, 0] + self.x
        ploty = rotated[:, :, 1] + self.y
        plotz = rotated[:, :, 2] + self.z

        ax.plot_surface(plotx, ploty, plotz, **self.plot_3d_kwargs)
        return ax


class NeuralinkThread(ProsthesisSystem, metaclass=ABCMeta):
    """Base class for Neuralink threads"""
    pass


class LinearEdgeThread(NeuralinkThread):
    
    # __slots__ = ('r', 'l', 'n', 'pitch', 'orient') # TODO
    
    def __init__(self, x=0, y=0, z=0, orient=np.array([0,0,1]), orient_mode='direction', 
                 r=5, n_elecs=32, spacing=50, insertion_depth=0, 
                 electrode=EllipsoidElectrode, name=None,
                 stim=None, preprocess=False, safe_mode=False):
        """
        Neuralink thread
        
        Parameters
        ----------
        x, y, z : float
            Coordinates of the thread insertion point on the surface of the cortex.
            z is optional and defaults to 0.
        orient : np.ndarray with shape (3) or (3, 3) 
            Orientation of the thread in 3D space. 

            orient can be:
            - A length 3 vector specifying the direction that the 
              thread should extend in (if orient_mode == 'direction')
            - A list of 3 angles, (r_x, r_y, r_z), specifying the rotation 
              in degrees about each axis (x rotation performed first). 
              (If orient_mode == 'angle')
            - 3D rotation matrix, specifying the direction that the thread 
              should extend in (i.e. a unit vector in the z direction will
              point in the direction after being rotated by this matrix)

        r : float
            Radius of the thread.
        n_elecs : int
            Number of electrodes along the thread.
        spacing : float
            Spacing between electrodes along the thread.
        insertion_depth : float
            Distance into cortex where electrodes start. Thread is assumed
            to end at insertion_depth + n_elecs*spacing
        electrode : Electrode
            Electrode class to use for the individual electrodes.
            Must accept x, y, z, and orient parameters, and contain a plot_patch
            and plot_kwargs if dim=2 or a plot_3d method if dim=3.
        """
        self.x, self.y, self.z = x, y, z
        self.loc = np.array([x, y, z])
        self.r = r
        self.n_elecs = n_elecs
        self.spacing = spacing
        self.electrode = electrode
        self.insertion_depth = insertion_depth
        # microns out of cortex that thread should extend (for visualization only)
        self.extracortical_depth = 1000
        self.thread_length = self.n_elecs * self.spacing + self.extracortical_depth + self.insertion_depth
        self.rot, self.angles, self.direction = parse_3d_orient(orient, orient_mode)
        self.plot_3d_kwargs = {'color': 'gray', 'alpha': 0.75,
                               'rstride': 2, 'cstride': 2}
        
        # calculate the coordinates of the electrodes
        electrodes = {}
        start = self.loc + self.insertion_depth * self.direction 
        # this is a little hacky, but basically, we don't want the electrodes
        # exactly on the thread, but rather, on the edge. 
        # This chooses an arbitrary angle (facing x axis), rotates the direction vector 
        # towards that angle, and puts the electrodes on the edge of the thread in that direction.
        # also, the exact specs are unclear from the paper here
        offset = parse_3d_orient([1, 0, 0], 'direction')[0] @ self.direction * (self.r + 7//2) 
        electrode_locs = [start + i*self.spacing*self.direction + offset for i in range(self.n_elecs)]
        for i, loc in enumerate(electrode_locs):
            electrodes[str(i)] = self.electrode(loc[0], loc[1], loc[2], orient=self.rot)
        
        self.earray = ElectrodeArray(electrodes)
        self.safe_mode = safe_mode
        self.preprocess = preprocess
        self.stim = stim


    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'angles' : self.angles, 'r': self.r, 'n_elecs': self.n_elecs, 'spacing': self.spacing})
        return params
    
    def plot3D(self, ax=None, **kwargs):
        """Plot the thread in 3D space

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, a new figure and axes will be created.
        """
        fig_kwargs = ['figsize']
        if ax is None:
            ax = plt.gca()
            if ax.name != '3d':
                plt.close()
                fig = plt.figure(**{k:v for k, v in kwargs.items() if k in fig_kwargs})
                ax = fig.add_subplot(111, projection='3d')
        else:
            if ax.name != '3d':
                raise ValueError('ax must be a 3D axis')
        
        # plot the cylindrical thread
        npoints = 15
        thetas = np.linspace(0, 2 * np.pi, npoints)
        zs = np.linspace(0, self.thread_length, npoints)
        xs = self.r * np.outer(np.cos(thetas), np.ones_like(zs))
        ys = self.r * np.outer(np.sin(thetas), np.ones_like(zs))
        zs = np.outer(np.ones_like(thetas), zs)
        stacked_points = np.stack([xs, ys, zs], axis=-1).reshape(xs.shape[0], xs.shape[1], 3, 1)
        rotated = np.matmul(self.rot, stacked_points).reshape(xs.shape[0], xs.shape[1], 3)
        start = self.loc - self.extracortical_depth * self.direction
        plotx = rotated[:, :, 0] + start[0]
        ploty = rotated[:, :, 1] + start[1]
        plotz = rotated[:, :, 2] + start[2]
        ax.plot_surface(plotx, ploty, plotz, **self.plot_3d_kwargs)

        for electrode in self.electrode_objects:
            electrode.plot3D(ax=ax, **{k:v for k, v in kwargs.items() if k not in fig_kwargs})

        return ax


class Neuralink(EnsembleImplant):

    @classmethod
    def from_neuropythy(cls, vfmap, locs=None, xrange=None, yrange=None, xystep=None, 
                        rand_insertion_angle=None, region='v1', Thread=LinearEdgeThread):
        """
        Create a neuralink implant from a neuropythy visual field map.

        The implant will be created by creating a NeuralinkThread for each
        visual field location specified either by locs or by xrange, yrange, 
        and xystep. Each thread will be inserted perpendicular to the 
        cortical surface at the corresponding location in cortex, with 
        up to rand_insertion_angle degrees of azimuthal rotation.

        Parameters
        ----------
        vfmap : p2p.topography.NeuropythyMap
            Visual field map to create implant from.
        locs : np.ndarray with shape (n, 2), optional
            Array of visual field locations to create threads at. Not
            needed if using xrange, yrange, and xystep.
        xrange, yrange: tuple of floats, optional
            Range of x and y coordinates to create threads at. If None, 
            defaults to the range of the visual field map.
        xystep : float, optional
            Spacing between threads. If None, defaults to the spacing of the 
            visual field map.
        rand_insertion_angle : float, optional
            If not none, insert threads at a random offset from perpendicular,
            with a maximum azimuthal rotation of rand_insertion_angle degrees.
        region : str, optional
            Region of cortex to create implant in.
        Thread : NeuralinkThread, optional
            Thread class to use for the implant. Must accept x, y, z, and orient
            parameters.

        Returns
        -------
        Neuralink : p2p.implants.Neuralink
            Neuralink ensemble implant created from the visual field map.
        """
        # import at runtime to avoid circular imports
        from ...topography import NeuropythyMap, Grid2D
        if not isinstance(vfmap, NeuropythyMap):
            raise TypeError("vfmap must be a p2p.topography.NeuropythyMap")
        
        if locs is None:
            if xrange is None:
                xrange = vfmap.xrange
            if yrange is None:
                yrange = vfmap.yrange
            if xystep is None:
                xystep = vfmap.xystep
            
            # make a grid of points
            grid = Grid2D(xrange, yrange, xystep)
            xlocs = grid.x.flatten()
            ylocs = grid.y.flatten()
        else:
            xlocs = locs[:, 0]
            ylocs = locs[:, 1]
        
        # thread will extend from the pial point to the intracortical point
        # will be (3, npoints) shape
        surface_points = np.array(vfmap.from_dva()[region](xlocs, ylocs, surface='pial'))
        intra_points = np.array(vfmap.from_dva()[region](xlocs, ylocs, surface='midgray'))
        surface_points= surface_points[:, np.isnan(surface_points).sum(axis=0) == 0]
        intra_points = intra_points[:, np.isnan(intra_points).sum(axis=0) == 0]
        if len(surface_points) != len(intra_points):
            raise ValueError('Unable to create implant, try using jitter_boundary=True')

        threads = {}
        for i in range(len(surface_points[0])):
            # get the direction vector of the thread
            direction = intra_points[:, i] - surface_points[:, i]
            direction /= np.linalg.norm(direction)

            # if rand_insertion_angle is not None, rotate the direction vector
            if rand_insertion_angle is not None:
                rho = np.random.uniform(-rand_insertion_angle, rand_insertion_angle)
                theta = np.random.uniform(0, 360)
                rot_rand, _, _ = parse_3d_orient([0, rho, theta], 'angle')
                rot_direction, _, _= parse_3d_orient(direction, 'direction')
                direction = rot_direction @ rot_rand

            location = surface_points[:, i]
            name = ''
            j = i
            while j >= 26:
                name = chr(65 + j % 26) + name
                j = j // 26 - 1
            name = chr(65 + j) + name

            threads[name] = Thread(x=location[0], y=location[1], z=location[2],
                                            orient=direction, orient_mode='direction')
        return cls(threads)
    

    def __init__(self, threads, stim=None, preprocess=False, safe_mode=False):
        """
        Neuralink implant, consisting of one or more 
        :py:class:`~pulse2percept.implants.cortex.NeuralinkThread`s.

        This is just a wrapper class for EnsembleImplant, with extra
        functionality for plotting in 3D and a factory method to easily create 
        a Neuralink implant (see :py:meth:`~Neuralink.from_neuropythy`).

        Parameters
        ----------
        threads : collection of NeuralinkThread
            Collection (list) of NeuralinkThread objects to combine into an \
            implant.
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            A valid source type for the :py:class:`~pulse2percept.stimuli.Stimulus`
            object (e.g., scalar, NumPy array, pulse train).
        preprocess : bool or callable, optional
            Either True/False to indicate whether to execute the implant's default
            preprocessing method whenever a new stimulus is assigned, or a custom
            function (callable).
        safe_mode : bool, optional
            If safe mode is enabled, only charge-balanced stimuli are allowed.
        """
        if isinstance(threads, dict):
            for key, thread in threads.items():
                if not isinstance(thread, NeuralinkThread):
                    raise TypeError("threads must be a collection of NeuralinkThread objects")
        else: 
            for thread in threads:
                if not isinstance(thread, NeuralinkThread):
                    raise TypeError("threads must be a collection of NeuralinkThread objects")
        super().__init__(threads, stim=stim, preprocess=preprocess, safe_mode=safe_mode)
    
    def plot3D(self, ax=None, **kwargs):
        """Plot the implant in 3D space
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, a new figure and axes will be created.
        """
        fig_kwargs = ['figsize']
        if ax is None:
            ax = plt.gca()
            if ax.name != '3d':
                plt.close()
                fig = plt.figure(**{k:v for k, v in kwargs.items() if k in fig_kwargs})
                ax = fig.add_subplot(111, projection='3d')
        else:
            if ax.name != '3d':
                raise ValueError('ax must be a 3D axis')
        
        for thread in self.implants.values():
            thread.plot3D(ax=ax, **{k:v for k, v in kwargs.items() if k not in fig_kwargs})
        return ax
