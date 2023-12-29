"""`EllipsoidElectrode`, `LinearEdgeThread`, `NeuralinkThread`"""
from abc import ABCMeta, abstractmethod
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from ..base import ProsthesisSystem
from ..electrodes import Electrode
from ..electrode_arrays import ElectrodeArray


def parse_3d_orient(orient, orient_mode):
    """Parse the orient parameter
    Given either a 3D rotation matrix, vector of angles of rotation,
    or direction vector, this function will calculate and return the 
    all three representations.
    
    Parameters
    ----------
    orient : np.ndarray with shape (3) or (3, 3)
        Orientation of the electrode in 3D space.
        orient can be:
        - A length 3 vector specifying the direction that the
            electrode should extend in (if orient_mode == 'direction')
        - A list of 3 angles, (r_x, r_y, r_z), specifying the rotation
            in degrees about each axis (starting with x).
            (If orient_mode == 'angle')
        - 3D rotation matrix, specifying the direction that the electrode
            should extend in (i.e. a unit vector in the x direction will
            point in the direction after being rotated by this matrix)
    orient_mode : str
        If 'direction', orient is a vector specifying the direction that the
        electrode should extend in. If 'angle', orient is a vector of 3 angles,
        (r_x, r_y, r_z), specifying the rotation in degrees about each axis
        (starting with x). Does not apply if orient is a 3D rotation matrix.

    Returns
    -------
    rot : np.ndarray with shape (3, 3)
        Rotation matrix
    angles : np.ndarray with shape (3)
        Angles of rotation (degrees) about each axis (x, y, z).
        Note that this mapping is not unique. This function will always
        set the rotation about the x axis to be 0, meaning that the
        returned coordinates will match spherical coordinates (i.e.
        r_y is phi and r_z is theta).
    direction : np.ndarray with shape (3)
        Unit vector specifying the direction of the orientation.
    """

    def construct_rot_matrix(angles):
        """Construct a rotation matrix from angles of rotation"""
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(angles[0]), -np.sin(angles[0])],
                          [0, np.sin(angles[0]), np.cos(angles[0])]])
        rot_y = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                          [0, 1, 0],
                          [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        rot_z = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                          [np.sin(angles[2]), np.cos(angles[2]), 0],
                          [0, 0, 1]])
        return rot_z @ rot_y @ rot_x
    
    def extract_direction(rot):
        """Extract direction vector from rotation matrix"""
        # Do this by rotating the point (0, 0, 1)
        direction = np.matmul(rot, np.array([0, 0, 1]))
        return direction
    
    def extract_angles(direction):
        """Extract angles of rotation from direction vector"""
        rot_x = 0
        rot_y = np.arctan2(direction[0], direction[2]) # i.e. phi
        rot_z = np.arctan2(direction[1], direction[0]) # i.e. theta
        angles = np.array([rot_x, rot_y, rot_z])
        return angles

    if isinstance(orient, list):
        orient = np.array(orient)
    if not isinstance(orient, np.ndarray) or orient.shape not in [(3,), (3, 3)]:
        raise ValueError(f'Incorrect value for orient parameter {orient}, ', 
                         'please pass an array with shape (3) or (3, 3)')
    if orient.ndim == 1:
        if orient_mode == 'direction':
            if not np.allclose(np.linalg.norm(orient), 1):
                # unnormalized
                if np.linalg.norm(orient) == 0:
                    raise ValueError('orient cannot be a zero vector if orient_mode is "direction"')
                orient = orient / np.linalg.norm(orient)

            direction = orient
            angles = extract_angles(direction)
            rot = construct_rot_matrix(angles)
        elif orient_mode == 'angle':
            angles = orient
            rot = construct_rot_matrix(angles)
            direction = extract_direction(rot)
        else:
            raise ValueError('orient_mode must be either "direction" or "angle".')
        
    elif orient.ndim == 2:
        if not np.allclose(np.linalg.inv(orient), orient.T) or orient.shape != (3, 3):
            raise ValueError(f'Invalid rotation matrix {orient}')
        rot = orient
        direction = extract_direction(rot)
        angles = extract_angles(direction)

    return rot, angles, direction


class EllipsoidElectrode(Electrode):
    
    __slots__ = ('rx', 'ry', 'rz', 'orient', 'plot_3d_kwargs', 
                 'plotx', 'ploty', 'plotz', 'rot', 'angles', 'direction')
    
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
            If dim=2, orient defaults to being perpendicular to cortical surface.
            If dim=3, orient defaults to being parallel to the z axis.

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
        self.plot_3d_kwargs = {'color': 'yellow', 'alpha': 0.9,
                               'rstride': 2, 'cstride': 2}
        
        self.rot, self.angles, self.direction = parse_3d_orient(orient, orient_mode)
        
        # prepare for plotting in 3d
        npoints = 15 # resolution, less is faster
        thetas = np.linspace(0, 2 * np.pi, npoints)
        phis = np.linspace(0, np.pi, npoints)
        plotx = self.rx * np.outer(np.cos(thetas), np.sin(phis))
        ploty = self.ry * np.outer(np.sin(thetas), np.sin(phis))
        plotz = self.rz * np.outer(np.ones_like(thetas), np.cos(phis))
        stacked_points = np.stack([plotx, ploty, plotz], axis=-1).reshape(plotx.shape[0], plotx.shape[1], 3, 1)
        rotated = np.matmul(self.rot, stacked_points).reshape(plotx.shape[0], plotx.shape[1], 3)
        self.plotx = rotated[:, :, 0] + self.x
        self.ploty = rotated[:, :, 1] + self.y
        self.plotz = rotated[:, :, 2] + self.z


    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'rx': self.rx, 'ry': self.ry, 'rz': self.rz, 'angles': self.angles})
        return params


    def electric_potential(self, x, y, z, v0):
        raise NotImplementedError
    

    def plot3D(self, ax=None):
        """Plot the electrode in 3D space
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, a new figure and axes will be created.
        .. note::
            Unlike with 2D plots, you must pass an Axes object to plot on the 
            same axes. This is because of how plt.gca() works with 3D plots.
        """
        if ax is None:
            ax = plt.gca()
            if ax.name != '3d':
                plt.close()
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
        else:
            if ax.name != '3d':
                raise ValueError('ax must be a 3D axis')
        
        ax.plot_surface(self.plotx, self.ploty, self.plotz, **self.plot_3d_kwargs)
        return ax


class NeuralinkThread(ElectrodeArray, metaclass=ABCMeta):
    """Base class for Neuralink threads"""
    pass


class LinearEdgeThread(NeuralinkThread):
    
    # __slots__ = ('r', 'l', 'n', 'pitch', 'orient') # TODO
    
    def __init__(self, x=0, y=0, z=0, orient=np.array([0,0,1]), orient_mode='direction', 
                 dim=3, r=5, n_elecs=32, spacing=50, insertion_depth=0, 
                 electrode=EllipsoidElectrode, name=None):
        """
        Neuralink thread
        
        Parameters
        ----------
        x, y, z : float
            Coordinates of the thread insertion point on the surface of the cortex.
            z is optional if dim==2 
        orient : np.ndarray with shape (3) or (3, 3) 
            Orientation of the thread in 3D space. 
            If dim=2, orient defaults to being perpendicular to cortical surface.
            If dim=3, orient defaults to being parallel to the z axis.

            orient can be:
            - A length 3 vector specifying the direction that the 
              thread should extend in (if orient_mode == 'direction')
            - A list of 3 angles, (r_x, r_y, r_z), specifying the rotation 
              in degrees about each axis (starting with x). 
              (If orient_mode == 'angle')
            - 3D rotation matrix, specifying the direction that the thread 
              should extend in (i.e. a unit vector in the x direction will
              point in the direction after being rotated by this matrix)

        dim : int
            Dimensionality of the simulation. Must be 2 or 3.
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
        name : str
            Name of the thread.
        """
        self.x, self.y, self.z = x, y, z
        self.loc = np.array([x, y, z])
        self.r = r
        self.n_elecs = n_elecs
        self.spacing = spacing
        self.electrode = electrode
        self.name = name
        self.dim = dim
        self.insertion_depth = insertion_depth
        # microns out of cortex that thread should extend (for visualization only)
        self.extracortical_depth = 1000
        self.thread_length = self.n_elecs * self.spacing + self.extracortical_depth + self.insertion_depth
        self.rot, self.angles, self.direction = parse_3d_orient(orient, orient_mode)
        self.plot_3d_kwargs = {'color': 'gray', 'alpha': 0.5,
                               'rstride': 2, 'cstride': 2}

        if self.dim not in [2, 3]:
            raise ValueError("dim must be either 2 or 3")

        if self.dim == 2:
            raise NotImplementedError("2D implant not yet implemented")
        
        # calculate the coordinates of the electrodes
        electrodes = []
        start = self.loc + self.insertion_depth * self.direction 
        # this is a little hacky, but basically, we don't want the electrodes
        # exactly on the thread, but rather, on the edge. 
        # This chooses an arbitrary angle (facing x axis), rotates the direction vector 
        # towards that angle, and puts the electrodes on the edge of the thread in that direction.
        # also, the exact specs are unclear from the paper here
        offset = parse_3d_orient([1, 0, 0], 'direction')[0] @ self.direction * (self.r + 7//2) 
        electrode_locs = [start + i*self.spacing*self.direction + offset for i in range(self.n_elecs)]
        for loc in electrode_locs:
            electrodes.append(self.electrode(loc[0], loc[1], loc[2], orient=self.rot))
        
        super().__init__(electrodes)


    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params.update({'angles' : self.angles, 'r': self.r, 'n_elecs': self.n_elecs, 'spacing': self.spacing, 
                       'name': self.name})
        return params
    
    def plot3D(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
            if ax.name != '3d':
                plt.close()
                fig = plt.figure(**kwargs)
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
            electrode.plot3D(ax=ax)

        return ax
