import numpy as np
from sys import platform
import matplotlib as mpl
if platform == "darwin":  # OS X
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.axes import Subplot
from matplotlib.animation import FuncAnimation
import imageio

from ..utils import Data, GridXY


class Percept(Data):
    """Visual percept

    .. versionadded:: 0.6

    Parameters
    ----------
    data : 3D NumPy array
        A NumPy array specifying the percept in (Y, X, T) dimensions
    space : :py:class:`~pulse2percept.utils.GridXY`
        A grid object specifying the (x,y) coordinates in space
    time : 1D array
        A list of time points
    metadata : dict, optional, default: None
        Additional stimulus metadata can be stored in a dictionary.

    """

    def __init__(self, data, space=None, time=None, metadata=None):
        xdva = None
        ydva = None
        if space is not None:
            if not isinstance(space, GridXY):
                raise TypeError("'space' must be a GridXY object, not "
                                "%s." % type(space))
            xdva = space._xflat
            ydva = space._yflat
        if time is not None:
            time = np.array([time]).flatten()
        self._internal = {
            'data': data,
            'axes': [('ydva', ydva), ('xdva', xdva), ('time', time)],
            'metadata': metadata
        }
        self._frame = 0
        # def f(a1, a2):
        #     # https://stackoverflow.com/a/26410051
        #     return (((a1 - a2[:,:,np.newaxis])).prod(axis=1)<=0).any(axis=0)

    def get_brightest_frame(self):
        """Return the brightest frame

        Looks for the brightest pixel in the percept, determines at what point
        in time it happened, and returns all brightness values at that point
        in a 2D NumPy array

        Returns
        -------
        frame : 2D NumPy array
            A slice ``percept.data[..., tmax]`` where ``tmax`` is the time at
            which the percept reached its maximum brightness.
        """
        return self.data[..., np.argmax(np.max(self.data, axis=(0, 1)))]

    def reset(self):
        self._frame = 0

    def __iter__(self):
        """Iterate over all frames in self.data"""
        self.reset()
        return self

    def __next__(self):
        frame = self._frame
        if frame >= self.data.shape[-1]:
            raise StopIteration
        self._frame += 1
        return self.data[..., frame]

    def plot(self, time=None, kind='pcolor', ax=None, **kwargs):
        """Plot the percept

        Parameters
        ----------
        kind : { 'pcolor' | 'hex' }, optional, default: 'pcolor'
            Kind of plot to draw:
            *  'pcolor': using Matplotlib's ``pcolor``. Additional parameters
               (e.g., ``vmin``, ``vmax``) can be passed as keyword arguments.
            *  'hex': using Matplotlib's ``hexbin``. Additional parameters
               (e.g., ``gridsize``) can be passed as keyword arguments.
        time : None, optional, default: None
            The time point to plot. If None, plots the brightest frame.
            Use ``animate`` to play the percept frame-by-frame.
        ax : matplotlib.axes.Axes; optional, default: None
            A Matplotlib Axes object. If None, a new Axes object will be
            created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Returns the axes with the plot on it

        """
        if time is None:
            idx = np.argmax(np.max(self.data, axis=(0, 1)))
            times = [self.time[idx]]
            frames = [self.data[..., idx]]
        else: 
            # Need to be smart about what to do when plotting more than one
            # frame.
            raise NotImplementedError
        if ax is None:
            if 'figsize' in kwargs:
                figsize = kwargs['figsize']
            else:
                figsize = np.int32(np.array(self.shape[:2][::-1]) / 15)
                figsize = np.maximum(figsize, 1)
            _, ax = plt.subplots(figsize=figsize)
        else:
            if not isinstance(ax, Subplot):
                raise TypeError("'ax' must be a Matplotlib axis, not "
                                "%s." % type(ax))

        cmap = kwargs['cmap'] if 'cmap' in kwargs else 'gray'
        if kind == 'pcolor':
            # Create a pseudocolor plot. Make sure to pass additional keyword
            # arguments that have not already been extracted:
            other_kwargs = {key: kwargs[key]
                            for key in (kwargs.keys() - ['figsize', 'cmap'])}
            ax.pcolor(np.flipud(frame), cmap=cmap, **other_kwargs)
            ax.set_xticks(np.linspace(0, self.shape[1], num=5))
            ax.set_yticks(np.linspace(self.shape[0], 0, num=5))
        elif kind == 'hex':
            # Create a hexbin plot:
            gridsize = kwargs['gridsize'] if 'gridsize' in kwargs else 80
            X, Y = np.meshgrid(self.xdva, self.ydva, indexing='xy')
            # Make sure to pass additional keyword arguments that have not
            # already been extracted:
            other_kwargs = {key: kwargs[key]
                            for key in (kwargs.keys() - ['figsize', 'cmap',
                                                         'gridsize'])}
            ax.hexbin(X.ravel(), Y.ravel()[::-1], frame.ravel(),
                      cmap=cmap, gridsize=gridsize, **other_kwargs)
            ax.axis([self.xdva[0], self.xdva[-1], self.ydva[0], self.ydva[-1]])
            ax.set_xticks(np.linspace(self.xdva[0], self.xdva[-1], num=5))
            ax.set_yticks(np.linspace(self.ydva[0], self.ydva[-1], num=5))
        else:
            raise ValueError("Unknown plot option '%s'. Choose either 'pcolor'"
                             "or 'hex'." % kind)
        ax.set_xticklabels(np.linspace(self.xdva[0], self.xdva[-1], num=5))
        ax.set_xlabel('x (dva)')
        ax.set_yticklabels(np.linspace(self.ydva[0], self.ydva[-1], num=5))
        ax.set_ylabel('y (dva)')
        return ax

    def play(self, ax=None, fps=None):
        """
        Parameters
        ----------
        fps : float or None
            If None, uses time axis. Not supported for non-homogeneous
            time axis.
        """
        # https://stackoverflow.com/a/46878531

        def update(data):
            mat.set_data(data)
            return mat

        def data_gen():
            while True:
                yield next(self)

        plt.rcParams["animation.html"] = 'jshtml'
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        mat = ax.imshow(next(self), cmap='gray', vmax=self.data.max())
        fig.colorbar(mat)
        plt.close(fig)
        if fps is None:
            interval = np.unique(np.diff(self.time))
            if len(interval) > 1:
                raise NotImplementedError
            interval = interval[0]
        else:
            interval = 1000.0 / fps

        ani = FuncAnimation(fig, update, data_gen, interval=interval)
        return ani

    def save(self, fname='percept.mp4', fps=None):
        if fps is None:
            interval = np.unique(np.diff(self.time))
            if len(interval) > 1:
                raise NotImplementedError
            fps = 1000.0 / interval[0]
        imageio.mimwrite(fname, self.data.transpose((2, 0, 1)), fps=fps)
        print('Created %s' % fname)
