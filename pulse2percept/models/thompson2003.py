"""`Thompson2003Model`, `Thompson2003Spatial` [Thompson2003]_"""

import numpy as np
import copy
from ..utils import sample
from ..topography import Curcio1990Map
from ..models import Model, SpatialModel
from ._thompson2003 import fast_thompson2003

# Log all warnings.warn() at the WARNING level:
import warnings
import logging
logging.captureWarnings(True)


class Thompson2003Spatial(SpatialModel):
    """Scoreboard model of [Thompson2003]_ (spatial module only)

    Implements the scoreboard model described in [Thompson2003]_, where all
    percepts are circular disks of a given size, and a fraction of electrodes
    may randomly drop out.

    .. note ::

        Use this class if you want to combine the spatial model with a temporal
        model.
        Use :py:class:`~pulse2percept.models.Thompson2003Model` if you want a
        a standalone model.

    Parameters
    ----------
    radius : double, optional
        Disk radius describing phosphene size (microns).
        If None, disk diameter is chosen as the electrode-to-electrode spacing
        (works only for implants with a ``shape`` attribute) with a 5% gap.
    dropout : int or float, optional
        If an int, number of electrodes to randomly drop out every frame.
        If a float between 0 and 1, the fraction of electrodes to randomly drop
        out every frame.
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : tuple, (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    xystep : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``xystep=0.5``.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    retinotopy : :py:class:`~pulse2percept.utils.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.utils.VisualFieldMap`
        object that provides retinotopic mappings.
        By default, :py:class:`~pulse2percept.utils.Curcio1990Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.

    .. important ::

        If you change important model parameters outside the constructor (e.g.,
        by directly setting ``model.xrange = (-10, 10)``), you will have to call
        ``model.build()`` again for your changes to take effect.
    """

    def get_default_params(self):
        """Returns all settable parameters of the model"""
        base_params = super(Thompson2003Spatial, self).get_default_params()
        params = {'radius': None, 'dropout': None,
                  'retinotopy': Curcio1990Map()}
        return {**base_params, **params}

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""
        if not np.allclose([e.z for e in earray.electrode_objects], 0):
            msg = ("Nonzero electrode-retina distances do not have any effect "
                   "on the model output.")
            warnings.warn(msg)
        radius = self.radius
        if radius is None:
            if not hasattr(earray, 'spacing'):
                raise NotImplementedError
            radius = 0.45 * earray.spacing
        dropout = np.zeros(stim.shape, dtype=np.uint8)
        if self.dropout is not None:
            for t in range(dropout.shape[1]):
                dropout[sample(np.arange(stim.shape[0]), k=self.dropout),
                        t] = 255
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        return fast_thompson2003(stim.data,
                                 np.array([earray[e].x for e in stim.electrodes],
                                          dtype=np.float32),
                                 np.array([earray[e].y for e in stim.electrodes],
                                          dtype=np.float32),
                                 self.grid.ret.x.ravel(),
                                 self.grid.ret.y.ravel(),
                                 dropout.astype(np.uint8),
                                 radius,
                                 self.thresh_percept)


class Thompson2003Model(Model):
    """Scoreboard model of [Thompson2003]_ (standalone model)

    Implements the scoreboard model described in [Thompson2003]_, where all
    percepts are circular disks of a given size, and a fraction of electrodes
    may randomly drop out.

    .. note ::

        Use this class if you want a standalone model.
        Use :py:class:`~pulse2percept.models.Thompson2003Spatial` if you want
        to combine the spatial model with a temporal model.

    radius : double, optional
        Disk radius describing phosphene size (microns).
        If None, disk diameter is chosen as the electrode-to-electrode spacing
        (works only for implants with a ``shape`` attribute) with a 5% gap.
    dropout : int or float, optional
        If an int, number of electrodes to randomly drop out every frame.
        If a float between 0 and 1, the fraction of electrodes to randomly drop
        out every frame.
    xrange : (x_min, x_max), optional
        A tuple indicating the range of x values to simulate (in degrees of
        visual angle). In a right eye, negative x values correspond to the
        temporal retina, and positive x values to the nasal retina. In a left
        eye, the opposite is true.
    yrange : tuple, (y_min, y_max), optional
        A tuple indicating the range of y values to simulate (in degrees of
        visual angle). Negative y values correspond to the superior retina,
        and positive y values to the inferior retina.
    xystep : int, double, tuple, optional
        Step size for the range of (x,y) values to simulate (in degrees of
        visual angle). For example, to create a grid with x values [0, 0.5, 1]
        use ``xrange=(0, 1)`` and ``xystep=0.5``.
    grid_type : {'rectangular', 'hexagonal'}, optional
        Whether to simulate points on a rectangular or hexagonal grid
    retinotopy : :py:class:`~pulse2percept.utils.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.utils.VisualFieldMap`
        object that provides retinotopic mappings.
        By default, :py:class:`~pulse2percept.utils.Watson2014Map` is
        used.
    n_gray : int, optional
        The number of gray levels to use. If an integer is given, k-means
        clustering is used to compress the color space of the percept into
        ``n_gray`` bins. If None, no compression is performed.

    .. important ::

        If you change important model parameters outside the constructor (e.g.,
        by directly setting ``model.xrange = (-10, 10)``), you will have to call
        ``model.build()`` again for your changes to take effect.

    """

    def __init__(self, **params):
        super(Thompson2003Model, self).__init__(spatial=Thompson2003Spatial(),
                                                temporal=None,
                                                **params)
