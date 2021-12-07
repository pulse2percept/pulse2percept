"""`Nanduri2012Model`, `Nanduri2012Spatial`, `Nanduri2012Temporal`
   [Nanduri2012]_"""
import numpy as np
from .base import Model, SpatialModel, TemporalModel
from ._nanduri2012 import spatial_fast, temporal_fast
from ..implants import ElectrodeArray, DiskElectrode
from ..stimuli import Stimulus


class Nanduri2012Spatial(SpatialModel):
    """Spatial response model of [Nanduri2012]_

    Implements the spatial response model described in [Nanduri2012]_, which
    assumes that the spatial activation of retinal tissue is equivalent to the
    "current spread" :math:`I`, described as a function of distance :math:`r`
    from the center of the stimulating electrode:

    .. math::

        I(r) =
        \\begin{cases}
            \\frac{\\verb!atten_a!}{\\verb!atten_a! + (r-a)^\\verb!atten_n!}
                & r > a \\\\
            1 & r \\leq a
        \\end{cases}

    where :math:`a` is the radius of the electrode (see Eq.2 in the paper).

    .. note::

        Use this class if you just want the spatial response model.
        Use :py:class:`~pulse2percept.models.Nanduri2012Model` if you want both
        the spatial and temporal model.

    Parameters
    ----------
    atten_a : float, optional
        Nominator of the attentuation function
    atten_n : float32, optional
        Exponent of the attenuation function's denominator
    retinotopy : :py:class:`~pulse2percept.utils.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.utils.VisualFieldMap`
        object that provides ``ret2dva`` and ``dva2ret`` methods.
        By default, :py:class:`~pulse2percept.utils.Curcio1990Map` is
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


    """

    def get_default_params(self):
        """Returns all settable parameters of the Nanduri model"""
        base_params = super(Nanduri2012Spatial, self).get_default_params()
        params = {'atten_a': 14000, 'atten_n': 1.69}
        return {**base_params, **params}

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        return spatial_fast(stim.data,
                            np.array([earray[e].x for e in stim.electrodes],
                                     dtype=np.float32),
                            np.array([earray[e].y for e in stim.electrodes],
                                     dtype=np.float32),
                            np.array([earray[e].z for e in stim.electrodes],
                                     dtype=np.float32),
                            np.array([earray[e].r for e in stim.electrodes],
                                     dtype=np.float32),
                            self.grid.xret.ravel(),
                            self.grid.yret.ravel(),
                            self.atten_a,
                            self.atten_n,
                            self.thresh_percept)

    def predict_percept(self, implant, t_percept=None):
        if not np.all([isinstance(e, DiskElectrode)
                       for e in implant.electrode_objects]):
            raise TypeError("The Nanduri2012 spatial model only supports "
                            "DiskElectrode arrays.")
        return super(Nanduri2012Spatial, self).predict_percept(
            implant, t_percept=t_percept
        )


class Nanduri2012Temporal(TemporalModel):
    """Temporal model of [Nanduri2012]_

    Implements the temporal response model described in [Nanduri2012]_, which
    assumes that the temporal activation of retinal tissue is the output of a
    linear-nonlinear model cascade (see Fig.6 in the paper).

    .. note::

        Use this class if you just want the temporal response model.
        Use :py:class:`~pulse2percept.models.Nanduri2012Model` if you want both
        the spatial and temporal model.

    Parameters
    ----------
    dt : float, optional
        Sampling time step (ms)
    tau1: float, optional
        Time decay constant for the fast leaky integrater.
    tau2: float, optional
        Time decay constant for the charge accumulation.
    tau3: float, optional
        Time decay constant for the slow leaky integrator.
    eps: float, optional
        Scaling factor applied to charge accumulation.
    asymptote: float, optional
        Asymptote of the logistic function used in the stationary nonlinearity
        stage.
    slope: float, optional
        Slope of the logistic function in the stationary nonlinearity stage.
    shift: float, optional
        Shift of the logistic function in the stationary nonlinearity stage.
    scale_out : float32, optional
        A scaling factor applied to the output of the model
    thresh_percept: float, optional
        Below threshold, the percept has brightness zero.

    """

    def get_default_params(self):
        base_params = super(Nanduri2012Temporal, self).get_default_params()
        params = {
            # Time decay for the ganglion cell impulse response:
            'tau1': 0.42,
            # Time decay for the charge accumulation:
            'tau2': 45.25,
            # Time decay for the slow leaky integrator:
            'tau3': 26.25,
            # Scaling factor applied to charge accumulation:
            'eps': 8.73,
            # Asymptote of the sigmoid:
            'asymptote': 14.0,
            # Slope of the sigmoid:
            'slope': 3.0,
            # Shift of the sigmoid:
            'shift': 16.0,
            # Scale the output:
            'scale_out': 1.0
        }
        return {**base_params, **params}

    def _predict_temporal(self, stim, t_percept):
        """Predict the temporal response"""
        # Pass the stimulus as a 2D NumPy array to the fast Cython function:
        stim_data = stim.data.reshape((-1, len(stim.time)))
        # Calculate at which simulation time steps we need to output a percept.
        # This is basically t_percept/self.dt, but we need to beware of
        # floating point rounding errors! 29.999 will be rounded down to 29 by
        # np.uint32, so we need to np.round it first:
        idx_percept = np.uint32(np.round(t_percept / self.dt))
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError("All times 't_percept' must be distinct multiples "
                             "of `dt`=%.2e" % self.dt)
        # Cython returns a 2D (space x time) NumPy array:
        return temporal_fast(stim_data.astype(np.float32),
                             stim.time.astype(np.float32),
                             idx_percept,
                             self.dt, self.tau1, self.tau2, self.tau3,
                             self.asymptote, self.shift, self.slope, self.eps,
                             self.scale_out, self.thresh_percept)


class Nanduri2012Model(Model):
    """[Nanduri2012]_ Model

    Implements the model described in [Nanduri2012]_, where percepts are
    circular and their brightness evolves over time.

    The model combines two parts:

    *  :py:class:`~pulse2percept.models.Nanduri2012Spatial` is used to
       calculate the spatial activation function, which is assumed to be
       equivalent to the "current spread" described as a function of distance
       from the center of the stimulating electrode (see Eq.2 in the paper).
    *  :py:class:`~pulse2percept.models.Nanduri2012Temporal` is used to
       calculate the temporal activation function, which is assumed to be the
       output of a linear-nonlinear cascade model (see Fig.6 in the paper).

    Parameters
    ----------
    atten_a : float, optional
        Nominator of the attentuation function (Eq.2 in the paper)
    atten_n : float32, optional
        Exponent of the attenuation function's denominator (Eq.2 in the paper)
    dt : float, optional
        Sampling time step (ms)
    tau1: float, optional
        Time decay constant for the fast leaky integrater.
    tau2: float, optional
        Time decay constant for the charge accumulation.
    tau3: float, optional
        Time decay constant for the slow leaky integrator.
    eps: float, optional
        Scaling factor applied to charge accumulation.
    asymptote: float, optional
        Asymptote of the logistic function used in the stationary nonlinearity
        stage.
    slope: float, optional
        Slope of the logistic function in the stationary nonlinearity stage.
    shift: float, optional
        Shift of the logistic function in the stationary nonlinearity stage.
    scale_out : float32, optional
        A scaling factor applied to the output of the model
    thresh_percept: float, optional
        Below threshold, the percept has brightness zero.
    retinotopy : :py:class:`~pulse2percept.utils.VisualFieldMap`, optional
        An instance of a :py:class:`~pulse2percept.utils.VisualFieldMap`
        object that provides ``ret2dva`` and ``dva2ret`` methods.
        By default, :py:class:`~pulse2percept.utils.Curcio1990Map` is
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


    """

    def __init__(self, **params):
        super(Nanduri2012Model, self).__init__(spatial=Nanduri2012Spatial(),
                                               temporal=Nanduri2012Temporal(),
                                               **params)
