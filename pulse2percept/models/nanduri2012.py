""":py:class:`~pulse2percept.models.Nanduri2012Model`, 
   :py:class:`~pulse2percept.models.Nanduri2012Spatial`, 
   :py:class:`~pulse2percept.models.Nanduri2012Temporal` [Nanduri2012]_"""
import numpy as np
from .base import Model, SpatialModel, TemporalModel
from ._nanduri2012 import spatial_fast, temporal_fast
from ..implants import ElectrodeArray, DiskElectrode
from ..stimuli import Stimulus
from ..units import ms


def _require_disk_electrodes(electrodes):
    """Require disk electrodes, whose radius is used by the Nanduri model."""
    if not all(isinstance(e, DiskElectrode) for e in electrodes):
        raise TypeError("The Nanduri2012 spatial model only supports "
                        "DiskElectrode arrays.")


class Nanduri2012Spatial(SpatialModel):
    r"""Spatial response model of [Nanduri2012]_.

        Models retinal activation as the sum of current spread from disk
        electrodes. For electrode :math:`e`, define the lateral distance from its
        center

        .. math::

            s_e(x,y) =
            \sqrt{(x-x_e)^2 + (y-y_e)^2}

        and the distance to the nearest point on the electrode disk

        .. math::

            d_e(x,y) =
            \sqrt{
                z_e^2 +
                \max\left[s_e(x,y)-a_e,\,0\right]^2
            },

        where :math:`a_e` is electrode radius and :math:`z_e` is electrode-retina
        distance. The spatial response is

        .. math::

            I(x,y,t) =
            \sum_{e \in E}
            A_e(t)
            \frac{\mathrm{atten\_a}}
                 {\mathrm{atten\_a} + d_e(x,y)^{\mathrm{atten\_n}}}.

        Thus activation is uniform beneath an electrode when :math:`z_e=0` and
        decays with distance from its edge. This is the p2p implementation of the
        current-spread model in Eq. 2 of [Nanduri2012]_, extended to include the
        electrode ``z`` coordinate.

        Only :py:class:`~pulse2percept.implants.DiskElectrode` arrays are
        supported because the model depends explicitly on electrode radius.

        Use this class for the spatial component alone. Use
        :py:class:`~pulse2percept.models.Nanduri2012Model` for the combined
        spatial-temporal model.

        Parameters
        ----------
        implant : :py:class:`~pulse2percept.implants.Implant`
            Implant whose electrode geometry is modeled.

            .. versionadded:: 0.11.0

        atten_a : float, optional
            Attenuation scale in Eq. 2. Current spread falls to half its maximum
            when :math:`d = \mathrm{atten\_a}^{1/\mathrm{atten\_n}}`.
            Distances are evaluated in microns. Default: 14000.
        atten_n : float, optional
            Exponent controlling the falloff of current spread with distance.
            Larger values produce a steeper tail. Default: 1.69.
        xrange : (float, float) or Quantity, optional
            Horizontal visual-field extent in degrees of visual angle. A physical
            retinal extent may instead be resolved through ``vfmap``.
        yrange : (float, float) or Quantity, optional
            Vertical visual-field extent in degrees of visual angle. A physical
            retinal extent may instead be resolved through ``vfmap``.
        step : float, (float, float), or Quantity, optional
            Grid spacing in degrees of visual angle. A pair specifies separate x
            and y spacing.

            .. versionchanged:: 0.10.0
                Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

        grid_type : {'rectangular', 'hexagonal'}, optional
            Sampling lattice used for the visual-field grid.
        thresh_percept : float, optional
            Brightness values below this threshold are set to zero.
        min_current_spread : float, optional
            Inherited Gaussian current-spread cutoff. This parameter is not used
            by ``Nanduri2012Spatial``.
        vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
            Retinotopic map between visual-field and retinal coordinates. Defaults
            to :py:class:`~pulse2percept.topography.Curcio1990Map`.
        n_gray : int or None, optional
            Number of gray levels in the returned percept. ``None`` disables
            gray-level quantization.
        noise : float, int, or None, optional
            Salt-and-pepper noise applied to each percept frame. An integer gives
            the number of affected pixels; a float in [0, 1] gives their fraction.
        verbose : bool, optional
            Whether to print status messages.
        ndim : list of int, optional
            Dimensionalities of ``vfmap`` accepted by the model.
        n_threads : int, optional
            Number of OpenMP threads.
        n_jobs : int or None, optional
            Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.
        """

    def get_default_params(self):
        """Return default model parameters."""
        base_params = super(Nanduri2012Spatial, self).get_default_params()
        params = {'atten_a': 14000, 'atten_n': 1.69}
        return {**base_params, **params}

    def _predict_spatial(self, earray, stim):
        """Predict the spatial response."""
        # The bound implant may have changed since the last build.
        _require_disk_electrodes(earray.electrode_objects)
        x_el, y_el, z_el = self._electrode_coords(earray, stim)
        # Radius is not part of the coordinate array.
        r_el = np.ascontiguousarray([earray[e].r for e in stim.electrodes],
                                    dtype=np.float32)
        return spatial_fast(self._stim_values(stim), x_el, y_el, z_el,
                            r_el,
                            self.grid.ret.x.ravel(),
                            self.grid.ret.y.ravel(),
                            self.atten_a,
                            self.atten_n,
                            self.thresh_percept,
                            self.n_threads)

    def _build(self):
        _require_disk_electrodes(self.implant.electrode_objects)


class Nanduri2012Temporal(TemporalModel):
    r"""Temporal response model of [Nanduri2012]_.

        Implements the linear-nonlinear cascade in Fig. 6 of [Nanduri2012]_.
        With stimulus amplitude :math:`A(t)`, the fast response and charge
        accumulation are

        .. math::

            \tau_1 \frac{dR_1}{dt} &= A(t) - R_1(t), \\

            \frac{dC}{dt} &= \max[A(t), 0], \\

            \tau_2 \frac{dR_2}{dt} &= C(t) - R_2(t).

        The two pathways are combined by half-wave rectification,

        .. math::

            R_3(t) =
            \max\left[
                R_1(t) - \epsilon_{\mathrm{ms}} R_2(t),\,0
            \right],

        where :math:`\epsilon_{\mathrm{ms}} = \epsilon / 1000` because p2p
        integrates time in milliseconds while the original parameterization used
        microseconds.

        A logistic nonlinearity sets the peak response. Let

        .. math::

            R_{3,\max} = \max_t R_3(t)

        and

        .. math::

            g =
            \frac{\mathrm{asymptote}}{R_{3,\max}}
            \sigma\left(
                \frac{R_{3,\max} - \mathrm{shift}}{\mathrm{slope}}
            \right),

        where :math:`\sigma(u)=1/(1+e^{-u})`. The entire :math:`R_3(t)` trace is
        multiplied by this gain, so the scaled peak equals the logistic response.

        The result then passes through three identical slow leaky integrators,

        .. math::

            \tau_3 \frac{dR_{4a}}{dt} &= gR_3 - R_{4a}, \\

            \tau_3 \frac{dR_{4b}}{dt} &= R_{4a} - R_{4b}, \\

            \tau_3 \frac{dB}{dt} &= R_{4b} - B,

        and the predicted brightness is ``scale_out`` :math:`\times B(t)`.

        Positive current drives the model. Use this class for the temporal
        component alone. Use :py:class:`~pulse2percept.models.Nanduri2012Model`
        for the combined spatial-temporal model.

        Parameters
        ----------
        dt : float or Quantity, optional
            Simulation time step, in milliseconds. Default: 0.005 ms.
        tau1 : float or Quantity, optional
            Time constant of the fast response :math:`R_1`, in milliseconds.
            Default: 0.42 ms.
        tau2 : float or Quantity, optional
            Time constant of the filtered charge accumulation :math:`R_2`, in
            milliseconds. Default: 45.25 ms.
        tau3 : float or Quantity, optional
            Time constant of each of the three final leaky-integrator stages, in
            milliseconds. Default: 26.25 ms.
        eps : float, optional
            Strength of the subtractive charge-accumulation pathway. The public
            value retains the original microsecond parameterization and is divided
            by 1000 internally for millisecond integration. Default: 8.73.
        asymptote : float, optional
            Upper asymptote of the logistic peak-response nonlinearity. Default:
            14.
        slope : float, optional
            Scale parameter controlling the steepness of the logistic
            nonlinearity. Default: 3.
        shift : float, optional
            Midpoint of the logistic nonlinearity along :math:`R_{3,\max}`.
            Default: 16.
        scale_out : float, optional
            Multiplicative scaling applied to the final brightness. Default: 1.
        thresh_percept : float, optional
            Brightness values below this threshold are set to zero. Default: 0.
        reduce : {'peak', 'last'}, optional
            How automatically chosen output points summarize the preceding
            interval. ``'last'`` reports brightness at the output instant;
            ``'peak'`` approximates the interval peak by subsampling. Explicit
            ``t_percept`` values always request those instants. Default:
            ``'last'``.
        verbose : bool, optional
            Whether to print status messages. Default: True.
        n_threads : int, optional
            Number of OpenMP threads. Defaults to all available CPU cores.
        n_jobs : int or None, optional
            Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.
        """

    # Positive current drives the Nanduri temporal cascade.
    _drive_sign = 1

    def get_default_params(self):
        base_params = super(Nanduri2012Temporal, self).get_default_params()
        params = {
            'tau1': 0.42,
            'tau2': 45.25,
            'tau3': 26.25,
            'eps': 8.73,
            'asymptote': 14.0,
            'slope': 3.0,
            'shift': 16.0,
            'scale_out': 1.0
        }
        return {**base_params, **params}

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'tau1': ms, 'tau2': ms,
                'tau3': ms}

    def _predict_temporal(self, stim, t_percept):
        """Predict the temporal response."""
        time = self._stim_times(stim)
        stim_data = self._stim_values(stim).reshape((-1, len(time)))
        # Round before casting so floating-point noise cannot shift a sample.
        idx_percept = np.uint32(np.round(t_percept / self.dt))
        if np.unique(idx_percept).size < t_percept.size:
            raise ValueError(f"All times 't_percept' must be distinct multiples "
                             f"of `dt`={self.dt:.2e}")
        return temporal_fast(stim_data.astype(np.float32),
                             time.astype(np.float32),
                             idx_percept,
                             self.dt, self.tau1, self.tau2, self.tau3,
                             self.asymptote, self.shift, self.slope, self.eps,
                             self.scale_out, self.thresh_percept, self.n_threads)


class Nanduri2012Model(Model):
    r"""Combined spatial-temporal model of [Nanduri2012]_.

        Combines :py:class:`~pulse2percept.models.Nanduri2012Spatial` with
        :py:class:`~pulse2percept.models.Nanduri2012Temporal`. See those classes
        for the spatial current-spread equation and temporal cascade.

        Parameters
        ----------
        implant : :py:class:`~pulse2percept.implants.Implant`
            Implant whose electrode geometry is modeled.

            .. versionadded:: 0.11.0

        atten_a : float, optional
            Spatial attenuation scale. Default: 14000.
        atten_n : float, optional
            Exponent controlling spatial attenuation. Default: 1.69.
        xrange : (float, float) or Quantity, optional
            Horizontal visual-field extent in degrees of visual angle. A physical
            retinal extent may instead be resolved through ``vfmap``.
        yrange : (float, float) or Quantity, optional
            Vertical visual-field extent in degrees of visual angle. A physical
            retinal extent may instead be resolved through ``vfmap``.
        step : float, (float, float), or Quantity, optional
            Grid spacing in degrees of visual angle. A pair specifies separate x
            and y spacing.

            .. versionchanged:: 0.10.0
                Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

        grid_type : {'rectangular', 'hexagonal'}, optional
            Sampling lattice used for the visual-field grid.
        vfmap : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
            Retinotopic map between visual-field and retinal coordinates. Defaults
            to :py:class:`~pulse2percept.topography.Curcio1990Map`.
        n_gray : int or None, optional
            Number of gray levels in the returned percept. ``None`` disables
            gray-level quantization.
        noise : float, int, or None, optional
            Salt-and-pepper noise applied to each percept frame.
        min_current_spread : float, optional
            Inherited Gaussian current-spread cutoff. Not used by the Nanduri
            spatial model.
        dt : float or Quantity, optional
            Simulation time step, in milliseconds. Default: 0.005 ms.
        tau1 : float or Quantity, optional
            Fast-response time constant, in milliseconds. Default: 0.42 ms.
        tau2 : float or Quantity, optional
            Charge-accumulation time constant, in milliseconds. Default:
            45.25 ms.
        tau3 : float or Quantity, optional
            Time constant of the final three-stage low-pass cascade, in
            milliseconds. Default: 26.25 ms.
        eps : float, optional
            Strength of the subtractive charge-accumulation pathway. Default:
            8.73.
        asymptote : float, optional
            Upper asymptote of the logistic peak-response nonlinearity. Default:
            14.
        slope : float, optional
            Scale parameter of the logistic nonlinearity. Default: 3.
        shift : float, optional
            Midpoint of the logistic nonlinearity. Default: 16.
        scale_out : float, optional
            Multiplicative scaling applied to final brightness. Default: 1.
        thresh_percept : float, optional
            Brightness values below this threshold are set to zero. Default: 0.
        reduce : {'peak', 'last'}, optional
            Temporal interval reduction used for automatically selected output
            times. Default: ``'last'``.
        verbose : bool, optional
            Whether to print status messages. Default: True.
        ndim : list of int, optional
            Dimensionalities of ``vfmap`` accepted by the spatial model.
        n_threads : int, optional
            Number of OpenMP threads.
        n_jobs : int or None, optional
            Alias for ``n_threads``. ``None`` and -1 use all available CPU cores.
        """

    def __init__(self, implant, **params):
        super(Nanduri2012Model, self).__init__(
            spatial=Nanduri2012Spatial(implant),
            temporal=Nanduri2012Temporal(), **params)
