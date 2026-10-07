""":py:class:`~pulse2percept.models.retina.Nanduri2012Model`, 
   :py:class:`~pulse2percept.models.retina.Nanduri2012Spatial`, 
   :py:class:`~pulse2percept.models.retina.Nanduri2012Temporal` [Nanduri2012]_"""
import numpy as np
from ..base import Model, TemporalModel
from ..temporal import _charge_chunks, _flush_denormals, _on_torch
from .base import RetinalSpatial
from ...implants import DiskElectrode
from ...topography.retina import Curcio1990Map
from ...units import ms


def _require_disk_electrodes(electrodes):
    """Require disk electrodes, whose radius is used by the Nanduri model."""
    if not all(isinstance(e, DiskElectrode) for e in electrodes):
        raise TypeError("The Nanduri2012 spatial model only supports "
                        "DiskElectrode arrays.")


def _radii(electrode_array, names):
    """Return the float32 radii (microns) of the named electrodes."""
    return np.array([electrode_array[e].radius for e in names],
                    dtype=np.float32)


class Nanduri2012Spatial(RetinalSpatial):
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
        :py:class:`~pulse2percept.models.retina.Nanduri2012Model` for the combined
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
            Horizontal visual-field extent in degrees of visual angle. A
            physical retinal extent may instead be resolved through
            ``visual_field_map``.
        yrange : (float, float) or Quantity, optional
            Vertical visual-field extent in degrees of visual angle. A physical
            retinal extent may instead be resolved through
            ``visual_field_map``.
        step : float, (float, float), or Quantity, optional
            Grid spacing in degrees of visual angle. A pair specifies separate x
            and y spacing.

            .. versionchanged:: 0.10.0
                Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

        grid_type : {'rect', 'hex'}, optional
            Sampling lattice used for the visual-field grid.
        thresh_percept : float, optional
            Brightness values below this threshold are set to zero.
        visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
            Retinotopic map between visual-field and retinal coordinates. Defaults
            to :py:class:`~pulse2percept.topography.retina.Curcio1990Map`.
        n_gray : int or None, optional
            Number of gray levels in the returned percept. ``None`` disables
            gray-level quantization.
        implant_position : (x, y) or Quantity, optional
            Position of the implant's local origin. A bare pair or length is a
            tissue position in microns; a dva position such as
            ``(6, -2) * dva`` is resolved through ``visual_field_map``.

            .. versionadded:: 0.11.0

        implant_rotation : float or Quantity, optional
            In-plane rotation (deg) about the implant's local origin,
            positive counter-clockwise.

            .. versionadded:: 0.11.0

        implant_depth : float or Quantity, optional
            Signed offset (um) along the tissue plane's normal, carried by
            the electrodes' local ``z``. Requires a 2D ``visual_field_map``.

            .. versionadded:: 0.11.0

        location_noise : float or None, optional
            Standard deviation of fixed electrode-specific phosphene offsets, in dva.
            Requires an invertible 2D ``visual_field_map``. ``None`` or 0 disables it.
            Location-dependent models may also change phosphene shape or size.
            
            .. versionadded:: 0.11.0

        verbose : bool, optional
            Whether to print status messages.
        ndim : list of int, optional
            Dimensionalities of ``visual_field_map`` accepted by the model.

        .. versionchanged:: 0.12.0

            Runs on Torch; ``n_threads`` and ``n_jobs`` were removed.
        """

    def __init__(self, implant, *, atten_a=14000, atten_n=1.69,
                 xrange=(-15, 15), yrange=(-15, 15), step=0.25,
                 grid_type='rect', thresh_percept=0,
                 visual_field_map=None, n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None,
                 verbose=True, ndim=None):
        super().__init__(
            implant, atten_a=atten_a, atten_n=atten_n, xrange=xrange,
            yrange=yrange, step=step, grid_type=grid_type,
            thresh_percept=thresh_percept,
            visual_field_map=(Curcio1990Map() if visual_field_map is None else
                              visual_field_map),
            n_gray=n_gray,
            implant_position=implant_position,
            implant_rotation=implant_rotation,
            implant_depth=implant_depth,
            location_noise=location_noise, verbose=verbose,
            ndim=[2] if ndim is None else ndim)

    def get_default_params(self):
        """Return default model parameters."""
        base_params = super(Nanduri2012Spatial, self).get_default_params()
        # Prediction runs on Torch's own thread pool:
        del base_params['n_threads'], base_params['n_jobs']
        params = {'atten_a': 14000, 'atten_n': 1.69}
        return {**base_params, **params}

    def _predict_spatial(self, electrode_array, stim):
        """Predict float32 brightness over the spatial grid."""
        import torch
        # The bound implant may have changed since the last build.
        _require_disk_electrodes(electrode_array.electrode_objects)
        x_el, y_el, z_el = self._electrode_coords(electrode_array, stim)
        r_el = _radii(electrode_array, stim.electrodes)
        values = self._stim_values(stim)
        # Silent electrodes add nothing here. `_predict_tensor` keeps them,
        # so they still receive gradients:
        active = np.any(np.abs(values) > 0, axis=1)
        waveform = torch.tensor(values[active], dtype=torch.float32)
        with torch.inference_mode():
            return self._predict_nanduri_tensor(
                waveform, x_el[active], y_el[active], z_el[active],
                r_el[active]).numpy()

    def _predict_tensor(self, waveform, time):
        """Return the flat Torch response to an ``(n_electrodes, T)`` waveform.

        Runs the ``predict_percept`` kernel on every implant electrode, so
        silent electrodes still receive gradients. Geometry is fixed.
        """
        if self.n_gray is not None:
            # Quantization is discrete and has no exact gradient:
            raise NotImplementedError("Tensor prediction does not support "
                                      "n_gray; set n_gray=None.")
        electrode_array = self.implant.electrode_array
        _require_disk_electrodes(electrode_array.electrode_objects)
        names = self.implant.electrode_names
        x_el, y_el, z_el = self._electrode_coords(electrode_array, None,
                                                  electrodes=names)
        resp = self._predict_nanduri_tensor(waveform, x_el, y_el, z_el,
                                            _radii(electrode_array, names))
        return self._spatial_response(resp, time, None)

    def _predict_nanduri_tensor(self, waveform, x_el, y_el, z_el, r_el):
        """Return the flat thresholded ``(P, T)`` response.

        ``waveform`` rows follow the float32 electrode coordinates and radii
        (microns). Geometry and weights are float32; the response has the
        dtype and device of ``waveform``.
        """
        import torch
        device = waveform.device
        x, y = (torch.as_tensor(np.ravel(c), dtype=torch.float32,
                                device=device)[:, None]
                for c in (self.grid.ret.x, self.grid.ret.y))
        x_el, y_el, z_el, r_el = (torch.as_tensor(c, dtype=torch.float32,
                                                  device=device)
                                  for c in (x_el, y_el, z_el, r_el))
        # Python floats holding float32 values keep Torch in float32:
        atten_a = float(np.float32(self.atten_a))
        atten_n = float(np.float32(self.atten_n))
        # Distance to the nearest point of the disk; depends on |z| only:
        edge = torch.clamp(((x - x_el) ** 2 + (y - y_el) ** 2).sqrt() - r_el,
                           min=0)
        dist = (edge ** 2 + z_el ** 2).sqrt()
        weights = atten_a / (atten_a + dist ** atten_n)
        # Unmapped grid points are zero:
        weights = torch.where(x.isnan() | y.isnan(), 0.0, weights)
        resp = weights.to(waveform.dtype) @ waveform
        thresh = float(np.float32(self.thresh_percept))
        # Zeroes only `|resp| < thresh`, so NaN propagates. `+ 0.0` turns -0.0
        # into 0.0:
        return torch.where(resp.abs() < thresh, 0.0, resp) + 0.0

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
        component alone. Use :py:class:`~pulse2percept.models.retina.Nanduri2012Model`
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

        .. versionchanged:: 0.12.0

            Runs on Torch; ``n_threads`` and ``n_jobs`` were removed.
        """

    # Positive current drives the Nanduri temporal cascade.
    _drive_sign = 1

    def __init__(self, *, dt=0.005, tau1=0.42, tau2=45.25, tau3=26.25,
                 eps=8.73, asymptote=14.0, slope=3.0, shift=16.0,
                 scale_out=1.0, thresh_percept=0, reduce='last', verbose=True):
        super().__init__(
            dt=dt, tau1=tau1, tau2=tau2, tau3=tau3, eps=eps,
            asymptote=asymptote, slope=slope, shift=shift,
            scale_out=scale_out, thresh_percept=thresh_percept, reduce=reduce,
            verbose=verbose)

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
        # Torch runs on its own thread pool:
        del base_params['n_threads'], base_params['n_jobs']
        return {**base_params, **params}

    def get_param_units(self):
        """Return units used to store model parameters."""
        return {**super().get_param_units(), 'tau1': ms, 'tau2': ms,
                'tau3': ms}

    def _predict_temporal(self, stim, t_percept):
        """Predict the float32 NumPy temporal response."""
        return _on_torch(self._predict_nanduri, self._stim_values(stim),
                         self._stim_times(stim), t_percept)

    def _predict_temporal_tensor(self, stim, t_percept, reduce='last'):
        """Predict the temporal response to tensor data, keeping its dtype
        and autograd graph. ``reduce`` is applied by the caller."""
        return self._predict_nanduri(stim.data, self._stim_times(stim),
                                     t_percept)

    @_flush_denormals()
    def _predict_nanduri(self, data, time, t_percept):
        """Return the Torch response to ``data`` sampled at ``time``.

        The logistic gain depends on the peak of ``R_3`` over the whole
        simulation, so the slow cascade runs in a second pass. The cascade is
        linear, so each chunk's input is projected once and scaled later.
        Timing is fixed.
        """
        import torch
        f32 = np.float32
        thresh = float(f32(self.thresh_percept))
        # Legacy: an output below threshold also resets the state, so each
        # output must end a chunk:
        merge = not thresh > 0
        n_space = data.reshape((-1, len(time))).shape[0]
        # Lower bound of `max_r3`, as in the legacy kernel:
        peak = data.new_full((n_space,), 1e-37)
        chunks = []
        # Time-major, as in `_charge_chunks`:
        for x, carry, weights, offs in _charge_chunks(self, data, time,
                                                      t_percept, 1, merge):
            if x is None:
                chunks.append((carry, None, len(offs)))
                continue
            r3 = torch.clamp(x, min=0)
            peak = torch.maximum(peak, r3.amax(dim=0))
            chunks.append((carry, weights @ r3, len(offs)))
        gain = (float(f32(self.asymptote)) *
                torch.sigmoid((peak - float(f32(self.shift))) /
                              float(f32(self.slope))) / peak)
        state = data.new_zeros((3, n_space))
        rows = []
        for carry, proj, m in chunks:
            resp = carry @ state
            if proj is not None:
                resp = torch.addcmul(resp, proj, gain)
            state = resp[m:]
            if m and not merge:
                last = torch.where(state[2:].abs() < thresh, 0.0, state[2:])
                state = torch.cat((state[:2], last))
                rows.append(last)
            else:
                rows.append(resp[:m])
        resp = torch.cat(rows).T.contiguous()
        return resp * float(f32(self.scale_out))


class Nanduri2012Model(Model):
    r"""Combined spatial-temporal model of [Nanduri2012]_.

        Combines :py:class:`~pulse2percept.models.retina.Nanduri2012Spatial` with
        :py:class:`~pulse2percept.models.retina.Nanduri2012Temporal`. See those classes
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
            Horizontal visual-field extent in degrees of visual angle. A
            physical retinal extent may instead be resolved through
            ``visual_field_map``.
        yrange : (float, float) or Quantity, optional
            Vertical visual-field extent in degrees of visual angle. A physical
            retinal extent may instead be resolved through
            ``visual_field_map``.
        step : float, (float, float), or Quantity, optional
            Grid spacing in degrees of visual angle. A pair specifies separate x
            and y spacing.

            .. versionchanged:: 0.10.0
                Renamed from ``xystep``; ``xystep`` was removed in 0.11.0.

        grid_type : {'rect', 'hex'}, optional
            Sampling lattice used for the visual-field grid.
        visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`, optional
            Retinotopic map between visual-field and retinal coordinates. Defaults
            to :py:class:`~pulse2percept.topography.retina.Curcio1990Map`.
        n_gray : int or None, optional
            Number of gray levels in the returned percept. ``None`` disables
            gray-level quantization.
        implant_position : (x, y) or Quantity, optional
            Position of the implant's local origin. A bare pair or length is a
            tissue position in microns; a dva position such as
            ``(6, -2) * dva`` is resolved through ``visual_field_map``.

            .. versionadded:: 0.11.0

        implant_rotation : float or Quantity, optional
            In-plane rotation (deg) about the implant's local origin,
            positive counter-clockwise.

            .. versionadded:: 0.11.0

        implant_depth : float or Quantity, optional
            Signed offset (um) along the tissue plane's normal, carried by
            the electrodes' local ``z``. Requires a 2D ``visual_field_map``.

            .. versionadded:: 0.11.0

        location_noise : float or None, optional
            Standard deviation of fixed electrode-specific phosphene offsets, in dva.
            Requires an invertible 2D ``visual_field_map``. ``None`` or 0 disables it.
            Location-dependent models may also change phosphene shape or size.
            
            .. versionadded:: 0.11.0

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
            Dimensionalities of ``visual_field_map`` accepted by the spatial
            model.

        .. versionchanged:: 0.12.0

            Runs on Torch; ``n_threads`` and ``n_jobs`` were removed.
        """

    def __init__(self, implant, *, atten_a=14000, atten_n=1.69,
                 xrange=(-15, 15), yrange=(-15, 15), step=0.25,
                 grid_type='rect',
                 visual_field_map=None,
                 n_gray=None,
                 implant_position=(0, 0), implant_rotation=0,
                 implant_depth=0,
                 location_noise=None, ndim=None, dt=0.005, tau1=0.42,
                 tau2=45.25, tau3=26.25, eps=8.73, asymptote=14.0, slope=3.0,
                 shift=16.0, scale_out=1.0, reduce='last', thresh_percept=0,
                 verbose=True):
        # `thresh_percept` and `verbose` are declared by both components and
        # are applied to both.
        super().__init__(
            spatial=Nanduri2012Spatial(
                implant, atten_a=atten_a, atten_n=atten_n, xrange=xrange,
                yrange=yrange, step=step, grid_type=grid_type,
                visual_field_map=visual_field_map,
                n_gray=n_gray,
                implant_position=implant_position,
                implant_rotation=implant_rotation,
                implant_depth=implant_depth,
                location_noise=location_noise, ndim=ndim,
                thresh_percept=thresh_percept, verbose=verbose),
            temporal=Nanduri2012Temporal(
                dt=dt, tau1=tau1, tau2=tau2, tau3=tau3, eps=eps,
                asymptote=asymptote, slope=slope, shift=shift,
                scale_out=scale_out, reduce=reduce,
                thresh_percept=thresh_percept, verbose=verbose))
