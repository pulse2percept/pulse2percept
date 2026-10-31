""":py:func:`~pulse2percept.to_torch`, :py:func:`~pulse2percept.export_onnx`"""
import importlib.util
import json
import logging
import numbers
import os
import warnings
from copy import deepcopy

import numpy as np

from .base import Model, SpatialModel
from ..implants.base import Implant, _bilinear_weights
from ..stimuli import AmplitudeEncoder, ImageStimulus
from ..stimuli.encoders import (_FrameClock, _frame_timing,
                                _implant_tensor_gap, _video_frames)
from ..units import DimensionMismatchError, Hz, as_value, ms, uA, um, xTh
from ..units.base import has_units

#: ONNX opset of exported graphs. Unity Inference Engine 2.6.1 reads opsets
#: up to 25; 15 covers every operator these graphs use.
_OPSET = 15

#: Agreement required of every exported graph, relative to the peak |response|
#: of the validation image. float32 matrix products with summation order left
#: to the runtime (BLAS, ONNX Runtime) differ by a few units in the last place
#: of each term.
_RTOL = 1e-5

#: Largest graph a single ``.onnx`` file can hold (protobuf limit, bytes).
_MAX_BYTES = 2 ** 31


def to_torch(model, input_shape=None, *, input_kind=None, fps=None,
             time=None, t_percept=None, freq=None, phase_dur=None):
    """Return a configured model as a batched, differentiable Torch module

    The module runs the supported stages of ``model.predict_percept`` on
    batches of Torch tensors: image sampling at the electrodes, the implant's
    encoder, implant constraints, and the spatial and temporal models.
    Gradients flow from the percept to the input through every continuous
    stage. The model is built first if needed.

    .. versionadded:: 0.12.0

    Parameters
    ----------
    model : :py:class:`~pulse2percept.models.Model` or SpatialModel
        Supported spatial models are
        :py:class:`~pulse2percept.models.retina.ScoreboardSpatial`,
        :py:class:`~pulse2percept.models.retina.AxonMapSpatial`,
        :py:class:`~pulse2percept.models.retina.Nanduri2012Spatial`,
        :py:class:`~pulse2percept.models.retina.Thompson2003Spatial` (without
        ``dropout``),
        :py:class:`~pulse2percept.models.cortex.ScoreboardSpatial`, and
        :py:class:`~pulse2percept.models.retina.Granley2023Model`, all with
        ``n_gray=None``. Supported temporal models are
        :py:class:`~pulse2percept.models.FadingTemporal`,
        :py:class:`~pulse2percept.models.AlphaTemporal`,
        :py:class:`~pulse2percept.models.retina.Nanduri2012Temporal`, and
        :py:class:`~pulse2percept.models.retina.Horsager2009Temporal`. The
        implant must be driven by electrical current, without custom
        ``preprocess``, ``reshape_stim``, ``check_stim``, or ``prepare_stim``.
    input_shape : (H, W) or (n_frames, H, W), optional
        Image or video size in pixels. Required unless
        ``input_kind='electrodes'``.
    input_kind : {'image', 'video', 'electrodes'}, optional
        Defaults to 'image' or 'video', from the length of ``input_shape``.
        'electrodes' starts after image sampling and encoding, which it
        bypasses; see Notes.
    fps : float, optional
        Video frame rate (Hz). If omitted, frames are 1 ms apart, as in
        :py:class:`~pulse2percept.stimuli.VideoStimulus`.
    time : array-like, optional
        Sample times (ms) of electrode waveforms. Required for
        ``input_kind='electrodes'`` with a temporal model, not allowed
        otherwise.
    t_percept : float or array-like, optional
        Output times, as in ``predict_percept``. Requires a temporal model.
    freq, phase_dur : float or array-like, optional
        Default pulse frequency (Hz) and phase duration (ms), scalar or one
        per electrode, of a Granley2023Model with
        ``input_kind='electrodes'``. Either may instead be passed to every
        call.

    Returns
    -------
    module : :py:class:`torch.nn.Module`
        Maps an input batch to raw percepts, not normalized for display:
        ``(B, 1, Hp, Wp)``, or ``(B, 1, T, Hp, Wp)`` for videos and temporal
        models (a one-frame video has ``T=1``).
        Attributes: ``electrode_names`` (order of electrode input),
        ``grid`` (percept grid, dva; row 0 at the largest y),
        ``t_percept`` (output times, ms, or None), and ``time`` (sample
        times of waveform input, ms, or None).

    Notes
    -----
    *  Image input is ``(B, C, H, W)`` and video input
       ``(B, C, n_frames, H, W)``: finite float gray levels in [0, 1].
       ``C=3`` (RGB) is converted to gray as in
       :py:meth:`~pulse2percept.implants.Implant.reshape_stim`; no supported
       model uses color. The image spans the electrode bounding box, row 0 at
       the smallest device y, as for an
       :py:class:`~pulse2percept.stimuli.ImageStimulus`.
    *  Images require the implant's encoder to be an
       :py:class:`~pulse2percept.stimuli.AmplitudeEncoder` without
       ``n_levels`` or ``stretch``. A spatial-only Scoreboard or AxonMap
       model without an encoder uses gray levels as relative drive.
    *  ``input_kind='electrodes'`` takes current (uA, negative is
       cathodic), rows in ``electrode_names`` order: ``(B, E)`` for a
       spatial-only model, ``(B, E, T)`` sampled at ``time`` with a temporal
       model. For a Granley2023Model, it takes amplitude in threshold
       multiples (xTh) ``(B, E)``; ``freq`` and ``phase_dur`` passed to a
       call may be ``(B, E)`` tensors that require gradients.
    *  Deactivated electrodes deliver nothing. Threshold multiples are
       converted to current with ``implant.thresholds``, which must cover
       every activated electrode or none. ``safe_mode`` and ``max_current``
       are checked on every call and raise ValueError, as in
       ``prepare_stim``; they have no gradient.
    *  Gradients are exact where the model is smooth. At discontinuities
       (gray-level clipping, thresholds, the Granley2023 amplitude cutoff,
       AxonMap's winning segment at ties), autograd returns the gradient of
       one side. Pulse timing is fixed and has no gradient, so
       :py:class:`~pulse2percept.stimuli.FrequencyEncoder` is not supported.
    *  The module is a snapshot: changing ``model`` afterward does not
       change it. Weights and geometry are float32 buffers; call ``.to``
       for another device and ``.double()`` for float64.
    *  Unlike ``predict_percept``, a video yields one output frame per input
       frame even if consecutive frames are identical.

    Examples
    --------
    >>> import torch
    >>> import pulse2percept as p2p
    >>> implant = p2p.implants.retina.ArgusII()
    >>> implant.encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50))
    >>> model = p2p.models.retina.ScoreboardModel(implant, rho=150)
    >>> module = p2p.to_torch(model, (28, 28))  # doctest: +SKIP
    >>> percepts = module(torch.rand(16, 1, 28, 28))  # doctest: +SKIP

    """
    import torch
    from ._deploy import _ImageSampler, _Temporal, _Workflow
    spatial, temporal = _stages(model)
    implant = spatial.implant
    if input_kind is None:
        input_kind = {2: 'image', 3: 'video'}.get(
            0 if input_shape is None else len(input_shape))
    if input_kind not in ('image', 'video', 'electrodes'):
        raise ValueError(f"'input_kind' must be 'image', 'video' or "
                         f"'electrodes', or follow from an 'input_shape' "
                         f"(H, W) or (n_frames, H, W); got input_kind="
                         f"{input_kind!r}, input_shape={input_shape}.")
    electrodes = input_kind == 'electrodes'
    n_dims = {'image': 2, 'video': 3}.get(input_kind)
    if (input_shape is None) != electrodes or (
            not electrodes and len(input_shape) != n_dims):
        raise ValueError(f"'input_shape' must be (H, W) for images, "
                         f"(n_frames, H, W) for videos, and None for "
                         f"electrode input; got {input_shape} for "
                         f"input_kind={input_kind!r}.")
    if fps is not None and input_kind != 'video':
        raise ValueError("'fps' requires video input.")
    if (time is not None) != (electrodes and temporal is not None):
        raise ValueError("'time' is required for, and only allowed with, "
                         "electrode input to a temporal model.")
    if t_percept is not None and temporal is None:
        raise ValueError("'t_percept' requires a temporal model.")
    granley = _is_granley2023(spatial)
    if (freq is not None or phase_dur is not None) and not (
            granley and electrodes):
        raise ValueError("'freq' and 'phase_dur' are only for a "
                         "Granley2023Model with input_kind='electrodes'.")
    gap = _implant_tensor_gap(implant, checked=True)
    if gap is not None:
        raise NotImplementedError(gap)
    names = implant.electrode_names
    on = np.array([e.activated for e in implant.electrode_objects])
    checks = (implant.safe_mode, implant.max_current)
    checked = implant.safe_mode or implant.max_current is not None
    args = {'shape': spatial.grid.x.shape, 'electrode_names': names,
            'grid': deepcopy(spatial.grid), 'on': on}
    if granley:
        from ._deploy import _Granley2023
        args['granley'] = _Granley2023(spatial)
    else:
        args['spatial'] = _spatial_stage(spatial)
    unit = uA
    if not electrodes:
        shape = _input_shape(input_shape[-2:])
        n_frames = 1 if input_kind == 'image' else _n_frames(input_shape[0])
        rows, cols, weights = _bilinear_weights(*implant._image_grid(shape))
        args['sampler'] = _ImageSampler(
            torch.as_tensor(rows * shape[1] + cols).reshape(-1),
            torch.as_tensor(weights, dtype=torch.float32).reshape(-1), shape,
            video=input_kind == 'video')
        encoder = implant.encoder
        if encoder is None:
            if (temporal is not None or granley or
                    not spatial._accepts_dimensionless_drive):
                raise NotImplementedError(
                    f"Image input to this model requires an implant encoder "
                    f"(an AmplitudeEncoder): {type(spatial).__name__}"
                    f"{' with a temporal model' if temporal else ''} needs "
                    f"physical stimulation.")
            if type(implant).reshape_stim is not Implant.reshape_stim:
                raise NotImplementedError(
                    "to_torch does not support a custom 'reshape_stim'.")
            # Dimensionless drive skips calibration and electrical checks:
            checked = False
            unit = None
            times = None
            if input_kind == 'video':
                times = _video_frames(n_frames, fps=fps)[0]
        else:
            if granley:
                _require_granley_pulses(encoder, input_kind)
            pulses, sched, unit = _pulse_stage(implant, encoder, input_kind,
                                               n_frames, fps)
            args['pulses'] = pulses
            # Videos keep their time axis, even with one frame:
            times = sched._frame_time if input_kind == 'video' else None
            if granley:
                # Realized pulse parameters, as `_biphasic_params` reads them:
                args['freq'] = torch.as_tensor(sched._freq[:, 0],
                                               dtype=torch.float32)
                args['phase_dur'] = torch.tensor(sched._phase_dur,
                                                 dtype=torch.float32)
        if temporal is not None:
            time, clock = pulses.time, pulses.frame_clock
    else:
        times = None
        if temporal is not None:
            time = np.asarray(as_value(time, spatial.time_unit, 'time'),
                              dtype=np.float64)
            if (time.ndim != 1 or time.size == 0 or
                    not np.all(np.isfinite(time)) or
                    np.any(np.diff(time) <= 0)):
                raise ValueError("'time' must be one-dimensional, nonempty, "
                                 "finite, and strictly increasing.")
            args['time'], clock = time, None
        if granley:
            unit = xTh
            if checked:
                raise NotImplementedError(
                    "to_torch does not check 'safe_mode' or 'max_current' "
                    "of Granley2023 pulse parameters.")
            for name, value in (('freq', freq), ('phase_dur', phase_dur)):
                if value is not None:
                    args[name] = _pulse_param(name, value, len(names))
        elif checked:
            args['checks'] = checks
    if checked and unit != uA:
        # As `Implant._require_current_stim`:
        raise DimensionMismatchError(
            "Safety checks need an electrical stimulus to check, and this "
            "one is a multiple of threshold. Set 'thresholds' on the "
            "implant for every activated electrode.")
    if granley and unit == uA:
        args['xth_scale'] = _threshold_scale(implant, on, 'Granley2023Model '
                                             'amplitudes in xTh')
    elif not granley and unit == xTh:
        raise DimensionMismatchError(
            f"{type(spatial).__name__} expects current, and the encoder "
            f"delivers threshold multiples. Set 'thresholds' on the implant "
            f"for every activated electrode, or give 'amp_range' in uA.")
    if temporal is not None:
        t_percept = as_value(t_percept, temporal.time_unit, 't_percept')
        args['temporal'] = _Temporal(_snapshot(temporal), time, clock,
                                     t_percept)
        times = args['temporal'](torch.zeros((1, len(time)))).time
    module = _Workflow(**args)
    module.t_percept = None if times is None else np.array(times)
    return module


def _stages(model):
    """Return the built spatial and temporal (or None) stages of a model."""
    if isinstance(model, Model):
        if not model.has_space:
            raise NotImplementedError("to_torch requires a spatial model: a "
                                      "temporal-only model has no implant.")
        model._build_stale()
        temporal = model.temporal
        if temporal is not None and (
                '_predict_temporal_tensor' not in vars(type(temporal))):
            raise NotImplementedError(f"to_torch does not support "
                                      f"{type(temporal).__name__}.")
        return model.spatial, temporal
    if isinstance(model, SpatialModel):
        if not model.is_built:
            model.build()
        return model, None
    raise TypeError(f"'model' must be a Model or SpatialModel, not "
                    f"{type(model)}.")


def _snapshot(model):
    """Return a built deep copy of a model stage, implant included."""
    implant = getattr(model, '_implant', None)
    # `BaseModel.__deepcopy__` shares the implant unless it is in `memo`:
    memo = {} if implant is None else {id(implant): deepcopy(implant)}
    return deepcopy(model, memo)


def _is_granley2023(spatial):
    """Return whether ``spatial`` is the Granley2023 stage (no subclass)."""
    from .retina.granley2023 import _Granley2023Spatial
    if type(spatial) is _Granley2023Spatial:
        if spatial.n_gray is not None:
            raise NotImplementedError(
                "to_torch does not support n_gray: gray-level quantization "
                "is discrete and has no gradient. Set n_gray=None.")
        return True
    return False


def _spatial_stage(spatial):
    """Return the ``(E, N) -> (P, N)`` Torch module of a built stage."""
    from ._deploy import _SpatialKernel
    name = type(spatial).__name__
    if spatial._needs_structured_stim:
        raise NotImplementedError(
            f"to_torch does not support {name}, which reads pulse-train "
            f"parameters this interface does not provide.")
    if spatial.n_gray is not None:
        raise NotImplementedError(
            "to_torch does not support n_gray: gray-level quantization is "
            "discrete and has no gradient. Set n_gray=None.")
    if not spatial._tensor_exact:
        raise NotImplementedError(f"to_torch does not support {name} with "
                                  f"the current parameters.")
    # Only classes that define their own: a subclass can change semantics
    # in ways an inherited one would ignore.
    if '_torch_adapter' in vars(type(spatial)):
        return spatial._torch_adapter()
    if '_predict_tensor' in vars(type(spatial)):
        return _SpatialKernel(_snapshot(spatial))
    raise NotImplementedError(f"to_torch does not support {name}.")


def _n_frames(n_frames):
    """Return a positive int number of video frames."""
    if not (float(n_frames).is_integer() and n_frames >= 1):
        raise ValueError(f"The number of video frames must be a positive "
                         f"integer, not {n_frames}.")
    return int(n_frames)


def _pulse_stage(implant, encoder, input_kind, n_frames, fps):
    """Return the ``_Pulses`` stage, its schedule, and its delivered unit."""
    import torch
    from ._deploy import _Pulses
    if type(encoder) is not AmplitudeEncoder:
        why = (" changes pulse timing with gray level, which has no gradient"
               if type(encoder).__name__ == 'FrequencyEncoder' else "")
        raise NotImplementedError(
            f"to_torch requires an AmplitudeEncoder; {type(encoder).__name__}"
            f"{why}.")
    gap = encoder._tensor_gap(checked=True)
    if gap is not None:
        raise NotImplementedError(gap)
    if input_kind == 'video':
        frame_time, frame_dur = _video_frames(n_frames, fps=fps,
                                              frame_dur=encoder.frame_dur)
    else:
        frame_time, frame_dur = _frame_timing(None, 1,
                                              frame_dur=encoder.frame_dur)
    names = implant.electrode_names
    # The schedule does not depend on amplitude:
    _, sched = encoder._tensor_schedule(
        torch.zeros((len(names), n_frames)), frame_time, frame_dur,
        timed=input_kind == 'video' and encoder.frame_dur is None)
    frame, wave = sched._pulse_index()
    # Identical columns give identical spatial responses; compute each once:
    key, columns = np.unique(np.concatenate([frame, wave]), axis=1,
                             return_inverse=True)
    on = np.array([e.activated for e in implant.electrode_objects])
    unit, scale = encoder.amp_unit, None
    if unit == xTh and implant.thresholds:
        scale = _threshold_scale(implant, on, 'threshold calibration')
        unit = uA
    lo, hi = encoder.amp_range
    pulses = _Pulses(lo, hi - lo, on, scale, sched._firing,
                     key[:len(names)].astype(np.int64), key[len(names):],
                     np.ravel(columns), sched.time.copy(),
                     _FrameClock(sched._frame_time, sched._frame_dur,
                                 sched._source_time, sched._source_dur),
                     implant.safe_mode, implant.max_current)
    return pulses, sched, unit


def _threshold_scale(implant, on, purpose):
    """Return float32 thresholds (uA) of all activated electrodes."""
    import torch
    thresholds = implant.thresholds
    missing = [n for n, a in zip(implant.electrode_names, on)
               if a and n not in thresholds]
    if missing:
        raise NotImplementedError(
            f"to_torch requires implant thresholds for every activated "
            f"electrode for {purpose}; missing: {', '.join(missing)}.")
    return torch.as_tensor(np.array(
        [thresholds.get(n, 1.0) for n in implant.electrode_names],
        dtype=np.float32))


def _require_granley_pulses(encoder, input_kind):
    """Require an encoder whose pulses Granley2023Model reads."""
    if input_kind != 'image':
        raise NotImplementedError("Granley2023Model predicts one percept per "
                                  "pulse train; to_torch supports still "
                                  "images only.")
    if encoder.pulse is not None or not encoder.cathodic_first:
        raise NotImplementedError(
            "Granley2023Model requires cathodic-first biphasic pulses: an "
            "encoder without a custom 'pulse' and with cathodic_first=True.")


def _pulse_param(name, value, n_el):
    """Return a fixed Granley2023 pulse parameter as a float32 tensor."""
    import torch
    value = np.asarray(as_value(value, Hz if name == 'freq' else ms, name),
                       dtype=np.float32)
    if value.shape not in ((), (n_el,)):
        raise ValueError(f"'{name}' must be a scalar or have one entry per "
                         f"electrode ({n_el}), not shape {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"'{name}' must be finite.")
    if name == 'phase_dur' and np.any(value <= 0):
        raise ValueError("'phase_dur' must be positive.")
    return torch.as_tensor(value)


def export_onnx(model, path, input_shape, *, white_point=None):
    """Export the image-to-percept workflow of a spatial model to ONNX

    Writes ``path`` and a JSON sidecar next to it (``model.onnx`` and
    ``model.json``). The graph reproduces ``model.predict_percept(image)``
    for a gray image of ``input_shape``: bilinear sampling of the image at
    every electrode, frame-level encoder modulation, and the spatial model.
    The model is built first if needed.

    .. versionadded:: 0.12.0

    Parameters
    ----------
    model : :py:class:`~pulse2percept.models.Model` or SpatialModel
        A spatial-only model. Supported spatial models are
        :py:class:`~pulse2percept.models.retina.ScoreboardSpatial`,
        :py:class:`~pulse2percept.models.cortex.ScoreboardSpatial`, and
        :py:class:`~pulse2percept.models.retina.AxonMapSpatial`, with
        ``n_gray=None``. The implant's encoder must
        be an :py:class:`~pulse2percept.stimuli.AmplitudeEncoder` with
        ``amp_range`` in uA, without ``n_levels`` or ``stretch``, on an
        implant without custom preprocessing or sampling, ``safe_mode``, or
        ``max_current``. An :py:class:`~pulse2percept.implants.EnsembleImplant`
        (not a subclass) is supported under the same conditions, with one
        image driving the whole ensemble; per-implant inputs are not
        exported.
    path : str or path-like
        Output ``.onnx`` file.
    input_shape : (int, int)
        Image height and width in pixels.
    white_point : float, optional
        Raw ``percept`` value to display as white, stored in the sidecar as
        ``display.white_point``. Must be finite and positive. By default,
        twice the peak response of ``model.predict_percept`` to a uniform
        gray image (0.5) of ``input_shape``, or 1.0 if that response is
        nowhere positive (e.g., ``freq=0``). Does not change the graph.

    Notes
    -----
    *  Input ``image`` is float32 ``(1, 1, H, W)``, gray levels in [0, 1].
       The image spans the electrode bounding box in device coordinates,
       row 0 at the smallest device y, as for an
       :py:class:`~pulse2percept.stimuli.ImageStimulus`. For an ensemble,
       the box spans the electrodes of all its implants.
    *  Output ``percept`` is float32 ``(1, 1, Hp, Wp)`` on the model's grid,
       row 0 at the largest y (dva). Values are the raw spatial response,
       not normalized for display, and may exceed 1.
    *  A viewer shows ``clip(max(0, percept) / white_point, 0, 1)``. The
       default white point is a visualization convention, not a
       psychophysical brightness calibration: the gray reference peaks at
       half brightness. Defaults differ between models, so displayed
       brightness is comparable across models only at a shared explicit
       ``white_point``.
    *  Weights are precomputed from the model's fixed geometry. No
       approximation is made: the graph agrees with ``predict_percept`` to
       float32 rounding, checked on a test image after export (with ONNX
       Runtime, if installed).
    *  Temporal models, color, video, scenes, and gaze are not exported.
    *  Requires the ``onnx`` and ``onnxscript`` packages.

    Examples
    --------
    >>> import pulse2percept as p2p
    >>> implant = p2p.implants.retina.ArgusII()
    >>> implant.encoder = p2p.stimuli.AmplitudeEncoder(amp_range=(0, 50))
    >>> model = p2p.models.retina.ScoreboardModel(implant)
    >>> p2p.export_onnx(model, 'argus.onnx', (60, 100))  # doctest: +SKIP

    """
    import torch
    spatial = _spatial_model(model)
    input_shape = _input_shape(input_shape)
    if white_point is not None:
        white_point = _white_point(white_point)
    if not spatial.is_built:
        spatial.build()
    module = _deploy_module(spatial, input_shape)
    n_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
    if n_bytes >= _MAX_BYTES:
        raise ValueError(
            f"The exact graph of this {type(spatial).__name__} holds "
            f"{n_bytes / 2 ** 30:.1f} GiB of weights, more than one ONNX file "
            f"can hold. Use a coarser grid or a smaller field of view.")
    _save_onnx(module, torch.zeros((1, 1) + input_shape), path)
    try:
        _validate(model, module, path, input_shape)
        if white_point is None:
            white_point = _reference_white_point(model, input_shape)
    except BaseException:
        os.remove(path)
        raise
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:
        json.dump(_sidecar(model, spatial, input_shape, white_point), f,
                  indent=1)


def _save_onnx(module, example, path):
    """Export ``module`` with input ``image`` and output ``percept``."""
    import torch
    # The exporter reports its opset conversion and skipped torchvision ops,
    # neither of which concerns this graph:
    loggers = [logging.getLogger(name)
               for name in ('torch.onnx', 'onnxscript')]
    levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            torch.onnx.export(module, (example,), path, dynamo=True,
                              opset_version=_OPSET, input_names=['image'],
                              output_names=['percept'], external_data=False,
                              verbose=False)
    finally:
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


def _spatial_model(model):
    """Return the spatial component of a spatial-only model."""
    if isinstance(model, Model):
        if model.has_time:
            raise NotImplementedError(
                f"ONNX export supports spatial-only models; this Model has "
                f"a {type(model.temporal).__name__} stage.")
        return model.spatial
    if isinstance(model, SpatialModel):
        return model
    raise TypeError(f"'model' must be a Model or SpatialModel, not "
                    f"{type(model)}.")


def _input_shape(input_shape):
    """Return ``(H, W)`` as positive ints."""
    shape = tuple(np.ravel(input_shape))
    if (len(shape) != 2 or not all(float(s).is_integer() and s >= 1
                                   for s in shape)):
        raise ValueError(f"'input_shape' must be a (height, width) pair of "
                         f"positive integers, not {input_shape}.")
    return tuple(int(s) for s in shape)


def _white_point(white_point):
    """Return an explicit ``white_point`` as a finite positive float."""
    # bool is an int, but True is not a deliberate white point:
    if (isinstance(white_point, bool) or
            not isinstance(white_point, numbers.Real)):
        raise TypeError(f"'white_point' must be a number or None, not "
                        f"{type(white_point).__name__}.")
    # Checked after conversion: a long double can overflow to inf.
    value = float(white_point)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"'white_point' must be finite and positive, not "
                         f"{white_point}.")
    return value


def _reference_white_point(model, input_shape):
    """Return twice the peak response to uniform 50% gray, or 1.0 if it is
    nowhere positive."""
    gray = np.full(input_shape, 0.5, dtype=np.float32)
    data = model.predict_percept(ImageStimulus(gray)).data
    # A zero response is legitimate; NaN or Inf is a numerical failure:
    if not np.all(np.isfinite(data)):
        raise ValueError("Cannot compute the default 'white_point': the 50% "
                         "gray reference percept contains NaN or Inf.")
    peak = float(np.max(data, initial=0))
    return 2 * peak if peak > 0 else 1.0


def _deploy_module(spatial, input_shape):
    """Return the Torch module of a built model's image-to-percept graph.

    The graph is that of ``to_torch``. Raises NotImplementedError for any
    configuration the graph would not reproduce.
    """
    # Only classes that define their own adapter: a subclass can change
    # semantics in ways an inherited one would ignore.
    if '_torch_adapter' not in vars(type(spatial)):
        raise NotImplementedError(
            f"ONNX export does not support {type(spatial).__name__}.")
    unsupported = [name for name, on in (
        ('n_gray', spatial.n_gray is not None),
        ('the current parameters', not spatial._tensor_exact)) if on]
    if unsupported:
        raise NotImplementedError(
            f"ONNX export does not support {type(spatial).__name__} with "
            f"{', '.join(unsupported)}.")
    implant = spatial.implant
    encoder = implant.encoder
    if type(encoder) is not AmplitudeEncoder:
        raise NotImplementedError(
            f"ONNX export requires an implant whose encoder is an "
            f"AmplitudeEncoder, not {type(encoder).__name__}.")
    # The tensor route `predict_percept` takes for gray images, without
    # calibration or safety checks:
    gap = encoder._tensor_gap()
    if gap is not None:
        raise NotImplementedError(gap)
    return to_torch(spatial, input_shape).eval()


def _validation_image(shape):
    """Return an asymmetric gray ramp, so flips and transposes show."""
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    return ((0.7 * (rows + 0.5) / shape[0] + 0.3 * (cols + 0.5) / shape[1]) **
            2).astype(np.float32)


def _require_close(actual, desired, thresh, what):
    """Raise RuntimeError unless ``actual`` matches ``desired``.

    Values may differ by ``_RTOL`` times the peak response, or flip across
    ``thresh`` within that tolerance.
    """
    actual, desired = np.ravel(actual), np.ravel(desired)
    # NaN compares False against any tolerance, so it would pass below:
    if not (np.all(np.isfinite(actual)) and np.all(np.isfinite(desired)) and
            np.isfinite(thresh)):
        raise RuntimeError(
            f"The exported graph cannot be validated against {what}: "
            f"non-finite response or threshold (NaN or Inf).")
    atol = _RTOL * max(np.abs(desired).max(initial=0), thresh)
    diff = np.abs(actual - desired)
    flip = (((actual == 0) | (desired == 0)) &
            (np.abs(diff - thresh) <= atol))
    bad = (diff > atol) & ~flip
    if np.any(bad):
        raise RuntimeError(
            f"The exported graph does not reproduce {what}: {bad.sum()} of "
            f"{bad.size} values differ by up to {diff[bad].max():.3g} "
            f"(tolerance {atol:.3g}). Please report this.")


def _validate(model, module, path, input_shape):
    """Check the module, and the ONNX file if ONNX Runtime is installed,
    against ``predict_percept`` on a test image."""
    import torch
    image = _validation_image(input_shape)
    expected = model.predict_percept(ImageStimulus(image)).data
    thresh = float(_spatial_model(model).thresh_percept)
    with torch.inference_mode():
        eager = module(torch.as_tensor(image)[None, None]).numpy()
    _require_close(eager, expected, thresh, 'predict_percept')
    if importlib.util.find_spec('onnxruntime') is None:
        return
    import onnxruntime
    session = onnxruntime.InferenceSession(
        os.fspath(path), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'image': image[None, None]})[0]
    _require_close(actual, eager, thresh, 'its Torch module')


def _json_value(value):
    """Return a JSON value for a model parameter, or a class name."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if has_units(value):
        # e.g. '[10. 10.] mm':
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return type(value).__name__
    return array.tolist() if array.ndim else type(value).__name__


def _sidecar(model, spatial, input_shape, white_point):
    """Return the JSON sidecar (schema version 2) of an exported model."""
    from .. import __version__
    implant = spatial.implant
    encoder = implant.encoder
    grid = spatial.grid
    x, y = implant.electrode_array.coordinates(um)[:, :2].T
    skip = ('verbose', 'ndim', 'axon_pickle', 'ignore_pickle')
    params = {name: _json_value(getattr(spatial, name))
              for name in spatial.get_default_params() if name not in skip}
    units = {name: str(unit) for name, unit in
             spatial.get_param_units().items() if name in params}
    return {
        'schema_version': 2,
        'pulse2percept_version': __version__,
        'model': type(model).__name__,
        'spatial_model': type(spatial).__name__,
        'implant': type(implant).__name__,
        'input': {
            'name': 'image',
            'shape': [1, 1, *input_shape],
            'dtype': 'float32',
            'range': [0.0, 1.0],
            'grayscale': True,
            'registration': (
                'The image spans the electrode bounding box in device '
                'coordinates: row 0 at the smallest y, column 0 at the '
                'smallest x. Each electrode samples it bilinearly.'),
            'device_extent_um': {'x': [float(x.min()), float(x.max())],
                                 'y': [float(y.min()), float(y.max())]},
        },
        'output': {
            'name': 'percept',
            'shape': [1, 1, *grid.x.shape],
            'dtype': 'float32',
            'values': 'raw spatial response, not normalized',
        },
        'display': {
            'white_point': white_point,
        },
        'percept_grid': {
            'units': 'dva',
            'x': grid.x[0].tolist(),
            'y': grid.y[:, 0].tolist(),
            'orientation': ('row 0 at the largest y, column 0 at the '
                            'smallest x'),
        },
        'electrodes': [str(name) for name in implant.electrode_names],
        'deactivated': [str(name) for name, e in implant.electrodes.items()
                        if not e.activated],
        'encoder': {
            'class': type(encoder).__name__,
            'amp_range': [float(a) for a in encoder.amp_range],
            'amp_unit': 'uA',
            'freq': float(encoder.freq),
            'freq_unit': 'Hz',
            'modulation': ('amp_range[0] + gray * (amp_range[1] - '
                           'amp_range[0]) per activated electrode; 0 if '
                           'freq is 0'),
        },
        'spatial_params': params,
        'spatial_param_units': units,
        'approximations': [],
    }
