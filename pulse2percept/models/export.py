""":py:func:`~pulse2percept.export_onnx`"""
import importlib.util
import json
import logging
import os
import warnings

import numpy as np

from .base import Model, SpatialModel
from ..implants.base import _bilinear_weights
from ..stimuli import AmplitudeEncoder, ImageStimulus
from ..units import um
from ..units.base import has_units

#: ONNX opset of exported graphs. Unity Inference Engine 2.x reads opsets
#: 7-15.
_OPSET = 15

#: Agreement required of every exported graph, relative to the peak |response|
#: of the validation image. float32 matrix products with summation order left
#: to the runtime (BLAS, ONNX Runtime) differ by a few units in the last place
#: of each term.
_RTOL = 1e-5

#: Largest graph a single ``.onnx`` file can hold (protobuf limit, bytes).
_MAX_BYTES = 2 ** 31


def export_onnx(model, path, input_shape):
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
        ``max_current``.
    path : str or path-like
        Output ``.onnx`` file.
    input_shape : (int, int)
        Image height and width in pixels.

    Notes
    -----
    *  Input ``image`` is float32 ``(1, 1, H, W)``, gray levels in [0, 1].
       The image spans the electrode bounding box in device coordinates,
       row 0 at the smallest device y, as for an
       :py:class:`~pulse2percept.stimuli.ImageStimulus`.
    *  Output ``percept`` is float32 ``(1, 1, Hp, Wp)`` on the model's grid,
       row 0 at the largest y (dva). Values are the raw spatial response,
       not normalized for display.
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
    except RuntimeError:
        os.remove(path)
        raise
    with open(os.path.splitext(path)[0] + '.json', 'w') as f:
        json.dump(_sidecar(model, spatial, input_shape), f, indent=1)


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


def _deploy_module(spatial, input_shape):
    """Return the Torch module of a built model's image-to-percept graph.

    Raises NotImplementedError for any configuration the graph would not
    reproduce.
    """
    import torch
    from ._deploy import _ImagePercept
    # Only classes that define their own adapter: a subclass can change
    # semantics in ways an inherited one would ignore.
    if '_onnx_adapter' not in vars(type(spatial)):
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
    # The tensor route `predict_percept` takes for gray images:
    gap = encoder._tensor_gap()
    if gap is not None:
        raise NotImplementedError(gap)
    rows, cols, weights = _bilinear_weights(*implant._image_grid(input_shape))
    pixel_index = torch.as_tensor(rows * input_shape[1] + cols).reshape(-1)
    pixel_weight = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
    n_el = implant.n_electrodes
    _, freq = encoder._modulate(np.zeros((n_el, 1), dtype=np.float32))
    # Deactivated or non-firing (0 Hz) electrodes deliver nothing:
    on = np.array([e.activated for e in implant.electrode_objects])
    on = (on[:, None] & (np.asarray(freq) > 0)).reshape((n_el, 1))
    return _ImagePercept(pixel_index, pixel_weight,
                         lambda gray: encoder._modulate(gray)[0],
                         torch.as_tensor(on), spatial._onnx_adapter(),
                         spatial.grid.x.shape).eval()


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


def _sidecar(model, spatial, input_shape):
    """Return the JSON sidecar (schema version 1) of an exported model."""
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
        'schema_version': 1,
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
