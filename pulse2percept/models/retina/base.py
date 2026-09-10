""":py:class:`~pulse2percept.models.retina.RetinalSpatial`"""
import warnings

import numpy as np

from ..base import SpatialModel, _length_valued, _placed_coords
from ...topography.retina import Curcio1990Map, RetinalMap
from ...units import DimensionMismatchError, as_value


def _visual_field_map_first(params):
    """Apply ``visual_field_map`` before parameters whose units need it."""
    if 'visual_field_map' not in params:
        return params
    return {'visual_field_map': params['visual_field_map'],
            **{key: val for key, val in params.items()
               if key != 'visual_field_map'}}


def _warn_ignores_z(model, electrode_array):
    """Warn when a model ignores nonzero electrode ``z`` coordinates.

    Reads placed coordinates, so ``implant_depth`` counts as depth.
    """
    if np.allclose(_placed_coords(model, electrode_array,
                                  model.space_unit)[:, 2], 0):
        return
    warnings.warn(
        f"{type(model).__name__} does not model electrode-retina distance: "
        f"nonzero z values do not change its response. In a real implant, "
        f"distance is expected to affect stimulation threshold and spatial "
        f"recruitment, but that relationship is not parameterized by this "
        f"model.")


class RetinalSpatial(SpatialModel):
    """Abstract base class for spatial models of retinal stimulation.

    Adds to :py:class:`~pulse2percept.models.SpatialModel` the behavior that is
    specific to simulating the retina:

    *  a retinotopic ``visual_field_map`` by default ([Curcio1990]_);
    *  ``xrange`` and ``yrange`` may be given as a physical retinal extent,
       which is resolved through ``visual_field_map``.

    Parameters
    ----------
    implant : :py:class:`~pulse2percept.implants.Implant`
        Implant whose electrode geometry is modeled.
    xrange : (float, float) or Quantity, optional
        Horizontal visual-field extent in degrees of visual angle. A physical
        retinal extent is accepted as shorthand and is resolved along the
        horizontal retinal meridian through ``visual_field_map``.
    yrange : (float, float) or Quantity, optional
        Vertical visual-field extent in degrees of visual angle. A physical
        retinal extent is resolved along the vertical meridian.
    visual_field_map : VisualFieldMap, optional
        Retinotopic map between visual-field and retinal coordinates.
    **params : keyword arguments
        Any other parameter of :py:class:`~pulse2percept.models.SpatialModel`.

    Notes
    -----
    ``xrange`` and ``yrange`` always describe the simulated visual field and
    are stored in degrees of visual angle. A retinal length is only shorthand
    for selecting that extent through ``visual_field_map``; the resulting grid
    is still uniformly sampled in visual angle. ``step`` therefore only accepts
    angular spacing.

    .. versionadded:: 0.11.0
    """

    def __init__(self, implant, **params):
        # `visual_field_map` first: `xrange`/`yrange` may be given as a retinal
        # extent, which is resolved through the map as it is assigned. See
        # `_visual_field_map_first`.
        super().__init__(implant, **_visual_field_map_first(params))
        # Laterality the grid was last built for; see `is_built`.
        self._built_map_eye = None

    def _validate_map_eye(self):
        """Require an eye-dependent visual_field_map to match the implant"""
        map_eye = getattr(self.visual_field_map, 'eye', None)
        if map_eye is None:
            return
        implant_eye = getattr(self.implant, 'eye', None)
        if implant_eye is None:
            raise TypeError(
                f"{type(self.visual_field_map).__name__} depends on retinal "
                f"laterality, but {type(self.implant).__name__} does not "
                f"carry an eye. Wrap a custom array in "
                f"pulse2percept.implants.retina.RetinalImplant, e.g. "
                f"RetinalImplant(ElectrodeGrid(...), eye='{map_eye}'), or "
                f"use an eye-independent visual_field_map.")
        if implant_eye != map_eye:
            raise ValueError(
                f"The implant sits in the {implant_eye} eye, but "
                f"{type(self.visual_field_map).__name__}(eye='{map_eye}') "
                f"maps the {map_eye} eye. Use a map with "
                f"eye='{implant_eye}', or an implant with eye='{map_eye}'.")

    @property
    def is_built(self):
        """Return whether the grid matches the current retinal laterality"""
        built = super().is_built
        map_eye = getattr(self.visual_field_map, 'eye', None)
        if map_eye is None:
            return built
        implant_eye = getattr(self.implant, 'eye', None)
        return (built and self._built_map_eye == map_eye and
                implant_eye == map_eye)

    def build(self, **build_params):
        """Build the model

        Parameters
        ----------
        **build_params : keyword arguments
            Declared model parameters to set before building.

        Returns
        -------
        self
        """
        self.set_params(**build_params)
        # Before the grid is laid out, so a mismatch cannot be baked into it:
        self._validate_map_eye()
        super().build()
        self._built_map_eye = getattr(self.visual_field_map, 'eye', None)
        return self

    def set_params(self, **params):
        """Set the parameters of this model"""
        super().set_params(**_visual_field_map_first(params))

    def get_default_params(self):
        """Return a dictionary of default values for all model parameters"""
        return {**super().get_default_params(),
                'visual_field_map': Curcio1990Map()}

    def _scene_sampling_points(self):
        """Return placed electrode positions in dva, through retinotopy."""
        visual_field_map = self.visual_field_map
        if not isinstance(visual_field_map, RetinalMap):
            raise ValueError(
                f"A scene reaches the electrodes through the model's "
                f"'visual_field_map', which has to say where on the retina "
                f"each degree of visual angle lands. This model's is a "
                f"{type(visual_field_map).__name__}.")
        xy = _placed_coords(self, self.implant.electrode_array,
                            visual_field_map.tissue_unit)[:, :2].T
        return visual_field_map.ret_to_dva(*xy)

    def _normalize_param_value(self, name, value):
        """Normalize a parameter to its stored unit.

        Physical ``xrange`` and ``yrange`` values are resolved through
        ``visual_field_map``; other unitful parameters use the generic
        conversion.
        """
        if name in ('xrange', 'yrange') and _length_valued(value):
            return self._retinal_range_to_dva(name, value)
        return super()._normalize_param_value(name, value)

    def _retinal_range_to_dva(self, name, value):
        """Resolve a retinal extent to a visual-field range.

        ``xrange`` is converted along the horizontal retinal meridian and
        ``yrange`` along the vertical meridian. The result is stored in degrees
        of visual angle and is not reinterpreted if ``visual_field_map``
        changes later.

        Parameters
        ----------
        name : {'xrange', 'yrange'}
            Range being assigned.
        value : (min, max)
            Retinal extent.

        Returns
        -------
        tuple of float
            Visual-field extent in increasing order.
        """
        visual_field_map = getattr(self, 'visual_field_map', None)
        if not isinstance(visual_field_map, RetinalMap):
            raise DimensionMismatchError(
                f"'{name}' is a visual field extent, measured in degrees of "
                f"visual angle. A physical length is shorthand for one only "
                f"on a retinal map, and this model's visual_field_map is a "
                f"{type(visual_field_map).__name__}. Specify '{name}' in "
                f"dva instead.")
        # In the unit the map's tissue side is measured in, which is what its
        # inverse transform below expects:
        extent = np.asarray(as_value(value, visual_field_map.tissue_unit,
                                     name),
                            dtype=np.float64).ravel()
        if extent.size != 2:
            raise ValueError(f"'{name}' must be a (min, max) pair, not "
                             f"{value}.")
        lo, hi = extent
        try:
            if name == 'xrange':
                lo_dva, _ = visual_field_map.ret_to_dva(lo, 0)
                hi_dva, _ = visual_field_map.ret_to_dva(hi, 0)
            else:
                _, lo_dva = visual_field_map.ret_to_dva(0, lo)
                _, hi_dva = visual_field_map.ret_to_dva(0, hi)
        except NotImplementedError:
            raise NotImplementedError(
                f"This visual field map "
                f"({type(visual_field_map).__name__}) cannot infer a visual "
                f"field range from retinal distance. Specify "
                f"'{name}' in dva instead.") from None
        # Sorted, because the retinal y axis points the opposite way from the
        # visual field's, so the two end points can come back swapped:
        return tuple(sorted((float(lo_dva), float(hi_dva))))
