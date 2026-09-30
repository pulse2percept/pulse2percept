""":py:class:`~pulse2percept.implants.cortex.CorticalImplant`"""
from ..base import Implant


def _validate_hemisphere(hemisphere):
    """Return ``hemisphere`` as 'left', 'right' or None."""
    if hemisphere is None:
        return None
    if not isinstance(hemisphere, str):
        raise TypeError(f"'hemisphere' must be a string or None, not "
                        f"{type(hemisphere)}.")
    hemisphere = hemisphere.lower()
    if hemisphere not in ('left', 'right'):
        raise ValueError(f"'hemisphere' must be 'left', 'right' or None, not "
                         f"{hemisphere}.")
    return hemisphere


class CorticalImplant(Implant):
    """Cortical prosthesis

    An :py:class:`~pulse2percept.implants.Implant` that stimulates visual
    cortex in one hemisphere. Base class for devices such as
    :py:class:`~pulse2percept.implants.cortex.Orion`. Can be used directly to
    assign a hemisphere to a custom electrode array.

    ``hemisphere`` is metadata only. Placement is set by the model's
    ``implant_position`` and the electrode coordinates. Setting a hemisphere
    does not move or mirror the array, and a hemisphere inconsistent with the
    coordinates is not rejected.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    electrode_array : :py:class:`~pulse2percept.implants.ElectrodeArray` or
                      :py:class:`~pulse2percept.implants.Electrode`
        The electrode array used to deliver electrical stimuli to cortex.
    hemisphere : 'left', 'right' or None, optional
        Which hemisphere the device is implanted in. Case-insensitive on
        input; stored lowercase. Defaults to None, i.e. unspecified.
    **kwargs :
        Keyword arguments accepted by
        :py:class:`~pulse2percept.implants.Implant`, such as ``preprocess``,
        ``safe_mode``, ``encoder``, ``raster``, ``max_current``,
        ``thresholds`` and ``scene_input_frame``.

    Examples
    --------
    Give a custom grid of electrodes a hemisphere:

    >>> from pulse2percept.implants import ElectrodeGrid
    >>> from pulse2percept.implants.cortex import CorticalImplant
    >>> implant = CorticalImplant(ElectrodeGrid((4, 4), 400),
    ...                           hemisphere='right')
    >>> implant.hemisphere
    'right'

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_hemisphere',)

    def __init__(self, electrode_array, hemisphere=None, **kwargs):
        super().__init__(electrode_array, **kwargs)
        self.hemisphere = hemisphere

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        if self.hemisphere is not None:
            params['hemisphere'] = self.hemisphere
        return params

    @property
    def hemisphere(self):
        """Implanted hemisphere

        'left', 'right', or None if unspecified. Metadata only: cortical
        models use electrode coordinates and ``implant_position``.
        """
        return getattr(self, '_hemisphere', None)

    @hemisphere.setter
    def hemisphere(self, hemisphere):
        """Hemisphere setter (called upon ``self.hemisphere = hemisphere``)"""
        self._hemisphere = _validate_hemisphere(hemisphere)
