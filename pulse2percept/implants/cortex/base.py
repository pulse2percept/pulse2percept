""":py:class:`~pulse2percept.implants.cortex.CorticalImplant`"""
from ..base import Implant


def _validate_hemisphere(hemisphere):
    """Return ``hemisphere`` as 'LH', 'RH' or None."""
    if hemisphere is None:
        return None
    if not isinstance(hemisphere, str):
        raise TypeError(f"'hemisphere' must be a string or None, not "
                        f"{type(hemisphere)}.")
    hemisphere = hemisphere.upper()
    if hemisphere not in ('LH', 'RH'):
        raise ValueError(f"'hemisphere' must be 'LH', 'RH' or None, not "
                         f"{hemisphere}.")
    return hemisphere


class CorticalImplant(Implant):
    """Cortical prosthesis

    A cortical prosthesis is an :py:class:`~pulse2percept.implants.Implant`
    that stimulates visual cortex, and therefore sits in one hemisphere. This
    is the base class for devices such as
    :py:class:`~pulse2percept.implants.cortex.Orion`, and can be used directly
    to give a custom electrode array a hemisphere.

    ``hemisphere`` is device metadata only. Where the array sits in cortex is
    set by the model's ``implant_position`` and by the electrode coordinates
    themselves, which stay authoritative: recording a hemisphere neither moves
    the array nor reflects its coordinates, and a hemisphere that disagrees
    with the coordinates is not rejected.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    electrode_array : :py:class:`~pulse2percept.implants.ElectrodeArray` or
                      :py:class:`~pulse2percept.implants.Electrode`
        The electrode array used to deliver electrical stimuli to cortex.
    hemisphere : 'LH', 'RH' or None, optional
        Which hemisphere the device is implanted in. Defaults to None, i.e.
        unspecified.
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
    ...                           hemisphere='RH')
    >>> implant.hemisphere
    'RH'

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_hemisphere',)

    def __init__(self, electrode_array, hemisphere=None, **kwargs):
        super().__init__(electrode_array, **kwargs)
        self.hemisphere = hemisphere

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        # Omitted when unspecified, which is the default:
        if self.hemisphere is not None:
            params['hemisphere'] = self.hemisphere
        return params

    @property
    def hemisphere(self):
        """Implanted hemisphere

        'LH', 'RH', or None if unspecified. Metadata: cortical models place
        the array from its coordinates and their own ``implant_position``, not
        from this attribute.
        """
        return getattr(self, '_hemisphere', None)

    @hemisphere.setter
    def hemisphere(self, hemisphere):
        """Hemisphere setter (called upon ``self.hemisphere = hemisphere``)"""
        self._hemisphere = _validate_hemisphere(hemisphere)
