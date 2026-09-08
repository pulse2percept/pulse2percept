""":py:class:`~pulse2percept.implants.retina.RetinalImplant`"""
from ..base import Implant


class RetinalImplant(Implant):
    """Retinal prosthesis

    A retinal prosthesis is an :py:class:`~pulse2percept.implants.Implant`
    that stimulates the retina, and therefore sits in one eye. This is the
    base class for devices such as
    :py:class:`~pulse2percept.implants.retina.ArgusII` and
    :py:class:`~pulse2percept.implants.retina.AlphaIMS`, and can be used
    directly to give a custom electrode array an eye.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    electrode_array : :py:class:`~pulse2percept.implants.ElectrodeArray` or
                      :py:class:`~pulse2percept.implants.Electrode`
        The electrode array used to deliver electrical stimuli to the retina.
    eye : 'left' or 'right', optional
        A string indicating whether the system is implanted in the left or
        right eye. Case-insensitive on input; stored lowercase.
    **kwargs :
        Keyword arguments accepted by
        :py:class:`~pulse2percept.implants.Implant`, such as ``preprocess``,
        ``safe_mode``, ``encoder``, ``raster``, ``max_current``,
        ``thresholds`` and ``scene_input_frame``.

    Examples
    --------
    Give a custom grid of electrodes a left eye:

    >>> from pulse2percept.implants import ElectrodeGrid
    >>> from pulse2percept.implants.retina import RetinalImplant
    >>> implant = RetinalImplant(ElectrodeGrid((4, 4), 400), eye='left')
    >>> implant.eye
    'left'

    """
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_eye',)

    def __init__(self, electrode_array, eye='right', **kwargs):
        super().__init__(electrode_array, **kwargs)
        self.eye = eye

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        params = super()._pprint_params()
        params['eye'] = self.eye
        return params

    @property
    def eye(self):
        """Implanted eye

        A :py:class:`~pulse2percept.implants.retina.RetinalImplant` can be
        implanted either in a left eye ('left') or right eye ('right').
        Models such as
        :py:class:`~pulse2percept.models.retina.AxonMapModel` will treat left
        and right eyes differently (for example, adjusting the location of the
        optic disc).

        Examples
        --------
        Implant Argus II in a left eye:

        >>> from pulse2percept.implants.retina import ArgusII
        >>> implant = ArgusII(eye='left')
        """
        return self._eye

    @eye.setter
    def eye(self, eye):
        """Eye setter (called upon `self.eye = eye`)"""
        if not isinstance(eye, str):
            raise TypeError(f"'eye' must be a string, not {type(eye)}.")
        eye = eye.lower()
        if eye not in ('left', 'right'):
            raise ValueError(f"'eye' must be either 'left' or 'right', not "
                             f"{eye}.")
        self._eye = eye
