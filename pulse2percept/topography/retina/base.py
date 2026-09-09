""":py:class:`~pulse2percept.topography.retina.RetinalMap`"""
from abc import abstractmethod

from ..base import VisualFieldMap


class RetinalMap(VisualFieldMap):
    """ Template class for retinal visual field maps, which only have 1 region.

    Retinal coordinates are signed, and the anatomical meaning of the sign of
    the horizontal coordinate depends on which eye is being mapped: for a right
    eye, negative x is temporal and positive x is nasal, and for a left eye the
    interpretation is mirrored. Not every retinal map is numerically
    eye-dependent (see :py:class:`~pulse2percept.topography.Curcio1990Map`),
    but ``eye`` is object state on all of them, because a map without a side is
    ambiguous wherever nasal and temporal retina differ.

    Parameters
    ----------
    eye : {'RE', 'LE'}, optional
        Whether the map describes a right eye ('RE') or a left eye ('LE').
        Case-insensitive on input; stored uppercase.

    .. versionchanged:: 0.11.0

        Takes ``eye``.

    """
    split_map = False
    regions = ['ret']

    def get_default_params(self):
        return {**super().get_default_params(), 'eye': 'RE'}

    @property
    def eye(self):
        """Mapped eye

        Which eye the map describes, either a right eye ('RE') or a left eye
        ('LE'). Maps whose published transformation distinguishes nasal from
        temporal retina need this to know which anatomical half-retina a
        signed x coordinate falls in; see
        :py:class:`~pulse2percept.topography.Watson2014DisplaceMap`.
        """
        return self._eye

    @eye.setter
    def eye(self, eye):
        if not isinstance(eye, str):
            raise TypeError(f"'eye' must be a string, not {type(eye)}.")
        eye = eye.upper()
        if eye not in ('RE', 'LE'):
            raise ValueError(f"'eye' must be either 'RE' or 'LE', not {eye}.")
        self._eye = eye

    def from_dva(self):
        return {'ret' : self.dva_to_ret}

    def to_dva(self):
        return {'ret' : self.ret_to_dva}

    @abstractmethod
    def dva_to_ret(self, x, y):
        """Convert degrees of visual angle (dva) to retinal coords (um)"""
        raise NotImplementedError

    def ret_to_dva(self, x, y):
        """Convert retinal coords (um) to degrees of visual angle (dva)"""
        raise NotImplementedError
