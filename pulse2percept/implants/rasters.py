""":py:class:`~pulse2percept.implants.Raster`,
   :py:class:`~pulse2percept.implants.SequentialRaster`,
   :py:class:`~pulse2percept.implants.CustomRaster`"""
from abc import ABCMeta, abstractmethod
import numpy as np

from ..utils import PrettyPrint


class Raster(PrettyPrint, metaclass=ABCMeta):
    """Abstract base class for all raster patterns

    A stimulator usually cannot drive every electrode at once, because the
    total current it can source at any instant is limited. Electrodes are
    therefore split into *raster groups* that take turns: group 0 fires, then
    group 1 some milliseconds later, and so on, with the whole sequence
    completing within one frame.

    An encoder asks a raster how long each electrode has to wait after the
    start of a frame before it may pulse; see
    :py:class:`~pulse2percept.stimuli.Encoder`.

    Subclasses only implement ``groups``.

    .. versionadded:: 0.9.2

    Parameters
    ----------
    group_dur : float, optional
        Time (ms) between one group firing and the next. If None, the groups
        are spread evenly over the frame, so that the sequence takes exactly
        one frame period to complete.

        .. note::

           Staggering the *onsets* of two groups only keeps them off the same
           time point if the first group is finished before the second starts.
           That holds whenever a group pulses at most once per frame, which is
           what amplitude modulation does. Under frequency modulation an
           electrode may pulse many times per frame, and pulses from different
           groups can then land on top of each other. Set ``max_current`` on
           the implant to find out when they do.

    """
    __slots__ = ('group_dur',)

    def __init__(self, group_dur=None):
        if group_dur is not None and group_dur <= 0:
            raise ValueError("'group_dur' must be positive.")
        self.group_dur = group_dur

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        return {'group_dur': self.group_dur, 'n_groups': self.n_groups}

    @property
    @abstractmethod
    def n_groups(self):
        """Number of raster groups"""
        raise NotImplementedError

    @abstractmethod
    def groups(self, electrodes):
        """Assign each electrode to a raster group

        Parameters
        ----------
        electrodes : array_like
            Electrode names, in the order they appear in the stimulus.

        Returns
        -------
        group : (n_electrodes,) int array
            The group each electrode belongs to, in ``0..n_groups-1``.

        """
        raise NotImplementedError

    def delays(self, electrodes, frame_dur):
        """How long each electrode waits before pulsing

        Parameters
        ----------
        electrodes : array_like
            Electrode names, in the order they appear in the stimulus.
        frame_dur : float
            Duration (ms) of a single frame. The whole raster sequence has to
            complete within it.

        Returns
        -------
        delay : (n_electrodes,) float array
            Time (ms) between the start of a frame and this electrode's first
            pulse.

        """
        group = np.asarray(self.groups(electrodes), dtype=np.int64)
        if group.min(initial=0) < 0 or group.max(initial=0) >= self.n_groups:
            raise ValueError(f"'groups' must be in 0..{self.n_groups - 1}.")
        dur = (frame_dur / self.n_groups if self.group_dur is None
               else self.group_dur)
        if self.n_groups * dur > frame_dur:
            raise ValueError(f"A raster of {self.n_groups} groups "
                             f"{dur:.3f} ms apart takes "
                             f"{self.n_groups * dur:.3f} ms, which does not "
                             f"fit into a frame (dur={frame_dur:.3f} ms). "
                             f"Shorten 'group_dur' or lower the frame rate.")
        return group * dur


class SequentialRaster(Raster):
    """Split electrodes into groups that fire one after another

    Electrodes are assigned to groups by their position in the stimulus, which
    for an :py:class:`~pulse2percept.implants.ElectrodeGrid` runs row by row.
    So on a 6x10 array such as :py:class:`~pulse2percept.implants.ArgusII`,
    ``SequentialRaster(6)`` puts each row in its own group -- a line raster.

    .. versionadded:: 0.9.2

    Parameters
    ----------
    n_groups : int
        Number of groups to split the electrodes into.
    interleave : bool, optional
        If False (the default), each group is a contiguous block of
        electrodes. If True, groups are interleaved, so that consecutive
        electrodes end up in different groups. Interleaving spreads each
        group's current further across the array.
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    A line raster for Argus II, one row of ten electrodes at a time:

    >>> from pulse2percept.implants import ArgusII, SequentialRaster
    >>> implant = ArgusII()
    >>> implant.raster = SequentialRaster(6)

    """
    __slots__ = ('_n_groups', 'interleave')

    def __init__(self, n_groups, interleave=False, group_dur=None):
        super().__init__(group_dur=group_dur)
        if int(n_groups) != n_groups or n_groups < 1:
            raise ValueError(f"'n_groups' must be a positive integer, not "
                             f"{n_groups}.")
        self._n_groups = int(n_groups)
        self.interleave = interleave

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'interleave': self.interleave})
        return params

    @property
    def n_groups(self):
        """Number of raster groups"""
        return self._n_groups

    def groups(self, electrodes):
        """Assign each electrode to a raster group"""
        idx = np.arange(len(electrodes))
        if self.interleave:
            return idx % self._n_groups
        # Contiguous blocks, as evenly sized as the electrode count allows:
        return idx * self._n_groups // max(1, len(electrodes))


class CustomRaster(Raster):
    """Assign electrodes to raster groups by name

    .. versionadded:: 0.9.2

    Parameters
    ----------
    groups : list of lists, or dict
        Either a list whose i-th element holds the names of the electrodes in
        group i, or a dict mapping each electrode name onto its group index.
        Every electrode in the stimulus must be accounted for.
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    Fire the four corners of Argus II before everything else:

    >>> from pulse2percept.implants import CustomRaster
    >>> raster = CustomRaster([['A1', 'A10', 'F1', 'F10'], ['B5', 'C5']])

    """
    __slots__ = ('_group_of', '_n_groups')

    def __init__(self, groups, group_dur=None):
        super().__init__(group_dur=group_dur)
        if isinstance(groups, dict):
            group_of = {str(k): int(v) for k, v in groups.items()}
        else:
            group_of = {}
            for idx, names in enumerate(groups):
                if isinstance(names, str):
                    raise TypeError(f"Group {idx} must be a list of electrode "
                                    f"names, not the string '{names}'.")
                for name in names:
                    group_of[str(name)] = idx
        if not group_of:
            raise ValueError("'groups' cannot be empty.")
        self._group_of = group_of
        self._n_groups = max(group_of.values()) + 1

    def _pprint_params(self):
        """Return a dict of class arguments to pretty-print"""
        params = super()._pprint_params()
        params.update({'n_electrodes': len(self._group_of)})
        return params

    @property
    def n_groups(self):
        """Number of raster groups"""
        return self._n_groups

    def groups(self, electrodes):
        """Assign each electrode to a raster group"""
        try:
            return np.array([self._group_of[str(e)] for e in electrodes])
        except KeyError:
            missing = sorted({str(e) for e in electrodes} -
                             set(self._group_of))
            raise ValueError(f"No raster group given for electrode(s) "
                             f"{missing[:10]}. Every electrode in the "
                             f"stimulus must be assigned to a group.")
