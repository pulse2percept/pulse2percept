""":py:class:`~pulse2percept.implants.Raster`,
   :py:class:`~pulse2percept.implants.SequentialRaster`,
   :py:class:`~pulse2percept.implants.CustomRaster`"""
from abc import ABCMeta, abstractmethod
import numpy as np

from ..utils import PrettyPrint


def _finite(name, value):
    """Reject NaN and infinity, which slip through every ``<`` comparison"""
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        raise ValueError(f"'{name}' must be finite, not {value}.")


def _whole(name, value):
    """Reject a non-integer count or index, which would silently truncate"""
    _finite(name, value)
    if int(value) != value:
        raise ValueError(f"'{name}' must be a whole number, not {value}.")
    return int(value)


class Raster(PrettyPrint, metaclass=ABCMeta):
    """Abstract base class for all raster patterns

    A stimulator usually cannot drive every electrode at once, because the
    total current it can source at any instant is limited. Electrodes are
    therefore split into *raster groups* that take turns.

    A raster is a **scheduling constraint, not a hardware state machine**. What
    it has to deliver is one property: no two groups are ever active at the same
    instant, so that the stimulator sources at most one group's worth of current
    however the video is modulated. It is not a switch that cyclically enables
    group 0, then group 1, then group 0 again forever, and a group's pulses do
    not have to land at the same phase of a repeating cycle.

    Taking turns is described by a **raster sweep**: group *g* starts its pulse
    ``g * group_dur`` after group 0 does, so a sweep spans ``n_groups *
    group_dur``. Two things then keep groups apart for good (see
    :py:class:`~pulse2percept.stimuli.Encoder`):

    1.  A pulse has to be short enough to finish before the next group's turn
        begins.
    2.  Electrodes on *different* pulse periods drift relative to one another,
        and would eventually collide however they started out. Their periods are
        therefore pinned to whole numbers of the sweep, which fixes their
        relative phase. Pinning rounds the period *up*, so multiplexing never
        drives an electrode faster -- and so never delivers more charge -- than
        asked.

        Electrodes that share one period cannot drift in the first place: their
        onsets stay ``group_dur`` apart forever, whatever that period is.
        Nothing is quantized in that case and the requested rate is delivered
        exactly, even when the period is not a whole number of sweeps. This is
        the usual case under amplitude modulation, and it is why rastering costs
        no frequency there.

        So with two groups 1.5 ms apart on a common 10 ms period, group 0 pulses
        at 0, 10, 20, ... and group 1 at 1.5, 11.5, 21.5, ... -- collision-free,
        but not a repeating 3 ms schedule, and the 10 ms period is left alone.

    The sweep belongs to the *stimulation* schedule, not to the video: it is
    tied to the pulse period, not to the frame rate. With no explicit
    ``group_dur`` the groups divide the shortest pulse period between them, so
    under amplitude modulation the sweep is exactly that period and each group
    pulses once per sweep. An explicit ``group_dur`` instead builds the sweep
    from the slot, which is generally much shorter than the period -- six groups
    of 1 ms sweep in 6 ms whatever rate the electrodes run at. Under frequency
    modulation the sweep is set by the fastest electrode, and slower ones pulse
    every *m*-th sweep.

    Subclasses only implement ``groups``.

    .. versionadded:: 0.10.0

    Parameters
    ----------
    group_dur : float, optional
        Duration (ms) of a single group's slot, and hence the spacing between
        one group's turn and the next. If None, the groups are spread evenly
        over the pulse period, so that a sweep takes exactly one period to
        complete -- which is what an encoder wants whenever every electrode
        pulses at the same rate.

        Setting it explicitly makes the sweep ``n_groups * group_dur``
        regardless of the pulse period, which is how you buy back frequency
        resolution under frequency modulation: a shorter slot means a shorter
        sweep, and the periods that have to be pinned are pinned onto a finer
        grid. It cannot be shorter than a single pulse.

        An encoder with a ``clock`` rounds the slot onto it and rebuilds the
        sweep from the result, so every group keeps a turn of the same length.

    """
    __slots__ = ('group_dur',)

    def __init__(self, group_dur=None):
        if group_dur is not None:
            _finite('group_dur', group_dur)
            if group_dur <= 0:
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

    def slot_dur(self, period):
        """Duration (ms) of one group's slot

        Parameters
        ----------
        period : float
            The pulse period (ms) a sweep has to fit into, so that every group
            gets its turn before the first one comes round again.

        Returns
        -------
        slot_dur : float
            ``group_dur`` if one was given, else the period split evenly
            between the groups.

        """
        if self.group_dur is not None:
            return float(self.group_dur)
        return float(period) / self.n_groups

    def offsets(self, electrodes, period):
        """How far behind group 0 each electrode's slot begins

        Parameters
        ----------
        electrodes : array_like
            Electrode names, in the order they appear in the stimulus.
        period : float
            The pulse period (ms) a sweep has to fit into.

        Returns
        -------
        offset : (n_electrodes,) float array
            Time (ms) between the start of a sweep and the start of this
            electrode's slot.

        """
        group = np.asarray(self.groups(electrodes), dtype=np.int64)
        if group.min(initial=0) < 0 or group.max(initial=0) >= self.n_groups:
            raise ValueError(f"'groups' must be in 0..{self.n_groups - 1}.")
        dur = self.slot_dur(period)
        # A tick of slack: the period is generally not a round number of ms
        # (a 300 Hz period is 3.333... ms), so an exact `>` would reject the
        # even split this class computes itself:
        if self.n_groups * dur > period * (1 + 1e-9):
            raise ValueError(f"A raster of {self.n_groups} groups "
                             f"{dur:.3f} ms apart sweeps in "
                             f"{self.n_groups * dur:.3f} ms, which does not "
                             f"fit into a pulse period of {period:.3f} ms. "
                             f"Shorten 'group_dur', use fewer groups, or lower "
                             f"the pulse frequency.")
        return group * dur


class SequentialRaster(Raster):
    """Split electrodes into groups that fire one after another

    Electrodes are assigned to groups by their position in the stimulus, which
    for an :py:class:`~pulse2percept.implants.ElectrodeGrid` runs row by row.
    So on a 6x10 array such as :py:class:`~pulse2percept.implants.ArgusII`,
    ``SequentialRaster(6)`` puts each row in its own group -- a line raster.

    .. versionadded:: 0.10.0

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
        _finite('n_groups', n_groups)
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

    .. versionadded:: 0.10.0

    Parameters
    ----------
    groups : list of lists, or dict
        Either a list whose i-th element holds the names of the electrodes in
        group i, or a dict mapping each electrode name onto its group index.
        Every electrode in the stimulus must be accounted for, and no electrode
        may appear in two groups.
    group_dur : float, optional
        See :py:class:`~pulse2percept.implants.Raster`.

    Examples
    --------
    Fire the four corners of Argus II before everything else. Every other
    electrode has to be given a group too, or the current limit that the raster
    exists to respect could be violated without anyone noticing:

    >>> from pulse2percept.implants import ArgusII, CustomRaster
    >>> corners = ['A1', 'A10', 'F1', 'F10']
    >>> rest = [e for e in ArgusII().electrode_names if e not in corners]
    >>> raster = CustomRaster([corners, rest])
    >>> raster.n_groups
    2

    """
    __slots__ = ('_group_of', '_n_groups')

    def __init__(self, groups, group_dur=None):
        super().__init__(group_dur=group_dur)
        if isinstance(groups, dict):
            group_of = {str(k): _whole(f'group of {k}', v)
                        for k, v in groups.items()}
        else:
            group_of = {}
            for idx, names in enumerate(groups):
                if isinstance(names, str):
                    raise TypeError(f"Group {idx} must be a list of electrode "
                                    f"names, not the string '{names}'.")
                for name in names:
                    name = str(name)
                    # Silently letting the last group win would break the very
                    # guarantee a raster exists to make, since the electrode
                    # would go on firing in the group it was taken out of:
                    if name in group_of:
                        raise ValueError(
                            f"Electrode '{name}' is in group "
                            f"{group_of[name]} and group {idx}. Every "
                            f"electrode belongs to exactly one group.")
                    group_of[name] = idx
        if not group_of:
            raise ValueError("'groups' cannot be empty.")
        if min(group_of.values()) < 0:
            raise ValueError("Group indices cannot be negative.")
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
