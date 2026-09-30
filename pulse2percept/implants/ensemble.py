""":py:class:`~pulse2percept.implants.EnsembleImplant`"""
import numpy as np
from .base import Implant, _ensemble_target
from .electrodes import Electrode
from .electrode_arrays import ElectrodeArray
from ..stimuli._merge import unique_time_points
from ..stimuli.base import _describe_unit
from ..units import DimensionMismatchError, as_value, dva, um


def _resolve_region(visual_field_map, region, ndim=2):
    """Return the map region to place implants in.

    ``region`` is required unless the map has exactly one region. ``ndim`` is
    the maximum map dimensionality the caller supports (2 for ``(x, y)``
    anchors, 3 for surface-normal placement).
    """
    if getattr(visual_field_map, 'ndim', 2) > ndim:
        raise NotImplementedError(
            f"{type(visual_field_map).__name__} is a "
            f"{visual_field_map.ndim}D map, and an ensemble places its "
            f"implants by an (x, y) anchor. Place them yourself with "
            f"'from_coords', or use a device that knows how to sit on a "
            f"surface (e.g. Neuralink.from_neuropythy).")
    regions = list(visual_field_map.from_dva().keys())
    if region is None:
        if len(regions) != 1:
            raise ValueError(f"{type(visual_field_map).__name__} maps to "
                             f"{len(regions)} regions ({', '.join(regions)}); "
                             f"pass the one to place implants in as "
                             f"'region'.")
        return regions[0]
    if region not in regions:
        raise ValueError(f"Unknown region {region!r}. "
                         f"{type(visual_field_map).__name__} maps to "
                         f"{', '.join(regions)}.")
    return region


class EnsembleImplant(Implant):
    
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_implants', '_electrode_array', 'safe_mode', 'preprocess')

    @staticmethod
    def _placed(implant_type, x, y):
        """Instantiate one constituent at ``(x, y)`` in ensemble coordinates."""
        implant = implant_type()
        for elec in implant.electrode_array.electrode_objects:
            elec.x += x
            elec.y += y
        return implant

    @classmethod
    def from_visual_field_map(cls, implant_type, visual_field_map, locs=None,
                              xrange=None, yrange=None, step=None,
                              region=None):
        """
        Create an ensemble implant from a visual field map.

        An implant of type ``implant_type`` is created for each visual field
        location specified either by ``locs`` or by ``xrange``, ``yrange`` and
        ``step``, and centered at the corresponding tissue coordinates.

        Accepts any 2D :py:class:`~pulse2percept.topography.VisualFieldMap`
        (retinal or cortical). 3D maps raise NotImplementedError; see
        :py:meth:`~pulse2percept.implants.cortex.Neuralink.from_neuropythy`.

        .. versionadded:: 0.11.0
            Replaces ``from_cortical_map``.

        Parameters
        ----------
        implant_type : type
            Type of implant to create for the ensemble. Must subclass
            p2p.implants.Implant
        visual_field_map : :py:class:`~pulse2percept.topography.VisualFieldMap`
            Visual field map to create the implant from.
        locs : np.ndarray with shape (n, 2), optional
            Array of visual field locations to create implants at (dva).
            Not needed if using xrange, yrange, and step.
        xrange, yrange: tuple of floats, optional
            Range of x and y coordinates (dva) to create implants at.
        step : float or (x_step, y_step), optional
            Spacing (dva) between implant centers.
        region : str, optional
            Region of tissue to create the implant in, e.g. ``'ret'`` or
            ``'v1'``. Required unless the map has exactly one region.

        Returns
        -------
        ensemble : p2p.implants.EnsembleImplant
            Ensemble implant created from the visual field map.

        Notes
        -----
        *  Visual field coordinates accept numbers (dva) or quantities (e.g.,
           ``xrange=(-3 * dva, 3 * dva)``). :py:meth:`from_coords` uses
           physical coordinates (um) instead. See
           :py:mod:`pulse2percept.units`.
        """
        from ..topography import Grid2D
        from ..topography.base import VisualFieldMap
        if not isinstance(visual_field_map, VisualFieldMap):
            raise TypeError("visual_field_map must be a "
                            "p2p.topography.VisualFieldMap")
        if not issubclass(implant_type, Implant):
            raise TypeError("implant_type must be a sub-type of Implant")
        region = _resolve_region(visual_field_map, region)

        # Locations in dva; the map converts them to tissue coordinates:
        locs = as_value(locs, dva, 'locs')
        xrange = as_value(xrange, dva, 'xrange')
        yrange = as_value(yrange, dva, 'yrange')
        step = as_value(step, dva, 'step')

        if locs is None:
            if xrange is None:
                xrange = (-3, 3)
            if yrange is None:
                yrange = (-3, 3)
            if step is None:
                step = 1

            # make a grid of points
            grid = Grid2D(xrange, yrange, step)
            xlocs = grid.x.flatten()
            ylocs = grid.y.flatten()
        else:
            xlocs = locs[:, 0]
            ylocs = locs[:, 1]

        implant_locations = np.array(
            visual_field_map.from_dva()[region](xlocs, ylocs)).T

        return cls.from_coords(implant_type=implant_type, locs=implant_locations)


    @classmethod
    def from_coords(cls, implant_type, locs=None, xrange=None, yrange=None, step=None):
        """
        Create an ensemble implant using physical (cortical or retinal) coordinates.

        Parameters
        ----------
        implant_type : type
            The type of implant to create for the ensemble.
        locs : np.ndarray with shape (n, 2), optional
            Array of physical locations (um) to create implants at. Not
            needed if using xrange, yrange, and step.
        xrange, yrange: tuple of floats, optional
            Range of x and y coordinates (um) to create implants at. Required
            (together with ``step``) if ``locs`` is not given.
        step : float or (x_step, y_step), optional
            Spacing (um) between implant centers.

        Raises
        ------
        ValueError
            If neither ``locs`` nor all three of ``xrange``, ``yrange`` and
            ``step`` are given.

        Notes
        -----
        *  Lengths may be given as plain numbers of microns or as unitful
           quantities (e.g. ``xrange=(-1 * mm, 1 * mm)``). See
           :py:mod:`pulse2percept.units`.

        .. versionchanged:: 0.10.0
            The grid arguments no longer have defaults. The old dva defaults
            ``(-3, 3)`` and ``1`` were read as um here.

        """
        from ..topography.base import _rectangular_mesh

        if not issubclass(implant_type, Implant):
            raise TypeError("implant_type must be a sub-type of Implant")

        # Physical coordinates (um):
        locs = as_value(locs, um, 'locs')
        xrange = as_value(xrange, um, 'xrange')
        yrange = as_value(yrange, um, 'yrange')
        step = as_value(step, um, 'step')

        if locs is None:
            # No default extent for a physical grid:
            missing = [name for name, value in [('xrange', xrange),
                                                ('yrange', yrange),
                                                ('step', step)]
                       if value is None]
            if missing:
                raise ValueError(
                    f"Pass either 'locs' or all of 'xrange', 'yrange' and "
                    f"'step' (missing: {', '.join(missing)}). Coordinates "
                    f"are physical, in microns.")

            # Not Grid2D, which would read um as dva:
            (xgrid, ygrid), _, _ = _rectangular_mesh(xrange, yrange, step)
            xlocs = xgrid.flatten()
            ylocs = ygrid.flatten()
        else:
            xlocs = locs[:, 0]
            ylocs = locs[:, 1]

        implant_list = [cls._placed(implant_type, x, y)
                        for x, y in zip(xlocs, ylocs)]
        
        return cls(implant_list)

    def __init__(self, implants, preprocess=False, safe_mode=False):
        """Ensemble implant

        Combines multiple implants in the same anatomical target into a single
        implant, to model tandem implants (e.g., ICVP, Neuralink).

        Constituents may differ in device type, but retinal and cortical
        implants cannot be mixed.

        Parameters
        ----------
        implants : list or dict
            A list or dict of implants to be combined.
        preprocess : bool or callable, optional
            Either True/False to indicate whether to execute the implant's default
            preprocessing method whenever a stimulus is prepared, or a custom
            function (callable).
        safe_mode : bool, optional
            If safe mode is enabled, only charge-balanced stimuli are allowed.

        Raises
        ------
        TypeError
            If ``implants`` mixes retinal and cortical implants.
        """
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.implants = implants

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'implants': self.implants,
                'electrode_array': self.electrode_array,
                'safe_mode': self.safe_mode, 'preprocess': self.preprocess}

    @property
    def implants(self):
        """Dict of implants

        """
        return self._implants
    
    @implants.setter
    def implants(self, implants):
        """Implant dict setter (called upon ``self.implants = implants``)"""
        # Assign the implant dict:
        if isinstance(implants, list):
            if not all(isinstance(implant, Implant) for implant in implants):
                raise TypeError(f"All elements in 'implants' must be Implant objects.")
            candidate = {i:implant for i,implant in enumerate(implants)}
        elif isinstance(implants, dict):
            if not all(isinstance(implant, Implant) for implant in implants.values()):
                raise TypeError(f"All elements in 'implants' must be Implant objects.")
            candidate = implants.copy()
        else:
            raise TypeError(f"'implants' must be a list or a dict object, not "
                            f"{type(implants)}.")
        # cannot mix retinal/cortical:
        _ensemble_target(candidate.values())
        self._implants = candidate
        # Create the electrode array
        electrodes = {}
        for i, implant in self._implants.items():
            for name, electrode in implant.electrode_array.electrodes.items():
                electrodes[str(i) + "-" + str(name)] = electrode
            
        self._electrode_array = ElectrodeArray(electrodes)

    def prepare_stim(self, source):
        """Prepare stimulation for an ensemble implant.

        ``source`` may address the combined electrode array directly, or be a
        dict keyed by constituent implant keys. Per-implant sources are
        prepared by each constituent, merged, then passed through
        ensemble-level preprocessing and safety checks. Missing implant keys
        contribute zeros.

        .. versionchanged:: 0.11.0
            Replaces ``merge_stimuli``.

        Parameters
        ----------
        source : dict or :py:class:`~pulse2percept.stimuli.Stimulus` source type
            One source for the whole ensemble, or ``{implant_key: source}``.

        Returns
        -------
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` or None
            Merged stimulation for the ensemble.

        Examples
        --------
        >>> import numpy as np
        >>> from pulse2percept.implants import EnsembleImplant
        >>> from pulse2percept.implants.cortex import Orion
        >>> ensemble = EnsembleImplant.from_coords(
        ...     Orion, locs=np.array([(0, 0), (-35000, 0)]))
        >>> ensemble.prepare_stim({0: np.ones(60),
        ...                        1: 2 * np.ones(60)}).data.shape
        (120, 1)
        """
        return self._prepare_stim(source)

    def _prepare_stim(self, source, allow_dimensionless=False):
        if isinstance(source, dict) and source and \
                all(key in self._implants for key in source):
            prepared = {
                key: implant._prepare_stim(
                    source.get(key), allow_dimensionless=allow_dimensionless)
                for key, implant in self._implants.items()}
            # Merge before ensemble-level preprocessing and safety checks:
            source = self._merged(prepared)
        return super()._prepare_stim(
            source, allow_dimensionless=allow_dimensionless)

    def _structured_children(self, prepared):
        """Return one source per driven ensemble electrode, or ``None``

        Electrodes of sparse or missing children are undriven.
        """
        sources = {}
        for i, implant in self._implants.items():
            stim = prepared.get(i)
            if stim is None:
                continue
            child = stim._structured_sources()
            if child is None:
                return None
            child = dict(child)
            for name in implant.electrode_names:
                if name in child:
                    sources[f"{i}-{name}"] = child[name]
        if not sources:
            return None
        # Use ensemble electrode order:
        return {name: sources[name] for name in self.electrode_names
                if name in sources}

    def _merged(self, prepared):
        """Merge the prepared stimuli of all constituent implants"""
        if not any(stim is not None for stim in prepared.values()):
            return None
        # Units come from implants that have a stimulus:
        present = [stim for stim in prepared.values() if stim is not None]
        if len({(s.unit, s.time_unit) for s in present}) > 1:
            names = ', '.join(sorted({_describe_unit(s.unit)
                                      for s in present}))
            raise DimensionMismatchError(
                f"Cannot merge stimuli measured in different units "
                f"({names}). Convert them to a common unit first.")

        # Collect each implant's 'user' metadata, keyed by implant:
        user_metadata = {str(i): stim.metadata['user']
                         for i, stim in prepared.items()
                         if stim is not None}

        # runtime import to avoid circular import
        from ..stimuli import Stimulus

        sources = self._structured_children(prepared)
        if sources is not None:
            # Every driven electrode has its own source:
            merged = Stimulus(sources, electrodes=list(sources),
                              metadata=user_metadata)
            return merged._inherit_units(present[0])

        # Interpolate each stim (n_electrodes, len(times[i])) onto the union of
        # all time points. Static stims go to the first time point; missing
        # stims are all zeros:
        stims = []
        times = []
        for i in self._implants:
            stim = prepared.get(i)
            if stim is not None:
                stims.append(stim)
                times.append(stim.time)
            else:
                stims.append(None)
                times.append(None)

        # Collect all time points, ignoring None
        valid_times = [t for t in times if t is not None]
        
        if valid_times:
            # Union of time points with tolerance, since `np.unique` would
            # keep float near-duplicates closer than DT:
            t_sorted, starts_group, _ = unique_time_points(valid_times)
            new_times = t_sorted[starts_group]
        else:
            new_times = None  # No time-dependent stimulation
        
        # Create a new list to hold interpolated stimuli
        new_stims = []
        num_timepoints = len(new_times) if new_times is not None else 1
        for implant, stim, t in zip(self._implants.values(), stims, times):
            names = implant.electrode_names
            # Electrodes the child stimulus does not name stay at zero:
            new_stim = np.zeros((len(names), num_timepoints))
            if stim is not None:
                # Match rows by name (sparse stimuli may be in any order):
                row = {e: j for j, e in enumerate(stim.electrodes)}
                data = stim.data
                for k, name in enumerate(names):
                    j = row.get(name)
                    if j is None:
                        continue
                    if t is None:
                        # A static stimulus lines up with the first time point:
                        new_stim[k, 0] = data[j, 0]
                    else:
                        # Zero outside the child's own time span:
                        new_stim[k] = np.interp(new_times, t, data[j],
                                                left=0, right=0)
            new_stims.append(new_stim)
        
        merged = Stimulus(np.concatenate(new_stims), time=new_times,
                          electrodes=self.electrode_names,
                          metadata=user_metadata)
        # Raw arrays would otherwise default to current units:
        return merged._inherit_units(present[0])
