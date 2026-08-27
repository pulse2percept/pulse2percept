""":py:class:`~pulse2percept.implants.EnsembleImplant`"""
import numpy as np
from .base import ProsthesisSystem
from .electrodes import Electrode
from .electrode_arrays import ElectrodeArray
from ..stimuli._merge import unique_time_points
from ..stimuli.base import _describe_unit
from ..units import DimensionMismatchError, as_value, dva, um

class EnsembleImplant(ProsthesisSystem):
    
    # Frozen class: User cannot add more class attributes
    __slots__ = ('_implants', '_earray', 'safe_mode', 'preprocess')

    @classmethod
    def from_cortical_map(cls, implant_type, vfmap, locs=None, xrange=None, yrange=None, step=None,
                        region='v1'):
        """
        Create an ensemble implant from a cortical visual field map.

        The implant will be created by creating an implant of type `implant_type`
        for each visual field location specified either by locs or by xrange, yrange,
        and step. Each implant will be centered at the given location.

        Parameters
        ----------
        vfmap : p2p.topography.CorticalMap
            Visual field map to create implant from.
        implant_type : type
            Type of implant to create for the ensemble. Must subclass
            p2p.implants.ProsthesisSystem
        locs : np.ndarray with shape (n, 2), optional
            Array of visual field locations to create implants at (dva).
            Not needed if using xrange, yrange, and step.
        xrange, yrange: tuple of floats, optional
            Range of x and y coordinates (dva) to create implants at.
        step : float or (x_step, y_step), optional
            Spacing (dva) between implant centers.
        region : str, optional
            Region of cortex to create implant in.

        Returns
        -------
        ensemble : p2p.implants.EnsembleImplant
            Ensemble implant created from the cortical visual field map.

        Notes
        -----
        *  These are visual field coordinates, so they may be given as plain
           numbers of degrees or as unitful quantities (e.g.
           ``xrange=(-3 * dva, 3 * dva)``). Contrast
           :py:meth:`from_coords`, which places implants by their physical
           position in microns. See :py:mod:`pulse2percept.units`.
        """
        from ..topography import CorticalMap, Grid2D
        if not isinstance(vfmap, CorticalMap):
            raise TypeError("vfmap must be a p2p.topography.CorticalMap")
        if not issubclass(implant_type, ProsthesisSystem):
            raise TypeError("implant_type must be a sub-type of ProsthesisSystem")

        # Where in the *visual field* the implants go; `vfmap` turns that into
        # a physical location further down:
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

        implant_locations = np.array(vfmap.from_dva()[region](xlocs, ylocs)).T

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
            The grid arguments no longer have defaults. They used to fall back
            on ``(-3, 3)`` and ``1``, which are the degrees of visual angle
            :py:meth:`from_cortical_map` works in; here they are microns, so
            the default laid every implant out inside a 6 um square.

        """
        from ..topography.base import _rectangular_mesh

        if not issubclass(implant_type, ProsthesisSystem):
            raise TypeError("implant_type must be a sub-type of ProsthesisSystem")

        # Physical coordinates, unlike the dva ranges `from_cortical_map`
        # takes:
        locs = as_value(locs, um, 'locs')
        xrange = as_value(xrange, um, 'xrange')
        yrange = as_value(yrange, um, 'yrange')
        step = as_value(step, um, 'step')

        if locs is None:
            # There are two ways to say where the implants go, and no default
            # for the second one: a physical grid has no universal extent the
            # way a visual field does, and the dva defaults `from_cortical_map`
            # uses would put every implant inside a 6 um square here.
            missing = [name for name, value in [('xrange', xrange),
                                                ('yrange', yrange),
                                                ('step', step)]
                       if value is None]
            if missing:
                raise ValueError(
                    f"Pass either 'locs' or all of 'xrange', 'yrange' and "
                    f"'step' (missing: {', '.join(missing)}). Coordinates "
                    f"are physical, in microns.")

            # Laid out directly rather than through a `Grid2D`, which is a
            # grid of *visual field* coordinates and would read these microns
            # as degrees:
            (xgrid, ygrid), _, _ = _rectangular_mesh(xrange, yrange, step)
            xlocs = xgrid.flatten()
            ylocs = ygrid.flatten()
        else:
            xlocs = locs[:, 0]
            ylocs = locs[:, 1]

        implant_list = [implant_type(x=x, y=y) for x,y in zip(xlocs, ylocs)]
        
        return cls(implant_list)

    def __init__(self, implants, preprocess=False, safe_mode=False):
        """Ensemble implant

        An ensemble implant combines multiple implants into one larger electrode array
        for the purpose of modeling tandem implants, e.g. ICVP, Neuralink

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
        """
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.implants = implants

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'implants': self.implants, 'earray': self.earray,
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
            if not all(isinstance(implant, ProsthesisSystem) for implant in implants):
                raise TypeError(f"All elements in 'implants' must be ProsthesisSystem objects.")
            self._implants = {i:implant for i,implant in enumerate(implants)}
        elif isinstance(implants, dict):
            if not all(isinstance(implant, ProsthesisSystem) for implant in implants.values()):
                raise TypeError(f"All elements in 'implants' must be ProsthesisSystem objects.")
            self._implants = implants.copy()
        else:
            raise TypeError(f"'implants' must be a list or a dict object, not "
                            f"{type(implants)}.")
        # Create the electrode array
        electrodes = {}
        for i, implant in self._implants.items():
            for name, electrode in implant.earray.electrodes.items():
                electrodes[str(i) + "-" + str(name)] = electrode
            
        self._earray = ElectrodeArray(electrodes)

    def prepare_stim(self, source):
        """Prepare stimulation for an ensemble implant.

        ``source`` may address the combined electrode array directly, or be a dict
        keyed by constituent implant keys. Per-implant sources are prepared by each
        constituent implant, merged, then passed through ensemble-level preprocessing
        and safety checks. Missing implant keys contribute zeros.

        .. versionchanged:: 0.11.0
            Replaces ``merge_stimuli`` and the stimuli previously stored on
            constituent implants.

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
        >>> ensemble = EnsembleImplant([Orion(), Orion(x=-35000)])
        >>> ensemble.prepare_stim({0: np.ones(60),
        ...                        1: 2 * np.ones(60)}).data.shape
        (120, 1)
        """
        if isinstance(source, dict) and source and \
                all(key in self._implants for key in source):
            prepared = {key: implant.prepare_stim(source.get(key))
                        for key, implant in self._implants.items()}
            # Merge per-implant results before applying ensemble-level
            # preprocessing and safety checks.
            source = self._merged(prepared)
        return super().prepare_stim(source)

    def _structured_children(self, prepared):
        """One source per ensemble electrode, or ``None``"""
        sources = {}
        for i, implant in self._implants.items():
            stim = prepared.get(i)
            if stim is None:
                return None
            child = stim._structured_sources()
            if child is None:
                return None
            child = {str(e): src for e, src in child}
            names = [str(e) for e in implant.electrode_names]
            if len(child) != len(names) or any(e not in child for e in names):
                return None
            for name in names:
                sources[f"{i}-{name}"] = child[name]
        if sorted(sources) != sorted(self.electrode_names):
            return None
        # Ensemble order, not the order the children happened to be built in:
        return {name: sources[name] for name in self.electrode_names}

    def _merged(self, prepared):
        """Combine one prepared stimulus per constituent implant into one"""
        if not any(stim is not None for stim in prepared.values()):
            return None
        # An implant with no stimulus contributes zeros and no
        # interpretation of them, so only the ones that have a stimulus
        # decide what the merged numbers mean:
        present = [stim for stim in prepared.values() if stim is not None]
        if len({(s.unit, s.time_unit) for s in present}) > 1:
            names = ', '.join(sorted({_describe_unit(s.unit)
                                      for s in present}))
            raise DimensionMismatchError(
                f"Cannot merge stimuli measured in different units "
                f"({names}). Convert them to a common unit first.")

        # The metadata of each implant is stored under 'user'; concatenate
        # those, keyed by which implant they came from:
        user_metadata = {str(i): stim.metadata['user']
                         for i, stim in prepared.items()
                         if stim is not None}

        # runtime import to avoid circular import
        from ..stimuli import Stimulus

        sources = self._structured_children(prepared)
        if sources is not None:
            # Every electrode has a source of its own, so the ensemble is
            # that collection
            merged = Stimulus(sources, electrodes=self.electrode_names,
                              metadata=user_metadata)
            return merged._inherit_units(present[0])

        # Need to combine all stimuli
        # The ith stim is a np array of shape (implant[i].n_electrodes, len(times[i]))
        # i.e. the amplitude of each electrode at each time point in times[i]
        # HOWEVER, the times are not necessarily the same across implants
        # So we need to create a new times array that is the union of all times
        # and then interpolate the stimuli for each implant to this new time array
        # Also, times[i] can be None if the stim is not temporal; in this case, we
        # just line it up with the first time point. Finally, if the
        # stim is none, then we just set it to all 0's, for all the time points
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
            # Get the union of all time points. Two implants that pulse at
            # the same instant get there by accumulating their own way, so
            # an exact `np.unique` would keep both copies and leave the
            # merged axis with points closer together than DT:
            t_sorted, starts_group, _ = unique_time_points(valid_times)
            new_times = t_sorted[starts_group]
        else:
            new_times = None  # No time-dependent stimulation
        
        # Create a new list to hold interpolated stimuli
        new_stims = []
        num_timepoints = len(new_times) if new_times is not None else 1
        for i, (stim, t) in enumerate(zip(stims, times)):
            n_electrodes = len(self._implants[list(self._implants.keys())[i]].electrode_names)
            if stim is None:
                # If stim is None, create a zero array of shape (n_electrodes, len(new_times))
                new_stim = np.zeros((n_electrodes, num_timepoints))
            elif t is None:
                # If stim exists but has no time information, assume all values correspond to first time point
                # fill the rest with 0s
                new_stim = np.zeros((n_electrodes, num_timepoints))
                new_stim[:, 0] = stim.data[:, 0]
            else:
                # Interpolate the stim data to new_times
                new_stim = np.zeros((n_electrodes, len(new_times)))
                for j in range(stim.data.shape[0]):  # Interpolate each electrode separately
                    # if the stim ends, make it 0 instead of repeating the last value. Only interpolate
                    # for the times that are in the original stim
                    new_stim[j] = np.interp(new_times, t, stim.data[j], left=0, right=0)
            
            new_stims.append(new_stim)
        
        # Combine all new_stims into a final array (stack along a new axis if needed)
        merged = Stimulus(np.concatenate(new_stims), time=new_times,
                          electrodes=self.electrode_names,
                          metadata=user_metadata)
        # The merge concatenates raw data arrays, so the result would
        # otherwise fall back to the default (current) reading of them:
        return merged._inherit_units(present[0])
