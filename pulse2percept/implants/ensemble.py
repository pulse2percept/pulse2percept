"""`EnsembleImplant`"""
import numpy as np
from .base import ProsthesisSystem
from .electrodes import Electrode
from .electrode_arrays import ElectrodeArray

class EnsembleImplant(ProsthesisSystem):
    

    # Frozen class: User cannot add more class attributes
    __slots__ = ('_implants', '_earray', '_stim', 'safe_mode', 'preprocess')

    @classmethod
    def from_cortical_map(cls, implant_type, vfmap, locs=None, xrange=None, yrange=None, xystep=None, 
                        region='v1'):
        """
        Create an ensemble implant from a cortical visual field map.

        The implant will be created by creating an implant of type `implant_type`
        for each visual field location specified either by locs or by xrange, yrange, 
        and xystep. Each implant will be centered at the given location.

        Parameters
        ----------
        vfmap : p2p.topography.CorticalMap
            Visual field map to create implant from.
        implant_type : type
            Type of implant to create for the ensemble. Must subclass 
            p2p.implants.ProsthesisSystem
        locs : np.ndarray with shape (n, 2), optional
            Array of visual field locations to create implants at (dva). 
            Not needed if using xrange, yrange, and xystep.
        xrange, yrange: tuple of floats, optional
            Range of x and y coordinates (dva) to create implants at.
        xystep : float, optional
            Spacing between implant centers. 
        region : str, optional
            Region of cortex to create implant in.

        Returns
        -------
        ensemble : p2p.implants.EnsembleImplant
            Ensemble implant created from the cortical visual field map.
        """
        from ..topography import CorticalMap, Grid2D
        if not isinstance(vfmap, CorticalMap):
            raise TypeError("vfmap must be a p2p.topography.CorticalMap")
        if not issubclass(implant_type, ProsthesisSystem):
            raise TypeError("implant_type must be a sub-type of ProsthesisSystem")

        if locs is None:
            if xrange is None:
                xrange = (-3, 3)
            if yrange is None:
                yrange = (-3, 3)
            if xystep is None:
                xystep = 1
            
            # make a grid of points
            grid = Grid2D(xrange, yrange, xystep)
            xlocs = grid.x.flatten()
            ylocs = grid.y.flatten()
        else:
            xlocs = locs[:, 0]
            ylocs = locs[:, 1]

        implant_locations = np.array(vfmap.from_dva()[region](xlocs, ylocs)).T

        return cls.from_coords(implant_type=implant_type, locs=implant_locations)


    @classmethod
    def from_coords(cls, implant_type, locs=None, xrange=None, yrange=None, xystep=None):
        """
        Create an ensemble implant using physical (cortical or retinal) coordinates.

        Parameters
        ----------
        implant_type : type
            The type of implant to create for the ensemble.
        locs : np.ndarray with shape (n, 2), optional
            Array of physical locations (um) to create implants at. Not
            needed if using xrange, yrange, and xystep.
        xrange, yrange: tuple of floats, optional
            Range of x and y coordinates to create implants at.
        xystep : float, optional
            Spacing between implant centers. 
        """
        from ..topography import Grid2D

        if not issubclass(implant_type, ProsthesisSystem):
            raise TypeError("implant_type must be a sub-type of ProsthesisSystem")
        
        if locs is None:
            if xrange is None:
                xrange = (-3, 3)
            if yrange is None:
                yrange = (-3, 3)
            if xystep is None:
                xystep = 1
            
            # make a grid of points
            grid = Grid2D(xrange, yrange, xystep)
            xlocs = grid.x.flatten()
            ylocs = grid.y.flatten()
        else:
            xlocs = locs[:, 0]
            ylocs = locs[:, 1]

        implant_list = [implant_type(x=x, y=y) for x,y in zip(xlocs, ylocs)]
        
        return cls(implant_list)

    def __init__(self, implants, stim=None, preprocess=False,safe_mode=False):
        """Ensemble implant

        An ensemble implant combines multiple implants into one larger electrode array
        for the purpose of modeling tandem implants, e.g. ICVP, Neuralink
        
        Parameters
        ----------
        implants : list or dict
            A list or dict of implants to be combined.
        stim : :py:class:`~pulse2percept.stimuli.Stimulus` source type
            A valid source type for the :py:class:`~pulse2percept.stimuli.Stimulus`
            object (e.g., scalar, NumPy array, pulse train).
        preprocess : bool or callable, optional
            Either True/False to indicate whether to execute the implant's default
            preprocessing method whenever a new stimulus is assigned, or a custom
            function (callable).
        safe_mode : bool, optional
            If safe mode is enabled, only charge-balanced stimuli are allowed.
        """
        self.preprocess = preprocess
        self.safe_mode = safe_mode
        self.implants = implants
        # self.stim might be set in self.implants = implants, so don't override it 
        # unless the user actually passes a stimulus
        if stim is not None or not hasattr(self, 'stim'):
            self.stim = stim

    def _pprint_params(self):
        """Return dict of class attributes to pretty-print"""
        return {'implants': self.implants, 'earray': self.earray, 'stim': self.stim,
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
        self.merge_stimuli()
        
    def merge_stimuli(self):
        """Constructs the combined stimulus for all implants in self._implants"""
        if any([i.stim for i in self._implants.values()]):
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
            for i, implant in self._implants.items():
                if implant.stim is not None:
                    stims.append(implant.stim)
                    times.append(implant.stim.time)
                else:
                    stims.append(None)
                    times.append(None)

            # Collect all time points, ignoring None
            valid_times = [t for t in times if t is not None]
            
            if valid_times:
                # Get the union of all time points
                new_times = np.unique(np.concatenate(valid_times))
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
            
            # The metadata for each implant is stored in implant.metadata, and has 'electrodes' and 'user' keys
            # We need to merge the 'electrodes' key across all implants, and simply concatenate the 'user' keys
            metadata = {}
            electrode_metadata = {}
            user_metadata = {}
            for i_name, implant in self._implants.items():
                if implant.stim is None:
                    continue
                # in the new implant, the the electrode names are i_name + "-" + electrode_name
                for e_name, e_metadata in implant.stim.metadata['electrodes'].items():
                    electrode_metadata[str(i_name) + "-" + e_name] = e_metadata
                user_metadata[str(i_name)] = implant.stim.metadata['user']
            metadata['electrodes'] = electrode_metadata
            metadata['user'] = user_metadata

            # Combine all new_stims into a final array (stack along a new axis if needed)
            # runtime import to avoid circular import
            from ..stimuli import Stimulus
            self.stim = Stimulus(np.concatenate(new_stims), time=new_times, electrodes=self.electrode_names, metadata=metadata)
