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
        self.implants = implants
        self.safe_mode = safe_mode
        self.preprocess = preprocess
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
