"""`EnsembleImplant`"""
from .base import ProsthesisSystem
from .electrodes import Electrode
from .electrode_arrays import ElectrodeArray

class EnsembleImplant(ProsthesisSystem):
    

    # Frozen class: User cannot add more class attributes
    __slots__ = ('_implants', '_earray', '_stim', 'safe_mode', 'preprocess')

    def __init__(self, implants, stim=None, preprocess=False,safe_mode=False):
        """Ensemble implant

        An ensemble implant combines multiple implants into one larger electrode array
        for the purpose of modeling tandem implants, e.g. CORTIVIS, ICVP
        
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
