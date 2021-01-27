"""`BiphasicAxonMapModel`""" 

from .beyeler2019 import AxonMapModel
from .base import TemporalModel
from ..stimuli import BiphasicPulseTrain

class BiphasicAxonMapModel(AxonMapModel):
    """
    AxonMapModel that scales percept size, brightness, and streak length according to
    the amplitude, frequency, and pulse duration of the BiphasicPulseTrain.

    All stimuli must be BiphasicPulseTrains.

    This model is different than other spatial models in that it calculates one percept from all time
    steps of the stimulus, and then returns this same percept at each time step. 

    The three new parameters are the models to be used to scale brightness, size, and streak length. 
    These models can be any function or object that is callable with function signature f(amp, freq, pdur).

    .. note: :

        Using this model in combination with a temporal model is not supported and may give unexpected results

    Parameters
    ----------
    bright_model: callable, optional
        Model used to modulate percept brightness with amplitude, frequency, and pulse duration
    size_model: callable, optional
        Model used to modulate percept size with amplitude, frequency, and pulse duration
    streak_model: callable, optional
        Model used to modulate percept streak length with amplitude, frequency, and pulse duration
    **params: dict, optional
        Arguments to be passed to AxonMapModel
    """

    def __init__(self, **params):
        super(BiphasicAxonMapModel, self).__init__(**params)
    
    def get_default_params(self):
        base_params = super(BiphasicAxonMapModel, self).get_default_params()
        params = {
            # Callable model used to modulate percept brightness with amplitude, frequency, and pulse duration
            'bright_model' : None, # TODO
            # Callable model used to modulate percept size with amplitude, frequency, and pulse duration
            'size_model' : None, # TODO
            # Callable model used to modulate percept streak length with amplitude, frequency, and pulse duration
            'streak_model' : None # TODO
        }
        params.update(base_params)
        return params

    def _predict_spatial(self, earray, stim):
        """Predicts the percept"""
        assert isinstance(earray, ElectrodeArray)
        assert isinstance(stim, Stimulus)

        
        return fast_axon_map(stim.data,
                             np.array([earray[e].x for e in stim.electrodes],
                                      dtype=np.float32),
                             np.array([earray[e].y for e in stim.electrodes],
                                      dtype=np.float32),
                             self.axon_contrib,
                             self.axon_idx_start.astype(np.uint32),
                             self.axon_idx_end.astype(np.uint32),
                             self.rho,
                             self.thresh_percept)


    def predict_percept(self, implant, t_percept=None):
        # Make sure stimulus is a BiphasicPulseTrain:
        if not isinstance(implant.stim, BiphasicPulseTrain):
            raise TypeError("Stimuli must be a BiphasicPulseTrain")
        
        return super(BiphasicAxonMapModel, self).predict_percept(implant,
                                                         t_percept=t_percept)
    