"""`BiphasicAxonMapModel`""" 
import numpy as np

from .beyeler2019 import AxonMapModel, AxonMapSpatial
from .base import TemporalModel, Model
from ._granley2021 import (fast_biphasic_axon_map)
from ..implants import ProsthesisSystem, ElectrodeArray
from ..stimuli import BiphasicPulseTrain, Stimulus

class BiphasicAxonMapSpatial(AxonMapSpatial):
    """
    AxonMapSpatial that scales percept size, brightness, and streak length according to
    the amplitude, frequency, and pulse duration of the BiphasicPulseTrain.

    All stimuli must be BiphasicPulseTrains.

    This model is different than other spatial models in that it calculates one percept from all time
    steps of the stimulus, and then returns this same percept at each time step. 

    The three new parameters are the models to be used to scale brightness, size, and streak length. 
    These models can be any python callable with function signature f(amp, freq, pdur) that returns a float.


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
        Arguments to be passed to AxonMapSpatial
    """

    def __init__(self, **params):
        super(BiphasicAxonMapSpatial, self).__init__(**params)
    
    def get_default_params(self):
        base_params = super(BiphasicAxonMapSpatial, self).get_default_params()
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

    def _build(self):
        # Fit models if needed
        if self.bright_model is None:
            pass
        if self.size_model is None:
            pass
        if self.streak_model is None:
            pass
        super(BiphasicAxonMapSpatial, self)._build()

    def _predict_spatial(self, earray, stim):
        """Predicts the percept"""
        assert isinstance(earray, ElectrodeArray)
        assert isinstance(stim, Stimulus)

        # stim.data is a NxM array of amps at time points, dont care about this at all
        data = 

        return fast_biphasic_axon_map(stim.data,
                                      np.array([earray[e].x for e in stim.electrodes],
                                            dtype=np.float32),
                                      np.array([earray[e].y for e in stim.electrodes],
                                            dtype=np.float32),
                                      self.axon_contrib,
                                      self.axon_idx_start.astype(np.uint32),
                                      self.axon_idx_end.astype(np.uint32),
                                      self.rho,
                                      self.thresh_percept,
                                      self.bright_model,
                                      self.size_model,
                                      self.streak_model)

class BiphasicAxonMapModel(Model):
    """
    AxonMapModel that scales percept size, brightness, and streak length according to
    the amplitude, frequency, and pulse duration of the BiphasicPulseTrain.

    All stimuli must be BiphasicPulseTrains.

    This model is different than other spatial models in that it calculates one percept from all time
    steps of the stimulus, and then returns this same percept at each time step. 

    The three new parameters are the models to be used to scale brightness, size, and streak length. 
    These models can be any python callable with function signature f(amp, freq, pdur) that returns a float.


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
        super(BiphasicAxonMapModel, self).__init__(spatial=BiphasicAxonMapSpatial(),
                                           temporal=None, **params)

    def predict_percept(self, implant, t_percept=None):
        # Make sure stimulus is a BiphasicPulseTrain:
        if not isinstance(implant.stim, BiphasicPulseTrain):
            # Could still be a stimulus where each electrode has a biphasic pulse train
            for ele, metadata in implant.stim.metadata['electrodes'].items():
                if not isinstance(metadata['type'], BiphasicPulseTrain): 
                    raise TypeError("Stimuli must be a BiphasicPulseTrain (Electrode %s is not)" % (ele)) 
        
        return super(BiphasicAxonMapModel, self).predict_percept(implant,
                                                         t_percept=t_percept)
    