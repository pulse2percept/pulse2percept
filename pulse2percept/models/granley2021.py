"""`BiphasicAxonMapModel`""" 
import numpy as np

from .beyeler2019 import AxonMapModel, AxonMapSpatial
from .base import TemporalModel, Model
from ._granley2021 import (fast_biphasic_axon_map)
from ..implants import ProsthesisSystem, ElectrodeArray
from ..stimuli import BiphasicPulseTrain, Stimulus
from ..datasets import load_nanduri2012

from sklearn.linear_model import LinearRegression

class DefaultBrightModel():
    def __init__(self):
        self.amp_freq_model = LinearRegression()

    def fit(self):
        self._fit_amp_freq()

    def _fit_amp_freq(self):
        data = load_nanduri2012()
        data = data[data['task'] == 'rate']
        x = data[['amp_factor', 'freq']]
        y = data['brightness']
        self.amp_freq_model.fit(x, y)

    def predict_pdur(self, pdur):
        # Fit using color threshold of weitz et al 2015
        return 1 / (0.952 + 0.215*pdur)
       

    def __call__(self, amp, freq, pdur):
        bright = self.amp_freq_model.predict([(amp, freq)])[0] * self.predict_pdur(pdur)

        # p = (1 / (1 + np.exp(-np.log(98)*x + np.log(96))) - 1/97) / (96/97)
        # Sigmoid with p[0] = 0, p[1] = 0.5, p[2] = 0.99
        p = (1 / (1 + np.exp(-4.58497 * amp + 4.56435)) - 0.0103093) / 0.989691
        if np.random.random() < p:
            return bright
        else:
            return 0


class DefaultSizeModel():
    def __init__(self):
        self.amp_model = LinearRegression()

    def fit(self):
        self._fit_amp()

    def _fit_amp(self):
        data = load_nanduri2012()
        data = data[data['task'] == 'size']
        x = data['amp_factor']
        y = data['size']
        self.amp_model.fit(np.array(x).reshape(-1, 1), y)

    def __call__(self, amp, freq, pdur):
        return self.amp_model.predict(np.array([amp]).reshape(1, -1))[0] 


class DefaultStreakModel():
    def __init__(self, axlambda):
        # never decrease lambda to less than 25
        self.min_lambda = 25
        self.min_scale = self.min_lambda / axlambda 
    def __call__(self, amp, freq, pdur):
        # Fit using streak lengths measure from weitz et al 2015
        scale = 1.56 - 0.54 * pdur ** 0.21
        if scale > self.min_scale:
            # Sqaured because it will be multiplied by axlambda squared, and we want to scale lambda
            return scale ** 2
        else:
            return self.min_scale ** 2




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
            'bright_model' : None, 
            # Callable model used to modulate percept size with amplitude, frequency, and pulse duration
            'size_model' : None, 
            # Callable model used to modulate percept streak length with amplitude, frequency, and pulse duration
            'streak_model' : None 
        }
        params.update(base_params)
        return params

    def _build(self):
        # Fit models if needed 
        if self.bright_model is None:
            self.bright_model = DefaultBrightModel()
            self.bright_model.fit()
        if self.size_model is None:
            self.size_model = DefaultSizeModel()
            self.size_model.fit()
        if self.streak_model is None:
            self.streak_model = DefaultStreakModel(self.axlambda)
        assert(callable(self.bright_model))
        assert(callable(self.size_model))
        assert(callable(self.streak_model))
        super(BiphasicAxonMapSpatial, self)._build()

    def _predict_spatial(self, earray, stim):
        """Predicts the percept"""
        assert isinstance(earray, ElectrodeArray)
        assert isinstance(stim, Stimulus)

        # Calculate model effects before losing GIL
        bright_effects = []
        size_effects = []
        streak_effects = []
        amps = []

        for e in stim.electrodes:
            amp = stim.metadata['electrodes'][str(e)]['metadata']['amp']
            freq = stim.metadata['electrodes'][str(e)]['metadata']['freq']
            pdur = stim.metadata['electrodes'][str(e)]['metadata']['phase_dur']
            bright_effects.append(self.bright_model(amp, freq, pdur))
            size_effects.append(self.size_model(amp, freq, pdur))
            streak_effects.append(self.streak_model(amp, freq, pdur))
            amps.append(amp)
                          

        return fast_biphasic_axon_map(np.array(amps, dtype=np.float32),
                                      np.array(bright_effects, dtype=np.float32),
                                      np.array(size_effects, dtype=np.float32),
                                      np.array(streak_effects, dtype=np.float32),
                                      np.array([earray[e].x for e in stim.electrodes],
                                            dtype=np.float32),
                                      np.array([earray[e].y for e in stim.electrodes],
                                            dtype=np.float32),
                                      self.axon_contrib,
                                      self.axon_idx_start.astype(np.uint32),
                                      self.axon_idx_end.astype(np.uint32),
                                      self.rho,
                                      self.thresh_percept,
                                      stim.shape[1])

class BiphasicAxonMapModel(Model):
    """
    AxonMapModel that scales percept size, brightness, and streak length according to
    the amplitude, frequency, and pulse duration of the BiphasicPulseTrain.

    All stimuli must be BiphasicPulseTrains with no delay dur.

    This model is different than other spatial models in that it calculates the brightest percept 
    from all time steps of the stimulus, and then returns this same percept at each time step. 

    The three new parameters are the models to be used to scale brightness, size, and streak length. 
    These models can be any python callable with function signature f(amp, freq, pdur) that return a float.


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
            for ele, params in implant.stim.metadata['electrodes'].items():
                if params['type'] != BiphasicPulseTrain or params['metadata']['delay_dur'] != 0: 
                    raise TypeError("All stimuli must be BiphasicPulseTrains with no delay dur (Failing electrode: %s)" % (ele)) 
        
        return super(BiphasicAxonMapModel, self).predict_percept(implant,
                                                         t_percept=t_percept)
    