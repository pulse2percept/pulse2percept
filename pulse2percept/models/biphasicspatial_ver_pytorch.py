"""`BiphasicAxonMapModel`, `BiphasicAxonMapSpatial`, [Granley2021]_"""
from functools import partial
import numpy as np
import sys
import torch
import torch.nn as nn
from . import AxonMapSpatial, Model
from ..implants import ProsthesisSystem, ElectrodeArray
from ..stimuli import BiphasicPulseTrain, Stimulus
from ..percepts import Percept
from ..utils import FreezeError
from .base import NotBuiltError, BaseModel
from ._granley2021 import fast_biphasic_axon_map

import torch
import torch.nn as nn

class FreezeError(AttributeError):
    """Custom error for attribute modification attempts outside the constructor."""
    pass

class TorchBiphasicAxonMapSpatial(nn.Module):
    """
    PyTorch version of BiphasicAxonMapSpatial model designed to simulate visual percepts 
    induced by retinal prostheses with adjustments for brightness, size, and streak length 
    based on electrical stimulation parameters.

    Parameters
    ----------
    See class documentation of BiphasicAxonMapSpatial for details on parameters.
    """
    def __init__(self, bright_model=None, size_model=None, streak_model=None, **params):
        super(TorchBiphasicAxonMapSpatial, self).__init__()
        self.is_built = False  
        self.bright_model = bright_model
        self.size_model = size_model
        self.streak_model = streak_model
        for key, val in params.items():
            setattr(self, key, val)
        self.is_built = True  
    
    def __getattr__(self, attr):
        # Mimic the JAX version's behavior for attribute access
        if not self.is_built and (attr in ['bright_model', 'size_model', 'streak_model']):
            raise AttributeError(f"{attr} not found. Required model components must be set during initialization.")
        return super().__getattr__(attr)
    
    def __setattr__(self, name, value):
        if not self.is_built:
            # Allow setting attributes freely during initialization
            object.__setattr__(self, name, value)
        else:
            # After initialization, restrict attribute setting
            if name in ['bright_model', 'size_model', 'streak_model', 'is_built']:
                object.__setattr__(self, name, value)
            else:
                # Check if attempting to set a parameter that exists
                try:
                    getattr(self, name)
                    super(TorchBiphasicAxonMapSpatial, self).__setattr__(name, value)
                except AttributeError as e:
                    # If the attribute doesn't exist or is not allowed, raise a FreezeError
                    raise FreezeError(f"'{name}' not found or cannot be modified after initialization.") from e

    @classmethod
    def get_default_params(cls):
        # Assuming there's a superclass with its own get_default_params method
        # This would call the superclass's get_default_params if it exists,
        # and start with an empty dict if it doesn't.
        base_params = super().get_default_params() if hasattr(super(), 'get_default_params') else {}
        params = {
            'bright_model': None,
            'size_model': None,
            'streak_model': None,
            # Additional specific parameters can be defined here.
        }
        # Merge the base parameters with this class's specific parameters
        return {**base_params, **params}

    def build(self):
        """Builds the model, ensuring all components are properly configured."""
        # Ensure models are callable
        for model in [self.bright_model, self.size_model, self.streak_model]:
            if not isinstance(model, torch.nn.Module):
                raise TypeError(f"{model} needs to be an instance of torch.nn.Module or callable")
                
        # Initialize or reconfigure models based on provided parameters
        # Placeholder for actual build logic that would initialize the model
        # for simulation based on the current set of parameters.
        print("Model built with current parameters.")

    def _predict_spatial(self, earray, stim):
            """Predicts the percept using PyTorch"""
            if not isinstance(earray, ElectrodeArray):
                raise TypeError("Implant must be of type ElectrodeArray but it is " + str(type(earray)))
    
            if not isinstance(stim, Stimulus):
                raise TypeError("Stim must be of type Stimulus but it is " + str(type(stim)))

            x = []
            y = []
            elec_params = []
            for e in stim.electrodes:
                amp = stim.metadata['electrodes'][str(e)]['metadata']['amp']
                if amp == 0:
                    continue
                freq = stim.metadata['electrodes'][str(e)]['metadata']['freq']
                pdur = stim.metadata['electrodes'][str(e)]['metadata']['phase_dur']
                elec_params.append([freq, amp, pdur])
                x.append(earray[e].x)
                y.append(earray[e].y)

            # Convert lists to PyTorch tensors
            elec_params = torch.tensor(elec_params, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            # Apply the models to compute effects
            bright_effects = self.bright_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]).view(-1)
            size_effects = self.size_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]).view(-1)
            streak_effects = self.streak_model(elec_params[:, 0], elec_params[:, 1], elec_params[:, 2]).view(-1)
            amps = elec_params[:, 1].view(-1)

            # Placeholder for the actual PyTorch equivalent of fast_biphasic_axon_map
            # This function needs to be implemented in PyTorch, taking into account
            # the inputs now available as PyTorch tensors.
            # Example: result = my_pytorch_biphasic_axon_map(amps, bright_effects, size_effects, streak_effects, x, y, ...)
            # return result

            # Since the PyTorch equivalent of fast_biphasic_axon_map is not defined,
            # this is a placeholder return statement.
            return None
    

    # only need handle tensor inputs
    # look at original biphasic axon map model for reference
    # so don't need set_atrributes

    def forward(self, inputs, like_jax=False):
        freq = inputs[0][:, :, 0]
        amp = inputs[0][:, :, 1]
        pdur = inputs[0][:, :, 2]

        rho = inputs[1][:, 0][:, None]
        axlambda = inputs[1][:, 1][:, None]
        a0 = inputs[1][:, 2][:, None]
        a1 = inputs[1][:, 3][:, None]
        a2 = inputs[1][:, 4][:, None]
        a3 = inputs[1][:, 5][:, None]
        a4 = inputs[1][:, 6][:, None]
        a5 = inputs[1][:, 7][:, None]
        a6 = inputs[1][:, 8][:, None]
        a7 = inputs[1][:, 9][:, None]
        a8 = inputs[1][:, 10][:, None]
        a9 = inputs[1][:, 11][:, None]

        scaled_amps = (a1 + a0*pdur) * amp

        # bright
        F_bright = a2 * scaled_amps + a3 * freq
        if self.amp_cutoff:
            F_bright = torch.where(scaled_amps > 0.25, F_bright, torch.zeros_like(F_bright))

        if not like_jax: # like pyx impl.
            F_bright = torch.where(amp > 0, F_bright, torch.zeros_like(F_bright))

        # size
        min_f_size = 10**2 / (rho**2)
        F_size = a5 * scaled_amps + a6
        F_size = torch.maximum(F_size, min_f_size)

        # streak
        min_f_streak = 10**2 / (axlambda ** 2)
        F_streak = a9 - a7 * pdur ** a8
        F_streak = torch.maximum(F_streak, min_f_streak)

        # eparams = torch.stack([F_bright, F_size, F_streak], axis=2) # 1, 225, 3

        # apply axon map
        intensities =   (
                        F_bright[:, None, None, :] * # 1, 1, 1, 225
                        torch.exp(
                                    -self.d2_el[None, :, :, :] / # dist2el 1, 2401, 118, 225
                                    (2. * rho**2 * F_size)[:, None, None, :] # 1, 1, 1, 225
                                    + # contribution of each electode to each axon segement of each
                                      # pixel by distance of segemnt to electrode
                                    self.axon_contrib[None, :, :, 2, None] / # sens 1, 2401, 118, 1
                                    (axlambda** 2 * F_streak)[:, None, None, :] # 1, 1, 1 , 225
                                      # contribution of each electode to each axon segement of each
                                      # pixel by sensitivity, which is scaled by axon distance
                                 ) # 1, 2401, 118, 225, scaling between 0, 1
                        ) # 1, 2401, 118, 225

        # after summing up...
        intensities = torch.max(torch.sum(intensities, axis=-1), axis=-1).values # sum over electrodes, max over segments
        intensities = torch.where(intensities > self.thresh_percept, intensities, torch.zeros_like(intensities))
        if self.clip:
            intensities = torch.clamp(intensities, self.clipmin, self.clipmax)

        batched_percept_shape = tuple([-1] + list(self.percept_shape))
        intensities = intensities.reshape(batched_percept_shape)
        return intensities 