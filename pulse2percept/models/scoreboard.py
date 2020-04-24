"""`ScoreboardModel`, `ScoreboardSpatial` [Beyeler2019]_"""
import numpy as np
from .base import Model, SpatialModel
from ._scoreboard import spatial_fast
from ..implants import ElectrodeArray
from ..stimuli import Stimulus
from ..utils import Watson2014Transform


class ScoreboardSpatial(SpatialModel):

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        params = super(ScoreboardSpatial, self).get_default_params()
        params.update({'rho': 100})
        return params

    @staticmethod
    def dva2ret(xdva):
        """Convert degrees of visual angle (dva) into retinal coords (um)"""
        return Watson2014Transform.dva2ret(xdva)

    @staticmethod
    def ret2dva(xret):
        """Convert retinal corods (um) to degrees of visual angle (dva)"""
        return Watson2014Transform.ret2dva(xret)

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        assert isinstance(earray, ElectrodeArray)
        assert isinstance(stim, Stimulus)
        return spatial_fast(stim.data,
                            np.array([earray[e].x for e in stim.electrodes],
                                     dtype=np.float32),
                            np.array([earray[e].y for e in stim.electrodes],
                                     dtype=np.float32),
                            self.grid.xret.ravel(),
                            self.grid.yret.ravel(),
                            self.rho,
                            self.thresh_percept)


class ScoreboardModel(Model):

    def __init__(self, **params):
        super(ScoreboardModel, self).__init__(spatial=ScoreboardSpatial(),
                                              temporal=None, **params)
