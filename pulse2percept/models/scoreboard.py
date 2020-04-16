"""`ScoreboardModel`, `ScoreboardSpatial` [Beyeler2019]_"""
import numpy as np
from ..models import Model, SpatialModel
from ..models._scoreboard import spatial_fast
from ..utils import Watson2014Transform


class ScoreboardSpatial(SpatialModel):

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        params = super().get_default_params()
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

    def predict_spatial(self, implant, t):
        """Predicts the brightness at spatial locations"""
        if t is None:
            t = implant.stim.time
        if implant.stim.time is None and t is not None:
            raise ValueError("Cannot calculate spatial response at times "
                             "t=%s, because stimulus does not have a time "
                             "component." % t)
        # Interpolate stimulus at desired time points:
        if implant.stim.time is None:
            stim = implant.stim.data.astype(np.float32)
        else:
            stim = implant.stim[:, np.array([t]).ravel()].astype(np.float32)
        # A Stimulus could be compressed to zero:
        if stim.size == 0:
            return np.zeros((np.array([t]).size, np.prod(self.grid.x.shape)),
                            dtype=np.float32)
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        electrodes = implant.stim.electrodes
        return spatial_fast(stim,
                            np.array([implant[e].x for e in electrodes],
                                     dtype=np.float32),
                            np.array([implant[e].y for e in electrodes],
                                     dtype=np.float32),
                            self.grid.xret.ravel(),
                            self.grid.yret.ravel(),
                            self.rho, self.thresh_percept)


class ScoreboardModel(Model):

    def __init__(self, **params):
        super().__init__(spatial=ScoreboardSpatial(), temporal=None, **params)
