"""`ScoreboardModel`"""
import numpy as np
from ..models import BaseModel, Watson2014ConversionMixin
from ..models._scoreboard import spatial_fast


class ScoreboardModel(Watson2014ConversionMixin, BaseModel):
    """Scoreboard model"""
    # Frozen class: User cannot add more class attributes
    __slots__ = ('rho',)

    def _get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        params = super(ScoreboardModel, self)._get_default_params()
        params.update({'rho': 100})
        return params

    def _predict_spatial(self, implant, t):
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
