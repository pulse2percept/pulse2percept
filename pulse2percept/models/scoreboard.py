"""`ScoreboardModel`"""
import numpy as np
from ..models import BaseModel, Watson2014ConversionMixin
from ..models._scoreboard import scoreboard_fast


class ScoreboardModel(Watson2014ConversionMixin, BaseModel):
    """Scoreboard model"""
    # Frozen class: User cannot add more class attributes
    __slots__ = ('rho',)

    def _get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        params = super(ScoreboardModel, self)._get_default_params()
        params.update({'rho': 100})
        return params

    def _predict_spatial(self, implant, t=0):
        """Predicts the brightness at spatial locations"""
        assert t is not None
        # Interpolate stimulus at desired time points:
        stim = implant.stim[:, np.array([t]).ravel()].astype(np.float32)
        # This does the expansion of a compact stimulus and a list of
        # electrodes to activation values at X,Y grid locations:
        electrodes = implant.stim.electrodes
        bright = scoreboard_fast(stim,
                                 np.array([implant[e].x for e in electrodes],
                                          dtype=np.float32),
                                 np.array([implant[e].y for e in electrodes],
                                          dtype=np.float32),
                                 self.grid.xret.ravel(),
                                 self.grid.yret.ravel(),
                                 self.rho, self.thresh_percept)
        # TODO:
        # return utils.Percept(self.xdva, self.ydva, brightness)
        # Reshape to T x X x Y:
        return bright.reshape([-1] + list(self.grid.x.shape))
