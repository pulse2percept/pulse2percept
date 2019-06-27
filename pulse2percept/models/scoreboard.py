import numpy as np
from pulse2percept.models.base import BaseModel
from pulse2percept.models.watson import WatsonConversionMixin
from pulse2percept.models._scoreboard import fast_score


class ScoreboardModel(WatsonConversionMixin, BaseModel):
    """Scoreboard model"""

    def _get_default_params(self):
        params = super(ScoreboardModel, self)._get_default_params()
        params.update({'rho': 100, 'thresh_percept': 1.0 / np.sqrt(np.e)})
        return params

    def _predict_pixel_percept(self, xygrid, stimulus, t=None):
        idx_xy, xydva = xygrid
        # Find all nonzero entries in the stimulus array:
        electrodes, pulses = stimulus.nonzero()
        # Call the Cython function for fast processing:
        bright = fast_score(pulses,
                            np.array([e.x for e in electrodes]),
                            np.array([e.y for e in electrodes]),
                            *self.get_tissue_coords(*xydva),
                            self.rho,
                            self.thresh_percept)
        # return utils.Percept(self.xdva, self.ydva, brightness)
        return bright
