"""`ScoreboardSpatial`, `ScoreboardModel"""

from ..base import Model, SpatialModel
from .._beyeler2019 import fast_scoreboard
import numpy as np

class ScoreboardSpatial(SpatialModel):
    """
    ``pulse2percept.models.ScoreboardSpatial`` adapted for cortical stimulation. 
    """
    def __init__(self, **params):
        super(ScoreboardSpatial, self).__init__(**params)

        # Use [Polemeni2006]_ visual field map by default
        if 'retinotopy' not in params.keys():
            self.retinotopy = None # TODO: Polimeni2006Map(regions=self.regions)

    def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        base_params = super(ScoreboardSpatial, self).get_default_params()
        params = {
                    # radial current spread
                    'rho': 200,  
                    # Visual field regions to simulate
                    'regions' : ['v1']
                 }
        return {**base_params, **params}


    

    def _predict_spatial(self, earray, stim):
        """Predicts the brightness at spatial locations"""

        x_el = np.array([earray[e].x for e in stim.electrodes],
                                        dtype=np.float32)
        y_el = np.array([earray[e].y for e in stim.electrodes],
                                        dtype=np.float32)
        return np.sum([
                fast_scoreboard(stim.data, x_el, y_el,
                                self.grid[region].x, self.grid[region].y,
                                self.rho, self.thresh_percept, self.n_threads)
                for region in self.regions ],
            axis = 0)
