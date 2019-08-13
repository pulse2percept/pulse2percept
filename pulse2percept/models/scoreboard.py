import numpy as np
from ..models import BaseModel, Watson2014ConversionMixin
from ..models._scoreboard import scoreboard


class ScoreboardModel(Watson2014ConversionMixin, BaseModel):
    """Scoreboard model"""

    def _get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        params = super()._get_default_params()
        params.update({'rho': 100, 'thresh_percept': 1.0 / np.sqrt(np.e)})
        return params

    def _predict_pixel_percept(self, xygrid, implant, t=None):
        """Predicts the brightness at a particular pixel location

        Parameters
        ----------
        xygrid : tuple
            (idx, (x, y)): Stemming from an enumerate of self.grid
        implant : ProsthesisSystem
            A ProsthesisSystem object with an assigned stimulus
        t : optional
            Not yet implemented.
        """
        idx_xy, xydva = xygrid
        # Call the Cython function for fast processing:
        electrodes = implant.stim.electrodes
        bright = scoreboard(implant.stim.interp(time=t).data.ravel(),
                            np.array([implant[e].x for e in electrodes],
                                     dtype=float),
                            np.array([implant[e].y for e in electrodes],
                                     dtype=float),
                            *self.get_tissue_coords(*xydva),
                            self.rho)
        # return utils.Percept(self.xdva, self.ydva, brightness)
        return bright
