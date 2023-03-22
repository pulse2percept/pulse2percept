"""`DynaphosModel`, `DynaphosSpatial`, `DynaphosTemporal`"""
import numpy as np
import warnings
from ..base import Model, SpatialModel
from ...utils import cart2pol
from ...topography import Polimeni2006Map

class DynaphosSpatial(SpatialModel):
   """Adaptation of the Dynaphos model from Grinten, Stevenick, Lozano (2022)

   Implements the spatial component of the Dynaphos model. Percepts from each
   electrode are Gaussian blobs, with the size dictated by a magnification factor
   M determined by the electrode's position in the visual cortex.
   """

   def __init__(self, **params):
         self._regions = None
         super(DynaphosSpatial, self).__init__(**params)

         # Use [Polemeni2006]_ visual field map with parameters specified in the paper
         self.retinotopy = Polimeni2006Map(a=0.75,k=17.3,b=120,alpha1=0.95,regions=self.regions)
   
   def get_default_params(self):
        """Returns all settable parameters of the scoreboard model"""
        base_params = super(DynaphosSpatial, self).get_default_params()
        params = {
                    'xrange' : (-5, 5),
                    'yrange' : (-5, 5),
                    'xystep' : 0.1,
                    # Visual field regions to simulate
                    'regions' : ['v1']
                 }
        return {**base_params, **params}
   
   def _build(self):
        # warn the user either that they are simulating points at discontinuous boundaries, 
        # or that the points will be moved by a small constant
        if np.any(self.grid['dva'].x == 0):
            if hasattr(self.retinotopy, 'jitter_boundary') and self.retinotopy.jitter_boundary:
                warnings.warn("Since the visual cortex is discontinuous " +
                    "across hemispheres, it is recommended to not simulate points " +
                    " at exactly x=0. Points on the boundary will be moved " +
                    "by a small constant") 
            else:
                warnings.warn("Since the visual cortex is discontinuous " +
                    "across hemispheres, it is recommended to not simulate points " +
                    " at exactly x=0. This can be avoided by adding a small " + 
                    "to both limits of xrange") 
        if (np.any([r in self.regions for r in self.grid.discontinuous_y]) and 
            np.any(self.grid['dva'].y == 0)):
            if hasattr(self.retinotopy, 'jitter_boundary') and self.retinotopy.jitter_boundary:
                warnings.warn("Since some simulated regions are discontinuous " +
                    "across the y axis, it is recommended to not simulate points " +
                    " at exactly y=0.  Points on the boundary will be moved " +
                    "by a small constant") 
            else:
                warnings.warn(f"Since some simulated regions are discontinuous " +
                    "across the y axis, it is recommended to not simulate points " +
                    " at exactly y=0. This can be avoided by adding a small " + 
                    "to both limits of yrange or setting " +
                    "self.retinotopy.jitter_boundary=True")
                
   def _predict_spatial(self, earray, stim):
      """Predicts the brightness at spatial locations"""
      x_el = np.array([earray[e].x for e in stim.electrodes],
                                       dtype=np.float32)
      y_el = np.array([earray[e].y for e in stim.electrodes],
                                          dtype=np.float32)
      # whether to allow current to spread between hemispheres
      separate = 0
      boundary = 0
      if self.retinotopy.split_map:
         separate = 1
         boundary = self.retinotopy.left_offset/2
      
      phosphene_locations = {}
      for region in self.regions:
         phosphene_locations[region] = self.retinotopy.to_dva()[region](x_el, y_el)

      theta, r = cart2pol(*phosphene_locations['v1'])

      # magnification factors (mm/dva)
      M = self.retinotopy.k * (self.retinotopy.b - self.retinotopy.a) / ((r + self.retinotopy.a) * (r + self.retinotopy.b))

      # excitability constant (from paper) uA/mm^2
      K = 675
      
      xgrid = self.grid['dva'].x.ravel()
      ygrid = self.grid['dva'].y.ravel()
      n_space = len(xgrid)
      n_time = stim.data.shape[1]

      # brightness array
      # holds (n_space) x (n_time)
      bright = np.zeros((n_space,n_time), dtype=np.float32)

      for space_idx in range(n_space):
         if np.isnan(xgrid[space_idx]) or np.isnan(ygrid[space_idx]):
            continue
         px_bright = 0.0
         for time_idx in range(n_time):
            for el_idx in range(stim.data.shape[0]):
               amp = stim.data[el_idx, time_idx]
               if np.abs(amp) > 0:
                  if separate and not ((x_el[el_idx] < boundary) == 
                           (xgrid[space_idx] < boundary)):
                     continue
                  D = 2 * np.sqrt(amp / K) # mm
                  P = (D / M[el_idx]) # dva
                  dist2 = np.power(xgrid[space_idx] - phosphene_locations['v1'][0][el_idx],2) + \
                          np.power(ygrid[space_idx] - phosphene_locations['v1'][1][el_idx],2)
                  sigma = P / 2
                  gauss = np.exp(-dist2 / (2 * sigma ** 2))
                  px_bright += amp * gauss
            # print('brightness at (', xgrid[space_idx], ',' , ygrid[space_idx],'):', px_bright)
            bright[space_idx, time_idx] = px_bright

      return np.asarray(bright)
