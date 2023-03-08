"""
`CorticalMap`
"""
import numpy as np
from abc import abstractmethod

from .base import VisualFieldMap
from ..utils import pol2cart, cart2pol


class CorticalMap(VisualFieldMap):
    """Template class for V1/V2/V3 visuotopic maps"""
    allowed_regions = {'v1', 'v2', 'v3'}
    split_map = True

    def __init__(self, **params):
        super(CorticalMap, self).__init__(**params)
        if not isinstance(self.regions, list):
            self.regions = list(self.regions)
        for region in self.regions:
            if region.lower() not in self.allowed_regions:
                raise ValueError(f"Specified region {region} not supported."\
                                 f" Options are {self.allowed_regions}")
        self.regions = [r.lower() for r in self.regions]

    def from_dva(self):
        mappings = dict()
        if 'v1' in self.regions:
            mappings['v1'] = self.dva_to_v1
        if 'v2' in self.regions:
            mappings['v2'] = self.dva_to_v2
        if 'v3' in self.regions:
            mappings['v3'] = self.dva_to_v3
        return mappings
    
    def to_dva(self):
        mappings = dict()
        if 'v1' in self.regions:
            mappings['v1'] = self.v1_to_dva
        if 'v2' in self.regions:
            mappings['v2'] = self.v2_to_dva
        if 'v3' in self.regions:
            mappings['v3'] = self.v3_to_dva
        return mappings
    
    def get_default_params(self):
        params = {
            'regions' : ['v1']
        }
        return params

    @abstractmethod
    def dva_to_v1(self, x, y):
        """Convert degrees visual angle (dva) to V1 coordinates (um)"""
        raise NotImplementedError

    @abstractmethod
    def dva_to_v2(self, x, y):
        """Convert degrees visual angle (dva) to V2 coordinates (um)"""
        raise NotImplementedError

    @abstractmethod
    def dva_to_v3(self, x, y):
        """Convert degrees visual angle (dva) to V3 coordinates (um)"""
        raise NotImplementedError

    def v1_to_dva(self, x, y):
        """Convert V1 coordinates (um) to degrees visual angle (dva)"""
        raise NotImplementedError

    def v2_to_dva(self, x, y):
        """Convert V2 coordinates (um) to degrees visual angle (dva)"""
        raise NotImplementedError

    def v3_to_dva(self, x, y):
        """Convert V3 coordinates (um) to degrees visual angle (dva)"""
        raise NotImplementedError


class Polimeni2006Map(CorticalMap):
    """Polimeni visual mapping"""
    def __init__(self, **params):
        super().__init__(**params)

    def get_default_params(self):
        base_params = super(Polimeni2006Map, self).get_default_params()
        params = {
            'k' : 15,
            'a' : 0.5,
            'b' : 90,
            'alpha1' : 1,
            'alpha2' : 0.333,
            'alpha3' : 0.25,
            'jitter_boundary' : False
        }
        return {**base_params, **params}

    def _invert_left_pol(self, theta, radius, inverted = None):
        """
        'Corrects' the mapping by flipping x axis if necessary, allowing for both
         left and right hemisphere to use the same map.
        """

        # Check if we're reverting from an existing inversion
        if inverted is None:
            inverted = (theta > (np.pi / 2)) | (theta < - (np.pi / 2))

        # Invert theta across y axis
        theta = np.where(inverted, np.pi - theta, theta)
        theta = np.where(theta > np.pi, theta - 2*np.pi, theta)
        theta = np.where(theta <= - np.pi, theta + 2*np.pi, theta)
        
        # Invert theta across x axis
        theta = -theta
        return theta, radius, inverted
    
    def _invert_left_cart(self, x, y, inverted = None):
        """
        'Corrects' the mapping by flipping x axis if neccesary, allowing for both
         left and right hemisphere to use the same map.
        """
        # Check if we're reverting from an existing inversion
        if inverted is None:
            inverted = x < 0

        # Invert across y axis
        x = np.where(inverted, -x, x)
        return x, y, inverted

    def add_nans(self, x, y, theta, radius, allow_zero=True):
        idx_nan = ((theta <= -np.pi/2) | (theta >= np.pi/2) | (radius < 0) |
                        (radius > 90) | (x < 0) | (x > 180))
        if not allow_zero:
            idx_nan = idx_nan | (theta == 0)
        else:
            idx_nan = idx_nan | ((theta == 0) & (radius == 0))
        x[idx_nan], y[idx_nan] = np.nan, np.nan
        return x, y

    def dva_to_v1(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            y = np.array(y)
        if self.jitter_boundary:
            # remove and discontinuities across x axis
            # shift to the same side as existing points
            x[x==0] += np.sign(np.mean(x)) * 1e-3 
        theta, radius = cart2pol(x, y)
        theta, radius, inverted = self._invert_left_pol(theta, radius)
        thetaV1 = self.alpha1 * theta
        zV1 = radius * np.exp(1j * thetaV1)
        wV1 = (self.k * np.log((zV1 + self.a) / (zV1 + self.b)) -
               self.k * np.log(self.a/self.b))
        xV1, yV1 = np.real(wV1), np.imag(wV1)
        xV1, yV1 = self.add_nans(xV1, yV1, theta, radius)
        xV1 *= 1000
        yV1 *= 1000
        return self._invert_left_cart(xV1, yV1, ~inverted)[:2]

    def dva_to_v2(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            y = np.array(y)
        if self.jitter_boundary:
            # remove and discontinuities across x and y axis
            # shift to the same side as existing points
            x[x==0] += np.sign(np.mean(x)) * 1e-3 
            y[y==0] += np.sign(np.mean(y)) * 1e-3 
        theta, radius = cart2pol(x, y)
        theta, radius, inverted = self._invert_left_pol(theta, radius)
        phi1 = np.pi / 2 * (1 - self.alpha1)
        phi2 = np.pi / 2 * (1 - self.alpha2)
        thetaV2 = self.alpha2 * theta + np.sign(theta) * (phi2 + phi1)
        zV2 = -np.conj(radius * np.exp(1j * thetaV2))
        wV2 = (self.k * np.log((zV2 + self.a) / (zV2 + self.b)) -
               self.k * np.log(self.a/self.b))
        xV2, yV2 = np.real(wV2), np.imag(wV2)
        xV2, yV2 = self.add_nans(xV2, yV2, theta, radius, allow_zero=False)
        xV2 *= 1000
        yV2 *= 1000
        return self._invert_left_cart(xV2, yV2, ~inverted)[:2]

    def dva_to_v3(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            y = np.array(y)
        if self.jitter_boundary:
            # remove and discontinuities across x and y axis
            # shift to the same side as existing points
            x[x==0] += np.sign(np.mean(x)) * 1e-3 
            y[y==0] += np.sign(np.mean(y)) * 1e-3 
        theta, radius = cart2pol(x, y)
        theta, radius, inverted = self._invert_left_pol(theta, radius)
        phi1 = np.pi / 2 * (1 - self.alpha1)
        phi2 = np.pi / 2 * (1 - self.alpha2)
        thetaV3 = self.alpha3 * theta + np.sign(theta) * (np.pi - phi1 - phi2)
        zV3 = radius * np.exp(1j * thetaV3)
        wV3 = (self.k * np.log((zV3 + self.a) / (zV3 + self.b)) -
               self.k * np.log(self.a/self.b))
        xV3, yV3 = np.real(wV3), np.imag(wV3)
        xV3, yV3 = self.add_nans(xV3, yV3, theta, radius, allow_zero=False)
        xV3 *= 1000
        yV3 *= 1000
        return self._invert_left_cart(xV3, yV3, ~inverted)[:2]

    def v1_to_dva(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            y = np.array(y)
        x, y, inverted = self._invert_left_cart(x, y)
        x /= 1000
        y /= 1000
        w = x + y*1j
        z = (self.a - self.a * np.exp(w/self.k)) / (self.a/self.b * np.exp(w/self.k) - 1)
        t1 = np.real(z)
        t2 = np.imag(z)
        r = np.sqrt(t1**2 + t2**2)
        thetav1 = np.arctan2(t2, t1)
        theta = thetav1 / self.alpha1
        return pol2cart(*self._invert_left_pol(theta, r, ~inverted)[:2])

    def v2_to_dva(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            y = np.array(y)
        x, y, inverted = self._invert_left_cart(x, y)
        x /= 1000
        y /= 1000
        w = x + y * 1j
        z = (self.a - self.a*np.exp(w / self.k)) / (self.a/self.b * np.exp(w/self.k) - 1)
        re = np.real(z)
        im = np.imag(z)
        r = np.sqrt(re**2 + im**2)
        thetav2 = np.arctan2(-im,re)
        thetav2 += np.sign(y)*np.pi
        phi1 = np.pi / 2 * (1 - self.alpha1)
        phi2 = np.pi / 2 * (1 - self.alpha2)
        theta = (thetav2 - (np.sign(y) * (phi1 + phi2))) / self.alpha2
        return pol2cart(*self._invert_left_pol(theta, r, ~inverted)[:2])

    def v3_to_dva(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(x, np.ndarray):
            y = np.array(y)
        x, y, inverted = self._invert_left_cart(x,y)
        x /= 1000
        y /= 1000
        w = x + y * 1j
        z = (self.a - self.a * np.exp(w/self.k)) / (self.a/self.b * np.exp(w/self.k) - 1)
        re, im = np.real(z), np.imag(z)
        r = np.sqrt(re**2 + im**2)
        thetav3 = np.arctan2(im,re)
        phi1 = np.pi / 2 * (1 - self.alpha1)
        phi2 = np.pi / 2 * (1 - self.alpha2)
        thetav3 -= np.sign(y) * (np.pi - phi1 - phi2)
        theta = thetav3 / self.alpha3
        return pol2cart(*self._invert_left_pol(theta, r, ~inverted)[:2])