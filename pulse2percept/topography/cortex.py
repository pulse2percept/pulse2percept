""":py:class:`~pulse2percept.topography.CorticalMap`, 
   :py:class:`~pulse2percept.topography.Polimeni2006Map`
"""
import numpy as np
from abc import abstractmethod

from .base import VisualFieldMap
from ..utils import pol2cart, cart2pol
import matplotlib.pyplot as plt


class CorticalMap(VisualFieldMap):
    """Template class for V1/V2/V3 visuotopic maps"""
    allowed_regions = {'v1', 'v2', 'v3'}

    # All 2D cortical maps are split into 2 separate grids for hemispheres
    split_map = True

    def __init__(self, **params):
        super(CorticalMap, self).__init__(**params)
        if not isinstance(self.regions, list):
            self.regions = [self.regions]
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
            'regions' : ['v1'],
            # Offset for the left hemisphere fovea
            'left_offset' : -20000
        }
        return {**super().get_default_params(),**params}

    @abstractmethod
    def dva_to_v1(self, x, y):
        """Convert degrees visual angle (dva) to V1 coordinates (um)"""
        raise NotImplementedError

    def dva_to_v2(self, x, y):
        """Abstract Method: Convert degrees visual angle (dva) to V2 coordinates (um)"""
        raise NotImplementedError("Must implement dva_to_v2 when creating a map with region 'v2'")

    def dva_to_v3(self, x, y):
        """Abstract Method: Convert degrees visual angle (dva) to V3 coordinates (um)"""
        raise NotImplementedError("Must implement dva_to_v3 when creating a map with region 'v3'")

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
            'a' : 0.69, # average of values in sec 4.6
            'b' : 80,
            'alpha1' : 1,
            'alpha2' : 0.333,
            'alpha3' : 0.25,
            'jitter_boundary' : True,
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
    
    def _invert_left_cart(self, x, y, inverted = None, boundary=0):
        """
        'Corrects' the mapping by flipping x axis if neccesary, allowing for both
         left and right hemisphere to use the same map.
        """
        # Check if we're reverting from an existing inversion
        if inverted is None:
            inverted = x < boundary
            x = np.where(inverted, -x + self.left_offset, x)
            return x, y, inverted

        # Invert across y axis
        x = np.where(inverted, -x + self.left_offset, x)
        return x, y, inverted

    def add_nans(self, x, y, theta, radius, allow_zero=True):
        idx_nan = ((theta <= -np.pi/2) | (theta >= np.pi/2) | (radius < 0) |
                        (radius > 90) )
        # use isclose for better numerical stability
        if not allow_zero:
            idx_nan = idx_nan | np.isclose(theta, 0, atol=1e-6)
        else:
            idx_nan = idx_nan | (np.isclose(theta, 0, atol=1e-6) & (radius == 0))
        x[idx_nan], y[idx_nan] = np.nan, np.nan
        return x, y

    def dva_to_v1(self, x, y):
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        if self.jitter_boundary:
            # remove and discontinuities across x axis
            # shift to the same side as existing points
            x[np.isclose(x, 0, rtol=0, atol=1e-7)] += np.copysign(1e-3, np.mean(x)) 
        theta, radius = cart2pol(x, y)
        theta, radius, inverted = self._invert_left_pol(theta, radius)
        thetaV1 = self.alpha1 * theta
        zV1 = radius * np.exp(1j * thetaV1)
        wV1 = (self.k * np.log((zV1 + self.a) / (zV1 + self.b)) -
               self.k * np.log(self.a/self.b, dtype='float32'))
        xV1, yV1 = np.real(wV1), np.imag(wV1)
        xV1, yV1 = self.add_nans(xV1, yV1, theta, radius)
        xV1 *= 1000
        yV1 *= 1000
        return self._invert_left_cart(xV1, yV1, ~inverted)[:2]

    def dva_to_v2(self, x, y):
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        if self.jitter_boundary:
            # remove and discontinuities across x and y axis
            # shift to the same side as existing points
            x[np.isclose(x, 0, rtol=0, atol=1e-7)] += np.copysign(1e-3, np.mean(x)) 
            y[np.isclose(y, 0, rtol=0, atol=1e-7)] += np.copysign(1e-3, np.mean(y)) 
        theta, radius = cart2pol(x, y)
        theta, radius, inverted = self._invert_left_pol(theta, radius)
        phi1 = np.pi / 2 * (1 - self.alpha1)
        phi2 = np.pi / 2 * (1 - self.alpha2)
        thetaV2 = self.alpha2 * theta + np.sign(theta) * (phi2 + phi1)
        zV2 = -np.conj(radius * np.exp(1j * thetaV2))
        wV2 = (self.k * np.log((zV2 + self.a) / (zV2 + self.b)) -
               self.k * np.log(self.a/self.b, dtype='float32'))
        xV2, yV2 = np.real(wV2), np.imag(wV2)
        xV2, yV2 = self.add_nans(xV2, yV2, theta, radius, allow_zero=False)
        xV2 *= 1000
        yV2 *= 1000
        return self._invert_left_cart(xV2, yV2, ~inverted)[:2]

    def dva_to_v3(self, x, y):
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        if self.jitter_boundary:
            # remove and discontinuities across x and y axis
            # shift to the same side as existing points
            x[np.isclose(x, 0, rtol=0, atol=1e-7)] += np.copysign(1e-3, np.mean(x)) 
            y[np.isclose(y, 0, rtol=0, atol=1e-7)] += np.copysign(1e-3, np.mean(y)) 
        theta, radius = cart2pol(x, y)
        theta, radius, inverted = self._invert_left_pol(theta, radius)
        phi1 = np.pi / 2 * (1 - self.alpha1)
        phi2 = np.pi / 2 * (1 - self.alpha2)
        thetaV3 = self.alpha3 * theta + np.sign(theta) * (np.pi - phi1 - phi2)
        zV3 = radius * np.exp(1j * thetaV3)
        wV3 = (self.k * np.log((zV3 + self.a) / (zV3 + self.b)) -
               self.k * np.log(self.a/self.b, dtype='float32'))
        xV3, yV3 = np.real(wV3), np.imag(wV3)
        xV3, yV3 = self.add_nans(xV3, yV3, theta, radius, allow_zero=False)
        xV3 *= 1000
        yV3 *= 1000
        return self._invert_left_cart(xV3, yV3, ~inverted)[:2]

    def v1_to_dva(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x, y, inverted = self._invert_left_cart(x, y, boundary=self.left_offset/2)
        x /= 1000
        y /= 1000
        w = x + y*1j
        z = (self.a - self.a * np.exp(w/self.k)) / (self.a/self.b * np.exp(w/self.k) - 1)
        z = z.astype('complex64')
        t1 = np.real(z)
        t2 = np.imag(z)
        r = np.sqrt(t1**2 + t2**2)
        thetav1 = np.arctan2(t2, t1)
        theta = thetav1 / self.alpha1
        return pol2cart(*self._invert_left_pol(theta, r, ~inverted)[:2])

    def v2_to_dva(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x, y, inverted = self._invert_left_cart(x, y, boundary=self.left_offset/2)
        x /= 1000
        y /= 1000
        w = x + y * 1j
        z = (self.a - self.a*np.exp(w / self.k)) / (self.a/self.b * np.exp(w/self.k) - 1)
        z = z.astype('complex64')
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
        x = np.array(x)
        y = np.array(y)
        x, y, inverted = self._invert_left_cart(x, y, boundary=self.left_offset/2)
        x /= 1000
        y /= 1000
        w = x + y * 1j
        z = (self.a - self.a * np.exp(w/self.k)) / (self.a/self.b * np.exp(w/self.k) - 1)
        z = z.astype('complex64')
        re, im = np.real(z), np.imag(z)
        r = np.sqrt(re**2 + im**2)
        thetav3 = np.arctan2(im,re)
        phi1 = np.pi / 2 * (1 - self.alpha1)
        phi2 = np.pi / 2 * (1 - self.alpha2)
        thetav3 -= np.sign(y) * (np.pi - phi1 - phi2)
        theta = thetav3 / self.alpha3
        return pol2cart(*self._invert_left_pol(theta, r, ~inverted)[:2])
    
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        theta = np.pi + np.linspace(-np.pi/2+0.001, np.pi/2-0.001, num=102).reshape((1, -1))
        radius = np.array([0.5, 2.5, 5, 10, 20, 40, 80]).reshape((-1, 1))
        x, y = self.dva_to_v1(*pol2cart(theta, radius))
        for i in range(x.shape[0]):
            ax.plot(x[i, :], y[i, :], 'gray', label='v1' if i == 0 else None)
            rad = f"{radius[i, 0] : .1f}" if radius[i, 0] < 5 else f"{radius[i, 0] : .0f}"
            x_val = x[i, np.argsort(np.abs(y[i]))[0]]
            ax.annotate(f"{rad}$\degree$", (x_val + 2000, 500),
                         ha='center')
        x, y = self.dva_to_v2(*pol2cart(theta, radius))
        for i in range(x.shape[0]):
            ax.plot(x[i, (theta < np.pi).ravel()], y[i, (theta < np.pi).ravel()],
                    'blue', linewidth=1, label='v2' if i == 0 else None)
            ax.plot(x[i, (theta > np.pi).ravel()], y[i, (theta > np.pi).ravel()],
                    'blue', linewidth=1)
        x, y = self.dva_to_v3(*pol2cart(theta, radius))
        for i in range(x.shape[0]):
            ax.plot(x[i, (theta < np.pi).ravel()], y[i, (theta < np.pi).ravel()],
                    'red', linewidth=1, label='v3' if i == 0 else None)
            ax.plot(x[i, (theta > np.pi).ravel()], y[i, (theta > np.pi).ravel()],
                    'red', linewidth=1)


        theta = np.pi + np.linspace(-np.pi/2, np.pi/2, 5).reshape((-1, 1))
        radius = np.logspace(np.log10(1e-5), np.log10(80), num=50).reshape((1,
                                                                            -1))
        x, y = self.dva_to_v1(*pol2cart(theta, radius))
        for i in range(x.shape[0]):
            ax.plot(x[i, :], y[i, :], 'gray', linewidth=1)
        theta = np.array([-np.pi/2, np.pi-1e-5, np.pi+1e-5, np.pi/2]).reshape((-1, 1))
        x, y = self.dva_to_v2(*pol2cart(theta, radius))
        for i in range(x.shape[0]):
            ax.plot(x[i, :], y[i, :], 'blue', linewidth=1)
        x, y = self.dva_to_v3(*pol2cart(theta, radius))
        for i in range(x.shape[0]):
            ax.plot(x[i, :], y[i, :], 'red', linewidth=1)
        

        ax.set_xticklabels(np.array(ax.get_xticks()) / 1000)
        ax.set_yticklabels(np.array(ax.get_yticks()) / 1000)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend()