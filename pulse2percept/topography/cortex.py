"""
`CorticalMap`
"""
import numpy as np
from abc import abstractmethod

from .base import VisualFieldMap


class CorticalMap(VisualFieldMap):
    """Template class for V1/V2/V3 visuotopic maps"""
    allowed_regions = {'v1', 'v2', 'v3'}

    def __init__(self, regions=['v1']):
        if not isinstance(regions, list):
            regions = list(regions)
        for region in regions:
            if region.lower() not in self.allowed_regions:
                raise ValueError(f"Specified region {region} not supported."\
                                 f" Options are {self.allowed_layers}")
        self.regions = [r.lower() for r in regions]

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
