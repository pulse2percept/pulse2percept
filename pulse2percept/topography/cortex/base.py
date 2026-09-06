""":py:class:`~pulse2percept.topography.cortex.CorticalMap`"""
from abc import abstractmethod

from ..base import VisualFieldMap
from ...units import um


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

    def get_param_units(self):
        """Return a dict of the units that parameters are stored in"""
        # Cortical coordinates are stored in microns, and the offset shifts
        # one hemisphere's x coordinates:
        return {**super().get_param_units(), 'left_offset': um}

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
