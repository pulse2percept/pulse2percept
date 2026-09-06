""":py:class:`~pulse2percept.topography.retina.RetinalMap`"""
from abc import abstractmethod

from ..base import VisualFieldMap


class RetinalMap(VisualFieldMap):
    """ Template class for retinal visual field maps, which only have 1 region."""
    split_map = False
    regions = ['ret']
    def __init__(self, **params):
        super().__init__(**params)

    def from_dva(self):
        return {'ret' : self.dva_to_ret}
    
    def to_dva(self):
        return {'ret' : self.ret_to_dva}
    
    @abstractmethod
    def dva_to_ret(self, x, y):
        """Convert degrees of visual angle (dva) to retinal coords (um)"""
        raise NotImplementedError
        
    def ret_to_dva(self, x, y):
        """Convert retinal coords (um) to degrees of visual angle (dva)"""
        raise NotImplementedError
