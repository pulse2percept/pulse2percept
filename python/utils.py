"""
Utility functions for pulse2percept
"""


class Parameters(object):
    def __init__(self, **params):
        for k, v in params.items():
            self.__dict__[k] = v
