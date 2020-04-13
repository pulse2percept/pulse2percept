import os
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    import numpy
    config = Configuration('datasets', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_data_dir('data')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())