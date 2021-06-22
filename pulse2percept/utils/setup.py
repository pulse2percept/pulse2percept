import os
import platform


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('utils', parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    if platform.python_implementation() != 'PyPy':
        config.add_extension('_fast_math',
                             sources=['_fast_math.pyx'],
                             include_dirs=[numpy.get_include()],
                             libraries=libraries),
        config.add_extension('_array',
                             sources=['_array.pyx'],
                             include_dirs=[numpy.get_include()],
                             libraries=libraries)
    config.add_subpackage('tests')

    return config
