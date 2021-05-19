import os
import platform


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration('models', parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    if platform.python_implementation() != 'PyPy':
        config.add_extension('_temporal',
                             sources=['_temporal.pyx'],
                             include_dirs=[numpy.get_include()],
                             libraries=libraries)
        config.add_extension('_beyeler2019',
                             sources=['_beyeler2019.pyx'],
                             include_dirs=[numpy.get_include()],
                             libraries=libraries)
        config.add_extension('_horsager2009',
                             sources=['_horsager2009.pyx'],
                             include_dirs=[numpy.get_include()],
                             libraries=libraries)
        config.add_extension('_nanduri2012',
                             sources=['_nanduri2012.pyx'],
                             include_dirs=[numpy.get_include()],
                             libraries=libraries)
    config.add_subpackage("tests")

    return config
