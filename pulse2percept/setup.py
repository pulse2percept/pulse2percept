import os
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('pulse2percept', parent_package, top_path)

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    config.add_subpackage('implants')
    config.add_subpackage('implants/tests')
    config.add_subpackage('percepts')
    config.add_subpackage('percepts/tests')
    config.add_subpackage('datasets')
    config.add_subpackage('datasets/tests')
    config.add_subpackage('model_selection')
    config.add_subpackage('model_selection/tests')
    config.add_subpackage('viz')
    config.add_subpackage('viz/tests')

    # Submodules which have their own setup.py; e.g., because they use Cython:
    config.add_subpackage('models')
    config.add_subpackage('stimuli')
    config.add_subpackage('utils')

    # Data directories
    config.add_data_dir('datasets/data')
    config.add_data_dir('stimuli/data')
    config.add_data_dir('viz/data')

    # https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
    config.ext_modules = cythonize(config.ext_modules,
                                   compiler_directives={
                                       'language_level': 3,  # use Py3 runtime
                                       'boundscheck': False,  # no IndexError
                                       'wraparound': False,  # no arr[-1]
                                       'initializedcheck': False,  # no None
                                   })
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
