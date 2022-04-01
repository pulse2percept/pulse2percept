#! /usr/bin/env python
# License: 3-clause BSD

import sys
import os
import platform
import shutil
import tempfile
from distutils.command.build_py import build_py
from distutils.command.sdist import sdist
from distutils.command.clean import clean as Clean
from distutils.errors import CompileError, LinkError
from pkg_resources import parse_version
import traceback

# Get version and release info, which is all stored in pulse2percept/info.py
ver_file = os.path.join('pulse2percept', 'version.py')
with open(ver_file) as f:
    exec(f.read())

VERSION = __version__

NUMPY_MIN_VERSION = '1.9.0'
SCIPY_MIN_VERSION = '1.0'
CYTHON_MIN_VERSION = '0.28'

DISTNAME = 'pulse2percept'
DESCRIPTION = 'A Python-based simulation framework for bionic vision'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Michael Beyeler, Ariel Rokem'
MAINTAINER_EMAIL = 'mbeyeler@ucsb.edu, arokem@gmail.com'
URL = 'https://github.com/pulse2percept/pulse2percept'
DOWNLOAD_URL = 'https://pypi.org/project/pulse2percept/#files'
LICENSE = 'new BDS'
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/pulse2percept/pulse2percept/issues',
    'Documentation': 'https://pulse2percept.github.io/pulse2percept',
    'Source Code': 'https://github.com/pulse2percept/pulse2percept'
}
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: C',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               ('Programming Language :: Python :: '
                'Implementation :: CPython'),
               ('Programming Language :: Python :: '
                'Implementation :: PyPy')
               ]

# Optional setuptools features
# We need to import setuptools early, if we want setuptools features,
# as it monkey-patches the 'setup' function
# For some commands, use setuptools
SETUPTOOLS_COMMANDS = {
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed',
}
if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools

    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        extras_require={
            'alldeps': (
                f'numpy >= {NUMPY_MIN_VERSION}',
                f'scipy >= {SCIPY_MIN_VERSION}'
            ),
        },
    )
else:
    extra_setuptools_args = dict()


class CleanCommand(Clean):
    """Custom clean command to remove build artifacts"""
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('pulse2percept'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


def openmp_build_ext():
    """Add support for OpenMP"""
    from numpy.distutils.command.build_ext import build_ext

    code = """#include <omp.h>
    int main(int argc, char** argv) { return(0); }"""

    class ConditionalOpenMP(build_ext):

        def can_compile_link(self, compile_flags, link_flags):

            cc = self.compiler
            fname = 'test.c'
            cwd = os.getcwd()
            tmpdir = tempfile.mkdtemp()

            try:
                os.chdir(tmpdir)
                with open(fname, 'wt') as fobj:
                    fobj.write(code)
                try:
                    objects = cc.compile([fname],
                                         extra_postargs=compile_flags)
                except CompileError:
                    return False
                try:
                    # Link shared lib rather then executable to avoid
                    # http://bugs.python.org/issue4431 with MSVC 10+
                    cc.link_shared_lib(objects, "testlib",
                                       extra_postargs=link_flags)
                except (LinkError, TypeError):
                    return False
            finally:
                os.chdir(cwd)
                shutil.rmtree(tmpdir)
            return True

        def build_extensions(self):
            """ Hook into extension building to check compiler flags """

            compile_flags = link_flags = []
            if self.compiler.compiler_type == 'msvc':
                compile_flags += ['/openmp']
                link_flags += ['/openmp']
            else:
                compile_flags += ['-fopenmp']
                link_flags += ['-fopenmp']

            if self.can_compile_link(compile_flags, link_flags):
                for ext in self.extensions:
                    ext.extra_compile_args += compile_flags
                    ext.extra_link_args += link_flags

            build_ext.build_extensions(self)

    return ConditionalOpenMP


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('pulse2percept')

    return config


def get_numpy_status():
    """
    Returns a dictionary containing a boolean specifying whether NumPy
    is up-to-date, along with the version string (empty string if
    not installed).
    """
    numpy_status = {}
    try:
        import numpy
        numpy_version = numpy.__version__
        numpy_status['up_to_date'] = parse_version(
            numpy_version) >= parse_version(NUMPY_MIN_VERSION)
        numpy_status['version'] = numpy_version
    except ImportError:
        traceback.print_exc()
        numpy_status['up_to_date'] = False
        numpy_status['version'] = ""
    return numpy_status


def get_cython_status():
    """
    Returns a dictionary containing a boolean specifying whether Cython is
    up-to-date, along with the version string (empty string if not installed).
    """
    cython_status = {}
    try:
        import Cython
        from Cython.Build import cythonize
        cython_version = Cython.__version__
        cython_status['up_to_date'] = parse_version(
            cython_version) >= parse_version(CYTHON_MIN_VERSION)
        cython_status['version'] = cython_version
    except ImportError:
        traceback.print_exc()
        cython_status['up_to_date'] = False
        cython_status['version'] = ""
    return cython_status


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    project_urls=PROJECT_URLS,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    classifiers=CLASSIFIERS,
                    cmdclass={
                        'clean': CleanCommand,
                        'build_py': build_py,
                        'build_ext': openmp_build_ext(),
                        'sdist': sdist
                    },
                    python_requires=">=3.7",
                    install_requires=[
                        f'numpy>={NUMPY_MIN_VERSION}',
                        f'scipy>={SCIPY_MIN_VERSION}',
                    ],
                    **extra_setuptools_args)

    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg_info',
                                                    '--version',
                                                    'clean'))):
        # For these actions, NumPy is not required
        #
        # They are required to succeed without NumPy for example when
        # pip is used to install pulse2percept when NumPy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION
    else:
        if sys.version_info < (3, 6):
            raise RuntimeError(
                f"pulse2percept requires Python 3.6 or later. The current"
                f" Python version is {platform.python_version()} installed in {sys.executable}.")
        instructions = ("Installation instructions are available on GitHub: "
                        "http://github.com/pulse2percept/pulse2percept\n")

        # Make sure NumPy is installed:
        numpy_status = get_numpy_status()
        numpy_req_str = f"pulse2percept erquires NumPy >= {NUMPY_MIN_VERSION}\n"
        if numpy_status['up_to_date'] is False:
            if numpy_status['version']:
                raise ImportError(f"Your installation of Numerical Python "
                                  f"(NumPy) {numpy_status['version']} is "
                                  f"out-of-date.\n{numpy_req_str}{instructions}")
            else:
                raise ImportError(f"Numerical Python (NumPy) is not "
                                  f"installed.\n{numpy_req_str}{instructions}")
        from numpy.distutils.core import setup

        # Make sure Cython is installed:
        cython_status = get_cython_status()
        cython_req_str = f"pulse2percept requires Cython >= {CYTHON_MIN_VERSION}.\n"
        if cython_status['up_to_date'] is False:
            if cython_status['version']:
                raise ImportError(f"Your installation of C-Extensions for "
                                  f"Python (Cython) {cython_status['version']} "
                                  f"is out-of-date.\n{cython_req_str}{instructions}")
            else:
                raise ImportError(f"C-Extensions for Python (Cython) is not "
                                  f"installed.\n{cython_req_str}{instructions}")
        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
