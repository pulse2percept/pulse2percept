from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
import os
import platform


# OpenMP flag detection
def get_openmp_flags(compiler):
    if os.name == "nt":  # Windows
        return ["/openmp"], ["/openmp"]
    return ["-fopenmp"], ["-fopenmp"]

class OpenMPBuildExt(build_ext):
    """Custom build extension to add OpenMP flags if supported."""
    def build_extensions(self):
        compile_flags, link_flags = get_openmp_flags(self.compiler)
        for ext in self.extensions:
            ext.extra_compile_args += compile_flags
            ext.extra_link_args += link_flags
        super().build_extensions()


# Dynamically discover all Cython extensions
def discover_cython_extensions():
    """Recursively find all .pyx files and create Extension objects."""
    extensions = []
    submodules = ["models", "stimuli", "utils"]  # Submodules with Cython
    for submodule in submodules:
        submodule_dir = os.path.join("pulse2percept", submodule)
        for root, _, files in os.walk(submodule_dir):
            for file in files:
                if file.endswith(".pyx"):
                    module_path = os.path.join(root, file).replace(os.path.sep, ".")[:-4]  # Remove ".pyx"
                    extensions.append(
                        Extension(
                            module_path,
                            sources=[os.path.join(root, file)],
                            include_dirs=[numpy.get_include()],
                            libraries=["m"] if os.name == "posix" else [],
                        )
                    )
    return extensions


# Collect all extensions
extensions = cythonize(
    discover_cython_extensions(),
    compiler_directives={
        "language_level": 3,  # Use Python 3 runtime
        "boundscheck": False,  # Disable bounds checking
        "wraparound": False,  # Disable negative indexing checks
        "initializedcheck": False,  # Disable uninitialized variable checks
    },
)

setup(
    name="pulse2percept",
    version="0.9.0.dev0",
    description="A Python-based simulation framework for bionic vision",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    author="Michael Beyeler, Jacob Granley, Apurv Varshney, Ariel Rokem",
    author_email="mbeyeler@ucsb.edu",
    url="https://github.com/pulse2percept/pulse2percept",
    license="BSD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    zip_safe=False,
    ext_modules=extensions,
    cmdclass={"build_ext": OpenMPBuildExt},
    python_requires=">=3.7",
    install_requires=[
        "cython>=0.28",
        "numpy>=1.11,<1.27",
        "setuptools>=42",
        "scipy<=1.7.1; python_version < '3.10'",
        "scipy>=1.0.1; python_version >= '3.10'",
        "scikit-image>=0.14",
        "matplotlib>=3.0.2",
        "imageio-ffmpeg>=0.4",
        "pandas",
        "joblib>=0.11",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
            "tox",
        ]
    },
)
