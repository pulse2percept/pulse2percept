from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
import os


class OpenMPBuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            if sys.platform == "darwin":  # macOS
                ext.extra_compile_args += ["-Xclang", "-fopenmp"]
                ext.extra_link_args += ["-lomp"]
            elif os.name == "posix":  # Linux
                ext.extra_compile_args += ["-fopenmp"]
                ext.extra_link_args += ["-fopenmp"]
            elif os.name == "nt":  # Windows
                ext.extra_compile_args += ["/openmp"]
            else:
                raise RuntimeError("Unsupported platform")
        super().build_extensions()


def find_pyx_modules(base_dir):
    """
    Recursively find all `.pyx` files in subdirectories of `base_dir`.
    """
    extensions = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pyx"):
                module_path = os.path.relpath(os.path.join(root, file), base_dir)
                module_name = module_path.replace(os.path.sep, ".").replace(".pyx", "")
                extensions.append(
                    Extension(
                        module_name,
                        [os.path.join(root, file)],
                        include_dirs=[numpy.get_include()],
                    )
                )
    return extensions


# Find all .pyx files in the relevant submodules
cython_extensions = find_pyx_modules("pulse2percept")

setup(
    ext_modules=cythonize(
        cython_extensions,
        compiler_directives={
            "language_level": 3,       # Use Python 3 syntax
            "boundscheck": False,      # Disable bounds checking for arrays
            "wraparound": False,       # Disable negative indexing
            "cdivision": True,         # Optimize division operations
            "initializedcheck": False  # Skip uninitialized variable checks
        },
    ),
    cmdclass={"build_ext": OpenMPBuildExt},
)
