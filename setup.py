import os
import sys
import platform
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy

# Define supported configurations
SUPPORTED_PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]
SUPPORTED_PLATFORMS = ["Linux", "Windows", "Darwin"]
UNSUPPORTED_CONFIGS = [
    {"os": "Darwin", "python_version": "3.9"}  # macOS + Python 3.9
]


def is_supported():
    current_os = platform.system()
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current_os not in SUPPORTED_PLATFORMS:
        return False, f"{current_os} is not a supported platform."
    if current_python not in SUPPORTED_PYTHON_VERSIONS:
        return False, f"Python {current_python} is not supported."
    for config in UNSUPPORTED_CONFIGS:
        if current_os == config["os"] and current_python == config["python_version"]:
            return False, f"Python {current_python} is not supported on {current_os}."
    return True, None

class OpenMPBuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            if sys.platform == "darwin":  # macOS
                # Fetch CPPFLAGS and LDFLAGS, providing defaults to avoid errors
                cppflags = os.getenv("CPPFLAGS", "")
                ldflags = os.getenv("LDFLAGS", "")
                
                if cppflags:
                    ext.extra_compile_args += ["-Xclang", "-fopenmp", "-I" + cppflags]
                else:
                    print("Warning: CPPFLAGS environment variable is not set.")

                if ldflags:
                    ext.extra_link_args += ["-lomp", "-L" + ldflags]
                else:
                    print("Warning: LDFLAGS environment variable is not set.")
            elif os.name == "posix":  # Linux
                try:
                    ext.extra_compile_args += ["-fopenmp"]
                    ext.extra_link_args += ["-fopenmp"]
                except RuntimeError:
                    print("Warning: OpenMP not supported on this platform. Compiling without OpenMP.")
            elif os.name == "nt":  # Windows
                ext.extra_compile_args += ["/openmp"]
                ext.extra_link_args += ["vcomp.lib"]
            else:
                print("Warning: OpenMP not supported on this platform. Compiling without OpenMP.")
        super().build_extensions()

def find_pyx_modules(base_dir, exclude_dirs=None):
    """
    Recursively find all `.pyx` files in subdirectories of `base_dir`, excluding certain directories.
    """
    if exclude_dirs is None:
        exclude_dirs = ["doc", "wheelhouse"]
    extensions = []
    for root, dirs, files in os.walk(base_dir):
        # Exclude specific directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".pyx"):
                module_path = os.path.relpath(os.path.join(root, file), base_dir)
                module_name = module_path.replace(os.path.sep, ".").replace(".pyx", "")
                module_name = f"pulse2percept.{module_name}"  # Ensure full module path
                extensions.append(
                    Extension(
                        module_name,
                        [os.path.join(root, file)],
                        include_dirs=[numpy.get_include()],
                    )
                )
    return extensions

# Run pre-build checks
is_supported, reason = is_supported()
if not is_supported:
    print(f"WARNING: {reason}\n"
          "Installation will proceed, but this configuration is not officially supported. "
          "Use at your own risk!")

# Find all .pyx files in the relevant submodules
cython_extensions = find_pyx_modules("pulse2percept")

for ext in cython_extensions:
    ext.define_macros = [("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")]
    # Ensure only files needing C++ are compiled as C++:
    if "_fast_array.pyx" in ext.sources[0]:  # This has been an issue
        ext.language = "c"
    elif any(file.endswith(".cpp") or file.endswith(".cxx") or file.endswith(".pyx") for file in ext.sources):
        # Force C++ compilation if the file requires it
        ext.language = "c++"


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
