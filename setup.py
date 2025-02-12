import os
import sys
import platform
import shutil
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
    """Check if the current platform and Python version are supported."""
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

def check_windows_build_tools():
    """Ensure Windows users have Microsoft Build Tools installed."""
    # Allow GitHub Actions to pass, since it has MSVC but may not expose `cl.exe`
    if os.name == "nt":
        if os.getenv("GITHUB_ACTIONS") == "true":
            print("Running in GitHub Actions, assuming Build Tools are installed.")
            return

        if not shutil.which("cl"):
            sys.stderr.write(
                "ERROR: Microsoft Build Tools for Visual Studio are required to build Cython extensions on Windows.\n"
                "Please install them from https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
                "Alternatively, use a pre-built wheel if available.\n"
            )
            sys.exit(1)

class OpenMPBuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            # Check if the extension has C++ sources
            is_cpp = any(file.endswith(".cpp") or file.endswith(".cxx") for file in ext.sources)

            if sys.platform == "darwin":  # macOS
                if is_cpp:
                    ext.extra_compile_args += ["-std=c++11", "-stdlib=libc++"]
                    ext.extra_link_args += ["-stdlib=libc++"]
                omp_include = os.popen("brew --prefix libomp").read().strip()
                if omp_include:
                    ext.extra_compile_args += ["-Xclang", "-fopenmp", "-I" + omp_include + "/include"]
                    ext.extra_link_args += ["-L" + omp_include + "/lib", "-lomp"]
                else:
                    print("Warning: OpenMP is not installed. Compiling without OpenMP support.")
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
    """Recursively find all `.pyx` files for Cython compilation."""
    if exclude_dirs is None:
        exclude_dirs = ["doc", "wheelhouse"]
    
    extensions = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]  # Exclude directories
        
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

check_windows_build_tools()

# Find all .pyx files in the relevant submodules
cython_extensions = find_pyx_modules("pulse2percept")

for ext in cython_extensions:
    ext.define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

setup(
    ext_modules=cythonize(
        cython_extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
    cmdclass={"build_ext": OpenMPBuildExt},
)
