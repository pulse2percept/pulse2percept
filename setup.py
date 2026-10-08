import os
import sys
import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as _np

# Supported matrix (purely informational warning)
SUPPORTED_PYTHON_VERSIONS = {"3.11", "3.12", "3.13", "3.14"}
SUPPORTED_PLATFORMS = {"Linux", "Windows", "Darwin"}
EXPLICITLY_UNSUPPORTED = set()  # e.g., {("Windows", "3.11")}


def _is_supported():
    current_os = platform.system()
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current_os not in SUPPORTED_PLATFORMS:
        return False, f"{current_os} is not a supported platform."
    if current_python not in SUPPORTED_PYTHON_VERSIONS:
        return False, f"Python {current_python} is not supported."
    if (current_os, current_python) in EXPLICITLY_UNSUPPORTED:
        return False, f"Python {current_python} is explicitly not supported on {current_os}."
    return True, None


_ok, _reason = _is_supported()
if not _ok:
    print(
        f"WARNING: {_reason}\n"
        "Installation will proceed, but this configuration is not officially supported."
    )


# Only NumPy 2.x is supported for build and runtime:
NUMPY_API_MACRO = ("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")

if int(_np.__version__.split(".")[0]) < 2:
    # Only reachable with --no-build-isolation (pyproject.toml requires
    # numpy>=2.0). Fail here instead of with a missing-macro compiler error:
    raise RuntimeError(
        f"Building pulse2percept requires NumPy 2.0 or newer, found "
        f"{_np.__version__}. Upgrade NumPy, or drop --no-build-isolation and "
        f"let pip install the declared build requirements."
    )


def _find_pyx_modules(base_dir, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {"doc", "wheelhouse"}
    extensions = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fn in files:
            if not fn.endswith(".pyx"):
                continue
            rel = os.path.relpath(os.path.join(root, fn), base_dir)
            mod = rel.replace(os.path.sep, ".")[:-4]  # strip .pyx
            fullmod = f"pulse2percept.{mod}"
            ext = Extension(
                name=fullmod,
                sources=[os.path.join(root, fn)],
                include_dirs=[_np.get_include()],
                define_macros=[NUMPY_API_MACRO],
            )
            # Heuristic: default to C; flip to C++ only if .cpp/.cxx sources exist
            if any(s.endswith((".cpp", ".cxx")) for s in ext.sources):
                ext.language = "c++"
            else:
                ext.language = "c"

            # Known sensitive module: force C
            if fullmod.endswith("utils._fast_array"):
                ext.language = "c"

            extensions.append(ext)
    return extensions


extensions = _find_pyx_modules("pulse2percept")

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
        annotate=bool(os.environ.get("CYTHON_ANNOTATE", "")),
    ),
)
