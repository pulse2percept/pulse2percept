import os
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as _np

# Supported matrix (purely informational warning)
SUPPORTED_PYTHON_VERSIONS = {"3.10", "3.11", "3.12", "3.13"}
SUPPORTED_PLATFORMS = {"Linux", "Windows", "Darwin"}
EXPLICITLY_UNSUPPORTED = set()  # e.g., {("Windows", "3.10")}


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


def _numpy_api_macro():
    """Pick the correct API macro for NumPy 1.x vs 2.x at *build* time."""
    major = int(_np.__version__.split(".")[0])
    if major >= 2:
        return ("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")
    # 1.7 is the canonical stable API for NumPy 1.x builds
    return ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")


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


class OpenMPBuildExt(build_ext):
    """Enable OpenMP when available; degrade gracefully otherwise."""

    def build_extensions(self):
        api_macro = _numpy_api_macro()
        disable_omp = os.environ.get("P2P_DISABLE_OPENMP", "0") == "1"

        for ext in self.extensions:
            ext.define_macros = list(getattr(ext, "define_macros", []))
            ext.extra_compile_args = list(getattr(ext, "extra_compile_args", []))
            ext.extra_link_args = list(getattr(ext, "extra_link_args", []))

            # Always add the NumPy API macro
            ext.define_macros.append(api_macro)

            if disable_omp:
                # Don’t add any OpenMP flags; useful on RTD
                continue

            try:
                if sys.platform == "darwin":
                    omp_prefix = os.environ.get("OMP_PREFIX")
                    if omp_prefix:
                        omp_inc = os.path.join(omp_prefix, "include")
                        omp_lib = os.path.join(omp_prefix, "lib")
                        if omp_inc and os.path.isdir(omp_inc):
                            ext.include_dirs.append(omp_inc)
                        ext.extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
                        if omp_lib and os.path.isdir(omp_lib):
                            ext.extra_link_args += [f"-L{omp_lib}"]
                        ext.extra_link_args += ["-lomp"]
                    else:
                        ext.extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
                        ext.extra_link_args += ["-lomp"]

                elif os.name == "posix":  # Linux
                    ext.extra_compile_args += ["-fopenmp"]
                    ext.extra_link_args += ["-fopenmp"]

                elif os.name == "nt":  # Windows (MSVC)
                    ext.extra_compile_args += ["/openmp"]

            except Exception as e:
                # Never fail just because OpenMP flags aren't available
                print(f"Warning: OpenMP flags not applied ({e}). Building without OpenMP.")

        super().build_extensions()


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
    cmdclass={"build_ext": OpenMPBuildExt},
)
