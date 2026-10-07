#!/usr/bin/env python3
"""Check that pulse2percept and torch share one OpenMP runtime (macOS).

torch bundles libomp. A second libomp in the same process aborts at the
first parallel region ("OMP: Error #15"), so pulse2percept links torch's copy
(see setup.py). Checks:

1. Static: extensions reference libomp only as ``@rpath/libomp.dylib`` (no
   Homebrew or ``/opt/llvm-openmp`` path), the OpenMP extensions reference
   it, and nothing else non-system is linked (the wheel repair ignores
   unresolved dependencies).
2. Runtime: after torch has used its OpenMP runtime, a pulse2percept OpenMP
   kernel runs, and exactly one libomp is loaded: torch's.

Prints the parsed load commands and the loaded libomp as evidence. Runs the
runtime check even if a static check failed.
"""

import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import pulse2percept
from pulse2percept.models.retina import Nanduri2012Temporal
from pulse2percept.stimuli import Stimulus

OPENMP = "@rpath/libomp.dylib"
SYSTEM = ("/usr/lib/", "/System/Library/")
# Extensions with `prange` loops, which must link OpenMP:
OPENMP_EXTS = ("_nanduri2012.",)

# As in PyTorch's tools/embed_libomp_macos.py:
_LOAD_RE = re.compile(r"(?:name|path) (.+) \(offset \d+\)")


def _load_commands(path, cmd):
    """Return the names or paths of ``cmd`` load commands, per ``otool -l``."""
    out = subprocess.run(["otool", "-l", str(path)], capture_output=True,
                         text=True, check=True).stdout.splitlines()
    found = []
    for i, line in enumerate(out):
        if line.strip() == f"cmd {cmd}":
            match = _LOAD_RE.match(out[i + 2].strip())
            if match:
                found.append(match.group(1))
    return found


def dependencies(path):
    """Return LC_LOAD_DYLIB install names from a Mach-O binary."""
    return set(_load_commands(path, "LC_LOAD_DYLIB"))


problems = []

# ---- Static linkage ----
exts = sorted(Path(pulse2percept.__file__).parent.rglob("*.so"))
if not exts:
    problems.append("no compiled extensions found")
for path in exts:
    deps = dependencies(path)
    print(f"{path.name}: LC_LOAD_DYLIB {sorted(deps)}")
    for dep in sorted(deps):
        if "libomp" in dep and dep != OPENMP:
            problems.append(f"{path.name} links {dep}, not {OPENMP}")
        elif dep != OPENMP and not dep.startswith(SYSTEM):
            problems.append(f"{path.name} links non-system library {dep}")
    if path.name.startswith(OPENMP_EXTS):
        print(f"{path.name}: LC_RPATH {_load_commands(path, 'LC_RPATH')}")
        if OPENMP not in deps:
            problems.append(f"{path.name} does not link {OPENMP}; built "
                            f"without OpenMP?")

# ---- Runtime ----
# Initialize torch's OpenMP first, then enter a pulse2percept parallel region:
torch.ones(512, 512) @ torch.ones(512, 512)
stim = Stimulus(np.ones((256, 3)), time=[0, 1, 2])
Nanduri2012Temporal(n_threads=4).predict_percept(stim, t_percept=[1, 2])

libc = ctypes.CDLL(None)
libc._dyld_get_image_name.restype = ctypes.c_char_p
images = [libc._dyld_get_image_name(i).decode()
          for i in range(libc._dyld_image_count())]
# Count loaded images, not distinct files: one file loaded twice is still two
# runtimes.
omp = [p for p in images if os.path.basename(p).startswith("libomp")]
torch_omp = os.path.realpath(os.path.join(os.path.dirname(torch.__file__),
                                          "lib", "libomp.dylib"))
print(f"loaded libomp: {omp}")
print(f"torch libomp:  {torch_omp}")
if len(omp) != 1:
    problems.append(f"expected exactly one loaded libomp, found {len(omp)}")
elif os.path.realpath(omp[0]) != torch_omp:
    problems.append(f"loaded libomp {omp[0]} is not torch's {torch_omp}")

if problems:
    sys.exit("check_openmp: FAILED\n  " + "\n  ".join(problems))
print("check_openmp: one OpenMP runtime (torch's)")
