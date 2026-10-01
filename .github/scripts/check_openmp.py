#!/usr/bin/env python3
"""Check that pulse2percept and torch share one OpenMP runtime (macOS).

torch bundles libomp. A second libomp in the same process aborts at the
first parallel region ("OMP: Error #15"), so pulse2percept links torch's copy
(see setup.py). Checks:

1. The extensions were built with OpenMP, i.e. link ``@rpath/libomp.dylib``.
2. An OpenMP kernel runs after torch has used its own thread pool.
3. Exactly one libomp is loaded, and it is torch's.
"""

import ctypes
import os
import subprocess
import sys

import numpy as np
import torch

import pulse2percept.models._temporal as ext
from pulse2percept.models import FadingTemporal
from pulse2percept.stimuli import Stimulus


def fail(msg):
    sys.exit(f"check_openmp: {msg}")


deps = subprocess.run(["otool", "-L", ext.__file__], capture_output=True,
                      text=True, check=True).stdout
if "@rpath/libomp.dylib" not in deps:
    fail(f"{ext.__file__} does not link @rpath/libomp.dylib; was it built "
         f"without OpenMP?\n{deps}")

# Initialize torch's OpenMP first, then enter a pulse2percept parallel region:
torch.ones(512, 512) @ torch.ones(512, 512)
stim = Stimulus(-np.ones((256, 3)), time=[0, 1, 2])
FadingTemporal(n_threads=4).predict_percept(stim, t_percept=[1, 2])

libc = ctypes.CDLL(None)
libc._dyld_get_image_name.restype = ctypes.c_char_p
images = [libc._dyld_get_image_name(i).decode()
          for i in range(libc._dyld_image_count())]
omp = {os.path.realpath(p) for p in images
       if os.path.basename(p).startswith("libomp")}
torch_lib = os.path.realpath(os.path.join(os.path.dirname(torch.__file__),
                                          "lib"))
if len(omp) != 1 or os.path.dirname(omp.pop()) != torch_lib:
    fail(f"expected only torch's libomp in {torch_lib}, found {sorted(omp)}")
print("check_openmp: one OpenMP runtime (torch's)")
