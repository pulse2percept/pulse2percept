from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 5
_version_micro = 0  # use '' for first of series, number for 1 and above
_version_extra = ''
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "pulse2percept: A Python-based simulation framework for bionic vision"
# Long description will go up on the pypi page
long_description = """

pulse2percept: A Python-based simulation framework for bionic vision
--------------------------------------------------------------------

By 2020 roughly 200 million people will suffer from retinal diseases such as
macular degeneration or retinitis pigmentosa. Consequently, a variety of
retinal sight restoration procedures are being developed to target these
diseases. Electronic prostheses (currently being implanted in patients)
directly stimulate remaining retinal cells using electrical current, analogous
to a cochlear implant. Optogenetic prostheses (soon to be implanted in human)
use optogenetic proteins to make remaining retinal cells responsive to light,
then use light diodes (natural illumination is inadequate) implanted in the
eye to stimulate these light sensitive cells.

However, these devices do not restore anything resembling natural vision:
Interactions between the electronics and the underlying neurophysiology result
in significant distortions of the perceptual experience.

We have developed a computer model that has the goal of predicting the
perceptual experience of retinal prosthesis patients. The model was developed
using a variety of patient data describing the brightness and shape of
phosphenes elicited by stimulating single electrodes, and validated against an
independent set of behavioral measures examining spatiotemporal interactions
across multiple electrodes.

More information can be found in
`Beyeler et al. (2017) <https://doi.org/10.25080/shinma-7f4c6e7-00c>`_
and in our
`Github repo <https://github.com/uwescience/pulse2percept>`_.

"""

NAME = "pulse2percept"
MAINTAINER = "Michael Beyeler, Ariel Rokem"
MAINTAINER_EMAIL = "mbeyeler@uw.edu, arokem@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/uwescience/pulse2percept"
DOWNLOAD_URL = ""
LICENSE = "BSD"
AUTHOR = "Michael Beyeler, Ariel Rokem"
AUTHOR_EMAIL = "mbeyeler@uw.edu, arokem@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {}
REQUIRES = ["numpy", "scipy", "joblib", "scikit_image", "sk_video", "cython"]
