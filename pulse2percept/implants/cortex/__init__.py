"""Cortical implants.

Base class
----------

.. autosummary::
    :toctree:

    CorticalImplant

Devices
-------

.. autosummary::
    :toctree:

    Orion
    ICVP
    NeuroPortArray
    Neuralink

Neuralink components
--------------------

.. autosummary::
    :toctree:

    NeuralinkThread
    LinearEdgeThread
    EllipsoidElectrode

.. seealso::

    *  :ref:`Core Concepts > Implants > Cortical Implants
       <topics-implants-cortex>`

"""

from .base import CorticalImplant
from .orion import Orion
from .neuroport import NeuroPortArray
from .icvp import ICVP
from .neuralink import EllipsoidElectrode, NeuralinkThread, LinearEdgeThread, Neuralink

__all__ = [
    "CorticalImplant",
    "Orion",
    "NeuroPortArray",
    "ICVP",
    "EllipsoidElectrode",
    "NeuralinkThread",
    "LinearEdgeThread",
    "Neuralink"
]
