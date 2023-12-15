"""
===============================================================================
Cortical implant gallery
===============================================================================

pulse2percept supports the following cortical implants:

Orion Prosthesis System (Cortigent Inc.)
----------------------------------------
:py:class:`~pulse2percept.implants.cortex.Orion` contains 60 electrodes in a hex shaped grid inspired by Argus II.
"""

import matplotlib.pyplot as plt
from pulse2percept.implants.cortex import *

orion = Orion()
orion.plot(annotate=True)
plt.show()

###############################################################################
# Cortivis Prosthesis System (Biomedical Technologies)
# ----------------------------------------------------
#
# :py:class:`~pulse2percept.implants.cortex.Cortivis` is an implant with 96 
# electrodes in a square shaped grid.

cortivis = Cortivis()
cortivis.plot(annotate=True)
plt.show()

###############################################################################
# ICVP Prosthesis System (Sigenics Inc.)
# --------------------------------------
#
# :py:class:`~pulse2percept.implants.cortex.ICVP` is an implant with 16 
# primary electrodes in a hex shaped grid, along with 2 additional "reference" 
# and "counter" electrodes.

icvp = ICVP()
icvp.plot(annotate=True)
plt.show()