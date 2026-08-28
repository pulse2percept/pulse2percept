# -*- coding: utf-8 -*-
"""
============================================================================
Simulating PRIMA in age-related macular degeneration
============================================================================

Someone with geographic atrophy has lost vision in the center of their visual
field but still sees normally around it. A subretinal implant such as
:py:class:`~pulse2percept.implants.PRIMAPivotal` sits inside that blind
spot and gives back a coarse, grayscale percept there.

Four objects say everything there is to say about that situation:

*  :py:class:`~pulse2percept.vision.Scene` -- what is visually present, and
   where native vision is lost.
*  :py:class:`~pulse2percept.implants.ProsthesisSystem` -- the device: where
   its electrodes are, and how it turns gray levels into stimulation.
*  :py:class:`~pulse2percept.models.Model` -- the retinotopy, which is what
   connects the two.
*  :py:class:`~pulse2percept.percepts.Percept` -- what the simulated observer
   sees.

None of the coordinate bookkeeping between them is yours to do.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

from pulse2percept.implants import PRIMAPivotal
from pulse2percept.models import ScoreboardModel
from pulse2percept.stimuli import AmplitudeEncoder, LogoBVL
from pulse2percept.units import dva
from pulse2percept.vision import Scene, Scotoma

###############################################################################

scotoma = Scotoma.circle(6 * dva)

scene = Scene(LogoBVL(resize=(240, 300)), fov=40 * dva, scotoma=scotoma)

scene.plot(gaze=(0, 0) * dva)
plt.title('Native vision alone')

###############################################################################

implant = PRIMAPivotal()
implant.encoder = AmplitudeEncoder(amp_range=(0, 40), freq=20)

###############################################################################

model = ScoreboardModel(implant=implant, rho=50, xrange=(-6, 6),
                        yrange=(-6, 6), step=0.05)

###############################################################################
# The scene is trial input, like any other stimulus:

percept = model.predict_percept(scene, gaze=(0, 0) * dva, vmax=40)

percept.plot()
plt.title('Native vision with a PRIMA percept in the scotoma')

###############################################################################

percept = model.predict_percept(scene, gaze=(8, -4) * dva, vmax=40)

percept.plot()
plt.title('Looking 8 degrees right and 4 degrees down')

###############################################################################

implant.preprocess = lambda stim: stim.filter('sobel')

percept = model.predict_percept(scene, gaze=(0, 0) * dva, vmax=40)

percept.plot()
plt.title('Edge-filtered device input, intact vision around it')

###############################################################################
