.. _examples-vision:

Residual vision
===============

The :mod:`pulse2percept.vision` module describes the visual world an implanted
eye is looking at: what is present (:py:class:`~pulse2percept.vision.Scene`)
and where native vision is lost
(:py:class:`~pulse2percept.vision.Scotoma`). A model registers a scene against
an implant and returns what the person sees; see :ref:`topics-models-scene`.
