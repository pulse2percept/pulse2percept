.. _examples-vision:

Visual scenes and residual vision
=================================

The :mod:`pulse2percept.vision` module describes the visual world an implanted
eye is looking at. A :py:class:`~pulse2percept.vision.Scene` is one monocular
visual field: what is present in it, and where native vision is lost
(:py:class:`~pulse2percept.vision.Scotoma`). A
:py:class:`~pulse2percept.vision.BinocularScene` holds one such field per eye,
side by side and uncombined. A model registers a scene against an implant and
returns what the person sees; see :ref:`topics-models-scene`.
