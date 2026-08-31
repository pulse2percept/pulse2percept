.. _users-faq:

==========================
Frequently Asked Questions
==========================

New to pulse2percept? This page gives a quick overview of the main concepts,
objects, and modeling choices. For complete examples, see the
:doc:`Example Gallery <../examples/index>`.

.. contents:: On this page
   :local:
   :depth: 1

.. note::

    Don't see your question here? Please `open an issue`_ on GitHub and ask
    your question there.

.. _open an issue: https://github.com/pulse2percept/pulse2percept/issues


Getting started
===============

What can I do with pulse2percept?
---------------------------------

pulse2percept is a simulation framework for visual neuroprostheses.

You can use it to describe an implant and its electrical stimulation, predict
the resulting neural response or visual percept using computational models,
explore how model and implant parameters affect those predictions, and develop
encoding strategies that turn images or videos into electrical stimulation.

A typical workflow looks roughly like this::

    image / video
         |
         |  optional StimulusEncoder (usually the implant's own)
         v
      Stimulus
         |
         v
       Implant
         |
         v
       Model
         |
         v
      Percept

.. note::

    You do not need every stage for every simulation. If you already know
    which electrodes you want to stimulate and with what current, for example,
    you can construct the stimulus directly and do not need an encoder.


What is the simplest simulation I can run?
-------------------------------------------

The :py:class:`~pulse2percept.models.ScoreboardModel` is a good place to start
for retinal stimulation. It assumes that each stimulated electrode produces a
localized blob of light.

For example, stimulate electrode A8 of an Argus II implant with 30 microamps
and predict the resulting percept:

.. code-block:: python

    from pulse2percept.implants import ArgusII
    from pulse2percept.models import ScoreboardModel

    model = ScoreboardModel(implant=ArgusII())

    percept = model.predict_percept({'A8': 30})
    percept.plot()

Here, ``30`` means 30 microamps because that is the documented unit for
electrical stimulation. You can also write the unit explicitly:

.. code-block:: python

    from pulse2percept.units import uA

    percept = model.predict_percept({'A8': 30 * uA})

Or, as a one-liner:

.. code-block:: python

    ScoreboardModel(implant=ArgusII()).predict_percept({'A8': 30}).plot()

.. important::

    The Scoreboard model is the simplest retinal spatial model, not
    automatically the best model for your scientific question. Use it to learn
    the workflow or when its assumptions are appropriate. If you are modeling
    cortical stimulation, start with a model from
    :py:mod:`pulse2percept.models.cortex` instead.


How do I look at a predicted percept?
-------------------------------------

:py:meth:`~pulse2percept.percepts.Percept.plot` displays a static percept:

.. code-block:: python

    percept.plot()

For a purely spatial percept, this displays the predicted brightness across the
visual field. For a spatiotemporal percept, ``plot()`` displays the brightest
frame.

To view how a percept evolves over time, use
:py:meth:`~pulse2percept.percepts.Percept.play`:

.. code-block:: python

    percept.play()

``play()`` creates an interactive animation in IPython or Jupyter.

A :py:class:`~pulse2percept.percepts.Percept` is also a data object. Its
``data``, spatial coordinates, and time axis can be accessed directly if you
want to perform your own analysis or visualization.


Which model should I use?
-------------------------

Start with two questions.

**1. Are you modeling retinal or cortical stimulation?**

Retinal and cortical prostheses stimulate different parts of the visual system
and require different models. Choose a model designed for the site of
stimulation you are studying. Retinal models live in
:py:mod:`pulse2percept.models`; cortical models live in
:py:mod:`pulse2percept.models.cortex`.

**2. Do you need to model space, time, or both?**

A **spatial model** predicts where stimulation produces activity or perceived
brightness and what the resulting spatial pattern looks like.

A **temporal model** predicts how the response evolves over time.

A **spatiotemporal model** does both.

If you only care about the spatial pattern produced by a stimulation pattern,
a spatial model may be sufficient. If your question depends on pulse
frequency, pulse timing, fading, persistence, or other dynamics, you need a
temporal component as well.

Once you have made those choices, consider which biological mechanisms the
model needs to capture. For retinal stimulation, for example:

* :py:class:`~pulse2percept.models.ScoreboardModel` assumes localized,
  electrode-centered activation and provides a useful simple baseline.
* :py:class:`~pulse2percept.models.AxonMapModel` additionally models activation
  of retinal ganglion cell axons and can therefore predict elongated
  phosphenes.
* Other retinal models capture temporal or spatiotemporal effects of
  electrical stimulation.

.. important::

    Choose the simplest model that captures the phenomenon relevant to your
    question. A more complicated model is not automatically a more accurate
    one: every additional mechanism introduces assumptions and parameters that
    must themselves be justified.

See :ref:`Computational Models <topics-models>` for the available models and
the publications on which they are based.


Can I just use the default model parameters?
--------------------------------------------

For learning the API or reproducing an example, yes.

For scientific conclusions, model parameters should be treated as scientific
assumptions, not generic software settings.

For example, the Axon Map model has two particularly important parameters:

``rho``
    Controls how quickly sensitivity falls off with distance away from an
    axon. Larger values generally produce wider phosphenes.

``lam``
    Controls how quickly sensitivity falls off along the axon. Larger values
    generally produce longer axonal streaks.

The current software defaults for
:py:class:`~pulse2percept.models.AxonMapModel` are ``rho=200`` microns and
``lam=500`` microns. These values make the model usable out of the box; they
should not be interpreted as universal values for every implant user.
In fact, every real prosthesis user has their onw ``rho`` and ``lam`` values
that best represent how "streaky" or "blobby" their vision appears
[Beyeler2019]_.

Likewise, the :py:class:`~pulse2percept.models.ScoreboardModel` has a ``rho``
parameter controlling the spatial extent of its electrode-centered blobs. Its
meaning is related to ``rho`` in the Axon Map model, but the two models make
different assumptions about how activation spreads through the retina.

If your conclusions depend on phosphene size, elongation, or another
model-dependent property, fit the relevant parameters to data when possible,
use published values appropriate to your population, or explicitly justify
the values you chose.

Other parameters can matter just as much. For example, optic-disc location and
retinotopic mapping affect the nerve fiber trajectories used by the Axon Map
model, while temporal models introduce their own parameters governing the
dynamics of the predicted response.

.. warning::

    Default parameters are defaults of the *software implementation*. They are
    not a substitute for subject-specific calibration or scientific
    justification.


How do I set model parameters?
------------------------------

Model parameters can usually be passed when the model is created:

.. code-block:: python

    from pulse2percept.implants import ArgusII
    from pulse2percept.models import AxonMapModel
    from pulse2percept.units import um

    model = AxonMapModel(ArgusII(), rho=250 * um, lam=700 * um)
    model.build()

Bare numbers also work and retain their documented units:

.. code-block:: python

    model = AxonMapModel(ArgusII(), rho=250, lam=700)

Many models perform expensive precomputations in ``build()``. If you change a
parameter that affects those computations after the model has been built, call
``build()`` again before predicting another percept.


How do I change the spatial resolution or field of view?
--------------------------------------------------------

Spatial models evaluate the predicted response on a grid in visual-field
coordinates. Three parameters control that grid:

``xrange``
    Horizontal extent of the simulated visual field, in degrees of visual
    angle.

``yrange``
    Vertical extent of the simulated visual field, in degrees of visual angle.

``step``
    Spacing between neighboring simulation points. Smaller values produce a
    finer spatial grid, but require more computation and memory.

For example:

.. code-block:: python

    from pulse2percept.implants import ArgusII
    from pulse2percept.models import AxonMapModel

    model = AxonMapModel(
        ArgusII(),
        xrange=(-15, 15),
        yrange=(-10, 10),
        step=0.25,
    )
    model.build()

This simulates a 30 x 20 degree region of visual space on a grid sampled every
0.25 degrees.

.. important::

    ``step`` controls the **numerical resolution of the simulation**. It
    does not change the physical resolution of the implant, the number of
    electrodes, or the biological size of a phosphene.

Likewise, ``xrange`` and ``yrange`` determine which part of visual space is
computed; they do not change the implant itself.

A smaller ``step`` can make plots look smoother and improve numerical
sampling, but it cannot add biological detail that is absent from the model.
Halving ``step`` in both dimensions also produces roughly four times as many
grid points, so unnecessarily fine grids can become expensive.

Because the simulation grid is created during ``build()``, call ``build()``
again after changing ``xrange``, ``yrange``, or ``step``.


Core concepts
=============

What are the main objects in pulse2percept?
-------------------------------------------

Most pulse2percept simulations involve four objects, plus an optional encoder:

:py:class:`~pulse2percept.stimuli.Stimulus`
    Describes the input supplied to the implant. For electrical stimulation,
    this specifies which electrodes are active, with what amplitudes, and
    optionally how stimulation changes over time.

:py:class:`~pulse2percept.implants.Implant`
    Describes the prosthetic device, including its electrode array, placement,
    and input/encoding pipeline. Specific devices such as
    :py:class:`~pulse2percept.implants.ArgusII` are subclasses of this object.

:py:class:`~pulse2percept.models.Model`
    A forward model that predicts a neural response or visual percept from the
    stimulated implant.

:py:class:`~pulse2percept.percepts.Percept`
    The predicted visual percept, represented across visual space and,
    optionally, time.

:py:class:`~pulse2percept.stimuli.StimulusEncoder`
    An optional step that converts higher-level input such as an image or video
    into the electrical stimulus delivered by an implant.


What is the difference between a stimulus and a percept?
--------------------------------------------------------

A **stimulus** describes what is delivered to the prosthesis.

A **percept** describes what a computational model predicts will result from
that stimulation.

For example, ``30 microamps on electrode A8`` is a stimulus. A small bright
phosphene at a particular location in the visual field is a percept predicted
from that stimulus.

.. note::

    Changing the model can change the predicted percept without changing the
    stimulus at all. The stimulus is an input to the forward model; the
    percept is its prediction.


Do I need a StimulusEncoder?
----------------------------

Only if you are assigning image or video content to an implant that does not
already have one.

A :py:class:`~pulse2percept.stimuli.StimulusEncoder` translates image or video
content into electrical stimulation. For example,
:py:class:`~pulse2percept.stimuli.AmplitudeEncoder` maps image intensity onto
pulse amplitude, whereas
:py:class:`~pulse2percept.stimuli.FrequencyEncoder` maps it onto pulse
frequency.

Devices whose video processing is known carry an encoder of their own, in
which case handing the model an image encodes it for you.
:py:class:`~pulse2percept.implants.ArgusII` is one of them. Assign a different
encoder to ``implant.encoder`` to say how the encoding should be done, or
``None`` to switch it off.

If you already know the electrical stimulation you want to simulate, construct
a :py:class:`~pulse2percept.stimuli.Stimulus` directly. An electrical stimulus
never goes through the encoder.

.. important::

    An **encoder** asks: *What stimulation should the device deliver?*

    A **model** asks: *What response or percept will that stimulation produce?*


Can I combine spatial and temporal models?
------------------------------------------

Yes. A :py:class:`~pulse2percept.models.Model` can contain a spatial component,
a temporal component, or both.

For example:

.. code-block:: python

    from pulse2percept.implants import ArgusII
    from pulse2percept.models import Model, ScoreboardSpatial, FadingTemporal

    model = Model(
        spatial=ScoreboardSpatial(ArgusII()),
        temporal=FadingTemporal(),
    )
    model.build()

The class names reflect this distinction:

* Classes ending in ``Model`` are stand-alone models that provide the usual
  high-level interface.
* Classes ending in ``Spatial`` implement a spatial model component.
* Classes ending in ``Temporal`` implement a temporal model component.

This lets you mix and match compatible components when the scientific question
calls for a combination that is not already provided as a stand-alone model.

See :ref:`Computational Models <topics-models>` for details.


What does ``build()`` do?
-------------------------

Some models need to perform expensive calculations that depend on their
parameters but do not need to be repeated for every stimulus.

Calling ``build()`` performs these calculations once. For example, the Axon Map
model can precompute retinal nerve fiber trajectories and their relationship
to the simulation grid.

You rarely have to call it yourself: ``predict_percept`` builds a model that
is not built yet, and giving a parameter a new value un-builds the model, so
the next prediction picks the change up.

.. code-block:: python

    model = SomeModel(implant=implant, ...)

    percept1 = model.predict_percept(stim1)
    model.rho = 200          # un-builds the model
    percept2 = model.predict_percept(stim2)   # builds it again

Call ``build()`` explicitly when you want to pay that cost at a moment of your
choosing, or to pass build-time parameters: ``model.build(rho=200)``.


Implants and stimulation
========================

How do I choose an implant?
---------------------------

The :py:mod:`pulse2percept.implants` module contains implementations of several
existing retinal and cortical prostheses.

Choose one of these when you want to simulate a particular device. If your
electrode layout does not correspond to an existing implant, you can construct
your own :py:class:`~pulse2percept.implants.ElectrodeArray` and
:py:class:`~pulse2percept.implants.Implant`.

The implant matters because electrode size, spacing, location, and orientation
can all affect the predicted response.

See :ref:`Basic Concepts: Implants <topics-implants>` for details.


How do I control which electrodes are stimulated?
-------------------------------------------------

For simple static stimulation, name the electrodes and their amplitudes:

.. code-block:: python

    percept = model.predict_percept({
        'A8': 30,
        'A9': 20,
    })

Only the listed electrodes are active.

For time-varying electrical stimulation, use objects such as
:py:class:`~pulse2percept.stimuli.BiphasicPulse` and
:py:class:`~pulse2percept.stimuli.BiphasicPulseTrain`, or construct a
:py:class:`~pulse2percept.stimuli.Stimulus` directly.

See :ref:`Electrical Stimuli <topics-stimuli>` for details.


Can I simulate images and videos?
---------------------------------

Yes, but it helps to distinguish **visual input** from **electrical
stimulation**.

:py:class:`~pulse2percept.stimuli.ImageStimulus` and
:py:class:`~pulse2percept.stimuli.VideoStimulus` can represent image and video
content. A :py:class:`~pulse2percept.stimuli.StimulusEncoder` can then sample
that content at the electrode locations and convert the resulting intensities
into electrical pulse trains.

For example:

.. code-block:: python

    from pulse2percept.implants import ArgusII
    from pulse2percept.models import ScoreboardModel
    from pulse2percept.stimuli import BostonTrain

    model = ScoreboardModel(implant=ArgusII())
    percept = model.predict_percept(BostonTrain())

:py:class:`~pulse2percept.implants.ArgusII` comes with an encoder of its own,
so the video is encoded for you. To say how, give the implant a different one:

.. code-block:: python

    from pulse2percept.stimuli import AmplitudeEncoder
    from pulse2percept.units import uA, Hz

    implant = ArgusII(encoder=AmplitudeEncoder(amp_range=(0, 50 * uA),
                                               freq=20 * Hz))

Either way ``implant.prepare_stim(BostonTrain())`` is the electrical
stimulation the device would deliver, which is also what the bound model
reads.

.. note::

    An image or video is not itself what a visual prosthesis delivers. The
    encoder defines how visual information is converted into a pattern of
    electrical stimulation.


Coordinates, units, and interpretation
======================================

How are retinal/cortical coordinates mapped to visual field coordinates?
------------------------------------------------------------------------

Retinal/cortical location and perceived visual-field location are not the
same coordinate system.

For example, stimulation of the inferior retina produces a percept in the
upper visual field, while stimulation of the superior retina produces a
percept in the lower visual field.
A disproportionate amount of cortical surface is dedicated to encoding
the fovea.

pulse2percept uses
:py:class:`~pulse2percept.topography.VisualFieldMap` objects to convert between
retinal or cortical coordinates and visual-field coordinates.

For retinal models, available mappings include
:py:class:`~pulse2percept.topography.Curcio1990Map`,
:py:class:`~pulse2percept.topography.Watson2014Map`, and
:py:class:`~pulse2percept.topography.Watson2014DisplaceMap`.

You can also implement your own
:py:class:`~pulse2percept.topography.VisualFieldMap`.

.. warning::

    Implant coordinates in microns or millimeters and percept coordinates in
    degrees of visual angle are different physical quantities. Do not treat
    them as interchangeable coordinate conventions.


What units does pulse2percept use?
----------------------------------

pulse2percept accepts both bare numbers and unitful values.

Bare numbers keep their documented historical meaning. Common conventions are:

==============================  =========================
Quantity                        Bare number means
==============================  =========================
Electrical current              microamps (uA)
Stimulus and percept time       milliseconds (ms)
Electrode/tissue geometry       microns (um)
Visual-field coordinates        degrees of visual angle
Frequency                       hertz (Hz)
Image/video intensity           dimensionless
Implant rotation                degrees
==============================  =========================

You can make units explicit:

.. code-block:: python

    from pulse2percept.units import mA, ms, um

    amplitude = 0.05 * mA
    phase_dur = 0.45 * ms
    rho = 200 * um

Compatible quantities are automatically converted to the units expected by the
API. For example, ``50 * uA`` and ``0.05 * mA`` describe the same current.

.. note::

    Explicit units are especially helpful when mixing electrode geometry,
    visual-field coordinates, or code from different experimental
    conventions. Bare numbers remain supported for backwards compatibility.

See :py:mod:`pulse2percept.units` for details.


What does brightness in a predicted Percept mean?
-------------------------------------------------

The values in :py:attr:`pulse2percept.percepts.Percept.data` are predicted
perceived brightness in **arbitrary units**. They are not physical luminance
values.

Their interpretation depends on the computational model that produced them.
The important question is therefore not only "what is the brightness value?"
but also "what quantity does this particular model predict, and how was it
calibrated?"

.. warning::

    Do not assume that numerical brightness values from different models are
    directly comparable unless the models explicitly define them that way.


Does a predicted percept show exactly what an implant user would see?
---------------------------------------------------------------------

No.

A predicted percept is the output of a computational model. It is conditional
on the model's assumptions, its parameters, the implant geometry, and the
stimulation supplied to it.

Some models have been fit and validated against measurements from visual
prosthesis users, but substantial differences can exist across subjects and
electrodes. Subject-specific model parameters can therefore be important when
the goal is to predict an individual user's percepts.

.. important::

    Simulated percepts are useful for testing hypotheses, comparing stimulation
    strategies, and studying the consequences of a model. They should not be
    interpreted as literal ground truth about what every implant user sees.


Troubleshooting and next steps
==============================

Where should I go after my first simulation?
--------------------------------------------

A useful progression is:

#. Try a simple :py:class:`~pulse2percept.models.ScoreboardModel` simulation
   and inspect the result with ``plot()``.
#. Change the stimulated electrode or current and see what changes.
#. Compare the same stimulation under
   :py:class:`~pulse2percept.models.ScoreboardModel` and
   :py:class:`~pulse2percept.models.AxonMapModel`.
#. Change ``rho`` and ``lam`` deliberately and inspect their effects.
#. Change ``step`` and the simulated field of view so you understand the
   difference between numerical sampling and model behavior.
#. Move to a temporal or spatiotemporal model if your scientific question
   depends on stimulation dynamics.
#. Use an encoder when you are ready to turn images or videos into pulse
   trains.

The :doc:`Example Gallery <../examples/index>` contains complete examples for
implants, stimuli, models, and encoding strategies.


The code I installed does not match the documentation. What gives?
-------------------------------------------------------------------

Make sure you are reading the documentation for the version of pulse2percept
that you installed.

* If you installed a release :ref:`with pip <install-release>`, use the
  `stable documentation`_.
* If you installed pulse2percept from source, you may be using newer,
  unreleased functionality. Use the `latest documentation`_ or the
  documentation corresponding to your branch.

.. _stable documentation: https://pulse2percept.readthedocs.io/en/stable/
.. _latest documentation: https://pulse2percept.readthedocs.io/en/latest/


I think I found a bug. What should I do?
----------------------------------------

Please `open an issue`_ on GitHub with a minimal example that reproduces the
problem, your pulse2percept version, and the full error message.

If you would like to contribute a fix or a new feature, see the
:ref:`Contribution Guidelines <dev-contributing-workflow>`.
