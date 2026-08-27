.. _getting-started:

===============
Getting Started
===============

Hello world
-----------

A pulse2percept simulation usually has three parts:

1. Choose a visual prosthesis.
2. Bind a model to it.
3. Predict the percept a stimulus produces.

For example:

.. code-block:: python

    import pulse2percept as p2p
    from pulse2percept.units import Hz, ms, uA

    implant = p2p.implants.ArgusII()
    model = p2p.models.AxonMapModel(implant=implant)

    stim = {'A5': p2p.stimuli.BiphasicPulseTrain(
        freq=20 * Hz,
        amp=50 * uA,
        phase_dur=0.45 * ms,
    )}
    percept = model.predict_percept(stim)

    percept.plot()

The implant describes where stimulation is delivered, the stimulus describes
what is presented, and the model predicts the resulting response. In short::

    source → implant → delivered stimulation → model → percept

The middle step is the implant's: it samples, encodes, schedules and calibrates
whatever it is given into the current its electrodes actually deliver. A model
does that for you, but you can ask for it directly:

.. code-block:: python

    delivered = implant.prepare_stim(stim)
    delivered.plot()

Physical quantities can be written explicitly with :mod:pulse2percept.units; 
bare numbers are also accepted in the documented canonical units.

Images and Videos
-----------------

Images and videos must first be translated into electrical stimulation. The
easiest way is to give the implant an encoder:

.. code-block:: python

    implant = p2p.implants.ArgusII(
        encoder=p2p.stimuli.AmplitudeEncoder(
            amp_range=(0, 50 * uA),
            freq=20 * Hz,
        )
    )
    model = p2p.models.AxonMapModel(implant=implant)

    percept = model.predict_percept(p2p.stimuli.BostonTrain())
    percept.play()

Building
--------

Models perform expensive one-time setup before they can predict anything.
That happens by itself:

.. code-block:: python

    model = p2p.models.AxonMapModel(implant=implant)

    # Builds automatically:
    percept = model.predict_percept(stim)

    # Changing a model parameter invalidates the existing build:
    model.rho = 250

    # Rebuilds automatically:
    percept = model.predict_percept(stim)

Call ``model.build()`` yourself when you would rather pay that cost at a moment
of your choosing -- before a timed loop, say -- or want to inspect the built
grid or axon map.

From here, the :ref:`basic concepts <topics-index>` explain each part of the
pipeline in more detail, and the :ref:`examples <sphx_glr_examples>` show
complete simulations.
