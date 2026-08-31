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

The implant describes the device, the stimulus describes the input, and the
model predicts the resulting percept::

    source → implant → delivered stimulation → model → percept

The implant converts a source into delivered stimulation by preprocessing,
encoding, scheduling, and threshold calibration. Models call this step
internally; use ``prepare_stim`` directly to inspect the delivered stimulus:

.. code-block:: python

    delivered = implant.prepare_stim(stim)
    delivered.plot()

Physical quantities can be written explicitly with :mod:`pulse2percept.units`; 
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

Models build automatically on first prediction. Changing a parameter
invalidates the affected build state, which is rebuilt on the next prediction:

.. code-block:: python

    model = p2p.models.AxonMapModel(implant=implant)

    # Builds automatically:
    percept = model.predict_percept(stim)

    # Rebuilds automatically after a parameter change:
    model.spatial.rho = 250
    percept = model.predict_percept(stim)

Call ``model.build()`` explicitly to build ahead of a timed loop or inspect
derived state such as the grid or axon map.

From here, the :ref:`basic concepts <topics-index>` explain each part of the
pipeline in more detail, and the :ref:`examples <sphx_glr_examples>` show
complete simulations.
