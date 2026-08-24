.. _getting-started:

===============
Getting Started
===============

Hello world
-----------

A pulse2percept simulation usually has three parts:

1. Choose a visual prosthesis.
2. Assign it a stimulus.
3. Use a model to predict the resulting percept.

For example:

.. code-block:: python

    import pulse2percept as p2p
    from pulse2percept.units import Hz, ms, uA

    implant = p2p.implants.ArgusII(
        stim={'A5': p2p.stimuli.BiphasicPulseTrain(
            freq=20 * Hz,
            amp=50 * uA,
            phase_dur=0.45 * ms,
        )}
    )

    model = p2p.models.AxonMapModel().build()
    percept = model.predict_percept(implant)

    percept.plot()

The implant describes where stimulation is delivered, the stimulus describes
what is delivered, and the model predicts the resulting response.

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
    implant.stim = p2p.stimuli.BostonTrain()

    percept = model.predict_percept(implant)
    percept.play()

From here, the :ref:`basic concepts <topics-index>` explain each part of the
pipeline in more detail, and the :ref:`examples <sphx_glr_examples>` show
complete simulations.