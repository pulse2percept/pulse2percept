.. _topics-implants:

=================
Visual Prostheses
=================

An implant in pulse2percept describes **where stimulation is delivered**:
the electrodes, their geometry, their location, and the stimulus assigned to
them. It fits into the modeling pipeline like so:

::

    electrical Stimulus -> implant -> model -> Percept

The implant says where the stimulation goes. The
:ref:`percept model <topics-models>` says how that stimulation is transformed
into a visual percept.

Choosing an implant and model
-----------------------------

The best model depends first on **where the implant stimulates**. A useful
starting point is:

.. list-table::
   :header-rows: 1

   * - Implant location
     - Good starting model
     - Why
   * - Epiretinal
     - :py:class:`~pulse2percept.models.AxonMapModel`
     - Epiretinal stimulation can activate retinal ganglion-cell axons,
       producing elongated percepts that follow nerve fiber bundles.
   * - Subretinal
     - :py:class:`~pulse2percept.models.ScoreboardModel`
     - A local "one electrode, one blob" model is a useful first approximation
       when axonal activation is not the main effect of interest.
   * - Suprachoroidal
     - :py:class:`~pulse2percept.models.ScoreboardModel`
     - pulse2percept does not currently provide a dedicated suprachoroidal
       phosphene model, so the scoreboard model is a simple geometry-first
       baseline.
   * - Cortical
     - :py:class:`~pulse2percept.models.cortex.ScoreboardModel`
     - Cortical stimulation is mapped through cortical retinotopy rather than
       the retinal nerve fiber layer.

These are **starting points, not compatibility rules**. For example,
:py:class:`~pulse2percept.models.ScoreboardModel` is also a useful baseline for
an epiretinal implant when axonal streaking is not part of the question. More
detailed retinal models should be chosen because their physiological
assumptions match the experiment, not simply because they are more complex.

A minimal example
-----------------

For an epiretinal implant such as Argus II, a typical simulation looks like:

.. code-block:: python

    import pulse2percept as p2p

    implant = p2p.implants.ArgusII()

    encoder = p2p.stimuli.AmplitudeEncoder(
        implant, amp_range=(0, 50), freq=20
    )
    implant.stim = encoder.encode(p2p.stimuli.BostonTrain())

    model = p2p.models.AxonMapModel().build()
    percept = model.predict_percept(implant)

    percept.play()

Changing the implant changes the electrode geometry. Changing the model changes
the assumptions about how stimulation becomes vision.

Available implants
------------------

pulse2percept includes software representations of several published visual
prostheses. The table below emphasizes **array geometry** and **which model to
start with**, rather than device manufacturer.

.. list-table::
   :header-rows: 1
   :widths: 18 22 18 42

   * - Implant
     - Array
     - Location
     - Suggested starting model

   * - :py:class:`~pulse2percept.implants.ArgusI`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.ArgusI().plot(annotate=False)
          plt.axis("off")

     - Epiretinal
     - :py:class:`~pulse2percept.models.AxonMapModel`

   * - :py:class:`~pulse2percept.implants.ArgusII`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.ArgusII().plot(annotate=False)
          plt.axis("off")

     - Epiretinal
     - :py:class:`~pulse2percept.models.AxonMapModel`

   * - :py:class:`~pulse2percept.implants.IMIE`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.IMIE().plot(annotate=False)
          plt.axis("off")

     - Epiretinal
     - :py:class:`~pulse2percept.models.AxonMapModel`

   * - :py:class:`~pulse2percept.implants.AlphaIMS`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.AlphaIMS().plot(annotate=False)
          plt.axis("off")

     - Subretinal
     - :py:class:`~pulse2percept.models.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.AlphaAMS`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.AlphaAMS().plot(annotate=False)
          plt.axis("off")

     - Subretinal
     - :py:class:`~pulse2percept.models.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.PRIMA`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.PRIMA().plot(annotate=False)
          plt.axis("off")

     - Subretinal
     - :py:class:`~pulse2percept.models.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.PRIMA75`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.PRIMA75().plot(annotate=False)
          plt.axis("off")

     - Subretinal
     - :py:class:`~pulse2percept.models.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.PRIMA55`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.PRIMA55().plot(annotate=False)
          plt.axis("off")

     - Subretinal
     - :py:class:`~pulse2percept.models.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.PRIMA40`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.PRIMA40().plot(annotate=False)
          plt.axis("off")

     - Subretinal
     - :py:class:`~pulse2percept.models.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.BVT24`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.BVT24().plot(annotate=False)
          plt.axis("off")

     - Suprachoroidal
     - :py:class:`~pulse2percept.models.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.BVT44`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.BVT44().plot(annotate=False)
          plt.axis("off")

     - Suprachoroidal
     - :py:class:`~pulse2percept.models.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.cortex.Orion`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.cortex.Orion().plot(annotate=False)
          plt.axis("off")

     - Cortical
     - :py:class:`~pulse2percept.models.cortex.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.cortex.Cortivis`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.cortex.Cortivis().plot(annotate=False)
          plt.axis("off")

     - Cortical
     - :py:class:`~pulse2percept.models.cortex.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.cortex.ICVP`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.cortex.ICVP().plot(annotate=False)
          plt.axis("off")

     - Cortical
     - :py:class:`~pulse2percept.models.cortex.ScoreboardModel`

   * - :py:class:`~pulse2percept.implants.cortex.Neuralink`
     -
       .. plot::
          :width: 110px
          :align: center
          :include-source: false
          :show-source-link: false
          :caption:

          import matplotlib.pyplot as plt
          import pulse2percept as p2p
          p2p.implants.cortex.Neuralink().plot(annotate=False)
          plt.axis("off")

     - Cortical
     - :py:class:`~pulse2percept.models.cortex.ScoreboardModel`

These classes are **research software representations based on published
descriptions**, not manufacturer-validated device simulators. Some geometries
necessarily rely on assumptions where complete device specifications are not
public; the API documentation for each class records those details.

What an implant contains
------------------------

Every visual prosthesis derives from
:py:class:`~pulse2percept.implants.ProsthesisSystem`. The pieces you will use
most often are:

``earray``
    The :py:class:`~pulse2percept.implants.ElectrodeArray` containing the
    electrodes and their locations.

``stim``
    The electrical :py:class:`~pulse2percept.stimuli.Stimulus` currently
    assigned to the implant.

``eye``
    The implanted eye for retinal systems.

``raster``
    An optional :py:class:`~pulse2percept.implants.Raster` describing which
    electrodes may stimulate at the same time.

Electrodes can be accessed by name or index:

.. code-block:: python

    implant = p2p.implants.ArgusII()

    implant['A1']
    implant[0]
    implant.electrode_names
    implant.earray.coordinates()

The easiest way to understand an implant geometry is often simply to plot it:

.. code-block:: python

    implant.plot(annotate=True)

Coordinate systems
------------------

Retinal implants use a coordinate system centered on the fovea. Distances are
stored in microns:

* positive ``x`` points toward the nasal retina;
* positive ``y`` points toward the superior retina;
* positive ``z`` moves away from the retina and into the vitreous.

The ``eye`` parameter handles the corresponding left- versus right-eye
geometry where needed.

Cortical implants live in physical cortical coordinates instead. A cortical
model combines those electrode locations with a
:py:class:`~pulse2percept.topography.VisualFieldMap` to determine where
stimulation falls in the visual field. That is why cortical implants use the
models in :py:mod:`pulse2percept.models.cortex`, rather than retinal models
such as the Axon Map Model.

Building your own implant
-------------------------

For a custom array, you usually do not need a new implant class. An
:py:class:`~pulse2percept.implants.ElectrodeGrid` can be wrapped directly in a
:py:class:`~pulse2percept.implants.ProsthesisSystem`:

.. code-block:: python

    from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem

    earray = ElectrodeGrid(
        shape=(10, 10),
        spacing=500,
        r=100,
    )
    implant = ProsthesisSystem(earray=earray)

For irregular arrays, build an
:py:class:`~pulse2percept.implants.ElectrodeArray` from individual electrode
objects. :py:class:`~pulse2percept.implants.EnsembleImplant` can combine
multiple implants into one system.

The implant geometry and percept model remain separate, so a custom implant can
be paired with whichever model best matches the stimulation target and the
scientific question.

Rastering
---------

Some stimulators cannot drive every electrode simultaneously. Raster strategies
split an array into groups that take turns:

.. code-block:: python

    implant.raster = p2p.implants.CheckerboardRaster(
        implant, n_groups=5
    )

The encoder uses that schedule when constructing the electrical stimulus. See
:ref:`topics-rasters` for the details.

.. seealso::

    * :py:mod:`pulse2percept.implants` for the implant API
    * :py:mod:`pulse2percept.implants.cortex` for cortical implant classes
    * :ref:`Stimulus Encoders <topics-encoders>`
    * :ref:`Raster Strategies <topics-rasters>`
    * :ref:`Electrical Stimuli <topics-stimuli>`
    * :ref:`Computational Models <topics-models>`
    * :ref:`Physical Units <topics-units>`
