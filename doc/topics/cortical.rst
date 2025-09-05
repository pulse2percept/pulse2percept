.. _topics-cortical:

==========================
Cortical Visual Prostheses
==========================
Visual prostheses can either be implanted on the retina or the cortex.
pulse2percept was originally designed for retinal implants, but now
supports cortical implants. In this section, we cover cortical
topography, implants, and models, with runnable examples.

.. _topics-cortical-topography:

Topography
----------
The visual cortex processes visual information and is split into left and
right hemispheres. It contains multiple regions (V1, V2, V3, …) that
arrange visual input retinotopically (a mapping from visual field coordinates
to cortical coordinates).

Model Plotting
^^^^^^^^^^^^^^
A simple way to visualize cortical retinotopy is with a model that samples
points in the visual field and maps them onto cortex.

.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt
    from pulse2percept.models.cortex import ScoreboardModel

    model = ScoreboardModel(regions=["v1", "v2", "v3"]).build()

Note the `model.build()` call; you must call this before plotting.

If we want to plot the model in the visual field, set `use_dva=True`. With
style `"scatter"`, you can see the sampling points in visual degrees (dva):

.. ipython:: python
    :okwarning:

    @savefig score.png align=center
    model.plot(style="scatter", use_dva=True)

If we omit `use_dva=True`, the points are shown **on cortex** (in mm). The
default scoreboard model uses :py:class:`~pulse2percept.topography.Polimeni2006Map`,
but you can also use :py:class:`~pulse2percept.topography.NeuropythyMap`
(3D, subject-specific, MRI-based) when available.

The cortex is represented as two hemifields with a 20 mm left-hemisphere
offset, and cortical maps have `split_map=True` so current doesn’t cross
between hemispheres.

.. ipython:: python
    :okwarning:

    @savefig model_scatter.png align=center
    model.plot(style="scatter")

Cortical magnification yields denser sampling near the fovea origin; another
useful style is `"hull"`:

.. ipython:: python
    :okwarning:

    @savefig model_hull.png align=center
    model.plot(style="hull")

And `"cell"`:

.. ipython:: python
    :okwarning:

    @savefig model_cell.png align=center
    model.plot(style="cell")

Visual Field Map Plotting
^^^^^^^^^^^^^^^^^^^^^^^^^
You can also plot a visual field map directly. We’ll **try** Neuropythy’s
MRI-based fsaverage V1 map, and gracefully fall back to Polimeni if
Neuropythy (or its data) isn’t usable at build time.

.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt
    from pulse2percept.topography import Polimeni2006Map
    try:
        from pulse2percept.topography import NeuropythyMap
        vfmap = NeuropythyMap('fsaverage', regions=['v1'])
        print("Using NeuropythyMap(fsaverage).")
    except Exception as e:
        print("Neuropythy unavailable on this build (falling back to Polimeni2006Map):", e)
        vfmap = Polimeni2006Map()

    @savefig neuropythy_or_polimeni.png align=center
    vfmap.plot()

.. _topics-cortical-implants:

Cortical Implants
-----------------
:py:class:`~pulse2percept.implants.cortex.Orion`,
:py:class:`~pulse2percept.implants.cortex.Cortivis`,
and :py:class:`~pulse2percept.implants.cortex.ICVP` are cortical implants.
Setting `annotate=True` shows electrode names, useful for assigning
per-electrode stimuli.

Orion
^^^^^
:py:class:`~pulse2percept.implants.cortex.Orion` has 60 electrodes in a hex grid.

.. ipython:: python

    from pulse2percept.implants.cortex import Orion

    orion = Orion()
    @savefig orion.png align=center
    orion.plot(annotate=True)

Cortivis
^^^^^^^^
:py:class:`~pulse2percept.implants.cortex.Cortivis` has 96 electrodes in a square grid.

.. ipython:: python

    from pulse2percept.implants.cortex import Cortivis

    cortivis = Cortivis()
    @savefig cortivis.png align=center
    cortivis.plot(annotate=True)

ICVP
^^^^
:py:class:`~pulse2percept.implants.cortex.ICVP` has 16 primary electrodes plus
reference/counter electrodes.

.. ipython:: python

    from pulse2percept.implants.cortex import ICVP

    icvp = ICVP()
    @savefig icvp.png align=center
    icvp.plot(annotate=True)

.. _topics-ensemble-implant:

Neuralink
^^^^^^^^^
:py:class:`~pulse2percept.implants.cortex.Neuralink` is composed of multiple
threads; currently :py:class:`~pulse2percept.implants.cortex.LinearEdgeThread`
(32 electrodes) is implemented.

.. ipython:: python

    import matplotlib.pyplot as plt
    from pulse2percept.implants.cortex import LinearEdgeThread

    thread = LinearEdgeThread()
    thread.plot3D()
    @savefig neuralink_thread.png align=center
    plt.axis('equal')

Neuropythy works well for 3D, subject-specific retinotopy. The code below
**tries** `NeuropythyMap('fsaverage', ['v1'])` and falls back to `Polimeni2006Map`
if Neuropythy (or its dataset) isn’t available during the docs build. The example
still produces a figure either way.

.. ipython:: python
    :okwarning:

    import matplotlib.pyplot as plt
    from pulse2percept.implants.cortex import Neuralink
    from pulse2percept.models.cortex import ScoreboardModel
    from pulse2percept.topography import Polimeni2006Map

    try:
        from pulse2percept.topography import NeuropythyMap
        nmap = NeuropythyMap('fsaverage', regions=['v1'])
        print("Using NeuropythyMap(fsaverage).")
    except Exception as e:
        print("Neuropythy unavailable on this build (falling back to Polimeni2006Map):", e)
        nmap = Polimeni2006Map()

    model = ScoreboardModel(vfmap=nmap, xrange=(-4, 0), yrange=(-4, 4), xystep=.25).build()
    neuralink = Neuralink.from_neuropythy(
        nmap, xrange=model.xrange, yrange=model.yrange, xystep=1, rand_insertion_angle=0
    )

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    neuralink.plot3D(ax=ax1)
    model.plot3D(style='cell', ax=ax1)
    ax2 = fig.add_subplot(122)
    neuralink.plot(ax=ax2)
    model.plot(style='cell', ax=ax2)

    @savefig neuralink.png align=center
    plt.show()

Ensemble Implants
-----------------
:py:class:`~pulse2percept.implants.EnsembleImplant` lets you combine multiple
implants (e.g., two :py:class:`~pulse2percept.implants.cortex.Cortivis`):

.. ipython:: python
    :okwarning:

    i1 = Cortivis(x=15000, y=0)
    i2 = Cortivis(x=20000, y=0)
    i1.plot(annotate=True)
    i2.plot(annotate=True)
    @savefig cortivis_multiple.png align=center
    plt.show()

.. ipython:: python

    from pulse2percept.implants import EnsembleImplant

    ensemble = EnsembleImplant(implants=[i1, i2])
    _, ax = plt.subplots(1, 1, figsize=(12, 7))
    @savefig ensemble.png align=center
    ensemble.plot(annotate=True, ax=ax)

Electrodes are renamed `index-electrode` by constructor order; with dict
input they’re `key-electrode`.

.. _topics-cortical-models:

Models
------
Apply :py:class:`~pulse2percept.models.cortex.ScoreboardModel` to a
:py:class:`~pulse2percept.implants.cortex.Cortivis` implant:

.. ipython:: python

    from pulse2percept.models.cortex import ScoreboardModel

    model = ScoreboardModel(rho=1000).build()

Create an implant:

.. ipython:: python

    from pulse2percept.implants.cortex import Cortivis

    implant = Cortivis()

Plot the model and implant together (Cortivis defaults to (15, 0)):

.. ipython:: python
    :okwarning:

    model.plot()
    implant.plot()
    @savefig model_implant_cortivis.png align=center
    plt.show()

Add a stimulus: here we apply 0, 1, and 2 (arbitrary units) to thirds of the array:

.. ipython:: python

    import numpy as np
    implant.stim = np.concatenate((np.zeros(32), np.zeros(32) + 1, np.zeros(32) + 2))
    @savefig model_stim.png align=center
    implant.plot(stim_cmap=True)

Or set a few electrodes explicitly:

.. ipython:: python

    implant.stim = {"15": 1, "37": 1.5, "61": 0.5}
    @savefig model_stim_specific.png align=center
    implant.plot(stim_cmap=True)

Use a larger Orion to make the pattern more obvious:

.. ipython:: python

    from pulse2percept.implants.cortex import Orion

    implant = Orion()
    implant.stim = np.concatenate((np.zeros(30), np.zeros(30) + 1))
    @savefig model_implant_orion.png align=center
    implant.plot(stim_cmap=True)

Predict a percept and plot it:

.. ipython:: python

    percept = model.predict_percept(implant)
    @savefig model_percept.png align=center
    percept.plot()

Stimulate the other half instead:

.. ipython:: python

    implant.stim = np.concatenate((np.zeros(30) + 1, np.zeros(30)))
    @savefig model_stim_bottom.png align=center
    implant.plot(stim_cmap=True)

.. ipython:: python

    percept = model.predict_percept(implant)
    @savefig model_percept_bottom.png align=center
    percept.plot()

Move the implant more peripheral to show cortical magnification:

.. ipython:: python

    implant = Orion(x=25000)
    implant.stim = np.concatenate((np.zeros(30) + 1, np.zeros(30)))
    percept = model.predict_percept(implant)
    @savefig model_stim_periphery.png align=center
    percept.plot()

pulse2percept currently has two cortical models,
:py:class:`~pulse2percept.models.cortex.ScoreboardModel` (simple radial spread)
and :py:class:`~pulse2percept.models.cortex.DynaphosModel` (adds temporal
and charge-related effects).

.. ipython:: python

    from pulse2percept.models.cortex import DynaphosModel
    from pulse2percept.stimuli import BiphasicPulseTrain
    from pulse2percept.implants.cortex import Orion

    model = DynaphosModel().build()
    implant = Orion()
    implant.stim = {e: BiphasicPulseTrain(20, 200, .45) for e in implant.electrode_names}
    percept = model.predict_percept(implant)
    @savefig model_dynaphos.png align=center
    percept.plot()

You can also play the percept as a video with `percept.play()`.

.. _topics-cortical-developers:

For Developers
--------------
Notes for implementers of cortical features.

Units
^^^^^
pulse2percept uses microns for length, microamps for current, and milliseconds
for time.

Topography
^^^^^^^^^^
Maps are subclasses of :py:class:`~pulse2percept.topography.CorticalMap`
(e.g., :py:class:`~pulse2percept.topography.Polimeni2006Map`). They typically
set `split_map=True` and `left_offset=20` mm, as visualized above.

To create a new map, subclass `CorticalMap` and implement `dva_to_v1`; add
`dva_to_v2`/`dva_to_v3` as applicable. Optionally implement the inverse
`v*_to_dva` methods.

.. code-block:: python

    from pulse2percept.topography import CorticalMap
    import numpy as np

    class TestMap(CorticalMap):
        def dva_to_v1(self, x, y):  # -> (x, y)
            return x, y
        def dva_to_v2(self, x, y):  # -> (2x, 2y)
            return 2 * x, 2 * y
        def dva_to_v3(self, x, y):  # -> (3x, 3y)
            return 3 * x, 3 * y

    m = TestMap(regions=["v1", "v2", "v3"])
    x = np.array([0, 1, 2]); y = np.array([3, 4, 5])
    print(m.from_dva()["v1"](x, y))
    print(m.from_dva()["v2"](x, y))
    print(m.from_dva()["v3"](x, y))
