.. _topics-datasets:

========
Datasets
========

:py:mod:`pulse2percept.datasets` provides published psychophysical and video
datasets from the bionic vision community. ``load_*`` functions read small
bundled data; ``fetch_*`` functions download larger datasets on first use.

.. list-table::
   :header-rows: 1
   :widths: 26 16 34 24

   * - Loader
     - Reference
     - Contents
     - Related model
   * - :py:func:`~pulse2percept.datasets.load_horsager2009`
     - [Horsager2009]_
     - Argus I single- and paired-pulse thresholds, 2 subjects
     - :py:class:`~pulse2percept.models.retina.Horsager2009Model`
   * - :py:func:`~pulse2percept.datasets.load_nanduri2012`
     - [Nanduri2012]_
     - Argus I brightness and size ratings vs amplitude and frequency,
       1 subject
     - :py:class:`~pulse2percept.models.retina.Nanduri2012Model`
   * - :py:func:`~pulse2percept.datasets.load_greenwald2009`
     - [Greenwald2009]_
     - Argus I brightness ratings and thresholds, 2 subjects
     - temporal models
   * - :py:func:`~pulse2percept.datasets.load_perezfornos2012`
     - [PerezFornos2012]_
     - Phosphene fading: joystick-reported brightness over time
     - temporal models
   * - :py:func:`~pulse2percept.datasets.fetch_beyeler2019`
     - [Beyeler2019]_
     - Phosphene drawings from Argus I and Argus II users (66 MB download)
     - :py:class:`~pulse2percept.models.retina.AxonMapModel`
   * - :py:func:`~pulse2percept.datasets.fetch_han2021`
     - [Han2021]_
     - Outdoor scene videos (303 MB download)
     - any model, as a
       :py:class:`~pulse2percept.stimuli.VideoStimulus`

The :ref:`example gallery <examples>` reproduces [Horsager2009]_,
[Nanduri2012]_ and [Beyeler2019]_ from these loaders.

Tabular loaders return a pandas DataFrame and accept filters, so a subject or
electrode can be selected without post-processing:

.. code-block:: python

    import pulse2percept as p2p

    data = p2p.datasets.load_horsager2009(subjects='S1', electrodes='C3')

Pandas is required for tabular datasets; some datasets also require HDF5 via
``h5py``.

Downloaded data
---------------

Fetched datasets are cached in ``~/pulse2percept_data`` by default. Set the
``PULSE2PERCEPT_DATA`` environment variable or pass a data directory directly
to a fetcher to use another location.

.. code-block:: python

    p2p.datasets.get_data_dir()
    p2p.datasets.clear_data_dir()

``clear_data_dir`` removes the cached files, so the next fetch downloads them
again.
