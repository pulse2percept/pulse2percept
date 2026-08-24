.. _topics-datasets:

========
Datasets
========

The :py:mod:`~pulse2percept.datasets` module provides loaders for small bundled
datasets and fetchers for larger downloadable datasets.

.. list-table::
   :header-rows: 1

   * - Function
     - Dataset
   * - :py:func:`~pulse2percept.datasets.load_horsager2009`
     - Threshold data from [Horsager2009]_
   * - :py:func:`~pulse2percept.datasets.load_nanduri2012`
     - Brightness ratings from [Nanduri2012]_
   * - :py:func:`~pulse2percept.datasets.load_perezfornos2012`
     - Phosphene fading from [PerezFornos2012]_
   * - :py:func:`~pulse2percept.datasets.fetch_beyeler2019`
     - Phosphene drawings from [Beyeler2019]_
   * - :py:func:`~pulse2percept.datasets.fetch_han2021`
     - Outdoor scenes from [Han2021]_

Pandas is required for tabular datasets; some datasets also require HDF5 via
``h5py``.

Downloaded data
---------------

Fetched datasets are cached in ``~/pulse2percept_data`` by default. Set the
``PULSE2PERCEPT_DATA`` environment variable or pass a data directory directly
to a fetcher to use another location.

.. code-block:: python

    import pulse2percept as p2p

    p2p.datasets.get_data_dir()
    p2p.datasets.clear_data_dir()

``clear_data_dir`` removes the cached files, so the next fetch downloads them
again.
