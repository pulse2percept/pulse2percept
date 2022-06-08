.. _topics-datasets:

========
Datasets
========

The :py:mod:`~pulse2percept.datasets` module provides two kinds of helper
functions that can be used to load datasets from the bionic vision community:

*  **Dataset loaders** can be used to load small datasets that come
   pre-packaged with the pulse2percept software.

   *  :py:func:`~pulse2percept.datasets.load_horsager2009`: Load the threshold
      data from [Horsager2009]_.

   *  :py:func:`~pulse2percept.datasets.load_nanduri2012`: Load the brightness
      rating data from [Nanduri2012]_.

   *  :py:func:`~pulse2percept.datasets.load_perezfornos2012`: Load the phosphene
      fading data from [PerezFornos2012]_.

*  **Dataset fetchers** can be used to download larger datasets from a given
   URL and directly import them into pulse2percept.

   *  :py:func:`~pulse2percept.datasets.fetch_beyeler2019`: Download and load
      the phosphene drawing dataset from [Beyeler2019]_.

   *  :py:func:`~pulse2percept.datasets.fetch_han2021`: Download and load
      the outdoor scenes from [Han2021]_.

.. note::

    You will need Pandas (``pip install pandas``) to load the data.
    Some datasets also require HDF5 (``pip install h5py``).

Local data directory
--------------------

By default, all datasets are downloaded to a directory called
'pulse2percept_data' located in the user home directory.
This directory is used as a cache so that large datasets don't have to be
downloaded repeatedly.

Alternatively, the directory can be set by a `PULSE2PERCEPT_DATA` environment
variable, or passed directly to the fetcher.

You can retrieve the current data directory as follows:

.. code-block:: python

    import pulse2percept as p2p
    p2p.datasets.get_data_dir()

You can delete the folder and all its contents as follows:

.. code-block:: python

    import pulse2percept as p2p
    p2p.datasets.clear_data_dir()

.. note ::

    Make sure you have write access to the specified data directory.

.. minigallery:: pulse2percept.datasets.load_horsager2009
    :add-heading: Examples using datasets
    :heading-level: -