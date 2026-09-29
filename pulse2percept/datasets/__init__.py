"""Published datasets from prosthetic vision studies.

Small datasets ship with pulse2percept (``load_*``). Larger datasets are
downloaded to the data directory on first use (``fetch_*``).

Loaders
-------

.. autosummary::
    :toctree:

    load_greenwald2009
    load_horsager2009
    load_nanduri2012
    load_perezfornos2012

Fetchers
--------

Require network access on first use.

.. autosummary::
    :toctree:

    fetch_beyeler2019
    fetch_han2021

Data Directory and Downloads
----------------------------

.. autosummary::
    :toctree:

    get_data_dir
    clear_data_dir
    fetch_url
    base.download_from_osf
    base.has_network
    base.osf_is_reachable

.. seealso::

    *  :ref:`Core Concepts > Datasets <topics-datasets>`

"""

from .base import clear_data_dir, get_data_dir, fetch_url
from .beyeler2019 import fetch_beyeler2019
from .han2021 import fetch_han2021
from .horsager2009 import load_horsager2009
from .nanduri2012 import load_nanduri2012
from .perezfornos2012 import load_perezfornos2012
from .greenwald2009 import load_greenwald2009


__all__ = [
    'clear_data_dir',
    'fetch_url',
    'fetch_beyeler2019',
    'fetch_han2021',
    'get_data_dir',
    'load_horsager2009',
    'load_nanduri2012',
    'load_perezfornos2012',
    'load_greenwald2009',
]
