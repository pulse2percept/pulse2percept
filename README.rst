.. |doi| image:: https://img.shields.io/badge/DOI-10.25080/shinma--7f4c6e7--00c-blue
   :target: https://doi.org/10.25080/shinma-7f4c6e7-00c
   :alt: DOI
.. |license| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/pulse2percept/pulse2percept/blob/master/LICENSE
   :alt: BSD 3-clause
.. |pypi| image:: https://img.shields.io/pypi/v/pulse2percept.svg
   :target: https://pypi.org/project/pulse2percept
   :alt: PyPI
.. |build| image:: https://github.com/pulse2percept/pulse2percept/actions/workflows/build.yml/badge.svg
   :target: https://github.com/pulse2percept/pulse2percept/actions
   :alt: build
.. |forks| image:: https://img.shields.io/github/forks/pulse2percept/pulse2percept
   :target: https://github.com/pulse2percept/pulse2percept/network/members
   :alt: GitHub forks
.. |stars| image:: https://img.shields.io/github/stars/pulse2percept/pulse2percept
   :target: https://github.com/pulse2percept/pulse2percept/stargazers
   :alt: GitHub stars

|doi| |license| |pypi| |build| |forks| |stars|

.. badges-end

====================================================================
pulse2percept: A Python-based simulation framework for bionic vision
====================================================================

.. intro-begin

An estimated 43 million people worldwide are blind. For some causes of
blindness, a visual neuroprosthesis (a retinal or cortical implant) is the
only treatment option. What implant users see depends on the device, the
stimulus, and the stimulated tissue, and is difficult to predict.

`pulse2percept`_ (p2p) is an open-source Python package for simulating
these percepts. It provides spatiotemporal models of common
`retinal and cortical implants`_.

.. _pulse2percept: https://github.com/pulse2percept/pulse2percept
.. _retinal and cortical implants: https://en.wikipedia.org/wiki/Visual_prosthesis

If you use p2p in a scholarly publication, please cite as:

.. epigraph::

    M Beyeler, GM Boynton, I Fine, A Rokem (2017). pulse2percept: A
    Python-based simulation framework for bionic vision. *Proceedings of the
    16th Python in Science Conference (SciPy)*, p.81-88,
    doi:`10.25080/shinma-7f4c6e7-00c <https://doi.org/10.25080/shinma-7f4c6e7-00c>`_.

Installation
============

.. quickstart-begin

To install the `stable release`_ of p2p, run:

.. code-block:: bash

    pip install pulse2percept

To install the `current development version`_ directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/pulse2percept/pulse2percept

``pip`` installs the dependencies and selects a release that supports your
Python version (see `Compatibility and Building from Source
<https://pulse2percept.readthedocs.io/en/latest/getting_started/install.html#install-compatibility>`__).

.. _stable release: https://pulse2percept.readthedocs.io/en/stable/index.html
.. _current development version: https://pulse2percept.readthedocs.io/en/latest/index.html

.. quickstart-end

You can find the full documentation
`here <https://pulse2percept.readthedocs.io/en/stable>`_.

Compatibility
-------------

.. compat-begin

+---------------------------+------+------+------+------+------+-----+-----+-----+
|           Python          | 3.14 | 3.13 | 3.12 | 3.11 | 3.10 | 3.9 | 3.8 | 3.7 |
+===========================+======+======+======+======+======+=====+=====+=====+
| p2p 0.11 Foundations      | Yes  | Yes  | Yes  | Yes  |      |     |     |     |
+---------------------------+------+------+------+------+------+-----+-----+-----+
| p2p 0.10 Encoders         | Yes  | Yes  | Yes  | Yes  |      |     |     |     |
+---------------------------+------+------+------+------+------+-----+-----+-----+
| p2p 0.9.1                 |      | Yes  | Yes  | Yes  | Yes  |     |     |     |
+---------------------------+------+------+------+------+------+-----+-----+-----+
| p2p 0.9 Cortex            |      |      | Yes  | Yes  | Yes  | Yes |     |     |
+---------------------------+------+------+------+------+------+-----+-----+-----+
| p2p 0.8 Retina            |      |      |      |      | Yes  | Yes | Yes | Yes |
+---------------------------+------+------+------+------+------+-----+-----+-----+

Prebuilt wheels are available for 64-bit Linux, macOS 11 and later
(Apple silicon and Intel), and 64-bit Windows. On other platforms, ``pip`` may
build pulse2percept from source, which requires a C compiler. NumPy, Cython,
and other build dependencies are installed automatically.

Our `GitHub Action Runners`_ test the current release on Linux, macOS, and
Windows for every supported Python version listed above.

.. _GitHub Action Runners: https://github.com/pulse2percept/pulse2percept/actions

.. compat-end

Upgrading and Uninstalling
--------------------------

.. upgrade-begin

To upgrade p2p to the latest stable version:

.. code-block:: bash

    pip install -U pulse2percept

To uninstall:

.. code-block:: bash

    pip uninstall pulse2percept -y

.. upgrade-end

Where to go from here
=====================

*  Run the `Quickstart`_.
*  Read the core concepts in order: `implants`_, `stimulation`_, and
   `models and percepts`_.
*  Work through the `Example Gallery`_, starting with the key examples.
*  Check the `FAQ`_ for common questions.
*  Request features or report bugs on the `Issue Tracker`_.

.. _Quickstart: https://pulse2percept.readthedocs.io/en/latest/examples/plot_quickstart.html
.. _implants: https://pulse2percept.readthedocs.io/en/latest/concepts/implants.html
.. _stimulation: https://pulse2percept.readthedocs.io/en/latest/concepts/stimulation.html
.. _models and percepts: https://pulse2percept.readthedocs.io/en/latest/concepts/models.html
.. _Example Gallery: https://pulse2percept.readthedocs.io/en/latest/examples/index.html
.. _FAQ: https://pulse2percept.readthedocs.io/en/latest/reference/faq.html
.. _Issue Tracker: https://github.com/pulse2percept/pulse2percept/issues
