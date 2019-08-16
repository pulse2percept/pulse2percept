.. _topics-index:

.. toctree::
   :caption: First steps
   :hidden:
   :maxdepth: 1

   Overview <self>
   install
   _examples/index

.. toctree::
   :caption: Basic concepts
   :hidden:

   topics/implants
   topics/stimuli
   topics/models

.. toctree::
   :caption: User Guide
   :hidden:

   users/api
   users/faq
   users/release_notes
   users/news

.. toctree::
   :caption: Developer Guide
   :hidden:

   developers/contributing
   developers/extending
   developers/debug

.. include:: ../README.rst
   :end-line: 19

|

=====================================
pulse2percept |version| documentation
=====================================

.. include:: ../README.rst
   :start-line: 24
   :end-line: 75

Installation
============

The latest stable release of *pulse2percept* can be installed with pip:

.. code-block:: bash

    pip3 install pulse2percept

In order to get the bleeding-edge version of *pulse2percept*, use the commands:

.. code-block:: bash

    git clone https://github.com/uwescience/pulse2percept.git
    cd pulse2percept
    make

Detailed instructions for different platforms can be found in our
:ref:`Installation Guide <install>`.

Where to go from here
=====================

*  Have a look at some code examples from our
   :ref:`Example Gallery <sphx_glr__examples>`.
*  Familiarize yourself with :ref:`visual prostheses <topics-implants>`,
   :ref:`electrical stimuli <topics-stimuli>`, and our
   :ref:`computational models <topics-models>`.
*  Request features or report bugs in our `issue tracker`_ on GitHub.

.. _issue tracker: https://github.com/uwescience/pulse2percept/issues

.. figure:: _static/eScience_Logo_HR.png
   :align: center
   :figclass: align-center
   :target: http://escience.washington.edu

   This work was supported by a grant from the
   `Gordon & Betty Moore Foundation`_ and the `Alfred P. Sloan Foundation`_
   to the University of Washington `eScience Institute`_.

.. _Gordon & Betty Moore Foundation: https://www.moore.org
.. _Alfred P. Sloan Foundation: http://www.sloan.org
.. _eScience Institute: http://escience.washington.edu
