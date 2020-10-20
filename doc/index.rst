.. _topics-index:

.. toctree::
   :caption: First steps
   :hidden:
   :maxdepth: 1

   Overview <self>
   install
   examples/index

.. toctree::
   :caption: Basic concepts
   :hidden:

   topics/implants
   topics/stimuli
   topics/models
   topics/datasets

.. toctree::
   :caption: User Guide
   :hidden:

   users/api
   users/faq
   users/news
   users/release_notes
   users/references

.. toctree::
   :caption: Developer Guide
   :hidden:

   developers/contributing
   developers/style_guide
   developers/releases

.. include:: ../README.rst
   :end-line: 24

|

=====================================
pulse2percept |version| documentation
=====================================

.. include:: ../README.rst
   :start-line: 28
   :end-line: 71

Installation
============

.. include:: ../README.rst
   :start-line: 74
   :end-line: 93

Detailed instructions for different platforms can be found in our
:ref:`Installation Guide <install>`.

.. note::

    You can also skip installation and run pulse2percept in a Jupyter Notebook
    on `Google Colab`_. Simply make the first cell in the notebook
    ``!pip install pulse2percept`` for the stable version or
    ``!pip install git+https://github.com/pulse2percept/pulse2percept.git``
    for the latest version.

.. _Google Colab: https://colab.research.google.com

Where to go from here
=====================

*  Have a look at some code examples from our
   :ref:`Example Gallery <sphx_glr_examples>`.
*  Familiarize yourself with :ref:`visual prostheses <topics-implants>`,
   :ref:`electrical stimuli <topics-stimuli>`, and our
   :ref:`computational models <topics-models>`.
*  See if your question has already been addressed in the
   :ref:`FAQ <users-faq>`.
*  Request features or report bugs in our `Issue Tracker`_ on GitHub.

.. _Issue Tracker: https://github.com/pulse2percept/pulse2percept/issues

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
