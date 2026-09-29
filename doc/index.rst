.. _topics-index:

.. toctree::
   :caption: Getting started
   :hidden:
   :maxdepth: 1

   Overview <self>
   getting_started/install
   Quickstart Guide <examples/plot_quickstart>

.. toctree::
   :caption: Core Concepts
   :hidden:

   concepts/implants
   concepts/stimulation
   concepts/models
   concepts/vision
   concepts/coordinates
   concepts/units
   concepts/datasets

.. toctree::
   :caption: Scientific examples
   :hidden:

   examples/models/index

.. toctree::
   :caption: Reference
   :hidden:

   reference/api
   reference/faq
   reference/release_notes
   reference/references
   reference/research

.. toctree::
   :caption: Developer Guide
   :hidden:

   developers/contributing
   developers/style_guide
   developers/benchmarks
   developers/releases

.. include:: ../README.rst
   :end-before: .. badges-end

|

=====================================
pulse2percept |version| documentation
=====================================

.. include:: ../README.rst
   :start-after: .. intro-begin
   :end-before: .. quickstart-end

Compatibility
=============

.. include:: ../README.rst
   :start-after: .. compat-begin
   :end-before: .. compat-end

Getting Started
===============

New to pulse2percept? Start with the
:doc:`Quickstart Guide <examples/plot_quickstart>` for some short
end-to-end examples.

From there, the core concepts are easiest to read in this order:
:doc:`implants <concepts/implants>`,
:doc:`stimulation <concepts/stimulation>`,
:doc:`models and percepts <concepts/models>`,
:doc:`visual input <concepts/vision>`,
:doc:`coordinates <concepts/coordinates>`, :doc:`units <concepts/units>`, and
:doc:`datasets <concepts/datasets>`.

The :doc:`model reproductions <examples/models/index>` show how published
models and experiments are implemented in pulse2percept.

For common questions, see the :doc:`FAQs <reference/faq>`.
If something is broken or missing, please open an issue on the 
`Issue Tracker`_.

.. _Issue Tracker: https://github.com/pulse2percept/pulse2percept/issues
