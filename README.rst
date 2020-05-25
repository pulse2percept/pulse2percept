.. image:: https://img.shields.io/badge/DOI-10.25080/shinma--7f4c6e7--00c-blue
   :target: https://doi.org/10.25080/shinma-7f4c6e7-00c
   :alt: DOI

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/pulse2percept/pulse2percept/blob/master/LICENSE
   :alt: BSD 3-clause

.. image:: https://img.shields.io/pypi/v/pulse2percept.svg
   :target: https://pypi.org/project/pulse2percept
   :alt: PyPI

.. image:: https://github.com/pulse2percept/pulse2percept/workflows/build/badge.svg
   :target: https://github.com/pulse2percept/pulse2percept/actions
   :alt: build

.. image:: https://img.shields.io/github/forks/pulse2percept/pulse2percept?style=social
   :target: https://github.com/pulse2percept/pulse2percept/network/members
   :alt: GitHub forks

.. image:: https://img.shields.io/github/stars/pulse2percept/pulse2percept?style=social
   :target: https://github.com/pulse2percept/pulse2percept/stargazers
   :alt: GitHub stars

====================================================================
pulse2percept: A Python-based simulation framework for bionic vision
====================================================================

By 2020 roughly 20 million people will suffer from retinal diseases such as
macular degeneration or retinitis pigmentosa, and a variety of sight
restoration technologies are being developed to target these diseases.

Retinal prostheses, now implanted in over 500 patients worldwide, electrically
stimulate surviving cells in order to evoke neuronal responses that are
interpreted by the brain as visual percepts ('phosphenes').
However, interactions between the device electronics and the retinal
neurophysiology result in perceptual distortions that may severely limit the
quality of the generated visual experience:

.. image:: https://raw.githubusercontent.com/pulse2percept/pulse2percept/master/doc/_static/boston-train-combined.gif
   :align: center
   :alt: Input stimulus and predicted percept

*(left: input stimulus, right: predicted percept)*

Built on the NumPy and SciPy stacks, `pulse2percept`_ provides an open-source
implementation of a number of computational models for state-of-the-art
`visual prostheses`_ (also known as the 'bionic eye'),
such as `ArgusII`_ and `AlphaIMS`_, to provide insight into the
visual experience provided by these devices.

.. _pulse2percept: https://github.com/pulse2percept/pulse2percept
.. _visual prostheses: https://en.wikipedia.org/wiki/Visual_prosthesis
.. _ArgusII: https://www.secondsight.com/discover-argus/
.. _AlphaIMS: https://www.retina-implant.de

Simulations such as the above are likely to be critical for providing realistic
estimates of prosthetic vision, thus providing regulatory bodies with guidance
into  what sort of visual tests are appropriate for evaluating prosthetic
performance, and improving current and future technology.

If you use pulse2percept in a scholarly publication, please cite as:

.. epigraph::

    M Beyeler, GM Boynton, I Fine, A Rokem (2017). pulse2percept: A
    Python-based simulation framework for bionic vision. *Proceedings of the
    16th Python in Science Conference (SciPy)*, p.81-88,
    doi:`10.25080/shinma-7f4c6e7-00c <https://doi.org/10.25080/shinma-7f4c6e7-00c>`_.

Installation
============

Once you have Python 3 and pip, the `stable release`_ of pulse2percept
can be installed with pip:

.. code-block:: bash

    pip3 install pulse2percept

In order to get the `bleeding-edge version`_ of pulse2percept, use the
commands:

.. code-block:: bash

    git clone https://github.com/pulse2percept/pulse2percept.git
    cd pulse2percept
    pip3 install .

.. _stable release: https://pulse2percept.readthedocs.io/en/stable/index.html
.. _bleeding-edge version: https://pulse2percept.readthedocs.io/en/latest/index.html

When installing the bleeding-edge version on Windows, note that you will have
to install your own C compiler first.
Detailed instructions for different platforms can be found in our
`Installation Guide`_.

.. _Installation Guide: https://pulse2percept.readthedocs.io/en/stable/install.html

Dependencies
------------

**pulse2percept 0.4.3 was the last version to support Python 2.7 and 3.4.**
pulse2percept 0.5+ requires Python 3.5+.

pulse2percept requires:

1.  `Python`_ (>= 3.5)
2.  `Cython`_ (>= 0.28)
3.  `NumPy`_ (>= 1.9)
4.  `SciPy`_ (>= 1.0)
5.  `Matplotlib`_ (>= 2.1)
6.  `JobLib`_ (>= 0.11)

Optional packages:

1.  `scikit-image`_ for image functionality in the ``io`` module.
2.  `scikit-video`_ for video functionality in the ``io`` module. You will also
    need an FFMPEG codec (see next bullet point).
3.  `ffmpeg codec`_ if you're on Windows and want to use functions in the `io`
    module.
4.  `Pandas`_ for loading datasets in the ``datasets`` module.
5.  `Dask`_ for parallel processing (a joblib alternative).
    Use conda to install.
6.  `Pytest`_ to run the test suite.

.. _Python: https://www.python.org
.. _Cython: https://www.cython.org
.. _NumPy: https://www.numpy.org
.. _SciPy: https://www.scipy.org
.. _Matplotlib: https://matplotlib.org
.. _JobLib: https://joblib.readthedocs.io
.. _scikit-image: https://scikit-image.org
.. _scikit-video: https://www.scikit-video.org
.. _ffmpeg codec: http://adaptivesamples.com/how-to-install-ffmpeg-on-windows
.. _Pandas: https://pandas.pydata.org
.. _Dask: https://github.com/dask/dask
.. _Pytest: https://docs.pytest.org/en/latest

All required packages are listed in ``requirements.txt`` in the root directory
of the git repository, and can be installed with the following command:

.. code-block:: bash

    git clone https://github.com/pulse2percept/pulse2percept.git
    cd pulse2percept
    pip3 install -r requirements.txt

All packages required for development (including all optional packages) are
listed in ``requirements-dev.txt`` and can be installed via:

.. code-block:: bash

    pip3 install -r requirements-dev.txt

Where to go from here
=====================

*  Have a look at some code examples from our `Example Gallery`_.
*  Familiarize yourself with `visual implants`_, `electrical stimuli`_,
   and our `computational models`_.
*  Check the `FAQ`_ to see if your question has already been answered.
*  Request features or report bugs in our `Issue Tracker`_ on GitHub.

.. _Example Gallery: https://pulse2percept.readthedocs.io/en/latest/examples/index.html
.. _visual implants: https://pulse2percept.readthedocs.io/en/latest/topics/implants.html
.. _electrical stimuli: https://pulse2percept.readthedocs.io/en/latest/topics/stimuli.html
.. _computational models: https://pulse2percept.readthedocs.io/en/latest/topics/models.html
.. _FAQ: https://pulse2percept.readthedocs.io/en/latest/users/faq.html
.. _Issue Tracker: https://github.com/pulse2percept/pulse2percept/issues
