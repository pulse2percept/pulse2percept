.. image:: https://img.shields.io/badge/DOI-10.25080/shinma--7f4c6e7--00c-blue
   :target: https://doi.org/10.25080/shinma-7f4c6e7-00c
   :alt: DOI

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/uwescience/pulse2percept/blob/master/LICENSE
   :alt: BSD 3-clause

.. image:: https://img.shields.io/pypi/v/pulse2percept.svg
   :target: https://pypi.org/project/pulse2percept
   :alt: PyPI

.. image:: https://img.shields.io/github/forks/uwescience/pulse2percept?style=social
   :target: https://github.com/uwescience/pulse2percept/network/members
   :alt: GitHub forks

.. image:: https://img.shields.io/github/stars/uwescience/pulse2percept?style=social
   :target: https://github.com/uwescience/pulse2percept/stargazers
   :alt: GitHub stars

====================================================================
pulse2percept: A Python-based simulation framework for bionic vision
====================================================================

By 2020 roughly 20 million people will suffer from retinal diseases such as
macular degeneration or retinitis pigmentosa, and a variety of sight
restoration technologies are being developed to target these diseases.

Retinal prostheses, now implanted in over 300 patients worldwide, electrically
stimulate surviving cells in order to evoke neuronal responses that are
interpreted by the brain as visual percepts ('phosphenes').
However, interactions between the device electronics and the retinal
neurophysiology result in perceptual distortions that may severely limit the
quality of the generated visual experience:

.. raw:: html

   <div style="width: 100%; margin: 0; padding: 0 30%">
    <div style="float: right; width: 20%; text-align: center">
      Predicted percept
    </div>
    <div style="width: 20%; text-align: center">
      Input stimulus
    </div>
  </div>

.. image:: https://raw.githubusercontent.com/uwescience/pulse2percept/master/doc/_static/boston-train-combined.gif
   :align: center
   :alt: Input stimulus and predicted percept

Built on the NumPy and SciPy stacks, *pulse2percept* provides an open-source
implementation of a number of computational models for state-of-the-art
`visual prostheses`_ (also known as the 'bionic eye'),
such as `ArgusII`_ and `AlphaIMS`_, to provide insight into the
visual experience provided by these devices.

.. _visual prostheses: https://en.wikipedia.org/wiki/Visual_prosthesis
.. _ArgusII: https://www.secondsight.com/discover-argus/
.. _AlphaIMS: https://www.retina-implant.de

Simulations such as the above are likely to be critical for providing realistic
estimates of prosthetic vision, thus providing regulatory bodies with guidance
into  what sort of visual tests are appropriate for evaluating prosthetic
performance, and improving current and future technology.

If you use *pulse2percept* in a scholarly publication, please cite as:

.. epigraph::

    M Beyeler, GM Boynton, I Fine, A Rokem (2017). pulse2percept: A
    Python-based simulation framework for bionic vision. *Proceedings of the
    16th Python in Science Conference (SciPy)*, p.81-88,
    doi:`10.25080/shinma-7f4c6e7-00c <https://doi.org/10.25080/shinma-7f4c6e7-00c>`_.

Installation
============

Dependencies
------------

pulse2percept requires:

1.  `Python`_ >= 3.5
2.  `Cython`_ >= 0.28
3.  `NumPy`_ >= 1.9
4.  `SciPy`_ >= 1.0
5.  `JobLib`_ >= 0.11

.. _Python: http://www.python.org
.. _Cython: http://www.cython.org
.. _NumPy: http://www.numpy.org
.. _SciPy: http://www.scipy.org
.. _JobLib: https://github.com/joblib/joblib

**pulse2percept 0.5 was the last version to support Python 2.7.**
pulse2percept 0.6 and later require Python 3.5 or newer.

Optional packages:

1.  `scikit-image`_ for image functionality in the `io` module.
2.  `scikit-video`_ for video functionality in the `io` module. You will also
    need an FFMPEG codec (see next bullet point).
3.  `ffmpeg codec`_ if you're on Windows and want to use functions in the `io`
    module.
4.  `Dask`_ for parallel processing (a joblib alternative).
    Use conda to install.
5.  `Numba`_ for just-in-time compilation. Use conda to install.
6.  `Pytest`_ to run the test suite.

.. _scikit-image: http://scikit-image.org
.. _scikit-video: http://www.scikit-video.org
.. _ffmpeg codec: http://adaptivesamples.com/how-to-install-ffmpeg-on-windows
.. _Dask: https://github.com/dask/dask
.. _Numba: http://numba.pydata.org
.. _Pytest: https://docs.pytest.org/en/latest

Stable version
--------------

The latest stable release of pulse2percept can be installed with pip:

.. code-block:: bash

    pip3 install pulse2percept

Development version
-------------------

In order to get the latest development version of pulse2percept, use the
following recipe.

1.  Go to `pulse2percept on GitHub`_
    and click on "Fork" in the top-right corner. This will allow you to work on
    your own copy of the code
    (``https://github.com/<Your User Name>/pulse2percept``)
    and contribute changes later on.

2.  Clone the repository to get a local copy on your computer:

    .. code-block:: bash

        git clone https://github.com/<Your User Name>/pulse2percept.git
        cd pulse2percept

    Make sure to replace ``<Your User Name>`` above with your actual GitHub
    user name.

3.  Install all packages listed in ``requirements.txt``:

    .. code-block:: bash

        pip3 install -r requirements.txt

    This includes Cython. If you are on Windows, you will also need a suitable
    C compiler (either Visual Studio or MinGW). See instructions `here`_.
    `Christoph Gohlke`_ maintains an unofficial set of Cython
    `Windows binaries`_ for various Python versions, in both 32 and 64 bits.

5.  On Unix platforms, you can compile pulse2percept using the Makefile:

    .. code-block:: bash

        make

    Type ``make help`` to see your options.

    On any other platforms (e.g., Windows), type:

    .. code-block:: bash

        pip3 install -e .

6.  You can run the test suite to make sure everything works as expected:

    .. code-block:: bash

        pip install pytest
        make tests

    Or, on Windows:

    .. code-block:: bash

        pip install pytest
        pytest --doctest-modules --showlocals -v pulse2percept

7.  To use pulse2percept after installation, execute in Python:

    .. code-block:: python

        import pulse2percept as p2p

.. _pulse2percept on GitHub: https://github.com/uwescience/pulse2percept
.. _here: https://github.com/cython/cython/wiki/InstallingOnWindows
.. _Christoph Gohlke: http://www.lfd.uci.edu/~gohlke
.. _Windows binaries: http://www.lfd.uci.edu/~gohlke/pythonlibs/#cython
