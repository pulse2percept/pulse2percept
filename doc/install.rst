.. _install:

============
Installation
============

pulse2percept is in active development.
To get the latest stable version, we recommend installing our
:ref:`latest release <install-release>`.
If you wish to :ref:`contribute to the project <dev-contributing>` or require
the bleeding-edge version, you will want to install
:ref:`from source <install-source>` instead.

.. note::

    Having trouble with the installation?
    Please refer to our :ref:`Troubleshooting Guide <install-troubleshooting>`.

.. _install-prerequisites:

Prerequisites
=============

*  **Python** (>=3.5): Before getting started, you will need to have `Python`_
   on your computer. Check if Python is installed on your system by typing
   ``python --version`` in a terminal or command prompt.

   .. important::

       pulse2percept 0.5 will be the last release to support Python 2.7 and
       3.4. pulse2percept 0.6+ will require **Python 3.5 or newer**.

*  **Cython** (>= 0.28): pulse2percept relies on C extension modules for code
   acceleration. These require a C compiler, which on Unix platforms is
   already installed (``gcc``). However, on Windows you will have to install a
   compiler yourself:

   1.  Install **Build Tools for Visual Studio 2019** from the
       `Microsoft website`_.
       Note that the build tools for Visual Studio 2015 or 2017 should work as
       well (Python >= 3.5 requires C++ 14.X to be exact).
       Also note that you don't need to install Visual Studio itself.

   2.  Install `Cython`_:

       .. code-block:: bash

           pip3 install Cython

       If you get an error saying ``unable to find vcvarsall.bat``, then there
       is a problem with your Build Tools installation, in which case you
       should follow `this guide`_.

   .. important::

       Some guides on the web tell you to install MinGW instead of Visual Studio.
       However, this is not recommended for 64-bit platforms.
       When in doubt, follow `this guide`_.

.. _Python: https://wiki.python.org/moin/BeginnersGuide/Download
.. _Microsoft website: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
.. _Cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
.. _this guide: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows

.. _install-release:

Installing a release
====================

After taking care of the :ref:`prerequisites <install-prerequisites>`,
the latest pulse2percept release can be installed using pip:

.. code-block:: bash

    pip3 install -U pulse2percept

Then from any Python console or script, try:

.. code-block:: python

    import pulse2percept as p2p

.. note::

    Find out what's new in the :ref:`Release Notes <users-release-notes>`.

.. _install-source:

Installing from source
======================

.. _install-source-prerequisites:

Prerequisites
-------------

1.  **Git**: On Unix, you can install git from the `command line`_. On Windows,
    make sure to download `Git for Windows`_.

2.  **make** (optional): pulse2percept provides a Makefile to simplify the
    build process.
    ``make`` is part of `build-essentials`_ on Ubuntu, `XCode`_ on Mac OS X,
    and can be downloaded from `ezwinports`_ on Windows.

.. _command line: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
.. _Git for Windows: https://gitforwindows.org
.. _build-essentials: https://stackoverflow.com/questions/11934997/how-to-install-make-in-ubuntu
.. _XCode: https://developer.apple.com/support/xcode
.. _ezwinports: https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make

Obtaining the latest code from GitHub
-------------------------------------

The latest code can be obtained from our `GitHub repository`_.
Open a terminal, navigate to the directory where you want to install
pulse2percept, and type:

.. code-block:: bash

    git clone https://github.com/uwescience/pulse2percept

This will download the git repository and place it into a directory called
"pulse2percept".

.. note::

   If you wish to contribute to the project, it is recommended to fork the
   repo instead (see our :ref:`Contribution Guidelines <dev-contributing>`).

.. _GitHub repository: https://github.com/uwescience/pulse2percept

Installing dependencies
-----------------------

pulse2percept requires:

*  :ref:`Python <install-prerequisites>` (>= 3.5)
*  :ref:`Cython <install-prerequisites>` (>= 0.28)
*  `NumPy`_ (>= 1.9)
*  `SciPy`_ (>= 1.0)
*  `JobLib`_ (>= 0.11)

Optional packages include:

*  `Dask`_: an alternative to JobLib
*  `Numba`_: for just-in-time compilation of several functions in the
   :py:mod:`~pulse2percept.utils.convolution` module
*  `scikit-image`_: for functions in the :py:mod:`~pulse2percept.io.image`
   module
*  `scikit-video`_: for functions in the :py:mod:`~pulse2percept.io.video`
   module

All required packages are listed in ``requirements.txt`` in the root directory
of the git repository, and can be installed with the following command:

.. code-block:: bash

    cd pulse2percept
    pip3 install -r requirements.txt

All packages required for development (including all optional packages) are
listed in ``requirements-dev.txt`` and can be installed via:

.. code-block:: bash

    pip3 install -r requirements-dev.txt

.. _NumPy: https://numpy.org
.. _SciPy: https://scipy.org
.. _JobLib: https://joblib.readthedocs.io
.. _Dask: https://dask.org
.. _Numba: https://numba.pydata.org
.. _scikit-image: https://scikit-image.org
.. _scikit-video: https://www.scikit-video.org

Building pulse2percept
----------------------

From the root directory of the git repo, type:

.. code-block:: bash

    pip3 install -e .

Then from any Python console or script, try:

.. code-block:: python

    import pulse2percept as p2p

Building with make
------------------

pulse2percept provides a Makefile to simplify the build process.
If you followed the :ref:`above guide <install-source-prerequisites>` to
install ``make``, the following commands are available:

*  ``make``: Installs pulse2percept
*  ``make uninstall``: Uninstalls pulse2percept
*  ``make tests``: Installs pulse2percept and runs the test suite
*  ``make doc``: Installs pulse2percept and generates the documentation
*  ``make clean``: Cleans out all build files
*  ``make help``: Brings up this message

.. _install-uninstall:

Uninstalling
============

You can uninstall pulse2percept using pip:

.. code-block:: python

   pip3 uninstall pulse2percept

In addition, you may want to manually delete the GitHub folder containing all
the source code if you installed :ref:`from source <install-source>`.

.. _install-troubleshooting:

Troubleshooting
===============

I'm getting an error in fast_retina.pyx when installing with pip on Windows
---------------------------------------------------------------------------

Early builds of pulse2percept 0.4 mistakingly omitted the Windows binary
for the Cython-dependent ``fast_retina`` module (see :issue:`88`).
The solution is to either pip install :ref:`a later version <install-release>`,
or to :ref:`build from source <install-source>`.

.. note::

   Still having trouble? Please `open an issue`_ on GitHub and describe your
   problem there. Make sure to mention your platform and whether you are
   installing using pip or from source.

.. _open an issue: https://github.com/uwescience/pulse2percept/issues