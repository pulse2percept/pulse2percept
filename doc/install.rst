.. _install:

============
Installation
============

pulse2percept is in active development.
To get the latest stable version, we recommend installing our
:ref:`latest release <install-release>`.
If you wish to contribute to the project or require
the bleeding-edge version, you will want to install
:ref:`from source <install-source>` instead.

.. note::

    Having trouble with the installation?
    Please refer to our :ref:`Troubleshooting Guide <install-troubleshooting>`.

.. _install-prerequisites:

Prerequisites
=============

*  **Python** (3.5 - 3.7): Before getting started, you will need to have to
   `install Python`_ on your computer. Check if Python is installed on your
   system by typing ``python --version`` in a terminal or command prompt.

   .. important::

       pulse2percept 0.4.3 was the last release to support Python 2.7 and 3.4.
       pulse2percept 0.5+ will require **Python 3.5 - 3.7**.

*  **Cython** (>= 0.28): pulse2percept relies on C extension modules for code
   acceleration. These require a C compiler, which on Unix platforms is
   already installed (``gcc``). However, on Windows you will have to install a
   compiler yourself:

   1.  Install **Build Tools for Visual Studio 2019** from the
       `Microsoft website`_.
       Note that the build tools for Visual Studio 2015 or 2017 should work as
       well (Python >= 3.5 requires C++ 14.X to be exact).
       Also note that you don't need to install Visual Studio itself.

   2.  `Install Cython`_:

       .. code-block:: bash

           pip3 install Cython

       If you get an error saying ``unable to find vcvarsall.bat``, then there
       is a problem with your Build Tools installation, in which case you
       should follow `this guide`_.

   .. warning::

       Some guides on the web tell you to install MinGW instead of Visual Studio.
       However, this is not recommended for 64-bit platforms.
       When in doubt, follow `this guide`_.

.. _install Python: https://wiki.python.org/moin/BeginnersGuide/Download
.. _Microsoft website: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
.. _Install Cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
.. _this guide: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows

.. _install-release:

Installing a release
====================

After taking care of the :ref:`prerequisites <install-prerequisites>`,
the latest pulse2percept release can be installed using pip:

.. code-block:: bash

    pip3 install pulse2percept

Then from any Python console or script, try:

.. code-block:: python

    import pulse2percept as p2p

.. _install-source:

Installing from source
======================

.. _install-source-prerequisites:

Prerequisites
-------------

**Git**: On Unix, you can install git from the `command line`_. On Windows,
make sure to download `Git for Windows`_.

.. _command line: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
.. _Git for Windows: https://gitforwindows.org

Dependencies
------------

**pulse2percept 0.4.3 was the last version to support Python 2.7 and 3.4. pulse2percept 0.5+ require Python 3.5 or newer.**

pulse2percept requires:

- Python (3.5 - 3.7)
- Cython (>= 0.28)
- NumPy (>= 1.9)
- SciPy (>= 1.0)
- Matplotlib (>= 2.0)
- JobLib (>= 0.11)

Optional packages:

- scikit-image for image functionality in the ``files`` module.
- scikit-video for video functionality in the ``files`` module. You will also need an FFMPEG codec (see next bullet point).
- ffmpeg codec if you're on Windows and want to use functions in the ``files`` module.
- Dask for parallel processing (a joblib alternative). Use ``conda`` to install.
- Numba for just-in-time compilation. Use ``conda`` to install.
- Pytest to run the test suite.

All required packages are listed in requirements.txt in the root directory of the git repository, and can be installed with the following command:

.. code-block:: bash

    git clone https://github.com/pulse2percept/pulse2percept.git
    cd pulse2percept
    pip3 install -r requirements.txt

Obtaining the latest code from GitHub
-------------------------------------

1.  Go to `pulse2percept on GitHub`_ and click on "Fork" in the top-right
    corner (you will need a `GitHub account`_ for this).
    This will allow you to work on your own copy of the code
    (``https://github.com/<username>/pulse2percept``)
    and contribute changes later on.

2.  Clone the repository to get a local copy on your computer:

    .. code-block:: bash

        git clone https://github.com/<username>/pulse2percept.git
        cd pulse2percept

    Make sure to replace ``<username>`` above with your actual GitHub user
    name.

    .. note::

        A "fork" is basically a "remote copy" of a GitHub repository; i.e.,
        creating "https://github.com/<username>/pulse2percept.git" from
        "https://github.com/pulse2percept/pulse2percept.git".

        A "clone" is basically a "local copy" of your GitHub repository; i.e.,
        creating a local "pulse2percept" directory (including all the git
        machinery and history) from
        "https://github.com/<username>/pulse2percept.git".

3.  Install all dependencies listed in ``requirements.txt``:

    .. code-block:: bash

        pip3 install -r requirements.txt

    This includes Cython. If you are on Windows, you will also need a suitable
    C compiler (see :ref:`Prerequisites <install-prerequisites>` above).

    If you plan on :ref:`contributing to pulse2percept <dev-contributing>`,
    you should also install all developer dependencies listed in
    ``requirements-dev.txt``:

    .. code-block:: bash

       pip3 install -r requirements-dev.txt

.. _pulse2percept on GitHub: https://github.com/pulse2percept/pulse2percept
.. _GitHub account: https://help.github.com/articles/signing-up-for-a-new-github-account

Building pulse2percept
----------------------

Assuming you are still in the root directory of the git clone, type:

.. code-block:: bash

    pip3 install -e .

Then from any Python console or script, try:

.. code-block:: python

    import pulse2percept as p2p

.. _install-upgrade:

Upgrading pulse2percept
=======================

If you have previously installed pulse2percept, but wish to upgrade to the
latest version, you have two options.

To upgrade to the latest stable release, use the ``-U`` option with pip:

.. code-block:: bash

    pip3 install -U pulse2percept

To upgrade to the bleedingest-edge version, navigate to the directory where you
cloned the git repository. If you have never upgraded your code before, add
a new `remote repository`_ named "upstream" (you need to do this only once):

.. code-block:: bash

    git remote add upstream https://github.com/pulse2percept/pulse2percept.git

Then you `sync your fork`_ by grabbing the latest code from the pulse2percept
master branch:

.. code-block:: bash

    git pull upstream master

.. _remote repository: https://help.github.com/articles/configuring-a-remote-for-a-fork
.. _sync your fork: https://help.github.com/articles/syncing-a-fork/

.. _install-uninstall:

Uninstalling pulse2percept
==========================

You can uninstall pulse2percept using pip:

.. code-block:: python

   pip3 uninstall pulse2percept

In addition, if you installed :ref:`from source <install-source>`, you may want
to manually delete the directory where you cloned the git repository that
contains all the source code.

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

.. _open an issue: https://github.com/pulse2percept/pulse2percept/issues
