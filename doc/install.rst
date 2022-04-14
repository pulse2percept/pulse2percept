.. _install:

============
Installation
============

Because pulse2percept is in active development, there are at least two versions
to choose from:

-  ``stable``: Our :ref:`stable release <install-release>` can be installed via
   pip (see `docs <https://pulse2percept.readthedocs.io/en/stable>`_,
   `code <https://github.com/pulse2percept/pulse2percept/tree/stable>`_).
-  ``latest``: Our bleeding-edge version must be installed 
   :ref:`from source <install-source>`, and may contain experimental features
   (see `docs <https://pulse2percept.readthedocs.io/en/latest/>`_,
   `code <https://github.com/pulse2percept/pulse2percept/tree/master>`_).

.. note::

    Having trouble with the installation?
    Please refer to our :ref:`Troubleshooting Guide <install-troubleshooting>`.

.. _install-python:

Installing Python
=================

Before getting started, you will need to install Python on your computer.
You can check if Python is already installed by typing ``python --version`` in
a terminal or command prompt.

If you don't have Python, you have several options:

- If you're unsure where to start, check out the `Python Wiki`_.
- `Python Anaconda`_ (good but slow in 2020): comes with the conda package
  manager and a range of scientific software pre-installed (NumPy, SciPy,
  Matplotlib, etc.).
- `Python Miniconda`_ (fast but minimal): comes with the conda package manager
  but nothing else.

pulse2percept supports these Python versions:

+----------------------+-----+-----+-----+-----+-----+-----+-----+-----+
|        Python        |3.10 | 3.9 | 3.8 | 3.7 | 3.6 | 3.5 | 3.4 | 2.7 |
+======================+=====+=====+=====+=====+=====+=====+=====+=====+
| p2p 0.8              | Yes | Yes | Yes | Yes |     |     |     |     |
+----------------------+-----+-----+-----+-----+-----+-----+-----+-----+
| p2p 0.7              |     | Yes | Yes | Yes | Yes |     |     |     |
+----------------------+-----+-----+-----+-----+-----+-----+-----+-----+
| p2p 0.6              |     |     | Yes | Yes | Yes | Yes |     |     |
+----------------------+-----+-----+-----+-----+-----+-----+-----+-----+
| p2p 0.5              |     |     |     | Yes | Yes | Yes |     |     |
+----------------------+-----+-----+-----+-----+-----+-----+-----+-----+
| p2p 0.4              |     |     |     |     |     | Yes | Yes | Yes |
+----------------------+-----+-----+-----+-----+-----+-----+-----+-----+

Following recent trends in the NumPy and SciPy community, we do not provide
wheels for 32-bit platforms (this includes all Unix platforms and Windows
starting with Python 3.10).
On these platforms, you will need to build pulse2percept from source.

On some platforms (e.g., macOS), you might also have to install pip yourself.
You can check if pip is installed on your system by typing ``pip --version``
in a terminal or command prompt.

If you don't have pip, do the following:

-  Download `get-pip.py`_ to your computer.
-  Open a terminal or command prompt and navigate to the directory containing
   ``get-pip.py``.
-  Run the following command:

   .. code-block:: bash

       python get-pip.py

.. _Python Anaconda: https://www.anaconda.com/distribution
.. _Python Wiki: https://wiki.python.org/moin/BeginnersGuide/Download
.. _Python Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _get-pip.py: https://bootstrap.pypa.io/get-pip.py

.. _install-release:

Installing the stable pulse2percept release
===========================================

After :ref:`installing Python <install-python>` above, the stable pulse2percept
release can be installed with pip:

.. code-block:: bash

    pip install pulse2percept

You can also install a specific version:

.. code-block:: bash

    pip install pulse2percept==0.6.0

Then from any Python console or script, try:

.. code-block:: python

    import pulse2percept as p2p

.. important::

    Make sure you are reading the right version of the documentation:
    https://pulse2percept.readthedocs.io/en/stable
    (<-- "stable", not "latest").

.. note::

    Find out what's new in the :ref:`Release Notes <users-release-notes>`.

.. _install-source:

Installing version |version| from source
========================================

.. _install-source-prerequisites:

Prerequisites
-------------

1.  **Python** (>= 3.7): Make sure to :ref:`install Python <install-python>`
    first.

2.  **XCode**: On macOS, make sure to install `Apple XCode`_.

3.  **Cython** (>= 0.28): pulse2percept relies on C extension modules for code
    acceleration. These require a C compiler, which on Unix platforms is
    already installed (``gcc``). However, on Windows you will have to install a
    compiler yourself:

    1.  Install **Build Tools for Visual Studio 2019** from the
        `Microsoft website`_.
        Note that the build tools for Visual Studio 2015 or 2017 should work as
        well (Python >= 3.7 requires C++ 14.X to be exact).
        Also note that you don't need to install Visual Studio itself.

    2.  `Install Cython`_:

        .. code-block:: bash

            pip install Cython

        If you get an error saying ``unable to find vcvarsall.bat``, then there
        is a problem with your Build Tools installation, in which case you
        should follow `this guide`_.

    .. warning::

        Some guides on the web tell you to install MinGW instead of Visual Studio.
        However, this is not recommended for 64-bit platforms.
        When in doubt, follow `this guide`_.

4.  **Git**: On Unix, you can install git from the `command line`_. On Windows,
    make sure to download `Git for Windows`_.

5.  **make** (optional): pulse2percept provides a Makefile to simplify the
    build process.
    ``make`` is part of `build-essentials`_ on Ubuntu, `XCode`_ on Mac OS X,
    and can be downloaded from `ezwinports`_ on Windows.

6.  **OpenMP** (optional): OpenMP is used to parallelize code written in Cython
    or C. OpenMP is part of the GCC compiler on Unix, and part of the
    `MinGW compiler <https://stackoverflow.com/a/38389181>`_ on Windows.
    Follow `these instructions 
    <https://dipy.org/documentation/1.0.0./installation/#openmp-with-osx>`_ 
    to get it to work on macOS.

.. _Apple XCode: https://developer.apple.com/xcode
.. _Microsoft website: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
.. _Install Cython: https://cython.readthedocs.io/en/latest/src/quickstart/install.html
.. _this guide: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
.. _command line: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
.. _Git for Windows: https://gitforwindows.org
.. _build-essentials: https://stackoverflow.com/questions/11934997/how-to-install-make-in-ubuntu
.. _XCode: https://developer.apple.com/support/xcode
.. _ezwinports: https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058#make

Dependencies
------------

.. include:: ../README.rst
   :start-line: 103
   :end-line: 155

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

        pip install -r requirements.txt

    This includes Cython. If you are on Windows, you will also need a suitable
    C compiler (see :ref:`Prerequisites <install-source-prerequisites>` above).

    If you plan on :ref:`contributing to pulse2percept <dev-contributing>`,
    you should also install all developer dependencies listed in
    ``requirements-dev.txt``:

    .. code-block:: bash

       pip install -r requirements-dev.txt

.. _pulse2percept on GitHub: https://github.com/pulse2percept/pulse2percept
.. _GitHub account: https://help.github.com/articles/signing-up-for-a-new-github-account

Building pulse2percept
----------------------

Assuming you are still in the root directory of the git clone, type
(note the ``.``):

.. code-block:: bash

    pip install -e .

Then from any Python console or script, try:

.. code-block:: python

    import pulse2percept as p2p

.. important::

    Make sure you are reading the right version of the documentation:
    https://pulse2percept.readthedocs.io/en/latest
    (<-- "latest", not "stable").

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
*  ``make help``: Prints a help message with this menu of options

.. _install-upgrade:

Upgrading pulse2percept
=======================

If you have previously installed pulse2percept, but wish to upgrade to the
newest release, you have two options.

To upgrade to the newest stable release, use the ``-U`` option with pip:

.. code-block:: bash

    pip install -U pulse2percept

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

   pip uninstall pulse2percept

This works for both stable and latest releases.

In addition, if you installed :ref:`from source <install-source>`, you may want
to manually delete the directory where you cloned the git repository that
contains all the source code.

.. _install-troubleshooting:

Troubleshooting
===============

Python ImportError: No module named pulse2percept
-------------------------------------------------

This is usually an issue related to ``$PATH``, the environment variable that
keeps track of all locations where pip should be looking for pulse2percept.
Chances are that pip installed pulse2percept somewhere outside of ``$PATH``.

You can check the installation location:

.. code-block:: python

   pip show pulse2percept

Then add the specificed location to ``$PATH``; see `PATH on Windows`_, 
`PATH on macOS`_, `PATH on Linux`_.

.. _PATH on Windows: https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/
.. _PATH on macOS: https://www.architectryan.com/2012/10/02/add-to-the-path-on-mac-os-x-mountain-lion/
.. _PATH on Linux: https://linuxize.com/post/how-to-add-directory-to-path-in-linux/

Error: numpy.ufunc size changed, may indicate binary incompatibility
--------------------------------------------------------------------

This issue may arise when one of the p2p dependencies was compiled using an 
older NumPy version. Upgrading to the latest NumPy version should fix the 
issue:

.. code-block:: python

  pip install -U numpy

.. note::

   Still having trouble? Please `open an issue`_ on GitHub and describe your
   problem there. Make sure to mention your platform and whether you are
   installing using pip or from source.

.. _open an issue: https://github.com/pulse2percept/pulse2percept/issues
