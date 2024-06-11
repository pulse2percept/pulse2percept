.. _install:

============
Installation
============

Because pulse2percept is in active development, there are at least three versions
to choose from:

-  ``stable``: Our :ref:`stable release <install-release>` can be installed
   with:

   .. code-block:: bash

       pip install pulse2percept

   It may not have all the latest features, but the ones it has are well
   tested.
   You can read the corresponding docs
   `here <https://pulse2percept.readthedocs.io/en/stable>`_.

   If you have previously installed p2p this way and wish to upgrade it to
   the latest *stable* version, use the ``-U`` option with pip:

   .. code-block:: bash

       pip install -U pulse2percept

-  ``latest``: Our bleeding-edge version can be installed with:

   .. code-block:: bash

       pip install git+https://github.com/pulse2percept/pulse2percept

   This will pull the latest code from the ``master`` branch of our
   `GitHub repo <https://github.com/pulse2percept/pulse2percept/tree/master>`_.
   You can read the corresponding docs
   `here <https://pulse2percept.readthedocs.io/en/stable>`_.

-  Developer version: Finally, if you wish to contribute to the p2p code
   base, you will need to install p2p :ref:`from source <install-source>`.

.. note::

    Having trouble with the installation?
    Please refer to our :ref:`Troubleshooting Guide <install-troubleshooting>`.

.. _install-source:

Installing version |version| from source
========================================

If you want to contribute to the p2p code base, you will need
to install p2p from source.

Similarly, if your platform is not officially supported (and hence we do
not have a wheel for it), you will need to build p2p from source.
Following recent trends in the NumPy and SciPy community, we do not provide
wheels for 32-bit platforms (this includes all Unix platforms and Windows
starting with Python 3.10).

Prerequisites
-------------

Before getting started, you will need the following:

-  **Python**: You can check whether Python is already installed by typing
   ``python --version`` in a terminal or command prompt.

   .. include:: ../README.rst
      :start-line: 97
      :end-line: 113

   If you don't have Python, there are several options:

   -  If you're unsure where to start, check out the `Python Wiki`_.
   
   -  `Python Anaconda`_ (good but slow in 2020): comes with the conda package
      manager and a range of scientific software pre-installed (NumPy, SciPy,
      Matplotlib, etc.).

   -  `Python Miniconda`_ (fast but minimal): comes with the conda package manager
      but nothing else.

-  **pip**: On some platforms (e.g., macOS), you may have to install pip yourself.
   You can check if pip is installed on your system by typing ``pip --version``
   in a terminal or command prompt.

   If you don't have pip, do the following:

   -  Download `get-pip.py`_ to your computer.
   -  Open a terminal or command prompt and navigate to the directory containing
      ``get-pip.py``.
   -  Run the following command:

      .. code-block:: bash

          python get-pip.py

-  **NumPy**: Once you have Python and pip, simply open a terminal and type 
   ``pip install numpy``.

-  **Cython** (>= 0.28): pulse2percept relies on C extension modules for code
   acceleration. These require a C compiler, which on Unix platforms is
   already installed (``gcc``). However, on Windows you will have to install a
   compiler yourself:

   1.  Install **Build Tools for Visual Studio 2019** from the `Microsoft website`_.
       Note that the build tools for Visual Studio 2015 or 2017 should work as
       well (Python >= 3.7 requires C++ 14.X to be exact).
       Also note that you don't need to install Visual Studio itself.

   2.  `Install Cython <https://cython.readthedocs.io/en/latest/src/quickstart/install.html>`_:

       .. code-block:: bash

           pip install Cython

       If you get an error saying ``unable to find vcvarsall.bat``, then there
       is a problem with your Build Tools installation, in which case you
       should follow `this guide`_.

   .. warning::

       Some guides on the web tell you to install MinGW instead of Visual Studio.
       However, this is not recommended for 64-bit platforms.
       When in doubt, follow `this guide <https://github.com/cython/cython/wiki/CythonExtensionsOnWindows>`_.

-  **Git**: On Unix, you can install git from the
   `command line <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_. 
   On Windows, make sure to download `Git for Windows <https://gitforwindows.org>`_.

-  **OpenMP** (optional): OpenMP is used to parallelize code written in Cython
   or C. OpenMP is part of the GCC compiler on Unix, and part of the
   `MinGW compiler <https://stackoverflow.com/a/38389181>`_ on Windows.
   Follow `these instructions 
   <https://dipy.org/documentation/1.0.0./installation/#openmp-with-osx>`_ 
   to get it to work on macOS.

.. _Python Anaconda: https://www.anaconda.com/distribution
.. _Python Wiki: https://wiki.python.org/moin/BeginnersGuide/Download
.. _Python Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _get-pip.py: https://bootstrap.pypa.io/get-pip.py
.. _Microsoft website: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019

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

3.  Install all dependencies listed in ``requirements.txt`` by using the
    following command:

    .. code-block:: bash

        pip install -r requirements.txt

    This includes Cython. If you are on Windows, make sure you followed the steps
    outlined above to install a suitable C compiler (see 
    :ref:`Prerequisites <install-source-prerequisites>` above).

    If you plan on :ref:`contributing to pulse2percept <dev-contributing>`,
    you should also install all developer dependencies listed in
    ``requirements-dev.txt``:

    .. code-block:: bash

       pip install -r requirements-dev.txt

.. _pulse2percept on GitHub: https://github.com/pulse2percept/pulse2percept
.. _GitHub account: https://help.github.com/articles/signing-up-for-a-new-github-account

Building pulse2percept
----------------------

Assuming you are still in the root directory of the git clone, type the
following (note the ``.``):

.. code-block:: bash

    pip install -e .

Then from any Python console or script, try:

.. code-block:: python

    import pulse2percept as p2p

.. important::

    Make sure you are reading the right version of the documentation:
    https://pulse2percept.readthedocs.io/en/latest
    (<-- "latest", not "stable").

Keeping your fork up to date
----------------------------

Assuming you are working on your own fork, you may want to integrate new
developments from the master branch from time to time.

If you have never upgraded your code before, add a new 
`remote repository <https://help.github.com/articles/configuring-a-remote-for-a-fork>`_ 
named "upstream" (you need to do this only once):

.. code-block:: bash

    git remote add upstream https://github.com/pulse2percept/pulse2percept.git

Then type ``git branch`` to make sure you are on the right local branch.
Finally, you can `"sync" your fork <https://help.github.com/articles/syncing-a-fork/>`_
by grabbing the latest code from the pulse2percept master branch:

.. code-block:: bash

    git pull upstream master

.. _install-uninstall:

Uninstalling pulse2percept
==========================

You can uninstall pulse2percept using pip:

.. code-block:: python

   pip uninstall -y pulse2percept

This works for both stable and latest releases.

In addition, if you installed :ref:`from source <install-source>`, you may want
to manually delete the directory where you cloned the git repository that
contains all the source code.

.. _install-troubleshooting:

Troubleshooting
===============

Cannot install platform-specific wheel
--------------------------------------

Following recent trends in the NumPy and SciPy community, we do not provide
wheels for 32-bit platforms (this includes all Unix platforms and Windows
starting with Python 3.10).

The main reason is that p2p heavily depends on NumPy, SciPy, Matplotlib,
and Scikit-Image. Since these packages no longer provide wheels for 32-bit
platforms, we cannot either.

In this case, you will have to install p2p :ref:`from source <install-source>`.

If you are getting this error message for a supposedly supported platform,
please `open an issue`_ on GitHub.
 
Python ImportError: No module named pulse2percept
-------------------------------------------------

This is usually an issue related to ``$PATH``, the environment variable that
keeps track of all locations where pip should be looking for pulse2percept.
Chances are that pip installed pulse2percept somewhere outside of ``$PATH``.

You can check the installation location:

.. code-block:: python

   pip show pulse2percept

Then add the specified location to ``$PATH``; see `PATH on Windows`_, 
`PATH on macOS`_, `PATH on Linux`_.

.. _PATH on Windows: https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/
.. _PATH on macOS: https://www.architectryan.com/2012/10/02/add-to-the-path-on-mac-os-x-mountain-lion/
.. _PATH on Linux: https://linuxize.com/post/how-to-add-directory-to-path-in-linux/

Error: numpy.ufunc size changed, may indicate binary incompatibility
--------------------------------------------------------------------

This issue may arise with older p2p versions, or if one of the p2p dependencies
was compiled using an  older NumPy version.

Upgrading to the latest NumPy version should fix the issue:

.. code-block:: python

  pip install -U numpy

Then reinstall p2p according to the guide above.

.. note::

   Still having trouble? Please `open an issue`_ on GitHub and describe your
   problem there. Make sure to mention your platform and whether you are
   installing using pip or from source.

.. _open an issue: https://github.com/pulse2percept/pulse2percept/issues
