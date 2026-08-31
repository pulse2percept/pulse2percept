.. _install:

============
Installation
============

pulse2percept requires Python 3.11 or newer.

Stable release
==============

Install the latest stable release from PyPI with ``pip``:

.. code-block:: bash

    pip install pulse2percept

If you use `uv`_, the equivalent command is:

.. code-block:: bash

    uv pip install pulse2percept

For a uv-managed project, add pulse2percept as a dependency with:

.. code-block:: bash

    uv add pulse2percept

The installer resolves NumPy and the other runtime dependencies automatically.

.. _uv: https://docs.astral.sh/uv/


Latest development version
==========================

To install the current development version directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/pulse2percept/pulse2percept

or with ``uv``:

.. code-block:: bash

    uv pip install git+https://github.com/pulse2percept/pulse2percept

This installs the latest code from the ``master`` branch. Unlike a PyPI wheel,
a GitHub installation builds pulse2percept locally, so it requires Git and a
working C compiler.

If you plan to modify pulse2percept itself, see the
:doc:`Developer Guide <developers/contributing>` instead. It covers cloning the
repository, editable installs, tests, and the contribution workflow.


Compatibility
=============

.. include:: ../README.rst
    :start-after: .. compat-begin
    :end-before: .. compat-end


Upgrading and uninstalling
==========================

Upgrade to the latest stable release with:

.. code-block:: bash

    pip install -U pulse2percept

or:

.. code-block:: bash

    uv pip install -U pulse2percept

To uninstall:

.. code-block:: bash

    pip uninstall pulse2percept

or:

.. code-block:: bash

    uv pip uninstall pulse2percept


Troubleshooting
===============

Unsupported Python version
--------------------------

pulse2percept requires Python 3.11 or newer. Check the interpreter you are
using with:

.. code-block:: bash

    python --version

If you have multiple Python installations, make sure you are installing into
the environment you intend to use.


Installed, but Python cannot import pulse2percept
-------------------------------------------------

The most common cause is installing into a different Python environment from
the one running your script or notebook.

Check which installation Python sees:

.. code-block:: bash

    python -c "import pulse2percept as p2p; print(p2p.__version__); print(p2p.__file__)"

If that command fails, compare:

.. code-block:: bash

    python -m pip show pulse2percept
    python -m pip --version

Both commands should refer to the Python environment you intend to use.


GitHub installation fails
-------------------------

Installing directly from GitHub requires Git:

.. code-block:: bash

    git --version

It also builds the Cython extensions locally, so a working C compiler is
required.

On Windows, install `Build Tools for Visual Studio`_ and select
``Desktop development with C++``.

On macOS, source builds may require OpenMP support. If the build fails while
linking OpenMP, you can build without OpenMP acceleration:

.. code-block:: bash

    P2P_DISABLE_OPENMP=1 pip install git+https://github.com/pulse2percept/pulse2percept

or:

.. code-block:: bash

    P2P_DISABLE_OPENMP=1 uv pip install git+https://github.com/pulse2percept/pulse2percept

.. _Build Tools for Visual Studio:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/


``Failed building wheel`` during a normal PyPI install
------------------------------------------------------

A normal installation on a supported platform should use a prebuilt wheel. If
``pip install pulse2percept`` unexpectedly tries to compile the package, first
make sure your Python version and platform are supported and update ``pip``:

.. code-block:: bash

    python -m pip install -U pip

Then try the installation again.

If the problem persists, please `open an issue`_ and include:

* your operating system;
* Python version;
* pulse2percept version;
* installer and version (for example, ``pip --version`` or ``uv --version``);
* the complete installation error.

.. _open an issue: https://github.com/pulse2percept/pulse2percept/issues
