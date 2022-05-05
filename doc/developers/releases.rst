.. _dev-releases:

=======================
Preparing a New Release
=======================

Before the Release
------------------

*  Make sure the version number is set correctly in "pulse2percept/version.py".
   In specific, the ``__version__`` variable should not contain a ``.dev0``
   substring.

*  Make sure the Release Notes in "doc/users/release_notes.rst" are up-to-date
   and complete.

Uploading the Release to PyPI
-----------------------------

pulse2percept wheels are built using GitHub Actions.

.. important::

    Before uploading the wheels to PyPI, make sure they work! You don't have to
    try all the wheels, but common problems are with Cython (and OpenMP) on
    Windows vs Unix.
    You can install a wheel via ``pip install <name>.wheel``

The following recipe will upload the files to TestPyPI:

.. code-block:: python

    cd pulse2percept

    # Clear out your 'dist' folder.
    rm -rf dist
    # Make a source distribution
    python setup.py sdist

    # Go and download your wheel files from wherever you put them. e.g. your CI
    # provider can be configured to store them for you. Put them all into the
    # 'dist' folder.
    wget ...

    # Or for TestPyPI:
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

Install the package from TestPyPI and make sure it works.
If everything looks good, upload the wheels to the real PyPI:

.. code-block:: python

    # Upload using 'twine' (you may need to 'pip install twine')
    twine upload dist/*

.. _cibuildwheel: https://github.com/joerick/cibuildwheel
.. _PR194: https://github.com/joerick/cibuildwheel/pull/194

Releasing the code on GitHub
----------------------------

*  Make a PR from ``master`` to ``stable``, call it "Release X.Y".
   Make sure to **squash and merge**, as every single commit on the stable
   branch should be a release.

*  Draft a new release on PR and tag it with "vX.Y".
   Upload all the wheels you downloaded as artifacts from Azure Pipelines
   above.

After the release
-----------------

*  Bump the version number in "pulse2percept/version.py", add back the ".0dev"
   appendix, and add an entry for the next release in 
   "doc/users/release_notes.rst". Then commit to ``master``.