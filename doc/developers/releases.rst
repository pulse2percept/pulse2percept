.. _dev-releases:

=======================
Preparing a New Release
=======================

Before the Release
------------------

*  Make a new PR to ``master`` with up-to-date Release Notes
   "doc/users/release_notes.rst". You might have to go through all the past PRs
   to make sure all work is adequately represented.

*  Make sure all new contributing authors are listed in the AUTHORS file.

*  Make sure the version number is set correctly in "pulse2percept/pyproject.toml".
   In specific, the ``version`` variable should not contain a ``.dev0`` substring.

Uploading the Release to PyPI
-----------------------------

pulse2percept wheels are built using GitHub Actions ``wheels.yml``.

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

Make sure the ``pip install`` command works e.g. on Google Colab.
If not, you will need to fix the wheels and upload them under a new 
patch number (e.g., v0.9.1 instead of v0.9.0).

.. _cibuildwheel: https://github.com/joerick/cibuildwheel
.. _PR194: https://github.com/joerick/cibuildwheel/pull/194

Releasing the code on GitHub
----------------------------

*  Make ``stable`` identical to ``master`` with a reset + force push:

   .. code-block:: bash

       git checkout stable
       git reset --hard master
       git push origin stable --force

   This is cleaner than squash and merge. Either way, it's important
   for ReadTheDocs that every single commit on the ``stable`` branch 
   matches a release.

*  Draft a new release on PR and tag it with "vX.Y".
   Upload all the wheels you downloaded as artifacts from GitHub Actions
   above.

After the release
-----------------

*  Bump the version number in "pulse2percept/pyproject.toml", add back the ".0dev"
   appendix, and add an entry for the next release in 
   "doc/users/release_notes.rst". Then commit to ``master``.
