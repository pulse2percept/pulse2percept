.. _dev-releases:

=======================
Preparing a New Release
=======================

A release is prepared on a short-lived release branch, merged into ``master``,
tagged, and then mirrored to ``stable``. The tag is the canonical release
commit; wheels and the source distribution must be built from that commit.

Prepare the release
-------------------

Create a release branch from an up-to-date ``master``:

.. code-block:: bash

    git checkout master
    git pull
    git checkout -b release-X.Y

On the release branch:

* Update ``doc/users/release_notes.rst``.
* Set the final version in ``pyproject.toml`` (for example, ``0.10.0`` rather
  than ``0.10.0.dev0``).
* Run the test suite and build the documentation.

Open a PR from the release branch into ``master`` and merge it once CI passes.

Tag the release
---------------

Update your local ``master`` and tag the merged release commit:

.. code-block:: bash

    git checkout master
    git pull
    git tag -a vX.Y.Z -m "pulse2percept X.Y.Z"
    git push origin vX.Y.Z

The ``v*`` tag triggers the GitHub Actions wheel build. These tag-built wheels,
rather than wheels from an earlier PR or ``master`` build, are the release
artifacts.

Update ``stable`` to point at the same commit:

.. code-block:: bash

    git push origin vX.Y.Z:stable --force

After this step,

.. code-block:: text

    stable == vX.Y.Z == release commit

This keeps the ReadTheDocs ``stable`` branch tied to a released version even if
new commits are merged into ``master`` immediately afterward.

Build and test the artifacts
----------------------------

While GitHub Actions builds the wheels, create the source distribution from the
tagged commit:

.. code-block:: bash

    git checkout vX.Y.Z
    rm -rf dist
    python -m build --sdist

Download the wheels produced by the tag-triggered ``Wheels`` workflow and place
them alongside the source distribution in ``dist/``:

.. code-block:: text

    dist/
    ├── pulse2percept-X.Y.Z.tar.gz
    ├── pulse2percept-X.Y.Z-cp311-...whl
    ├── pulse2percept-X.Y.Z-cp312-...whl
    └── ...

Check the artifacts before uploading them:

.. code-block:: bash

    twine check dist/*

It is strongly recommended to upload to TestPyPI first:

.. code-block:: bash

    twine upload --repository testpypi dist/*

Install the package from TestPyPI and smoke-test it on at least one common
platform. Pay particular attention to Cython/OpenMP behavior on Windows,
Linux, and macOS.

If the artifacts are good, upload the same files to PyPI:

.. code-block:: bash

    twine upload dist/*

Once a version has been uploaded to PyPI, its files cannot be replaced. If a
release artifact is broken, fix the problem and publish a new patch release.

Publish the GitHub release
--------------------------

Create a GitHub Release for ``vX.Y.Z`` using the release notes. Attach the same
source distribution and wheels if desired.

After the release
-----------------

Return to ``master`` and prepare the next development version:

* Set the version in ``pyproject.toml`` to the next ``.dev0`` version.
* Add an empty section for the next release to
  ``doc/users/release_notes.rst``.
* Commit these changes to ``master``.

For example, after releasing ``0.10.0``, development continues as
``0.11.0.dev0``.
