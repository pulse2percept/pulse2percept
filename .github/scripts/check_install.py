#!/usr/bin/env python3
"""Verify an *installed* p2p beyond `import pulse2percept`.

Run from a directory other than the repo root. Checks:

1. The installed package is imported, not a source checkout on sys.path
   (the checkout has no compiled extensions).
2. Every C extension loads from a compiled binary, not a pure-Python
   fallback.
3. Installed dependencies satisfy the distribution metadata.
4. A model builds.

Usage:
    python check_install.py [--source-root /path/to/repo]

--source-root is optional. When given, the expected C extensions are derived
from the .pyx files in that checkout.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import sys
from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

PKG = "pulse2percept"

# PyPI name -> import name, for the few that differ.
IMPORT_NAME_OVERRIDES = {
    "scikit-image": "skimage",
    "pillow": "PIL",
}


def import_name(dist_name: str) -> str:
    return IMPORT_NAME_OVERRIDES.get(dist_name, dist_name.replace("-", "_"))


def check_not_shadowed(source_root: Path | None) -> list[str]:
    """Return failures if the package resolves to the source checkout."""
    spec = importlib.util.find_spec(PKG)
    if spec is None or not spec.origin:
        return [f"{PKG} is not importable at all"]

    origin = Path(spec.origin).resolve()
    print(f"{PKG} resolves to: {origin}")

    if source_root is not None:
        root = Path(source_root).resolve()
        if root in origin.parents:
            return [
                f"{PKG} resolved to the source checkout at {origin}, not the "
                f"installed package. Run this from outside {root} -- as "
                "written the test never touches what was installed."
            ]

    if not any(part in ("site-packages", "dist-packages") for part in origin.parts):
        # Not fatal: an editable install resolves outside site-packages
        print(f"note: {origin} is outside site-packages (editable install?)")
    return []


def expected_extensions(source_root: Path | None) -> list[str]:
    """Return module names for every Cython extension, from the .pyx files."""
    if source_root is None:
        return []
    root = Path(source_root).resolve()
    modules = []
    for pyx in sorted(root.glob(f"{PKG}/**/*.pyx")):
        modules.append(".".join(pyx.relative_to(root).with_suffix("").parts))
    return modules


def check_extensions(modules: list[str]) -> list[str]:
    """Return failures for extensions that fail to import or are not compiled."""
    failures = []
    if not modules:
        print("\nNo .pyx sources given, skipping the compiled-extension check.")
        return failures

    print(f"\nChecking {len(modules)} compiled extension(s):")
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - want the reason in the log
            failures.append(f"{name}: failed to import ({exc.__class__.__name__}: {exc})")
            print(f"  {name}: FAILED to import")
            continue

        origin = getattr(mod, "__file__", "") or ""
        if not origin.endswith(tuple(EXTENSION_SUFFIXES)):
            failures.append(
                f"{name}: loaded from {origin!r}, which is not a compiled "
                "extension. A pure-Python fallback is masking a broken build."
            )
            print(f"  {name}: NOT COMPILED ({origin})")
        else:
            print(f"  {name}: ok ({Path(origin).name})")
    return failures


def check_dependencies() -> list[str]:
    """Return failures for runtime deps that fail to import or violate specifiers."""
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version

    try:
        dist = metadata.distribution(PKG)
    except metadata.PackageNotFoundError:
        return [f"{PKG} is not installed in this environment"]

    failures = []
    print("\nRuntime dependencies:")
    for line in dist.requires or []:
        req = Requirement(line)
        # Extras are not runtime requirements of the base install.
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue

        module = import_name(req.name)
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{req.name}: import {module!r} failed ({exc})")
            print(f"  {req.name}: IMPORT FAILED")
            continue

        # Use installed metadata (what pip resolved), not __version__:
        try:
            version = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            print(f"  {req.name}: imported, no distribution metadata to check")
            continue

        if not req.specifier:
            print(f"  {req.name} == {version}")
            continue

        try:
            ok = Version(version) in req.specifier
        except InvalidVersion:
            print(f"  {req.name} == {version} (unparseable, not checked)")
            continue

        print(f"  {req.name} == {version} {'ok' if ok else 'VIOLATES'} {req.specifier}")
        if not ok:
            failures.append(f"{req.name}=={version} does not satisfy {req.specifier}")
    return failures


def check_model_builds() -> list[str]:
    """Return failures from building small example models."""
    print("\nBuilding models:")
    return _check_scoreboard() + _check_prima_ho2018()


def _check_scoreboard() -> list[str]:
    """ArgusII + ScoreboardModel -> nonempty percept."""
    try:
        from pulse2percept.implants.retina import ArgusII
        from pulse2percept.models.retina import ScoreboardModel

        # Also runs against releases: 0.10.0 renamed `xystep` -> `step`.
        # Inspect the constructor, since 0.11 parameters live on components:
        params = inspect.signature(ScoreboardModel).parameters
        spacing = "step" if "step" in params else "xystep"
        model = ScoreboardModel(implant=ArgusII(), xrange=(-4, 4),
                                yrange=(-4, 4), **{spacing: 1})
        percept = model.predict_percept({e: 1 for e in ("A1", "F10")})
        if percept is None or percept.data.size == 0:
            return ["ScoreboardModel produced an empty percept"]
        print(f"  ScoreboardModel -> percept {percept.data.shape}, ok")
    except Exception as exc:  # noqa: BLE001
        return [f"ScoreboardModel failed ({exc.__class__.__name__}: {exc})"]
    return []


def _check_prima_ho2018() -> list[str]:
    """ImageStimulus -> PRIMAPivotal optical encoder -> Ho2018Model.

    Skipped below 0.11 (Ho2018Model added), since this script also runs
    against PyPI releases. Skips by version, because an ImportError on 0.11+
    is a real failure.
    """
    from packaging.version import InvalidVersion, Version

    try:
        installed = Version(metadata.version(PKG))
    except (metadata.PackageNotFoundError, InvalidVersion):
        installed = None
    if installed is not None and installed < Version("0.11.0.dev0"):
        print(f"  PRIMAPivotal + Ho2018Model: skipped, needs 0.11.0.dev0 or "
              f"newer (installed {installed})")
        return []

    try:
        import numpy as np

        from pulse2percept.implants.retina import PRIMAPivotal
        from pulse2percept.models.retina import Ho2018Model
        from pulse2percept.stimuli import ImageStimulus

        # Tiny grid and image: smoke test, not a numerical regression.
        model = Ho2018Model(PRIMAPivotal(), xrange=(-1, 1), yrange=(-1, 1),
                            step=0.5, verbose=False)
        stim = ImageStimulus(np.ones((4, 4), dtype=np.float32))
        percept = model.predict_percept(stim)
        if percept is None or percept.data.size == 0:
            return ["Ho2018Model produced an empty percept"]
        if not np.any(percept.data > 0):
            return ["Ho2018Model produced no positive brightness"]
        print(f"  PRIMAPivotal + Ho2018Model -> percept "
              f"{percept.data.shape}, ok")
    except Exception as exc:  # noqa: BLE001
        return [f"PRIMAPivotal + Ho2018Model failed "
                f"({exc.__class__.__name__}: {exc})"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=None,
        help="repo checkout, used to detect shadowing and to find .pyx files",
    )
    args = parser.parse_args()
    source_root = Path(args.source_root) if args.source_root else None

    print(f"Python: {sys.version}")
    print(f"cwd   : {Path.cwd()}")

    failures = check_not_shadowed(source_root)
    if failures:
        # Later checks would test the wrong package:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    import pulse2percept

    print(f"{PKG} version: {getattr(pulse2percept, '__version__', 'unknown')}")

    failures += check_extensions(expected_extensions(source_root))
    failures += check_dependencies()
    failures += check_model_builds()

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nAll install checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
