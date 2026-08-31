#!/usr/bin/env python3
"""Verify an *installed* pulse2percept, beyond what `import pulse2percept` proves.

Run this from a directory that is not the repo root. It answers four things a
plain import does not:

1. Are we testing the installed package, or did a source checkout on sys.path
   shadow it? (Running a smoke test from the repo root silently tests the
   checkout, which has no compiled extensions in it.)
2. Did every C extension load as a real binary, or did something quietly fall
   back to pure Python? A broken extension that degrades silently is worse
   than one that raises, because the install looks healthy.
3. Do the installed dependencies satisfy the versions the distribution
   metadata asks for?
4. Does a model actually build?

Usage:
    python check_install.py [--source-root /path/to/repo]

--source-root is optional. When given, the set of C extensions to expect is
derived from the .pyx files in that checkout, so adding a new .pyx extends
this check automatically rather than needing a hardcoded list updated.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
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
    """Confirm we resolved the installed package, not a source checkout."""
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
        # Not fatal on its own: an editable install legitimately resolves
        # outside site-packages. The source-root check above is the one that
        # catches the real mistake.
        print(f"note: {origin} is outside site-packages (editable install?)")
    return []


def expected_extensions(source_root: Path | None) -> list[str]:
    """Module names for every Cython extension, derived from the .pyx files."""
    if source_root is None:
        return []
    root = Path(source_root).resolve()
    modules = []
    for pyx in sorted(root.glob(f"{PKG}/**/*.pyx")):
        modules.append(".".join(pyx.relative_to(root).with_suffix("").parts))
    return modules


def check_extensions(modules: list[str]) -> list[str]:
    """Every extension must load, and must load from a compiled binary."""
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
    """Every declared runtime dep must be importable and satisfy its specifier."""
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

        # Trust installed metadata over a module's __version__ attribute; the
        # two can disagree, and metadata is what pip resolved against.
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
    """The install is only useful if a model actually builds."""
    print("\nBuilding a model:")
    try:
        from pulse2percept.implants import ArgusII
        from pulse2percept.models import ScoreboardModel

        # This runs against released versions too, and 0.10.0 renamed the
        # grid spacing parameter `xystep` -> `step`. Ask the installed model
        # which spelling it takes rather than assuming the current one.
        # `implant` is passed by keyword because that spelling works both
        # before and after 0.11.0 made it a required positional argument.
        probe = ScoreboardModel(implant=ArgusII())
        spacing = "step" if hasattr(probe, "step") else "xystep"
        model = ScoreboardModel(implant=ArgusII(), xrange=(-4, 4),
                                yrange=(-4, 4), **{spacing: 1})
        percept = model.predict_percept({e: 1 for e in ("A1", "F10")})
        if percept is None or percept.data.size == 0:
            return ["ScoreboardModel produced an empty percept"]
        print(f"  ScoreboardModel -> percept {percept.data.shape}, ok")
    except Exception as exc:  # noqa: BLE001
        return [f"model build failed ({exc.__class__.__name__}: {exc})"]
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
        # Everything downstream would be testing the wrong package.
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
