#!/usr/bin/env python3
"""Install pulse2percept inside the real Colab runtime and check what it broke.

This runs *inside* Google's published Colab image (see colab.yml). It is
Python rather than shell so it can be run and debugged locally against the
very same image a user gets in a notebook:

    docker run --rm -v "$PWD:/repo:ro" -w /tmp \
        us-docker.pkg.dev/colab-images/public/cpu-runtime:latest \
        python3 /repo/.github/scripts/colab_smoke.py pulse2percept \
            --source-root /repo

The working directory must not be the checkout: from there a source tree
shadows the installed package on sys.path and the whole run tests the wrong
thing (quietly).

Why this exists
---------------
An install that fails outright is the easy case, and the regular test matrix
already catches it. The failure users actually hit in Colab is quieter: pip
resolves a dependency, upgrades numpy or scipy, and the TensorFlow / JAX /
PyTorch that Colab preinstalled against the old ABI stops importing -- or the
notebook starts demanding a runtime restart. A clean virtualenv has nothing
preinstalled, so it structurally cannot reproduce that. So we install into the
real thing and compare `pip freeze` either side.

Packages that appear are fine. Packages that *change version* are not.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def canonicalize(name: str) -> str:
    """Normalize a distribution name (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, echoing it so CI logs show exactly what happened."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, text=True)


def pip(*args: str, capture: bool = False) -> str:
    """Invoke pip for the interpreter running this script."""
    cmd = [sys.executable, "-m", "pip", *args]
    if capture:
        out = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return out.stdout
    run(cmd)
    return ""


def freeze() -> dict[str, str]:
    """Snapshot the environment as {canonical name: version}."""
    installed: dict[str, str] = {}
    for line in pip("freeze", "--all", capture=True).splitlines():
        line = line.strip()
        # Skip blanks, comments, editable installs and bare VCS/URL entries
        # that carry no comparable version.
        if not line or line.startswith(("#", "-e ", "-")):
            continue
        if "==" in line:
            name, _, version = line.partition("==")
        elif " @ " in line:
            name, _, version = line.partition(" @ ")
        else:
            continue
        installed[canonicalize(name)] = version.strip()
    return installed


def pip_check() -> set[str]:
    """Whatever `pip check` complains about, as a set of lines.

    Colab's image routinely ships with dependency conflicts of its own, so an
    absolute `pip check` result is not a signal. Only conflicts that appear
    *because of* our install are.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pip", "check"], text=True, capture_output=True
    )
    if out.returncode == 0:
        # Exit 0 means no conflicts; the stdout is a success message, not a
        # finding, and must not end up in the diff as a phantom conflict.
        return set()
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def report(title: str, rows: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        print(f"  {row}")
    if not rows:
        print("  (none)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "spec",
        help='what to hand pip, e.g. "pulse2percept" or "git+https://…@<sha>"',
    )
    parser.add_argument(
        "--allow-change",
        action="append",
        default=[],
        metavar="NAME",
        help="tolerate a version change in this package (repeatable)",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="repo checkout, used to work out which C extensions to expect",
    )
    parser.add_argument(
        "--require-wheel",
        action="store_true",
        help="fail unless a prebuilt wheel exists for this interpreter",
    )
    args = parser.parse_args()
    tolerated = {canonicalize(n) for n in args.allow_change}

    print(f"Python     : {sys.version}")
    print(f"Executable : {sys.executable}")

    before = freeze()
    conflicts_before = pip_check()
    print(f"\nColab ships {len(before)} packages before we touch anything.")
    if conflicts_before:
        print(f"({len(conflicts_before)} pre-existing pip conflicts, not ours)")

    # No wheel for Colab's interpreter means every user pays for a source
    # build: minutes of compiling, and a hard failure if the image ever stops
    # shipping a compiler. Asking pip resolves this against Colab's *actual*
    # Python and platform, so nothing here needs updating when Colab moves.
    if args.require_wheel:
        probe = run(
            [
                sys.executable, "-m", "pip", "install",
                "--only-binary=:all:", "--dry-run", "--quiet", args.spec,
            ],
            check=False,
        )
        if probe.returncode != 0:
            print(
                f"\nFAILED:\n  - no prebuilt wheel for {args.spec} on this "
                f"interpreter ({sys.implementation.cache_tag}). Colab users "
                "would fall back to building from source."
            )
            return 1
        print("A prebuilt wheel is available for Colab's interpreter.")

    # Install exactly the way a notebook user would: plain pip, build
    # isolation left on. Deliberately *not* --no-build-isolation, which would
    # test a path no Colab user is on.
    install = run(
        [sys.executable, "-m", "pip", "install", "--root-user-action=ignore", args.spec],
        check=False,
    )
    if install.returncode != 0:
        print(
            f"\nFAILED:\n  - `pip install {args.spec}` failed in the Colab "
            "runtime (pip's own error is above). This is what a notebook user "
            "would see."
        )
        return 1

    after = freeze()

    changed = [
        f"{name}: {before[name]} -> {after[name]}"
        for name in sorted(before.keys() & after.keys())
        if before[name] != after[name] and name not in tolerated
    ]
    removed = sorted(before.keys() - after.keys() - tolerated)
    added = sorted(after.keys() - before.keys())

    report("Added by the install (expected)", added)
    report("Changed versions (must be empty)", changed)
    report("Removed (must be empty)", removed)

    # Only conflicts we introduced count; Colab's own are not our problem.
    new_conflicts = sorted(pip_check() - conflicts_before)
    report("New dependency conflicts (must be empty)", new_conflicts)

    problems = []
    if changed:
        problems.append(
            f"{len(changed)} package(s) Colab preinstalled changed version. "
            "This is what breaks TensorFlow/JAX/PyTorch in a notebook."
        )
    if removed:
        problems.append(f"{len(removed)} package(s) were removed: {removed}")
    if new_conflicts:
        problems.append(f"{len(new_conflicts)} new dependency conflict(s).")

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nInstall left Colab's preinstalled packages untouched.")

    # Hand off to the installed-package verifier, found next to this file so
    # the mount point is not baked in. It is invoked as a script path (not
    # `python -c`), which keeps the current directory off sys.path.
    verify = [sys.executable, str(Path(__file__).resolve().parent / "check_install.py")]
    if args.source_root:
        verify += ["--source-root", args.source_root]
    return run(verify, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
