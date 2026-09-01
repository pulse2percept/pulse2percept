#!/usr/bin/env python3
"""Install p2p inside the real Colab runtime and check what it broke.

This runs inside Google's published Colab image (see colab.yml). It is
Python rather than shell so it can be run and debugged locally against the
very same image a user gets in a notebook:

    docker run --rm -v "$PWD:/repo:ro" -w /tmp \
        us-docker.pkg.dev/colab-images/public/cpu-runtime:latest \
        python3 /repo/.github/scripts/colab_smoke.py pulse2percept \
            --source-root /repo

The working directory must not be the checkout: from there a source tree
shadows the installed package on sys.path and the whole run tests the wrong
thing (quietly).

"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# The distribution under test:
DIST_NAME = "pulse2percept"

# `Collecting numpy<3,>=2.0 (from pulse2percept)`, or with a chain of
# requirers: `(from scikit-image>=0.24->pulse2percept)`
COLLECTING = re.compile(
    r"^\s*Collecting\s+(?P<req>[^\s(]+)"
    r"(?:\s+\(from\s+(?P<chain>[^)]*)\))?\s*$"
)
# Leading distribution name of a requirement, e.g. `numpy` of `numpy<1.27`.
REQ_NAME = re.compile(r"^[A-Za-z0-9._-]+")
# Any version specifier:
HAS_SPECIFIER = re.compile(r"[<>=!~]")


def canonicalize(name: str) -> str:
    """Normalize a distribution name (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def run(cmd: list[str], *, check: bool = True,
        tee: bool = False) -> subprocess.CompletedProcess:
    """Run a command, echoing it so CI logs show exactly what happened"""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if not tee:
        return subprocess.run(cmd, check=check, text=True)
    proc = subprocess.run(cmd, check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    return proc


def parse_requirements(output: str) -> dict[str, list[tuple[str, str]]]:
    """Map each package pip resolved to the requirement(s) it was satisfying

    Returns ``{canonical name: [(requirement, requirer chain), ...]}``, where
    the chain is pip's own ``a->b`` rendering, nearest requirer first, and is
    empty for a package asked for on the command line.
    """
    found: dict[str, list[tuple[str, str]]] = {}
    for line in output.splitlines():
        match = COLLECTING.match(line)
        if not match:
            continue
        req = match.group("req")
        name = REQ_NAME.match(req)
        if not name:
            continue
        entry = (req, (match.group("chain") or "").strip())
        bucket = found.setdefault(canonicalize(name.group(0)), [])
        if entry not in bucket:
            bucket.append(entry)
    return found


def explain(names: list[str],
            requirements: dict[str, list[tuple[str, str]]]) -> list[str]:
    """Say which requirement pip was resolving for each of ``names``."""
    rows: list[str] = []
    for name in names:
        reqs = requirements.get(name)
        if not reqs:
            # pip only prints `Collecting` for packages it (re)installs by
            # name; a version can also move as a side effect of backtracking.
            rows.append(f"{name}: no requirement line from pip - most likely "
                        f"resolver backtracking to satisfy another pin")
            continue
        for req, chain in reqs:
            who = chain.replace("->", " -> ") if chain else "the install target"
            rows.append(f"{name}: required by {who} as `{req}`")
            # Whoever asked for it first is the one holding the constraint.
            immediate = REQ_NAME.match(chain.split("->")[0].strip()) if chain else None
            specifier = req[len(REQ_NAME.match(req).group(0)):]
            if (immediate and canonicalize(immediate.group(0)) == DIST_NAME
                    and HAS_SPECIFIER.search(specifier)):
                rows.append(f"    ^ that bound is declared by {DIST_NAME} "
                            f"itself -- ours to fix, in pyproject.toml")
    return rows


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
    """Whatever `pip check` complains about, as a set of lines

    Colab's image routinely ships with dependency conflicts of its own, so 
    only conflicts that appear because of our install are important.
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
    # isolation left on
    install = run(
        [sys.executable, "-m", "pip", "install", "--root-user-action=ignore", args.spec],
        check=False,
        tee=True,
    )
    if install.returncode != 0:
        print(
            f"\nFAILED:\n  - `pip install {args.spec}` failed in the Colab "
            "runtime (pip's own error is above). This is what a notebook user "
            "would see."
        )
        return 1

    after = freeze()

    changed_names = [
        name for name in sorted(before.keys() & after.keys())
        if before[name] != after[name] and name not in tolerated
    ]
    changed = [f"{name}: {before[name]} -> {after[name]}"
               for name in changed_names]
    removed = sorted(before.keys() - after.keys() - tolerated)
    added = sorted(after.keys() - before.keys())

    report("Added by the install (expected)", added)
    report("Changed versions (must be empty)", changed)
    report("Removed (must be empty)", removed)

    # Only conflicts we introduced count; Colab's own are not our problem.
    new_conflicts = sorted(pip_check() - conflicts_before)
    report("New dependency conflicts (must be empty)", new_conflicts)

    if changed_names or removed:
        report("Why those versions moved",
               explain(changed_names + removed,
                       parse_requirements(install.stdout or "")))

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

    # Hand off to the installed-package verifier:
    verify = [sys.executable, str(Path(__file__).resolve().parent / "check_install.py")]
    if args.source_root:
        verify += ["--source-root", args.source_root]
    return run(verify, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
