# scripts/verify_runtime_deps.py
# Validate runtime deps from installed metadata (no duplication of pyproject).
import sys, importlib, re
from importlib import metadata
from packaging.requirements import Requirement
from packaging.version import Version
from packaging.specifiers import SpecifierSet

PKG = "pulse2percept"

# Minimal overrides for import name mismatches (PyPI -> module)
IMPORT_NAME_OVERRIDES = {
    "scikit-image": "skimage",
    "imageio-ffmpeg": "imageio_ffmpeg",
    # add more if you truly depend on one with a different import name
}

def _import_name(dist_name: str) -> str:
    return IMPORT_NAME_OVERRIDES.get(dist_name, dist_name.replace("-", "_"))

def main() -> None:
    try:
        dist = metadata.distribution(PKG)
    except metadata.PackageNotFoundError:
        print(f"{PKG} is not installed in this environment.", file=sys.stderr)
        sys.exit(1)

    requires = dist.requires or []
    failures = []

    for line in requires:
        req = Requirement(line)

        # Skip extras (e.g., something; extra == "dev")
        if req.marker and 'extra' in str(req.marker):
            continue

        # Evaluate environment markers (e.g., python_version >= "3.12")
        if req.marker and not req.marker.evaluate():
            continue

        dist_name = req.name
        mod_name = _import_name(dist_name)

        # Try to import module to confirm presence
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            failures.append(f"{dist_name}: import '{mod_name}' failed ({e})")
            continue

        # Try to get version from either module or installed dist
        ver = getattr(mod, "__version__", None)
        if ver is None:
            try:
                ver = metadata.version(dist_name)
            except metadata.PackageNotFoundError:
                # some packages don’t expose a dist with the same name; best effort
                ver = "0+unknown"

        print(f"{dist_name} (module {mod_name}) == {ver}")

        # Enforce version spec (if present)
        if req.specifier:
            try:
                if Version(ver) not in req.specifier:
                    failures.append(f"{dist_name}=={ver} !∈ {req.specifier}")
            except Exception as e:
                failures.append(f"{dist_name}: version/spec check error ({e})")

    if failures:
        print("Dependency validation failures:\n  - " + "\n  - ".join(failures))
        sys.exit(1)

    print("All runtime dependencies satisfy installed metadata.")

if __name__ == "__main__":
    main()
