"""Read-only verification of the packaged scene snapshot."""

from hashlib import sha256
import json

from .api import scene_root


def verify_resources():
    root = scene_root()
    manifest = json.loads((root / "asset-manifest.json").read_text())
    missing, changed = [], []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            changed.append(name)
    return {"ok": not missing and not changed, "files": len(manifest["files"]),
            "missing": missing, "changed": changed,
            "simulation_tested": False}
