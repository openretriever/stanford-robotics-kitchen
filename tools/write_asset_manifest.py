"""Record a reviewed scene snapshot; never called implicitly by the runtime."""

from hashlib import sha256
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "src/retriever_src_kitchen/_scene"
files = {}
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "asset-manifest.json" or "__pycache__" in path.parts:
        continue
    data = path.read_bytes()
    files[path.relative_to(root).as_posix()] = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
(root / "asset-manifest.json").write_text(json.dumps({"schema": "src-kitchen.assets/1", "files": files}, indent=2) + "\n")
print(f"Recorded {len(files)} files, {sum(item['bytes'] for item in files.values()) / 1e6:.1f} MB")
