"""Remove authoring-machine metadata while preserving decoded texture pixels."""

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "src/retriever_src_kitchen/_scene"

for path in sorted(ROOT.rglob("*.png")):
    with Image.open(path) as original:
        pixels, mode, size = original.tobytes(), original.mode, original.size
        profile = original.info.get("icc_profile")
        dpi = original.info.get("dpi")
        clean = original.copy()
    clean.info.clear()
    options = {}
    if profile is not None:
        options["icc_profile"] = profile
    if dpi is not None:
        options["dpi"] = dpi
    clean.save(path, **options)
    with Image.open(path) as verified:
        assert (verified.mode, verified.size, verified.tobytes()) == (mode, size, pixels)
    print("Pixel-preserving metadata cleanup:", path.relative_to(ROOT))
