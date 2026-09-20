"""Import a curated scene snapshot and make its robot references portable.

This is a one-time migration helper, not a runtime dependency or a live sync.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "src/retriever_src_kitchen/_scene"
FILES = (
    "build_scene.py", "integration.py", "export_scene.py", "simulation_visuals.py",
    "motion_demo.py", "organizer_search.py", "monitor_branding.py",
    "seasoning_demo.py", "drawer_motion.py", "drawer_inspection.py",
    "teaser.py", "teaser.html", "console.js", "console.css", "record.js",
    "build_mobile_aloha.py", "layout.json", "requirements.txt",
    "vendor/lucide.js", "vendor/html2canvas.js",
    "reference/stanford-robotics-center-logo.png",
    "output/kitchen.xml", "output/kitchen-sim.xml",
    "output/robots/manifest.json", "output/robots/mobile-aloha.glb",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--preview-support", type=Path, required=True)
    parser.add_argument("--preview-license", type=Path, required=True)
    parser.add_argument("--robot-assets", type=Path, required=True)
    parser.add_argument("--robot-license", type=Path, required=True)
    args = parser.parse_args()
    if SCENE.exists():
        raise SystemExit("Refusing to overwrite an existing snapshot")
    SCENE.mkdir(parents=True)

    def copy(relative):
        target = SCENE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source / relative, target)

    for relative in FILES:
        copy(relative)
    for folder in ("output/simulation-assets", "vendor/mobile_aloha"):
        for path in sorted((args.source / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                copy(path.relative_to(args.source))
    for path in sorted((args.source / "vendor/motion").rglob("*")):
        if path.is_file() and (path.suffix == ".py" or path.name in ("LICENSE", "METADATA", "WHEEL", "top_level.txt")):
            copy(path.relative_to(args.source))

    manifest = json.loads((SCENE / "output/robots/manifest.json").read_text())
    for name, entry in manifest.items():
        copy(Path("output/robots") / entry["glb"])
        path = args.source / "output/robots" / entry["xml"]
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        settings = dict(compiler.attrib) if compiler is not None else {}
        for element in root.findall("./asset/*"):
            if "file" not in element.attrib:
                continue
            relative = Path(element.get("file"))
            if relative.is_absolute():
                parts = relative.parts
                try:
                    index = next(i for i in range(len(parts) - 2)
                                 if parts[i:i + 3] == ("robosuite", "models", "assets"))
                except StopIteration:
                    raise ValueError("Unexpected absolute robot asset") from None
                source = args.robot_assets.joinpath(*parts[index + 3:])
            else:
                directory = settings.get("meshdir" if element.tag == "mesh" else "texturedir", "")
                source = path.parent / directory / relative
            content = source.read_bytes()
            destination = Path("assets") / name.lower() / (hashlib.sha256(content).hexdigest()[:12] + "-" + source.name)
            target = SCENE / "output/robots" / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            element.set("file", destination.as_posix())
        if compiler is not None:
            compiler.attrib.pop("meshdir", None)
            compiler.attrib.pop("texturedir", None)
        ET.indent(root)
        ET.ElementTree(root).write(SCENE / "output/robots" / entry["xml"], encoding="unicode")

    support = SCENE / "preview_support"
    support.mkdir()
    (support / "__init__.py").write_text('"""Attributed preview support; never imported by scene discovery."""\n')
    for name in ("embodied.py", "runtime.py", "web_console.py"):
        shutil.copy2(args.preview_support / name, support / name)
    shutil.copy2(args.preview_license, support / "LICENSE")
    shutil.copy2(args.robot_license, SCENE / "output/robots/LICENSE.robosuite")
    print("Imported curated snapshot into", SCENE)


if __name__ == "__main__":
    main()
