"""Verify an extracted wheel from an unrelated directory, not the editable tree."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="src-kitchen-wheel-") as directory:
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(directory)
        code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from retriever_src_kitchen import scene_root, create_scene
from retriever_src_kitchen.resources import verify_resources
assert scene_root().is_relative_to(Path(sys.argv[1]).resolve())
report = verify_resources()
assert report["ok"], report
model, data, spec = create_scene()
assert model.body("parked_mobile_aloha").id >= 0
assert model.nu == 9
print("PASS: wheel resources and native scene are independent of the editable source")
"""
        subprocess.run([sys.executable, "-I", "-c", code, directory], cwd=directory,
                       check=True, timeout=180)


if __name__ == "__main__":
    main()
