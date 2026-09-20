import json
from pathlib import Path
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET

import pytest

from retriever_src_kitchen import describe_scene, harness_config, scene_root
from retriever_src_kitchen.resources import verify_resources

REPO = Path(__file__).resolve().parents[1]


def run_python(code):
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_metadata_and_discovery_are_import_safe():
    run_python('''
import socket, sys, threading
def forbidden(*args, **kwargs):
    raise AssertionError("Discovery attempted external I/O")
socket.socket = forbidden
before = set(sys.modules)
import retriever_src_kitchen as kitchen
assert kitchen.scene_root().is_dir()
assert kitchen.describe_scene()["hardware_access"] is False
assert not ({"mujoco", "viser", "mink", "retriever", "numpy"} & (set(sys.modules) - before))
assert len(threading.enumerate()) == 1
''')


def test_hub_exports_load_without_simulation():
    run_python('''
from pathlib import Path
import sys, tomllib
from retriever.hub._loader import load_exports
manifest = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["retriever"]["module"]
before = set(sys.modules)
exports = load_exports(Path("."), manifest["module"], manifest["exports"], namespace="src_kitchen_test")
assert set(exports) == set(manifest["exports"])
assert all(callable(value) for value in exports.values())
assert exports["describe_scene"]()["name"] == "SRC Kitchen"
assert exports["scene_root"]().joinpath("output/kitchen-sim.xml").is_file()
assert not ({"mujoco", "viser", "mink"} & (set(sys.modules) - before))
''')


def test_scene_metadata_matches_assets():
    info = describe_scene()
    assert info["drawers"] == {"kitchen": 12, "tabletop_organizer": 4}
    assert info["parked_robot"]["visual_only"] is True
    assert info["controlled_robot"] == "Panda"


def test_resource_inventory():
    report = verify_resources()
    assert report["ok"], report
    assert report["files"] > 100
    assert not report["simulation_tested"]


@pytest.mark.parametrize("bad", [None, "", "newest", "A" * 64, "a" * 63, 123])
def test_harness_config_rejects_unreviewed_shapes(bad):
    with pytest.raises(ValueError):
        harness_config(bad)


def test_harness_config_keeps_operator_pin():
    pin = "a" * 64
    assert harness_config(pin) == {"source_root": str(scene_root()), "expected_source_digest": pin}


def test_all_xml_assets_are_local_and_present():
    root = scene_root()
    paths = [root / "output/kitchen.xml", root / "output/kitchen-sim.xml"]
    paths += list((root / "output/robots").glob("*.xml"))
    for path in paths:
        document = ET.parse(path).getroot()
        compiler = document.find("compiler")
        settings = {} if compiler is None else compiler.attrib
        for asset in document.findall("./asset/*"):
            if "file" not in asset.attrib:
                continue
            relative = Path(asset.get("file"))
            assert not relative.is_absolute(), (path, relative)
            directory = settings.get("meshdir" if asset.tag == "mesh" else "texturedir", "")
            target = (path.parent / directory / relative).resolve()
            assert target.is_relative_to(root) and target.is_file(), target


def test_source_contract_is_present():
    for name in ("organizer_search.py", "motion_demo.py", "integration.py", "build_scene.py",
                 "export_scene.py", "simulation_visuals.py", "monitor_branding.py", "layout.json",
                 "output/robots/manifest.json", "reference/stanford-robotics-center-logo.png",
                 "vendor/motion/mink/__init__.py"):
        assert scene_root().joinpath(name).is_file(), name


def test_cli_metadata():
    output = subprocess.check_output([sys.executable, "-m", "retriever_src_kitchen.cli", "info"], text=True)
    assert json.loads(output)["name"] == "SRC Kitchen"


def test_module_conflicts_fail_closed():
    run_python('''
import sys, types
from retriever_src_kitchen import create_scene
sys.modules["integration"] = types.SimpleNamespace(__file__="/unrelated/integration.py")
try:
    create_scene()
except RuntimeError as error:
    assert "conflict" in str(error)
else:
    raise AssertionError("Conflicting scene module was accepted")
assert "mujoco" not in sys.modules
''')


def test_preview_port_validation():
    from retriever_src_kitchen import preview
    for first, second in ((0, 8106), (8105, 8105), (True, 8106), (8105, 70000)):
        with pytest.raises(ValueError):
            preview(port=first, viewer_port=second)


def test_manifest_and_project_version_agree():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert project["project"]["version"] == describe_scene()["version"]
