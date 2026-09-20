"""Package boundary around the existing, explicitly selected scene source."""

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import re
import sys
from threading import RLock

_IMPORT_LOCK = RLock()


def scene_root():
    """Return the packaged source directory; do not import or start simulation."""
    return Path(__file__).resolve().parent / "_scene"


def describe_scene():
    """Read descriptive metadata without importing native libraries."""
    return json.loads((scene_root() / "scene.json").read_text())


@contextmanager
def _scene_imports():
    # Keep the established flat source contract for existing consumers. Mixing
    # different scene snapshots in one interpreter is deliberately rejected.
    with _IMPORT_LOCK:
        root = scene_root()
        names = {p.stem for p in root.glob("*.py")} | {"preview_support"}
        for name in names:
            module = sys.modules.get(name)
            if module is not None:
                filename = getattr(module, "__file__", None)
                if filename is None or not Path(filename).resolve().is_relative_to(root):
                    raise RuntimeError(f"Scene module conflict: {name}; use a fresh interpreter")
        for name, module in tuple(sys.modules.items()):
            if module is not None and (name == "mink" or name.startswith("mink.")):
                filename = getattr(module, "__file__", None)
                if filename is None or not Path(filename).resolve().is_relative_to(root / "vendor/motion"):
                    raise RuntimeError("Motion dependency conflict; use a fresh interpreter")
        previous = list(sys.path)
        sys.path.insert(0, str(root))
        try:
            yield
        finally:
            sys.path[:] = previous


def create_scene(robot="Panda", *, position=None, yaw_degrees=None, mount_height=None):
    """Construct native (model, data, spec); start no clocks, viewer, or policy."""
    with _scene_imports():
        module = importlib.import_module("integration")
        return module.load_scene(robot, position, yaw_degrees, mount_height=mount_height)


def create_demo(task="search"):
    """Construct a paused scripted demonstration, not a Harness task provider.

    The search demo contains scripted target knowledge. Use the consuming
    host's bounded kitchen adapter for reviewed, observation-driven steps.
    """
    choices = {"search": ("organizer_search", "OrganizerSearch"),
               "cup": ("motion_demo", "CupTransfer"),
               "drawer": ("drawer_motion", "DrawerPull"),
               "seasoning": ("seasoning_demo", "SeasoningDemo")}
    if task not in choices:
        raise ValueError(f"Unknown demo: {task}")
    module, name = choices[task]
    with _scene_imports():
        return getattr(importlib.import_module(module), name)()


def harness_config(expected_source_digest):
    """Bind this source to an operator-reviewed pin; never compute/accept a new pin."""
    if type(expected_source_digest) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_source_digest) is None:
        raise ValueError("Expected a reviewed lowercase SHA-256 scene fingerprint")
    return {"source_root": str(scene_root()), "expected_source_digest": expected_source_digest}


def preview(*, port=8105, viewer_port=8106, task="search", output_dir=None):
    """Run the local standalone demo until interrupted; never start real hardware."""
    if any(type(p) is not int or not 1 <= p <= 65535 for p in (port, viewer_port)) or port == viewer_port:
        raise ValueError("Choose distinct valid HTTP and viewer ports")
    with _scene_imports():
        module = importlib.import_module("teaser")
    args = ["--port", str(port), "--viewer-port", str(viewer_port), "--task", task]
    if output_dir is not None:
        args += ["--output-dir", str(output_dir)]
    module.main(args)
