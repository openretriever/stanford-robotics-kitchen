"""Lazy Retriever Pipeline entrypoints for the packaged kitchen simulation."""

from copy import deepcopy
import importlib
from pathlib import Path

from .api import _scene_imports

_TASKS = ("search", "drawer", "cup", "seasoning")
_DESCRIPTION = {
    "name": "src-kitchen",
    "runtime": "paused MuJoCo simulation",
    "rate_hz": 50,
    "tasks": list(_TASKS),
    "nodes": [
        {"name": "skill_dispatcher", "flow": "Dispatch", "trigger": "Rate(50 Hz)"},
        {"name": "inverse_kinematics", "flow": "InverseKinematics", "trigger": "Tick"},
        {"name": "mujoco_simulator", "flow": "Simulation", "trigger": "Command"},
        {"name": "task_verifier", "flow": "Verify", "trigger": "Observation"},
        {"name": "event_sink", "flow": "Events", "trigger": "Observation"},
    ],
    "edges": [
        ["skill_dispatcher", "inverse_kinematics"],
        ["inverse_kinematics", "mujoco_simulator"],
        ["mujoco_simulator", "task_verifier"],
        ["task_verifier", "event_sink"],
    ],
    "hardware_access": False,
    "starts_paused": True,
}


def describe_pipeline():
    """Describe the Flow graph without importing Retriever or native simulation."""
    return deepcopy(_DESCRIPTION)


def create_pipeline(*, task="search", viewer_port=8106, output_dir=None):
    """Build the existing kitchen Flow graph without starting clocks or servers.

    The returned object is a regular ``retriever.flow.Pipeline``. Its paused
    runtime is available as ``pipeline.kitchen_runtime`` for explicit control.
    """
    if task not in _TASKS:
        raise ValueError(f"Unknown task: {task}")
    if type(viewer_port) is not int or not 1 <= viewer_port <= 65535:
        raise ValueError("Choose a valid viewer port")
    if output_dir is not None:
        output_dir = Path(output_dir)

    with _scene_imports():
        teaser = importlib.import_module("teaser")
    runtime = teaser.Runtime(task, viewer_port=viewer_port, output_dir=output_dir)
    pipeline = teaser.make_pipeline(runtime)
    pipeline.kitchen_runtime = runtime
    return pipeline
