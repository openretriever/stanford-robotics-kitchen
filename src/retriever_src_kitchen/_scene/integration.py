"""Native MuJoCo scene composition and a minimal Retriever mjviser adapter."""

import argparse
import json
import math
import hashlib
import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from build_scene import ROOT


def resolve_robot_asset(path):
    path = Path(path)
    marker = ("robosuite", "models", "assets")
    if not path.exists():
        for i in range(len(path.parts) - 2):
            if path.parts[i:i + 3] == marker:
                package = importlib.util.find_spec("robosuite")
                if package and package.origin:
                    return str(Path(package.origin).parent.joinpath(*path.parts[i + 1:]))
    return str(path)


def load_scene(robot=None, position=None, yaw_degrees=None, model_path=None, mount_height=None):
    config = json.loads((ROOT / "layout.json").read_text())
    position = list(position if position is not None else config["robot_position"])
    yaw = math.radians(yaw_degrees if yaw_degrees is not None else config["robot_yaw_degrees"])
    model_path = Path(model_path or ROOT / "output/kitchen-sim.xml").resolve()
    spec = mujoco.MjSpec.from_file(str(model_path))
    for asset, directory in ((spec.meshes, spec.meshdir), (spec.textures, spec.texturedir)):
        for item in asset:
            if item.file:
                item.file = str((model_path.parent / directory / item.file).resolve())
    spec.meshdir = spec.texturedir = ""
    entry = None
    if robot and robot != "None":
        manifest = json.loads((ROOT / "output/robots/manifest.json").read_text())
        entry = manifest[robot]
        robot_path = (ROOT / "output/robots" / entry["xml"]).resolve()
        child = mujoco.MjSpec.from_file(str(robot_path))
        for mesh in child.meshes:
            if mesh.file:
                mesh.file = resolve_robot_asset((robot_path.parent / child.meshdir / mesh.file).resolve())
        for texture in child.textures:
            if texture.file:
                texture.file = resolve_robot_asset((robot_path.parent / child.texturedir / texture.file).resolve())
        child.meshdir = ""
        child.texturedir = ""
        # Preserve the visual group convention used by the console bridge.
        if robot == "ALOHA 2":
            for geom in child.geoms:
                geom.group = 1 if geom.group == 2 else 0
        mount = entry["mount_height"] if mount_height is None else float(mount_height)
        if mount < 0 or (0 < mount < 0.04):
            raise ValueError("Mount height must be zero or at least 0.04 m")
        quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        if mount:
            spec.worldbody.add_geom(name="robot_pedestal", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=[0.26, 0.26, (mount - 0.03) / 2], pos=[*position[:2], position[2] + (mount - 0.03) / 2],
                                    quat=quat, rgba=[0.28, 0.30, 0.31, 1], group=1)
            spec.worldbody.add_geom(name="robot_mount_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=[0.56 if robot == "ALOHA 2" else 0.325, 0.325, 0.015],
                                    pos=[*position[:2], position[2] + mount - 0.015], quat=quat,
                                    rgba=[0.72, 0.74, 0.74, 1], group=1)
        frame = spec.worldbody.add_frame(name="robot_mount", pos=[*position[:2], position[2] + mount], quat=quat)
        spec.attach(child, prefix="robot/", frame=frame)
    model = spec.compile()
    data = mujoco.MjData(model)
    if entry:
        robot_addresses = []
        for j in range(model.njnt):
            if model.joint(j).name.startswith("robot/"):
                start = model.jnt_qposadr[j]
                end = model.jnt_qposadr[j + 1] if j + 1 < model.njnt else model.nq
                robot_addresses.extend(range(start, end))
        if len(robot_addresses) != len(entry["qpos"]):
            raise ValueError("Robot pose does not match the compiled joint layout")
        data.qpos[robot_addresses] = entry["qpos"]
    mujoco.mj_forward(model, data)
    return model, data, spec


def console_state(model, data):
    """Pass this object to the existing MjviserBridge.start/update methods."""
    return SimpleNamespace(model=model, data=data)


def main():
    from export_scene import export_glb
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--position", type=float, nargs=3)
    parser.add_argument("--yaw", type=float, default=0)
    args = parser.parse_args()
    model, data, spec = load_scene(args.robot, args.position, args.yaw)
    stem = "kitchen-" + args.robot.lower().replace(" ", "")
    out = ROOT / "output" / stem
    export_glb(model, data, out.with_suffix(".glb"))
    # Include a keyframe so native MuJoCo viewers can recover the preview pose.
    spec.add_key(name="kitchen_preview", qpos=data.qpos, ctrl=data.ctrl)
    root = ET.fromstring(spec.to_xml())
    asset_dir = out.parent / "robot-assets" / args.robot.lower().replace(" ", "")
    asset_dir.mkdir(parents=True, exist_ok=True)
    for element in root.findall("./asset/*"):
        if "file" in element.attrib:
            source = Path(element.attrib["file"])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:10]
            dest = asset_dir / f"{digest}-{source.name}"
            if not dest.exists():
                shutil.copy2(source, dest)
            element.set("file", str(dest.relative_to(out.parent)))
    ET.indent(root)
    ET.ElementTree(root).write(out.with_suffix(".xml"), encoding="unicode", xml_declaration=True)
    print(json.dumps({"robot": args.robot, "geoms": model.ngeom, "nq": model.nq,
                      "finite": bool(np.isfinite(data.geom_xpos).all()), "xml": str(out.with_suffix('.xml'))}))


if __name__ == "__main__":
    main()
