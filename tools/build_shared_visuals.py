"""Bake the parked mobile robot into visual-only MJCF shared by all viewers."""

from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1] / "src/retriever_src_kitchen/_scene"


def build():
    scene = trimesh.load(ROOT / "output/robots/mobile-aloha.glb", force="scene")
    document = ET.Element("mujoco")
    assets = ET.SubElement(document, "asset")
    world = ET.SubElement(document, "worldbody")
    body = ET.SubElement(world, "body", name="parked_mobile_aloha",
                         pos="-0.92 2.0 0.005", quat="0.7071067811865476 0 0 0.7071067811865476")
    # Explicit inertia lets MuJoCo compile this mesh-only body with massless
    # visual geoms. The fixed body has no DOFs and cannot affect robot dynamics.
    ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="0.01 0.01 0.01")
    destination = ROOT / "output/robots/mobile-aloha-assets"
    destination.mkdir(parents=True, exist_ok=True)
    z_up = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    for i, node in enumerate(scene.graph.nodes_geometry):
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        material = getattr(mesh.visual, "material", None)
        rgba = getattr(material, "baseColorFactor", None)
        if rgba is None:
            rgba = getattr(material, "diffuse", None)
        if rgba is None:
            rgba = mesh.visual.main_color
        rgba = np.asarray(rgba, dtype=float)
        if rgba.max() > 1:
            rgba /= 255
        mesh.apply_transform(z_up @ transform)
        mesh = trimesh.graph.smooth_shade(mesh, angle=np.deg2rad(35))
        name = f"mobile_aloha_visual_{i:03d}"
        (destination / f"{name}.obj").write_text(trimesh.exchange.obj.export_obj(mesh, include_texture=False))
        ET.SubElement(assets, "mesh", name=name, file=f"mobile-aloha-assets/{name}.obj")
        finish_name = material.name
        if assets.find(f"material[@name='{finish_name}']") is None:
            ET.SubElement(assets, "material", name=finish_name,
                          rgba=" ".join(f"{v:.7g}" for v in rgba),
                          metallic=str(material.metallicFactor),
                          roughness=str(material.roughnessFactor), specular="0.45", shininess="0.35")
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name,
                      group="1", contype="0", conaffinity="0", mass="0",
                      material=finish_name,
                      rgba=" ".join(f"{v:.7g}" for v in rgba))
    ET.indent(document)
    ET.ElementTree(document).write(ROOT / "output/robots/mobile-aloha.xml", encoding="unicode")
    for name in ("kitchen.xml", "kitchen-sim.xml"):
        path = ROOT / "output" / name
        document = ET.parse(path).getroot()
        for include in document.findall("include"):
            if include.get("file") == "robots/mobile-aloha.xml":
                document.remove(include)
        scene_assets = document.find("asset")
        scene_world = document.find("worldbody")
        for element in list(scene_assets):
            if element.get("name", "").startswith(("mobile_aloha_visual_", "mobile_aloha_finish_")):
                scene_assets.remove(element)
        for element in list(scene_world):
            if element.get("name") == "parked_mobile_aloha":
                scene_world.remove(element)
        for element in assets:
            element = deepcopy(element)
            if element.get("file"):
                element.set("file", "robots/" + element.get("file"))
            scene_assets.append(element)
        scene_world.append(deepcopy(body))
        ET.indent(document)
        ET.ElementTree(document).write(path, encoding="unicode")
    build_cooktop()
    print(f"Embedded {len(scene.graph.nodes_geometry)} passive visual meshes; no joints or actuators added")


def build_cooktop():
    """Add a shallow open pan and visible burners, without changing contacts."""
    destination = ROOT / "output/robots/mobile-aloha-assets"
    profile = [[0, 0], [0.105, 0], [0.139, 0.042], [0.137, 0.047],
               [0.132, 0.047], [0.101, 0.008], [0, 0.008], [0, 0]]
    pan = trimesh.creation.revolve(profile, sections=96)
    (destination / "cooking_pan.obj").write_text(trimesh.exchange.obj.export_obj(pan))
    for name in ("kitchen.xml", "kitchen-sim.xml"):
        path = ROOT / "output" / name
        document = ET.parse(path).getroot()
        assets, world = document.find("asset"), document.find("worldbody")
        for parent in (assets, world):
            for element in list(parent):
                if element.get("name", "").startswith("cooking_"):
                    parent.remove(element)
        ET.SubElement(assets, "mesh", name="cooking_pan_mesh",
                      file="robots/mobile-aloha-assets/cooking_pan.obj")
        cooking = ET.SubElement(world, "body", name="cooking_station")
        ET.SubElement(cooking, "inertial", pos="-1.10 2.76 0.93", mass="1",
                      diaginertia="0.01 0.01 0.01")

        def geom(name, **attributes):
            return ET.SubElement(cooking, "geom", name="cooking_" + name,
                                 group="1", contype="0", conaffinity="0", mass="0", **attributes)

        for index, (x, y) in enumerate(((-1.10, 2.76), (-0.74, 2.76), (-1.10, 3.08), (-0.74, 3.08))):
            geom(f"burner{index}", type="cylinder", pos=f"{x} {y} 0.887",
                 size="0.092 0.009", rgba="0.075 0.08 0.085 1")
            geom(f"burner_cap{index}", type="cylinder", pos=f"{x} {y} 0.898",
                 size="0.067 0.002", rgba="0.23 0.24 0.25 1")
        geom("pan", type="mesh", mesh="cooking_pan_mesh", pos="-1.10 2.76 0.903",
             rgba="0.16 0.17 0.18 1")
        geom("handle_neck", type="box", pos="-1.10 2.595 0.935",
             size="0.014 0.047 0.007", rgba="0.65 0.67 0.69 1")
        geom("handle", type="capsule", fromto="-1.10 2.43 0.944 -1.10 2.555 0.936",
             size="0.017", rgba="0.035 0.04 0.045 1")
        for i, (x, y, angle) in enumerate(((0, 0, 0.3), (-0.034, 0.019, -0.4), (0.031, 0.028, 0.7),
                                          (0.008, -0.04, -0.2), (0.045, -0.03, 1.1))):
            geom(f"food{i}", type="ellipsoid", pos=f"{-1.10+x} {2.76+y} 0.925",
                 size="0.025 0.011 0.008", euler=f"0 0 {angle}", rgba="0.84 0.49 0.22 1")
        ET.indent(document)
        ET.ElementTree(document).write(path, encoding="unicode")


if __name__ == "__main__":
    build()
