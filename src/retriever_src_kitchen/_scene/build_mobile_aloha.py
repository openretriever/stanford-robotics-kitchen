"""Build a parked Mobile ALOHA preview from AgileX's public robot description."""

import argparse
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from yourdfpy import URDF

from build_scene import ROOT

REVISION = "2843ff11d2695c1563a1c9847f632aa9734f5bbc"
SOURCE = f"https://raw.githubusercontent.com/agilexrobotics/mobile_aloha_sim/{REVISION}/"
VENDOR = ROOT / "vendor/mobile_aloha"
DESCRIPTION = "aloha_isaac_sim/urdf/arx5_description_isaac.urdf"


def apply_finishes(scene):
    # Manufacturer-photo-inspired finishes, not calibrated color measurements.
    finishes = {
        "arm": ([48, 53, 58, 255], .45, .36),
        "joint": ([30, 33, 37, 255], .25, .42),
        "frame": ([43, 47, 51, 255], .3, .48),
        "panel": ([222, 227, 230, 255], .12, .40),
        "steel": ([167, 180, 189, 255], .75, .28),
        "rubber": ([17, 19, 22, 255], 0, .82),
    }

    def finish(mesh, name):
        color, metal, rough = finishes[name]
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name="mobile_aloha_finish_" + name, baseColorFactor=color,
            metallicFactor=metal, roughnessFactor=rough))
        return mesh

    parents = {child: parent for parent, child, _ in scene.graph.to_edgelist()}
    for node in list(scene.graph.nodes_geometry):
        _, geometry = scene.graph[node]
        mesh = scene.geometry[geometry]
        link = parents[node]
        if link == "box2_Link":
            # The source STL combines the body panels and upper mounting deck.
            deck_faces = mesh.triangles_center[:, 2] > .57
            deck = mesh.submesh([np.flatnonzero(deck_faces)], append=True)
            panels = mesh.submesh([np.flatnonzero(~deck_faces)], append=True)
            scene.geometry[geometry] = finish(panels, "panel")
            scene.add_geometry(finish(deck, "frame"), node_name=node + "_deck",
                               parent_node_name=node)
        elif link == "box1_Link":
            finish(mesh, "frame")
        elif link.endswith(("_link7", "_link8")):
            finish(mesh, "steel")
        elif link.endswith(("_link1", "_link4", "_link5", "_link6", "_base_link")) and link != "base_link":
            finish(mesh, "joint")
        elif link.endswith(("_link2", "_link3")):
            finish(mesh, "arm")
        elif "wheel" in link:
            finish(mesh, "rubber")
        elif "castor" in link:
            finish(mesh, "steel")
        else:
            material = getattr(mesh.visual, "material", None)
            color = np.asarray(material.main_color if material else mesh.visual.main_color)[:3]
            # Keep the base's original dark/light partitions, replacing CAD cyan.
            finish(mesh, "rubber" if color.mean() < 90 else "panel")


def fetch_assets():
    def download(relative):
        path = VENDOR / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(SOURCE + relative, timeout=60) as response:
                path.write_bytes(response.read())
            print("Downloaded", relative, flush=True)
        return path

    source = download(DESCRIPTION)
    download("LICENSE")
    root = ET.parse(source).getroot()
    for relative in sorted({mesh.get("filename").removeprefix("package://")
                            for mesh in root.findall(".//mesh")}):
        if not relative.startswith("aloha_isaac_sim/meshes/") or ".." in Path(relative).parts:
            raise ValueError(f"Unexpected mesh reference: {relative}")
        download(relative)


def build():
    robot = URDF.load(str(VENDOR / DESCRIPTION),
                      filename_handler=lambda fname: str(VENDOR / fname.removeprefix("package://")))
    # Front follower wrists sit near counter height; rear leader arms fold low.
    robot.update_cfg({f"{side}_joint{joint}": angle
                      for side in ("fl", "fr", "lr", "rr")
                      for joint, angle in ((2, 0.0), (3, 0.0 if side in ("fl", "fr") else -0.6),
                                           (4, 0.0))})
    scene = robot.scene.copy()
    apply_finishes(scene)
    bounds = scene.bounds
    if not np.isfinite(bounds).all():
        raise ValueError("Nonfinite Mobile ALOHA geometry")
    scene.apply_translation([0, 0, -bounds[0, 2]])
    # Match the kitchen exporter: glTF Y-up, then Viser restores Z-up.
    scene.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                    [0, -1, 0, 0], [0, 0, 0, 1]]))
    output = ROOT / "output/robots/mobile-aloha.glb"
    scene.export(str(output))
    print(f"GLB: {output}; {len(scene.geometry)} meshes; bounds={scene.bounds.tolist()}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    if parser.parse_args().fetch:
        fetch_assets()
    build()
