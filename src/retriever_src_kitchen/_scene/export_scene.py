"""Export the same compiled MuJoCo scene to GLB, PNG and an MP4 camera tour."""

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

from build_scene import ROOT
from simulation_visuals import textured_mesh


def export_glb(model, data, output, groups=(1, 3, 4)):
    scene = trimesh.Scene()
    for i in range(model.ngeom):
        if model.geom_group[i] not in groups:
            continue
        mesh = (textured_mesh(model, i) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                else create_primitive_mesh(model, i))
        matid = model.geom_matid[i]
        rgba = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[i]
        if rgba[3] < 1:
            mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=rgba, alphaMode="BLEND", doubleSided=True, roughnessFactor=0.2))
        transform = np.eye(4)
        transform[:3, :3] = data.geom_xmat[i].reshape(3, 3)
        transform[:3, 3] = data.geom_xpos[i]
        # glTF uses Y-up; MuJoCo and the placement manifest use Z-up.
        basis = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        scene.add_geometry(mesh, node_name=model.geom(i).name or f"geom_{i}", transform=basis @ transform)
    scene.export(str(output))
    print(f"GLB: {output}")


def caption(frame, subtitle="Kitchen reconstruction | Estimated scale"):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        sub_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    except OSError:
        title_font = sub_font = ImageFont.load_default()
    draw.rectangle((24, 23, 31, 82), fill=(160, 25, 35))
    draw.text((44, 20), "Stanford Robotics Center", font=title_font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(50, 55, 55))
    draw.text((45, 60), subtitle, font=sub_font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(50, 55, 55))
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "output/kitchen.xml")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--glb-only", action="store_true")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    if model.nkey:
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "kitchen_preview")
        mujoco.mj_resetDataKeyframe(model, data, max(0, kid))
    mujoco.mj_forward(model, data)
    export_glb(model, data, args.model.with_suffix(".glb"))
    if args.glb_only:
        return
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [0, 1, 0, 1, 1, 0]
    opt.sitegroup[:] = 0
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    views = {
        "overview": ((0, 0.3, 0.85), 8.8, 125, -27),
        "entrance": ((0, 1.1, 1.14), 5.55, 110, -12),
        "island": ((0, 1.7, 1.12), 4.35, 68, -13),
        "top": ((0, 0, 0), 9.5, 90, -89.9),
    }
    for name, (lookat, distance, azimuth, elevation) in views.items():
        camera.lookat[:] = lookat
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        renderer.update_scene(data, camera=camera, scene_option=opt)
        frame = caption(renderer.render())
        prefix = "" if args.model.stem == "kitchen" else args.model.stem + "-"
        Image.fromarray(frame).save(args.model.parent / f"{prefix}{name}.png")
    if args.video:
        target = args.model.parent / f"{args.model.stem}-tour.mp4"
        with imageio.get_writer(str(target), fps=30, codec="libx264", quality=8, macro_block_size=1) as writer:
            for i in range(round(args.seconds * 30)):
                t = i / max(1, round(args.seconds * 30) - 1)
                s = t * t * (3 - 2 * t)
                camera.lookat[:] = (0, 0.75 + 0.65 * s, 1.0)
                camera.distance = 7.7 - 2.0 * math.sin(math.pi * s)
                camera.azimuth = 122 - 64 * s
                camera.elevation = -22 + 9 * math.sin(math.pi * s)
                renderer.update_scene(data, camera=camera, scene_option=opt)
                writer.append_data(caption(renderer.render()))
        print(f"Video: {target}")
    renderer.close()


if __name__ == "__main__":
    main()
