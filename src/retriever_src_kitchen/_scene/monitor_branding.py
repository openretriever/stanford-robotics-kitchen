"""Matching wall TVs with the supplied logo on a single white display surface."""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
import trimesh


def add_monitor_logos(spec):
    spec.add_texture(name="src_logo", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     width=1600, height=900, rgb1=[1, 1, 1])
    material = spec.add_material(name="baked_src_display", rgba=[1, 1, 1, 1],
                                 emission=1, roughness=1, specular=0)
    material.textures[1] = "src_logo"
    w, h = 1.66, 1.66 * 9 / 16
    for name in ("back_monitor", "left_monitor"):
        body = next(body for body in spec.bodies if body.name == name)
        # Cover the unequal legacy frames already included in the room bake.
        body.add_geom(name=name + "_matching_frame", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[0, -.025, 0], size=[(w + .04) / 2, .033, (h + .04) / 2],
                      rgba=[.025, .025, .025, 1], group=1, contype=0, conaffinity=0)
        mesh = trimesh.creation.box(extents=[w, .001, h])
        uv = np.column_stack((mesh.vertices[:, 0] / w + .5, .5 - mesh.vertices[:, 2] / h))
        spec.add_mesh(name=name + "_logo_mesh", uservert=mesh.vertices.ravel(),
                      userface=mesh.faces.ravel(), usertexcoord=uv.ravel(),
                      userfacetexcoord=mesh.faces.ravel())
        body.add_geom(name=name + "_logo", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=name + "_logo_mesh", material="baked_src_display",
                      pos=[0, -.060, 0], group=1, contype=0, conaffinity=0)


def upload_monitor_logos(model):
    path = Path(__file__).parent / "reference/stanford-robotics-center-logo.png"
    tid = model.texture("src_logo").id
    width, height = int(model.tex_width[tid]), int(model.tex_height[tid])
    # Letterbox in the display buffer, preserving the original transparent asset.
    screen = Image.new("RGBA", (width, height), "white")
    with Image.open(path) as image:
        logo = image.convert("RGBA")
        logo.thumbnail((int(width * .86), int(height * .80)), Image.Resampling.LANCZOS)
        screen.alpha_composite(logo, ((width - logo.width) // 2, (height - logo.height) // 2))
    pixels = np.asarray(screen.convert("RGB"))
    start = model.tex_adr[tid]
    assert model.tex_nchannel[tid] == 3
    model.tex_data[start:start + pixels.size] = pixels.ravel()
