"""Preserve baked lighting, PBR factors, and split normals in mjviser 0.0.14.

Stock mjviser still displays the textured MJCF. This small local extension adds
its emission and material finish without modifying the installed dependency.
"""

import mujoco
import numpy as np
import trimesh
from mjviser import ViserMujocoScene
from mjviser.conversions import group_geoms_by_visual_compat, mujoco_mesh_to_trimesh


def is_baked(model, geom_id):
    mat_id = model.geom_matid[geom_id]
    return mat_id >= 0 and model.mat(mat_id).name.startswith("baked_")


def textured_mesh(model, geom_id):
    mesh = mujoco_mesh_to_trimesh(model, geom_id)
    if not is_baked(model, geom_id):
        return mesh
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_faceadr[mesh_id]
    count = model.mesh_facenum[mesh_id]
    normal_start = model.mesh_normaladr[mesh_id]
    normal_ids = model.mesh_facenormal[start:start + count].ravel()
    if np.all(normal_ids >= 0) and len(mesh.vertices) == len(normal_ids):
        mesh.vertex_normals = model.mesh_normal[normal_start + normal_ids]
    mat_id = model.geom_matid[geom_id]
    material = mesh.visual.material
    material.metallicFactor = float(model.mat_metallic[mat_id])
    material.roughnessFactor = float(model.mat_roughness[mat_id])
    material.emissiveTexture = material.baseColorTexture
    strength = 0.10 if model.mat_metallic[mat_id] > 0.5 else float(model.mat_emission[mat_id])
    material.emissiveFactor = np.full(3, strength)
    if model.mat(mat_id).name == "baked_src_display":
        # A powered display emits its image; room lighting must not tint it.
        material.baseColorFactor = [0, 0, 0, 255]
        material.emissiveFactor = np.ones(3)
    return mesh


def finished_robot_mesh(model, geom_id):
    mesh_id = int(model.geom_dataid[geom_id])
    start, count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    faces = model.mesh_face[start:start + count]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh_id] + faces.ravel()]
    normal_ids = model.mesh_facenormal[start:start + count].ravel()
    normals = model.mesh_normal[model.mesh_normaladr[mesh_id] + normal_ids]
    # Preserve the exported per-corner normals instead of recomputing at launch.
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3),
                          vertex_normals=normals, process=False)
    mat = int(model.geom_matid[geom_id])
    mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        name=model.mat(mat).name,
        baseColorFactor=np.rint(model.mat_rgba[mat] * 255).astype(np.uint8),
        metallicFactor=float(model.mat_metallic[mat]),
        roughnessFactor=float(model.mat_roughness[mat])))
    return mesh


class KitchenMujocoScene(ViserMujocoScene):
    def _add_fixed_geometry(self):
        super()._add_fixed_geometry()
        groups = {}
        for geom_id in range(self.mj_model.ngeom):
            if self.mj_model.geom_rgba[geom_id, 3] == 0:
                continue
            key = (int(self.mj_model.geom_bodyid[geom_id]), int(self.mj_model.geom_group[geom_id]))
            groups.setdefault(key, []).append(geom_id)
        for (body_id, group_id), geom_ids in groups.items():
            for sub_id, subgroup in enumerate(group_geoms_by_visual_compat(self.mj_model, geom_ids)):
                key = (body_id, group_id, sub_id)
                if key in self._fixed_geom_handles and self.mj_model.body(body_id).name == "parked_mobile_aloha":
                    scene = trimesh.Scene()
                    for geom_id in subgroup:
                        mesh = finished_robot_mesh(self.mj_model, geom_id)
                        transform = np.eye(4)
                        rotation = np.empty(9)
                        mujoco.mju_quat2Mat(rotation, self.mj_model.geom_quat[geom_id])
                        transform[:3, :3] = rotation.reshape(3, 3)
                        transform[:3, 3] = self.mj_model.geom_pos[geom_id]
                        scene.add_geometry(mesh, transform=transform)
                    previous = self._fixed_geom_handles[key]
                    name, visible = previous.name, previous.visible
                    previous.remove()
                    self._fixed_geom_handles[key] = self.server.scene.add_glb(
                        name, scene.export(file_type="glb"), position=self.mj_data.xpos[body_id],
                        wxyz=self.mj_data.xquat[body_id], visible=visible,
                        cast_shadow=False, receive_shadow=.3)
                    continue
                if key not in self._fixed_geom_handles or len(subgroup) != 1:
                    continue
                geom_id = subgroup[0]
                if not is_baked(self.mj_model, geom_id):
                    continue
                mesh = textured_mesh(self.mj_model, geom_id)
                transform = np.eye(4)
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, self.mj_model.geom_quat[geom_id])
                transform[:3, :3] = rotation.reshape(3, 3)
                transform[:3, 3] = self.mj_model.geom_pos[geom_id]
                mesh.apply_transform(transform)
                previous = self._fixed_geom_handles[key]
                name, visible = previous.name, previous.visible
                previous.remove()
                self._fixed_geom_handles[key] = self.server.scene.add_mesh_trimesh(
                    name, mesh, position=self.mj_data.xpos[body_id],
                    wxyz=self.mj_data.xquat[body_id], visible=visible,
                    cast_shadow=False, receive_shadow=(
                        0.0 if self.mj_model.mat(self.mj_model.geom_matid[geom_id]).name
                        == "baked_src_display" else 0.3))
