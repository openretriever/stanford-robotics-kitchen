import importlib
import sys

import numpy as np
import pytest

from retriever_src_kitchen import create_demo, create_scene
from retriever_src_kitchen.api import _scene_imports

pytestmark = pytest.mark.simulation


@pytest.fixture(scope="module")
def demo():
    previous = list(sys.path)
    result = create_demo("search")
    assert sys.path == previous
    return result


def test_native_scene_has_one_passive_mobile_robot():
    model, data, _ = create_scene()
    body = model.body("parked_mobile_aloha").id
    assert model.body_jntnum[body] == 0
    geoms = np.where(model.geom_bodyid == body)[0]
    assert len(geoms) == 166
    assert np.all(model.geom_group[geoms] == 1)
    assert np.all(model.geom_contype[geoms] == 0)
    assert np.all(model.geom_conaffinity[geoms] == 0)
    materials = set(model.geom_matid[geoms])
    assert len(materials) >= 5 and -1 not in materials
    assert any(model.mat_metallic[mat] > .5 for mat in materials)
    assert any(model.mat_rgba[mat, :3].mean() < .2 for mat in materials)
    assert any(model.mat_rgba[mat, :3].mean() > .8 for mat in materials)
    assert model.nq == 21 and model.nu == 9
    assert np.isfinite(data.geom_xpos).all()
    np.testing.assert_allclose(data.xpos[body], [-0.92, 2.0, .005])
    rotation = data.xmat[body].reshape(3, 3)
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    cooking = model.body("cooking_station").id
    cooking_geoms = np.where(model.geom_bodyid == cooking)[0]
    assert len(cooking_geoms) == 16
    assert np.all(model.geom_contype[cooking_geoms] == 0)
    assert np.all(model.geom_conaffinity[cooking_geoms] == 0)
    pan = model.geom("cooking_pan").id
    mesh = model.geom_dataid[pan]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start:start + count] @ data.geom_xmat[pan].reshape(3, 3).T + data.geom_xpos[pan]
    np.testing.assert_allclose(data.geom_xpos[pan, :2], [-1.10, 2.76])
    assert vertices[:, 2].min() == pytest.approx(.903, abs=1e-6)


def test_scripted_demo_remains_paused(demo):
    assert demo.frame == 0
    assert demo.data.time == 0
    assert len(demo.arm_q) == 7


def test_all_drawers_and_contents_roundtrip(demo):
    demo.reset()
    with _scene_imports():
        cls = importlib.import_module("drawer_inspection").DrawerInspection
    inspector = cls(demo.model, demo.data)
    assert len(inspector.drawers) == 16
    original = demo.data.qpos.copy()
    for name, drawer in inspector.drawers.items():
        inspector.move(name, 1, now=0)
        assert inspector.update(now=.375)
        assert demo.data.qpos[drawer["q"]] == pytest.approx(drawer["limit"] / 2)
        inspector.update(now=1)
        assert demo.data.qpos[drawer["q"]] == pytest.approx(drawer["limit"])
        inspector.move(name, 0, now=2)
        inspector.update(now=3)
    np.testing.assert_allclose(demo.data.qpos, original, atol=1e-12)
    assert demo.data.time == 0 and demo.frame == 0
    assert not demo.observations


def test_mobile_body_does_not_move_during_physics(demo):
    demo.reset()
    body = demo.model.body("parked_mobile_aloha").id
    before = demo.data.xpos[body].copy()
    joints = demo.data.qpos[demo.arm_q].copy()
    for _ in range(10):
        demo.advance_physics(joints, .04)
    np.testing.assert_array_equal(demo.data.xpos[body], before)
    assert np.isfinite(demo.data.qpos).all()
    demo.reset()


def test_native_viewer_binary_contains_mobile_geometry(demo, tmp_path):
    import mujoco
    path = tmp_path / "shared-scene.mjb"
    mujoco.mj_saveModel(demo.model, str(path))
    received = mujoco.MjModel.from_binary_path(str(path))
    assert received.body("parked_mobile_aloha").id >= 0
    assert received.ngeom == demo.model.ngeom
    assert received.nu == demo.model.nu


def test_preview_preserves_robot_finish(demo):
    with _scene_imports():
        convert = importlib.import_module("simulation_visuals").finished_robot_mesh
    body = demo.model.body("parked_mobile_aloha").id
    for geom in np.where(demo.model.geom_bodyid == body)[0][::20]:
        mesh = convert(demo.model, geom)
        material = mesh.visual.material
        mat = demo.model.geom_matid[geom]
        assert material.metallicFactor == pytest.approx(demo.model.mat_metallic[mat])
        assert material.roughnessFactor == pytest.approx(demo.model.mat_roughness[mat])
        np.testing.assert_allclose(material.baseColorFactor / 255, demo.model.mat_rgba[mat], atol=1/255)
        assert np.isfinite(mesh.vertex_normals).all()
