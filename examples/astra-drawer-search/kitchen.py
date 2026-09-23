"""The SRC kitchen as a searchable drawer world: src-kitchen's contact-driven DrawerPull,
plus an injected camera, hidden visual-only jars, and retargeting that keeps opened drawers open."""
from __future__ import annotations

import io as _io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from retriever_src_kitchen import api

# src-kitchen keeps its scene source private; api._scene_imports() is the
# sanctioned way in, and it is what api.create_demo() itself uses.
with api._scene_imports():
    import importlib
    _motion = importlib.import_module("motion_demo")
    _drawer = importlib.import_module("drawer_motion")
CupTransfer = _motion.CupTransfer
DrawerPull = _drawer.DrawerPull

IMAGE = 320                      # Astra sees 320x320, matching robot-sim's contract
JAR_HALF = 0.038                 # half-height; keeps the jar under the 0.052 wall top

# The twelve kitchen drawer slide joints, as three stacks of four.
# The beverage bank is at x=-2.5, ~2.9 m from the Panda's base: unreachable, reported not searched.
DRAWERS = tuple(f"range_drawers_drawer{col}_{row}" for col in (0, 1) for row in range(4))
UNREACHABLE_DRAWERS = tuple(f"beverage_drawers_drawer0_{row}" for row in range(4))

# Measured at mount 0.30: these five pass DrawerPull's own verification; the other three stay
# in DRAWERS so failure recovery gets exercised.
OPENABLE = ("range_drawers_drawer0_1", "range_drawers_drawer0_2", "range_drawers_drawer0_3",
            "range_drawers_drawer1_2", "range_drawers_drawer1_3")

JAR_COLOURS = {
    "paprika": (0.62, 0.18, 0.09),
    "herb seasoning": (0.15, 0.33, 0.14),
    "coarse salt": (0.88, 0.87, 0.84),
}


def look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera xyaxes (right, up) for a camera at pos aimed at target."""
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return [*right, *true_up]


@dataclass
class Placement:
    """Where a target jar was hidden. Scoring only -- never sent to a model."""
    drawer: str
    label: str


class DrawerWorld(DrawerPull):
    """A retargetable contact-driven pull over a kitchen holding hidden jars."""

    robot_mount_height = 0.30      # 0.58 reaches 1 of 4 drawers; 0.30 reaches 3 of 4 (measured)

    def __init__(self, placements, camera_pos=None, camera_target=None):
        self._placements = list(placements)
        self._labels = [place.label for place in self._placements]
        self._camera_pos = list(camera_pos or (0.30, 1.95, 1.62))
        self._camera_target = list(camera_target or (-0.20, 2.45, 0.70))
        # DrawerPull.__init__ hardcodes one drawer; go straight to CupTransfer.
        self.drawer_name = self._placements[0].drawer if self._placements else DRAWERS[0]
        CupTransfer.__init__(self)
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.impratio = 10
        self.model.opt.noslip_iterations = 10
        self._renderer = None
        self._video_renderer = None
        self.visited: list[str] = []
        self._light_scene()
        self._scene_option = self._clean_view()

    # -- what the camera should and should not show -------------------------

    def _light_scene(self):
        """Compiled lights are diffuse 0.06 with a headlight doing the work; raise the lights, drop the headlight."""
        head = self.model.vis.headlight
        head.ambient[:] = [0.28, 0.27, 0.25]
        head.diffuse[:] = [0.03, 0.03, 0.03]
        head.specular[:] = [0.03, 0.03, 0.03]
        for index in range(self.model.nlight):
            light = self.model.light(index)
            light.diffuse[:] = [0.62, 0.58, 0.52]
            light.specular[:] = [0.25, 0.25, 0.25]
            light.castshadow[:] = 1

    @staticmethod
    def _clean_view():
        """Hide site groups 1 (a 10 m green grip-axis cylinder Astra reported as an object) and 2 (annotation markers)."""
        option = mujoco.MjvOption()
        option.sitegroup[1] = 0
        option.sitegroup[2] = 0
        return option

    # -- world construction -------------------------------------------------

    def decorate(self, spec, body):
        """Add a camera (the model has none) and hide the jars. Overhead pose: a level camera cannot see into an open drawer."""
        spec.worldbody.add_camera(
            name="drawer_view", pos=self._camera_pos,
            xyaxes=look_at(self._camera_pos, self._camera_target),
        )
        # Every drawer gets a copy of every jar, drawn only where it was placed, so relocate() can
        # move a jar without recompiling (a body cannot change parent at run time). Look the drawers
        # up before adding anything: mujoco 3.3.1's spec.body(name) returns the last body added once
        # there is one, which chained the jars under each other.
        truth = self.ground_truth()
        drawers = {drawer: spec.body(drawer) for drawer in DRAWERS}
        for drawer, drawer_body in drawers.items():
            for index, label in enumerate(self._labels):
                # Body origin is the front panel; floor z=-0.035, walls top at +0.052. Seat the jar on the floor.
                jar = drawer_body.add_body(
                    name=f"target_{index}_{drawer}", pos=[0.0, 0.26, -0.035 + JAR_HALF])
                # Visual-only: a collidable jar jams the pull at 0.014 m instead of 0.30 m.
                jar.add_geom(
                    name=f"target_{index}_{drawer}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[0.026, JAR_HALF, 0], mass=0, contype=0, conaffinity=0, group=1,
                    rgba=[*JAR_COLOURS.get(label, (0.5, 0.5, 0.5)), float(truth.get(drawer) == label)],
                )

    # -- mid-task faults (E2's perturbation study) ----------------------------

    def relocate(self, label, drawer):
        place = next(p for p in self._placements if p.label == label)
        index = self._labels.index(label)
        self.model.geom(f"target_{index}_{place.drawer}_geom").rgba[3] = 0.0
        self.model.geom(f"target_{index}_{drawer}_geom").rgba[3] = 1.0
        place.drawer = drawer

    def jam(self, drawer, stop_m=0.02):
        """The slide stops at stop_m: under DrawerPull's 0.25 m, so the pull fails verification."""
        self.model.joint(drawer + "_slide").range[:] = [0.0, stop_m]

    # -- retargeting without losing accumulated world state -----------------

    def retarget(self, drawer_name):
        """Aim at another drawer without resetting the world; send only the arm home (the scripted
        trajectory assumes the home pose -- without this every pull after the first reads 0.000)."""
        if drawer_name not in DRAWERS:
            raise ValueError(f"Unknown drawer: {drawer_name}")
        self.drawer_name = drawer_name
        self.data.qpos[self.arm_q] = self.initial_q[self.arm_q]
        self.data.qvel[self.arm_v] = 0.0
        self.data.qpos[self.finger_q] = [0.04, -0.04]
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.drawer_id = self.model.body(drawer_name).id
        self.drawer_q = self.model.jnt_qposadr[self.model.joint(drawer_name + "_slide").id]
        self.handle = self.data.site_xpos[self.model.site(drawer_name + "_handle").id].copy()
        self.grip_frames = self.pull_frames = 0
        self.phase = 0
        self.phase_time = 0.0
        self.phase_origin = self.data.site_xpos[self.ee_id].copy()
        self.finished = self.success = False
        self.error = ""
        self.phase_log = []
        self.visited.append(drawer_name)

    # -- observation --------------------------------------------------------

    def render(self, camera="drawer_view"):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, IMAGE, IMAGE)
        self._renderer.update_scene(self.data, camera=camera,
                                    scene_option=self._scene_option)
        return Image.fromarray(self._renderer.render())

    def render_large(self, camera="drawer_view", width=1280, height=720):
        """Video renderer, separate from the 320x320 one that feeds the model."""
        if self._video_renderer is None or self._video_renderer.width != width:
            self._video_renderer = mujoco.Renderer(self.model, height, width)
        self._video_renderer.update_scene(self.data, camera=camera,
                                          scene_option=self._scene_option)
        return Image.fromarray(self._video_renderer.render())

    def jpeg(self, camera="drawer_view", quality=85):
        buffer = _io.BytesIO()
        self.render(camera).save(buffer, "JPEG", quality=quality)
        return buffer.getvalue()

    def geometry(self):
        """Handle height and stack per drawer: names are not visually inferable, heights are."""
        rows = {}
        for name in DRAWERS:
            site = self.model.site(name + "_handle").id
            rows[name] = {"handle_height_m": round(float(self.data.site_xpos[site][2]), 3),
                          "stack": "left" if name.startswith("range_drawers_drawer0") else "right"}
        return rows

    def openings(self):
        """Measured slide position of every drawer, in metres."""
        return {
            name: float(self.data.qpos[self.model.jnt_qposadr[self.model.joint(name + "_slide").id]])
            for name in DRAWERS
        }

    # -- scoring only -------------------------------------------------------

    def ground_truth(self):
        return {place.drawer: place.label for place in self._placements}


def cameras(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
