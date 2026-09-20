"""The SRC kitchen as a searchable drawer world, for an Astra-driven search.

What this owns and what it borrows
----------------------------------
The drawer opening is src-kitchen's own `DrawerPull`: a scripted, contact-driven
Panda pull whose success test is `opening > 0.25` AND two-finger contact held for
more than 90% of the pull frames. That primitive is reused, not reinvented, and
its author's framing stands -- "Scripted contact-driven opening; not a learned
search policy". What is added here is only the world the search happens in:

  * a `drawer_view` camera, because the compiled model defines none (`model.ncam == 0`);
  * target jars hidden inside chosen drawers, so there is something to find;
  * `retarget()`, which moves the skill to a different drawer WITHOUT resetting
    physics, so drawers opened earlier in a run stay open and accumulate;
  * `ground_truth()`, used only to score a finished run, never shown to a model.

So the robot really opens the drawer, and the choice of which drawer and the
reading of what is inside are the model's problem.
"""
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
# Reachable search space. The four beverage_drawers sit at x=-2.5, nearly three
# metres from the Panda's fixed base at (0, 1.90, 0): measured, not assumed. They
# are listed separately so a run can report what it could not have searched.
DRAWERS = tuple(f"range_drawers_drawer{col}_{row}" for col in (0, 1) for row in range(4))
UNREACHABLE_DRAWERS = tuple(f"beverage_drawers_drawer0_{row}" for row in range(4))

# Which of the eight the scripted pull actually opens, at robot_mount_height 0.30.
# Measured by sweeping all eight to completion and keeping only the ones that pass
# DrawerPull's own verification (opening > 0.25m AND two-finger contact held for
# more than 90% of the pull). The other three fail in "Align with handle" or
# "Approach drawer" and are left in DRAWERS on purpose: a search that can only be
# pointed at drawers guaranteed to open would never exercise failure recovery.
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

    # DrawerPull mounts the Panda at 0.58m, which can only reach the TOP drawer of
    # a stack: measured 1/4 at 0.58 against 3/4 at 0.30, with the lower rows
    # failing in "Align with handle" at a 0.043m gap that is almost entirely
    # vertical. Dropping the mount is what makes a multi-drawer search possible.
    robot_mount_height = 0.30

    def __init__(self, placements, camera_pos=None, camera_target=None):
        self._placements = list(placements)
        self._camera_pos = list(camera_pos or (0.30, 1.95, 1.62))
        self._camera_target = list(camera_target or (-0.20, 2.45, 0.70))
        # DrawerPull.__init__ hardcodes one drawer, so go straight to CupTransfer
        # and supply the target ourselves. Everything else about DrawerPull -- its
        # phases, IK targets, grasp verification -- is inherited untouched.
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
        """Light the room properly instead of relying on the headlight.

        As compiled, the three ceiling lights carry diffuse 0.06 each while a
        camera-mounted headlight (ambient 0.18, diffuse 0.10) does most of the
        work. A headlight casts no shadows and moves with the eye, so everything
        renders flat and washed out. Push the real lights up, drop the headlight
        to a fill, and warm the key slightly.
        """
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
        """Hide visualisation-only sites from every rendered frame.

        Site group 1 holds robosuite's `gripperright_grip_site_cylinder`: a green
        cylinder 10 METRES long that crosses the whole image. Group 2 holds the
        scene's teal annotation markers. Neither is part of the kitchen, and the
        green one is not merely ugly -- a model asked what it sees reports "a
        thin green elongated object", so leaving it in corrupts perception as
        well as the video.
        """
        option = mujoco.MjvOption()
        option.sitegroup[1] = 0
        option.sitegroup[2] = 0
        return option

    # -- world construction -------------------------------------------------

    def decorate(self, spec, body):
        """Add the camera the model lacks, and hide the jars to be found.

        The default pose looks DOWN into the bank from above-right. A camera level
        with the drawer fronts sees the fronts perfectly and the open drawer's
        contents not at all -- verified by rendering both: level views report an
        open drawer as a dark cavity, this one shows the jar inside it.
        """
        spec.worldbody.add_camera(
            name="drawer_view", pos=self._camera_pos,
            xyaxes=look_at(self._camera_pos, self._camera_target),
        )
        for index, place in enumerate(self._placements):
            # A jar is parented to its drawer body, so it travels with the drawer
            # and only becomes visible once that drawer is actually pulled open.
            drawer_body = spec.body(place.drawer)
            # Seat the jar on the drawer floor, deep inside the box. The body
            # origin is the FRONT PANEL, not the box centre: the box runs +y to
            # y=0.53 with its floor at z=-0.035 and side walls topping out at
            # z=+0.052. A jar at z=+0.055 -- one z of guesswork -- clears those
            # walls and is visible with every drawer shut, which makes the search
            # solvable without opening anything. Sit it at JAR_HALF above the
            # floor so the walls and front panel hide it until the drawer is out.
            jar = drawer_body.add_body(
                name=f"target_{index}", pos=[0.0, 0.26, -0.035 + JAR_HALF])
            # Visual-only, exactly like this scene's parked Mobile ALOHA geometry.
            # A collidable jar wedges against the cabinet frame and stops the pull
            # dead at 0.0144m instead of 0.2995m -- measured, not theorised. The
            # jar exists to be SEEN, so it takes no part in contact: contype and
            # conaffinity 0, zero mass. Nothing here can be picked up.
            jar.add_geom(
                name=f"target_{index}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.026, JAR_HALF, 0], mass=0, contype=0, conaffinity=0, group=1,
                rgba=[*JAR_COLOURS.get(place.label, (0.5, 0.5, 0.5)), 1.0],
            )

    # -- retargeting without losing accumulated world state -----------------

    def retarget(self, drawer_name):
        """Aim the skill at another drawer, preserving every drawer already open.

        This is DrawerPull.reset() minus CupTransfer.reset()'s mj_resetData: the
        handle, joint address and phase counters are recomputed, but the drawers'
        qpos is left alone. A search that re-closed the kitchen on every step
        would not be a long-horizon search.

        The ARM, however, must go home. DrawerPull's phases blend from
        phase_origin toward handle-relative targets on the assumption that the
        arm starts at the solved home pose; starting from wherever the previous
        attempt ended invalidates the whole trajectory. Left unfixed this is
        silent and total: the first pull of a run behaves exactly as measured in
        isolation and every pull after it reports opening=0.000.
        """
        if drawer_name not in DRAWERS:
            raise ValueError(f"Unknown drawer: {drawer_name}")
        self.drawer_name = drawer_name
        # Restore only the robot's own degrees of freedom, never the drawers'.
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
        """A separate, higher-resolution renderer for video.

        Kept apart from render(): that one is pinned to 320x320 because it feeds
        the model, and the contract with Astra should not silently change when
        somebody wants a nicer picture.
        """
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
        """Handle height and stack for each drawer, from the model.

        The belief role is given this because drawer NAMES are not visually
        inferable: nothing in an image distinguishes drawer0_1 from drawer0_3,
        and without a mapping the model attributes what it sees to whichever
        drawer it can pick out -- in testing, consistently reporting the jar in
        an empty drawer. Heights are mechanism geometry, not target knowledge.
        """
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
