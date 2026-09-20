"""Contact-driven Panda drawer pull, separate from the seasoning placement."""

import json

import mujoco
import numpy as np

from motion_demo import CupTransfer, ROOT, mink


class DrawerPull(CupTransfer):
    robot_position = (0, 1.90, 0)
    robot_yaw = 90
    robot_mount_height = 0.58
    home_position = (0, 2.18, 1.0)
    durations = (3.0, 2.0, 1.0, 4.0, 1.0, 1.5, 1.0)
    labels = ("Approach drawer", "Align with handle", "Grasp handle", "Pull drawer",
              "Release handle", "Retract", "Verify opening")

    def __init__(self):
        self.drawer_name = "range_drawers_drawer0_3"
        super().__init__()
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.impratio = 10
        self.model.opt.noslip_iterations = 10

    def reset(self):
        super().reset()
        self.drawer_id = self.model.body(self.drawer_name).id
        self.drawer_q = self.model.jnt_qposadr[self.model.joint(self.drawer_name + "_slide").id]
        self.handle = self.data.site_xpos[self.model.site(self.drawer_name + "_handle").id].copy()
        self.grip_frames = self.pull_frames = 0

    def target(self):
        handle = self.handle + [0, 0.0036, 0]
        return np.array((handle + [0, -0.15, 0.18], handle, handle,
                         handle + [0, -0.30, 0], handle + [0, -0.30, 0],
                         handle + [0, -0.40, 0.13], handle + [0, -0.40, 0.13])[self.phase])

    def prepare(self):
        self.phase_time += self.dt
        u = min(1.0, self.phase_time / self.durations[self.phase])
        blend = u * u * u * (10 + u * (-15 + 6 * u))
        self.rotation = mink.SO3.from_matrix(np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]]))
        goal = self.phase_origin + blend * (self.target() - self.phase_origin)
        return self.solve(goal), 0.0 if 2 <= self.phase <= 3 else 0.04

    def integrate(self, joints, opening):
        self.advance_physics(joints, opening)
        self.frame += 1
        self.last_error = float(np.linalg.norm(self.data.site_xpos[self.ee_id] - self.target()))
        if self.phase == 3:
            self.pull_frames += 1
            fingers = set()
            for contact in self.data.contact:
                if self.drawer_id in self.model.geom_bodyid[list(contact.geom)]:
                    fingers.update(self.model.geom(g).name for g in contact.geom
                                   if "pad_collision" in self.model.geom(g).name)
            self.grip_frames += len(fingers) == 2
        if self.phase_time > self.durations[self.phase] + 3:
            self.finished, self.error = True, "Drawer target not reached"
        elif self.phase_time >= self.durations[self.phase] and self.last_error < 0.04:
            self.phase_log.append({"phase": self.labels[self.phase], "opening": float(self.data.qpos[self.drawer_q]),
                                   "error": self.last_error})
            if self.phase == len(self.labels) - 1:
                self.finished = True
                self.success = bool(self.data.qpos[self.drawer_q] > 0.25 and self.grip_frames / max(1, self.pull_frames) > 0.9)
                if not self.success:
                    self.error = "Drawer opening or finger contact verification failed"
            else:
                self.phase += 1
                self.phase_time = 0
                self.phase_origin = self.data.site_xpos[self.ee_id].copy()
        return self.snapshot()

    def snapshot(self):
        state = super().snapshot()
        state.update(opening_m=float(self.data.qpos[self.drawer_q]),
                     contact_fraction=self.grip_frames / max(1, self.pull_frames),
                     progress=1.0 if self.finished else (self.phase + min(1, self.phase_time / self.durations[self.phase])) / len(self.labels))
        return state


if __name__ == "__main__":
    demo = DrawerPull()
    for _ in range(1200):
        demo.step()
        if demo.finished:
            break
    report = {**demo.snapshot(), "phases": demo.phase_log,
              "mount_height_m": demo.robot_mount_height,
              "limitations": "Scripted contact-driven opening; not a learned search policy"}
    (ROOT / "output/drawer-motion-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if demo.success else 1)
