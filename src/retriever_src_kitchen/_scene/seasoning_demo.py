"""Contact-driven shaker choreography, using the kitchen's Mink/Panda controller."""

import json
import math

import mujoco
import numpy as np

from motion_demo import CupTransfer, ROOT, mink


class SeasoningDemo(CupTransfer):
    destination = CupTransfer.source.copy()
    plate = np.array([0.25, 1.18, 0.875])
    work = np.array([0.25, 1.24, 1.30])
    durations = (2.2, 1.6, 1.0, 1.7, 2.4, 2.8, 4.0, 2.8, 2.4, 1.7, 1.0, 1.7, 1.4)
    labels = ("Approach shaker", "Align gripper", "Grasp shaker", "Lift shaker",
              "Move over plate", "Tilt shaker", "Season the food", "Return upright",
              "Return shaker", "Lower shaker", "Release shaker", "Retract", "Verify seasoning")
    tilt_angle = math.radians(112)

    def __init__(self):
        super().__init__()
        # Resist soft-contact rotational creep during the sustained tilted grip.
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.impratio = 10
        self.model.opt.noslip_iterations = 10
        self.model.opt.iterations = 100

    def decorate(self, spec, body):
        # Finger hulls also touch the jar; match their torsional contact to the pads.
        for geom in spec.geoms:
            if "gripperright_finger" in geom.name and "collision" in geom.name:
                geom.friction = [1.5, 0.03, 0.001]
                geom.condim = 6
                geom.priority = 2
                geom.solref = [0.01, 1]
        for geom in body.geoms:
            if geom.name.startswith("cup_shell") or geom.name == "cup_bottom":
                geom.rgba = [0.33, 0.19, 0.08, 1]
        body.add_geom(name="shaker_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      pos=[0, 0, 0.044], size=[0.031, 0.006, 0], mass=0,
                      contype=0, conaffinity=0, group=1, rgba=[0.65, 0.67, 0.65, 1])
        for i in range(7):
            a = math.tau * i / 7
            body.add_geom(name=f"shaker_hole{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          pos=[0.018 * math.cos(a), 0.018 * math.sin(a), 0.0502],
                          size=[0.0025, 0.0003, 0], mass=0, contype=0, conaffinity=0,
                          group=1, rgba=[0.025, 0.028, 0.022, 1])
        plate = spec.worldbody.add_body(name="seasoning_plate", pos=self.plate)
        plate.add_geom(name="plate_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.14, 0.006, 0], contype=1, conaffinity=1,
                       group=1, rgba=[0.90, 0.93, 0.94, 1])
        for i in range(64):
            a, b = math.tau * i / 64, math.tau * (i + 1) / 64
            plate.add_geom(name=f"plate_rim{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                           fromto=[0.13 * math.cos(a), 0.13 * math.sin(a), 0.009,
                                   0.13 * math.cos(b), 0.13 * math.sin(b), 0.009],
                           size=[0.009, 0, 0], contype=0, conaffinity=0,
                           group=1, rgba=[0.90, 0.93, 0.94, 1])
        for i, (x, y, radius) in enumerate(((-0.045, 0.025, 0.037), (0.035, 0.045, 0.031),
                                           (0.027, -0.035, 0.035), (-0.047, -0.045, 0.028))):
            plate.add_geom(name=f"food{i}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                           pos=[x, y, 0.026], size=[radius, radius * 0.75, 0.022],
                           group=1, contype=0, conaffinity=0,
                           rgba=[0.29 + i * 0.025, 0.46 + i * 0.018, 0.11, 1])

    def reset(self):
        super().reset()
        self.max_tilt = 0.0
        self.seasoning_frames = 0
        self.shake_heights = []
        self.grip_frames = self.carry_frames = 0
        self.rotation = mink.SO3.from_matrix(np.diag([1., -1., -1.]))

    def target(self):
        s, work = self.source, self.work
        points = [s + [0, 0, 0.20], s, s, s + [0, 0, 0.24], work, work, work,
                  work, s + [0, 0, 0.24], s, s, s + [0, 0, 0.23], s + [0, 0, 0.23]]
        if self.phase == 9:
            offset = self.data.site_xpos[self.ee_id] - self.data.xpos[self.cup_id]
            points[9] = s + np.clip(offset, -0.06, 0.06) + [0, 0, 0.002]
        elif self.phase == 10:
            points[10] = self.place_ee
        return np.array(points[self.phase])

    def prepare(self):
        self.phase_time += self.dt
        u = min(1.0, self.phase_time / self.durations[self.phase])
        blend = u * u * u * (10 + u * (-15 + 6 * u))
        target = self.phase_origin + blend * (self.target() - self.phase_origin)
        angle = 0.0
        if self.phase == 5:
            angle = self.tilt_angle * blend
        elif self.phase == 6:
            wave = math.sin(math.tau * 1.25 * self.phase_time)
            angle = self.tilt_angle + math.radians(5) * wave
            target = self.work + [0, 0, 0.012 * wave]
        elif self.phase == 7:
            angle = self.tilt_angle * (1 - blend)
        c, s = math.cos(angle), math.sin(angle)
        self.rotation = mink.SO3.from_matrix(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ np.diag([1., -1., -1.]))
        return self.solve(target), 0.025 if 2 <= self.phase <= 9 else 0.04

    def integrate(self, joints, opening):
        self.advance_physics(joints, opening)
        self.frame += 1
        cup = self.data.xpos[self.cup_id]
        tilt = math.acos(float(np.clip(self.data.xmat[self.cup_id, 8], -1, 1)))
        self.max_tilt = max(self.max_tilt, tilt)
        self.peak_height = max(self.peak_height, float(cup[2]))
        if 3 <= self.phase <= 9:
            self.carry_frames += 1
            touching = set()
            for contact in self.data.contact:
                names = [self.model.geom(g).name for g in contact.geom]
                if any(self.model.geom_bodyid[g] == self.cup_id for g in contact.geom):
                    touching.update(n for n in names if "pad_collision" in n)
            if len(touching) >= 2:
                self.grip_frames += 1
        if self.phase == 6:
            cap = cup + self.data.xmat[self.cup_id].reshape(3, 3) @ [0, 0, 0.05]
            if tilt > math.radians(100) and np.linalg.norm(cap[:2] - self.plate[:2]) < 0.12 and cap[2] > self.plate[2] + 0.05:
                self.seasoning_frames += 1
                self.shake_heights.append(float(cap[2]))
        if self.phase == 12:
            self.settle_positions.append(cup.copy())
            self.settle_positions = self.settle_positions[-25:]
        self.last_error = float(np.linalg.norm(self.data.site_xpos[self.ee_id] - self.target()))
        if not np.isfinite(self.data.qpos).all():
            self.finished, self.error = True, "Simulation became unstable"
        elif self.phase_time > self.durations[self.phase] + 3.0:
            self.finished, self.error = True, "Motion target was not reached"
        elif self.phase_time >= self.durations[self.phase] and self.last_error < 0.04:
            self.phase_log.append({"phase": self.labels[self.phase], "frame": self.frame,
                                   "shaker": cup.tolist(), "tilt_degrees": math.degrees(tilt)})
            if self.phase == 12:
                self.finished = True
                self.success = bool(np.linalg.norm(cup[:2] - self.source[:2]) < 0.03 and
                                    abs(cup[2] - self.source[2]) < 0.015 and tilt < math.radians(8) and
                                    self.seasoning_frames >= 100 and np.ptp(self.shake_heights) > 0.015 and
                                    self.grip_frames / max(1, self.carry_frames) > 0.95 and
                                    len(self.settle_positions) == 25 and
                                    np.max(np.ptp(self.settle_positions, axis=0)) < 0.001)
                if not self.success:
                    self.error = "Seasoning or shaker return did not meet verification tolerances"
            else:
                if self.phase == 9:
                    self.place_ee = self.data.site_xpos[self.ee_id].copy()
                self.phase += 1
                self.phase_time = 0.0
                self.phase_origin = self.data.site_xpos[self.ee_id].copy()
        return self.snapshot()

    def snapshot(self):
        state = super().snapshot()
        state.update(progress=1.0 if self.finished else
                     (self.phase + min(1., self.phase_time / self.durations[self.phase])) / len(self.labels),
                     max_tilt_degrees=math.degrees(self.max_tilt),
                     seasoning_seconds=self.seasoning_frames * self.dt,
                     shake_travel_m=float(np.ptp(self.shake_heights)) if self.shake_heights else 0.,
                     contact_fraction=self.grip_frames / max(1, self.carry_frames))
        return state


if __name__ == "__main__":
    demo = SeasoningDemo()
    for _ in range(3000):
        state = demo.step()
        if demo.frame % 100 == 0:
            print(json.dumps(state), flush=True)
        if demo.finished:
            break
    report = {**demo.snapshot(), "phases": demo.phase_log}
    (ROOT / "output/seasoning-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if demo.success else 1)
