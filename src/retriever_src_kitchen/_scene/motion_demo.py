"""A small physical cup-transfer demo using Mink IK and MuJoCo contacts."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/motion"))
import mink

from integration import load_scene


class CupTransfer:
    robot_position = (0.5, 1.77, 0)
    robot_yaw = -90
    robot_mount_height = None
    home_position = (0.50, 1.47, 1.23)
    home_rotation = np.diag([1.0, -1.0, -1.0])
    dt = 1 / 50
    durations = (2.2, 1.6, 1.2, 1.7, 2.8, 1.7, 1.0, 1.7, 1.2)
    source = np.array([0.70, 1.15, 0.908])
    destination = np.array([0.20, 1.15, 0.908])
    labels = ("Approach cup", "Align gripper", "Grasp cup", "Lift cup",
              "Move beside sink", "Lower cup", "Release cup", "Retract", "Verify placement")

    def __init__(self):
        _, data, spec = load_scene("Panda", position=self.robot_position, yaw_degrees=self.robot_yaw,
                                   mount_height=self.robot_mount_height)
        robot_q = data.qpos.copy()
        for geom in spec.geoms:
            if "pad_collision" in geom.name:
                geom.friction = [1.5, 0.02, 0.001]
                geom.condim = 6
                geom.solref = [0.01, 1]
                geom.priority = 2
        body = spec.worldbody.add_body(name="demo_cup", pos=self.source)
        body.add_freejoint(name="cup_joint")
        body.add_geom(name="cup_collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[0.030, 0.045, 0], mass=0.08, group=0,
                      rgba=[0, 0, 0, 0], friction=[0.9, 0.02, 0.001], condim=6,
                      solref=[0.01, 1])
        # A thin open ceramic cup shell; the interior is visible in the video.
        for i in range(32):
            a = 2 * np.pi * i / 32
            body.add_geom(name=f"cup_shell_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[0.028 * np.cos(a), 0.028 * np.sin(a), 0],
                          quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
                          size=[0.003, 0.003, 0.045], mass=0, contype=0, conaffinity=0,
                          group=1, rgba=[0.05, 0.38, 0.30, 1])
        body.add_geom(name="cup_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      pos=[0, 0, -0.041], size=[0.028, 0.004, 0], mass=0,
                      contype=0, conaffinity=0, group=1, rgba=[0.05, 0.38, 0.30, 1])
        self.decorate(spec, body)
        self.model = spec.compile()
        self.model.opt.timestep = 0.001
        # High finger servo gains maintain contact during the scripted pull.
        self.model.actuator_gainprm[7:9, 0] = 8000
        self.model.actuator_biasprm[7:9, 1:3] = [-8000, -200]
        self.model.actuator_forcerange[7:9] = [-100, 100]
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [self.model.joint(f"robot/robot0_joint{i}").id for i in range(1, 8)]
        self.arm_q = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.arm_v = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.finger_q = [self.model.jnt_qposadr[self.model.joint(f"robot/gripperright_finger_joint{i}").id]
                         for i in (1, 2)]
        self.cup_q = self.model.jnt_qposadr[self.model.joint("cup_joint").id]
        self.cup_id = self.model.body("demo_cup").id
        self.ee_id = self.model.site("robot/gripperright_grip_site").id
        self.initial_q = self.model.qpos0.copy()
        self.initial_q[:len(robot_q)] = robot_q
        self.initial_q[self.finger_q] = [0.04, -0.04]
        self.configuration = mink.Configuration(self.model, q=self.initial_q)
        self.task = mink.FrameTask("robot/gripperright_grip_site", "site", 1.0, 0.5, lm_damping=1e-3)
        self.posture = mink.PostureTask(self.model, cost=1e-3)
        self.posture.set_target(self.initial_q)
        self.limits = [mink.ConfigurationLimit(self.model), mink.VelocityLimit(
            self.model, {self.model.joint(j).name: 1.2 for j in self.joint_ids})]
        self.rotation = mink.SO3.from_matrix(self.home_rotation)
        # Solve a raised home pose before enabling physics.
        for _ in range(150):
            self.solve(np.array(self.home_position))
        self.initial_q[self.arm_q] = self.configuration.q[self.arm_q]
        self.reset()

    def decorate(self, spec, body):
        pass

    def solve(self, target):
        self.task.set_target(mink.SE3.from_rotation_and_translation(self.rotation, target))
        velocity = mink.solve_ik(self.configuration, [self.task, self.posture], self.dt,
                                 solver="quadprog", damping=1e-4, limits=self.limits)
        self.configuration.integrate_inplace(velocity, self.dt)
        return self.configuration.q[self.arm_q].copy()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_q
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.phase = self.frame = 0
        self.phase_time = 0.0
        self.phase_origin = self.data.site_xpos[self.ee_id].copy()
        self.finished = self.success = False
        self.error = ""
        self.peak_height = float(self.source[2])
        self.last_error = 0.0
        self.phase_log = []
        self.place_ee = None
        self.settle_positions = []

    def target(self):
        s, d = self.source, self.destination
        points = [s + [0, 0, 0.20], s, s,
                  s + [0, 0, 0.22], d + [0, 0, 0.22], d,
                  d, d + [0, 0, 0.23], d + [0, 0, 0.23]]
        if self.phase == 5:
            offset = self.data.site_xpos[self.ee_id] - self.data.xpos[self.cup_id]
            points[5] = d + np.clip(offset, -0.08, 0.08) + [0, 0, 0.002]
        elif self.phase == 6:
            points[6] = self.place_ee
        return np.array(points[self.phase])

    def prepare(self):
        self.phase_time += self.dt
        u = min(1.0, self.phase_time / self.durations[self.phase])
        smooth = u * u * u * (10 + u * (-15 + 6 * u))
        target = self.phase_origin + smooth * (self.target() - self.phase_origin)
        q_target = self.solve(target)
        opening = 0.0 if 2 <= self.phase <= 5 else 0.04
        return q_target, opening

    def advance_physics(self, q_target, opening):
        # Torque-controlled Panda with gravity compensation, plus finger servos.
        for _ in range(round(self.dt / self.model.opt.timestep)):
            torque = (np.array([700, 700, 550, 550, 280, 180, 120]) *
                      (q_target - self.data.qpos[self.arm_q]) -
                      np.array([100, 90, 80, 70, 35, 28, 20]) * self.data.qvel[self.arm_v] +
                      self.data.qfrc_bias[self.arm_v])
            self.data.ctrl[:7] = np.clip(torque, self.model.actuator_ctrlrange[:7, 0], self.model.actuator_ctrlrange[:7, 1])
            self.data.ctrl[7:9] = [opening, -opening]
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def integrate(self, q_target, opening):
        self.advance_physics(q_target, opening)
        self.frame += 1
        cup = self.data.xpos[self.cup_id]
        if self.phase == 8:
            self.settle_positions.append(cup.copy())
            self.settle_positions = self.settle_positions[-25:]
        self.peak_height = max(self.peak_height, float(cup[2]))
        self.last_error = float(np.linalg.norm(self.data.site_xpos[self.ee_id] - self.target()))
        if not np.isfinite(self.data.qpos).all():
            self.error = "Simulation became unstable"
            self.finished = True
        elif self.phase_time > self.durations[self.phase] + 2.5:
            self.error = "Motion target was not reached"
            self.finished = True
        elif self.phase_time >= self.durations[self.phase] and self.last_error < 0.035:
            self.phase_log.append({"phase": self.labels[self.phase], "frame": self.frame,
                                   "ee_error": self.last_error, "cup": cup.tolist()})
            if self.phase == 8:
                self.finished = True
                self.success = bool(np.linalg.norm(cup[:2] - self.destination[:2]) < 0.05 and
                                    abs(cup[2] - self.destination[2]) < 0.018 and
                                    self.peak_height > self.source[2] + 0.10 and
                                    self.data.xmat[self.cup_id, 8] > 0.98 and
                                    len(self.settle_positions) == 25 and
                                    np.max(np.ptp(self.settle_positions, axis=0)) < 0.001 and
                                    np.linalg.norm(self.data.qvel[-6:-3]) < 0.02)
                if not self.success:
                    self.error = "Cup placement did not meet the verification tolerance"
            else:
                if self.phase == 5:
                    self.place_ee = self.data.site_xpos[self.ee_id].copy()
                self.phase += 1
                self.phase_time = 0.0
                self.phase_origin = self.data.site_xpos[self.ee_id].copy()
        return self.snapshot()

    def step(self):
        if self.finished:
            return self.snapshot()
        return self.integrate(*self.prepare())

    def snapshot(self):
        return {"frame": self.frame, "phase": self.phase, "skill": self.labels[self.phase],
                "finished": self.finished, "success": self.success, "error": self.error,
                "cup_position": self.data.xpos[self.cup_id].tolist(),
                "ee_position": self.data.site_xpos[self.ee_id].tolist(),
                "target_error_m": self.last_error, "sim_time": self.data.time,
                "peak_cup_height": self.peak_height,
                "cup_upright": float(self.data.xmat[self.cup_id, 8]),
                "settle_drift_m": float(np.max(np.ptp(self.settle_positions, axis=0)))
                    if len(self.settle_positions) == 25 else None,
                "progress": 1.0 if self.finished else
                    (self.phase + min(1.0, self.phase_time / self.durations[self.phase])) / 9}


if __name__ == "__main__":
    demo = CupTransfer()
    for _ in range(2500):
        state = demo.step()
        if state["frame"] % 100 == 0:
            print(json.dumps(state), flush=True)
        if demo.finished:
            break
    report = {**demo.snapshot(), "phases": demo.phase_log}
    (ROOT / "output/motion-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if demo.success else 1)
