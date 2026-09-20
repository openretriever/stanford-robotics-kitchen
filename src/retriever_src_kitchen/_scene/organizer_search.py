"""Contact-driven search in the small countertop organizer from the reference."""

import json

import mujoco
import numpy as np

from motion_demo import CupTransfer, ROOT, mink
from monitor_branding import add_monitor_logos, upload_monitor_logos


class OrganizerSearch(CupTransfer):
    placement_rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    riser_height = .12
    organizer_center = np.array([0.72, 0.85, riser_height])
    placement_quat = [np.sqrt(.5), 0, 0, -np.sqrt(.5)]
    robot_position = (1.63, 0.85, 0)
    robot_yaw = -180
    robot_mount_height = 0.70
    home_position = (1.26, 0.85, 1.36)
    home_rotation = placement_rotation @ CupTransfer.home_rotation
    grasp_rotation = placement_rotation @ np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
    source = np.array([0, 0, -1.0])
    drawer_coordinates = ((0, 1), (1, 1), (0, 0), (1, 0))
    drawer_locations = ("top left", "top right", "bottom left", "bottom right")
    inventory = ("paprika", "garlic powder", "herb seasoning", "black pepper")
    target_index = 2
    action_labels = ("Approach", "Align with", "Grasp", "Open", "Inspect", "Close", "Release", "Clear", "Verify")
    action_durations = (2.5, 1.5, .8, 2.5, 1.2, 2.5, .6, 1.2, .5)
    transfer_labels = ("Approach seasoning", "Align with seasoning", "Grasp seasoning",
                       "Lift seasoning from drawer", "Carry seasoning to counter", "Lower seasoning",
                       "Release seasoning on counter", "Clear placed seasoning",
                       "Approach open drawer 3", "Align with drawer 3 handle", "Grasp drawer 3 handle",
                       "Close drawer 3", "Release drawer 3 handle", "Retract from closed drawer",
                       "Raise clear of organizer", "Return above organizer", "Turn toward drawer")
    transfer_durations = (2.5, 2.5, 1., 2., 3., 2., 1., 1.5, 2.5, 4., .8, 2.5, .6, 1.5, 2., 2.5, 2.5)
    placement = np.array([.98, .43, .887])

    def decorate(self, spec, body):
        add_monitor_logos(spec)
        # The shared controller's legacy cup is inactive and outside the scene.
        body.gravcomp = 1
        for geom in body.geoms:
            geom.contype = geom.conaffinity = 0
            geom.group = 0
        wood = [0.80, 0.68, 0.49, 1]
        inner = [0.88, 0.77, 0.59, 1]
        white = [0.93, 0.93, 0.88, 1]

        def box(parent, name, pos, size, color, mass=0.03):
            return parent.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                                   pos=pos, size=np.array(size) / 2, rgba=color,
                                   mass=mass, group=1, friction=[1.2, 0.02, 0.001], condim=6)

        riser = spec.worldbody.add_body(name="organizer_riser",
            pos=self.organizer_center + [0, 0, .867 - self.riser_height / 2], quat=self.placement_quat)
        box(riser, "organizer_riser_base", [0, 0, 0], [.58, .36, self.riser_height], [.89, .90, .88, 1])
        unit = spec.worldbody.add_body(name="countertop_organizer",
            pos=self.organizer_center + [0, 0, .867], quat=self.placement_quat)
        for name, pos, size in (
            ("bottom", [0, 0, 0.009], [.54, .32, .018]),
            ("top", [0, 0, .331], [.54, .32, .018]),
            ("left", [-.263, 0, .17], [.014, .32, .34]),
            ("right", [.263, 0, .17], [.014, .32, .34]),
            ("divider", [0, 0, .17], [.012, .32, .31]),
            ("back", [0, -.154, .17], [.512, .012, .31]),
            ("shelf", [0, 0, .17], [.512, .30, .012]),
        ):
            box(unit, "organizer_" + name, pos, size, wood)
        # Fine straight bamboo laminations are geometry, shared by all renderers.
        for i in range(25):
            box(unit, f"organizer_grain_{i}", [-.255 + i * .021, 0, .3402],
                [.0004, .316, .0004], [.69, .57, .40, 1], mass=0)
        for row, z in enumerate((.095, .249)):
            for column, x in enumerate((.135, -.135)):
                name = f"organizer_drawer_{column}_{row}"
                drawer = unit.add_body(name=name, pos=[x, .16, z])
                drawer.add_joint(name=name + "_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                                 axis=[0, 1, 0], range=[0, .23], limited=True,
                                 damping=2, frictionloss=.12, armature=.01)
                for part, pos, size in (
                    ("front", [0, .007, 0], [.240, .014, .132]),
                    ("bottom", [0, -.136, -.054], [.222, .272, .012]),
                    ("left", [-.109, -.136, .002], [.012, .272, .100]),
                    ("right", [.109, -.136, .002], [.012, .272, .100]),
                    ("back", [0, -.266, .002], [.218, .012, .100]),
                ):
                    box(drawer, name + "_" + part, pos, size, wood if part == "front" else inner)
                for side in (-1, 1):
                    box(drawer, name + f"_handle_support_{side}", [side * .046, .032, 0],
                        [.014, .043, .015], white)
                box(drawer, name + "_handle", [0, .054, 0], [.106, .014, .015], white)
                drawer.add_site(name=name + "_grip", pos=[0, .054, 0], size=[.004, .004, .004], group=2)
                drawer.add_site(name=name + "_inside", pos=[0, -.10, -.043], size=[.004, .004, .004], group=2)
                index = self.drawer_coordinates.index((column, row))
                spice = spec.worldbody.add_body(name=f"organizer_spice_{column}_{row}",
                    pos=self.organizer_center + self.placement_rotation @ np.array([x, .11 if index == self.target_index else .06, .867 + z - .019]),
                    quat=self.placement_quat)
                spice.add_freejoint(name=f"organizer_spice_{column}_{row}_free")
                color = ([.58, .19, .06, 1], [.72, .62, .37, 1],
                         [.12, .38, .18, 1], [.20, .19, .17, 1])[index]
                box(spice, f"organizer_spice_{column}_{row}_jar", [0, 0, 0], [.040, .075, .040], color, .05)
                box(spice, f"organizer_spice_{column}_{row}_cap", [0, .042, 0], [.042, .012, .042], white, .006)
                box(spice, f"organizer_spice_{column}_{row}_label", [0, 0, .0202], [.032, .041, .0004], white, 0)

        plate = spec.worldbody.add_body(name="search_food_plate", pos=[.30, .52, .874])
        plate.add_geom(name="search_plate_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[.14, .006, 0], group=1, rgba=[.48, .64, .24, 1])
        for i in range(48):
            a, b = 2 * np.pi * i / 48, 2 * np.pi * (i + 1) / 48
            plate.add_geom(name=f"search_plate_rim_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[.13 * np.cos(a), .13 * np.sin(a), .009, .13 * np.cos(b), .13 * np.sin(b), .009],
                size=[.009, 0, 0], group=1, contype=0, conaffinity=0, rgba=[.48, .64, .24, 1])
        plate.add_geom(name="search_food_bread", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=[.045, .025, .023], size=[.048, .069, .014], group=1, rgba=[.92, .83, .57, 1], contype=0, conaffinity=0)
        for i in range(3):
            plate.add_geom(name=f"search_food_steak_{i}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                pos=[-.047, -.023 + i * .036, .025], size=[.034, .021, .016], group=1,
                rgba=[.33, .105 + i * .012, .045, 1], contype=0, conaffinity=0)
        for i in range(9):
            for j in range(5):
                a = np.pi * j / 4
                plate.add_geom(name=f"search_food_corn_{i}_{j}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                    pos=[-.05 + i * .012, -.076 + .017 * np.cos(a), .018 + .017 * np.sin(a)],
                    size=[.008, .006, .006], group=1, rgba=[.96, .60 + .025 * (i % 3), .04, 1], contype=0, conaffinity=0)
        plate.add_site(name="search_seasoning_target", pos=[0, 0, .15], size=[.008, .008, .008], group=2)

    def __init__(self, use_memory=False):
        self.use_memory = use_memory
        super().__init__()
        upload_monitor_logos(self.model)
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.impratio = 10
        self.model.opt.noslip_iterations = 10

    def reset(self):
        super().reset()
        if not hasattr(self, "memory"):
            self.memory = {}
            self.run_number = 0
        self.run_number += 1
        self.run_memory = {}
        self.memory_used = self.use_memory and self.memory.get(self.drawer_locations[self.target_index]) == "herb seasoning"
        self.search_order = (self.target_index,) if self.memory_used else (0, 1, 2)
        self.sequence = tuple((drawer, action) for drawer in self.search_order
                              for action in ((0, 1, 2, 3, 4, 6, 7, 23, *range(9, 17), 24, 25, *range(17, 23), 8)
                                             if drawer == self.target_index else range(8)))
        self.labels = tuple(self.transfer_labels[action - 9] if action >= 9 else
                            "Verify retrieval and closed drawers" if action == 8 else
                            f"{self.action_labels[action]} drawer {drawer + 1} ({self.drawer_locations[drawer]})"
                            for drawer, action in self.sequence)
        self.durations = tuple(self.transfer_durations[action - 9] if action >= 9 else self.action_durations[action]
                               for _, action in self.sequence)
        self.drawers = [f"organizer_drawer_{column}_{row}" for column, row in self.drawer_coordinates]
        self.handles = [self.data.site_xpos[self.model.site(name + "_grip").id].copy() for name in self.drawers]
        self.qaddrs = [self.model.jnt_qposadr[self.model.joint(name + "_slide").id] for name in self.drawers]
        self.observations = []
        self.found = False
        self.pull_frames = self.contact_frames = 0
        column, row = self.drawer_coordinates[self.target_index]
        self.spice_id = self.model.body(f"organizer_spice_{column}_{row}").id
        self.spice_dof = self.model.jnt_dofadr[self.model.joint(f"organizer_spice_{column}_{row}_free").id]
        self.pick_position = self.data.xpos[self.spice_id].copy()
        self.pick_ee = self.pick_position + [0, 0, .014]
        self.placement_ee = self.placement + [0, 0, .014]
        self.lift_peak = float(self.pick_position[2])
        self.carry_frames = self.carry_contact_frames = 0
        self.object_placed = False
        self.phase_origin_q = self.data.qpos[self.arm_q].copy()
        self.open_handle_q = self.phase_origin_q.copy()

    @property
    def action(self):
        return self.sequence[self.phase][1]

    @property
    def drawer_index(self):
        return self.sequence[self.phase][0]

    def target(self):
        handle = self.handles[self.drawer_index] + self.placement_rotation @ [0, -.0036, 0]
        raised = self.placement_rotation @ [0, .14, .18]
        opened = self.placement_rotation @ [0, .20, 0]
        retreat = self.placement_rotation @ [0, .10, .08]
        release = handle + opened if self.drawer_index == self.target_index else handle
        clear = handle + opened + retreat if self.drawer_index == self.target_index else handle + raised
        if self.action == 24:
            return np.array(self.home_position)
        if self.action in (23, 25):
            return handle + opened + retreat + [0, 0, .25]
        if self.action >= 9:
            return np.array((self.pick_ee + [0, 0, .20], self.pick_ee, self.pick_ee,
                             self.pick_ee + [0, 0, .18], self.placement_ee + [0, 0, .22],
                             self.placement_ee, self.placement_ee, self.placement_ee + [0, 0, .20],
                             handle + opened + retreat, handle + opened, handle + opened,
                             handle, handle, handle + raised)[self.action - 9])
        if self.action == 8:
            return handle + raised
        return np.array((handle + raised, handle, handle, handle + opened, handle + opened,
                         handle, release, clear, clear)[self.action])

    def prepare(self):
        self.phase_time += self.dt
        u = min(1, self.phase_time / self.durations[self.phase])
        blend = u ** 3 * (10 + u * (-15 + 6 * u))
        self.rotation = mink.SO3.from_matrix(self.home_rotation if 9 <= self.action <= 16 or self.action == 24 else self.grasp_rotation)
        if self.action == 18:
            # Return through actuators to the measured handle pose, preserving
            # the arm configuration used for opening instead of switching IK branches.
            return self.phase_origin_q + blend * (self.open_handle_q - self.phase_origin_q), .04
        goal = self.phase_origin + blend * (self.target() - self.phase_origin)
        return self.solve(goal), 0 if self.action in (2, 3, 4, 5, 11, 12, 13, 14, 19, 20) else .04

    def inspect(self, index):
        index = int(index)
        opening = float(self.data.qpos[self.qaddrs[index]])
        center = self.data.site_xpos[self.model.site(self.drawers[index] + "_inside").id]
        column, row = self.drawer_coordinates[index]
        position = self.data.xpos[self.model.body(f"organizer_spice_{column}_{row}").id]
        visible = opening > .16 and np.linalg.norm(position[:2] - center[:2]) < .10
        item = self.inventory[index] if visible else None
        observation = {"drawer": self.drawer_locations[index], "opening_m": opening,
                       "item": item, "match": item == "herb seasoning",
                       "source": "Simulator-state observation after opening, not camera recognition"}
        self.observations.append(observation)
        if visible:
            self.memory[observation["drawer"]] = item
            self.run_memory[observation["drawer"]] = item
        self.found = observation["match"]
        if self.found:
            self.pick_position = position.copy()
            self.pick_ee = position + [0, 0, .014]
            self.open_handle_q = self.data.qpos[self.arm_q].copy()
        return visible

    def spice_is_placed(self):
        return bool(np.linalg.norm(self.data.xpos[self.spice_id] - self.placement) < .025 and
                    np.linalg.norm(self.data.qvel[self.spice_dof:self.spice_dof + 6]) < .03 and
                    all(abs(self.data.qpos[q]) > .035 for q in self.finger_q))

    def integrate(self, joints, opening):
        self.advance_physics(joints, opening)
        self.frame += 1
        self.last_error = float(np.linalg.norm(self.data.site_xpos[self.ee_id] - self.target()))
        spice = self.data.xpos[self.spice_id]
        self.lift_peak = max(self.lift_peak, float(spice[2]))
        self.settle_positions.append(spice.copy())
        self.settle_positions = self.settle_positions[-25:]
        if self.action in (12, 13):
            pads = set()
            for contact in self.data.contact:
                if self.spice_id in self.model.geom_bodyid[list(contact.geom)]:
                    pads.update(self.model.geom(g).name for g in contact.geom if "pad_collision" in self.model.geom(g).name)
            self.carry_frames += 1
            self.carry_contact_frames += len(pads) == 2
        if self.action == 3:
            drawer = self.model.body(self.drawers[self.drawer_index]).id
            pads = set()
            for contact in self.data.contact:
                if drawer in self.model.geom_bodyid[list(contact.geom)]:
                    pads.update(self.model.geom(g).name for g in contact.geom if "pad_collision" in self.model.geom(g).name)
            self.pull_frames += 1
            self.contact_frames += len(pads) == 2
        if not np.isfinite(self.data.qpos).all() or self.phase_time > self.durations[self.phase] + 3:
            self.finished, self.error = True, "Organizer motion target was not reached"
        elif self.phase_time >= self.durations[self.phase] and self.last_error < .025:
            if self.action == 4 and not self.inspect(self.drawer_index):
                self.finished, self.error = True, "Drawer did not expose its contents"
            self.phase_log.append({"phase": self.labels[self.phase], "opening_m": [float(self.data.qpos[q]) for q in self.qaddrs],
                                   "spice": spice.tolist(), "ee": self.data.site_xpos[self.ee_id].tolist(),
                                   "carry_contact_fraction": self.carry_contact_frames / max(1, self.carry_frames)})
            if self.action == 12:
                offset = self.data.site_xpos[self.ee_id] - spice
                self.placement_ee = self.placement + np.clip(offset, -.05, .05)
            if self.action == 16:
                self.object_placed = self.spice_is_placed()
                if not self.object_placed:
                    self.finished, self.error = True, "Seasoning was not released on the counter"
            if self.action == 18:
                self.configuration.update(self.data.qpos)
            if self.action == 8:
                self.finished = True
                self.success = bool(self.found and len(self.observations) == len(self.search_order) and
                    all(abs(self.data.qpos[q]) < .012 for q in self.qaddrs) and
                    self.contact_frames / max(1, self.pull_frames) > .90 and
                    self.carry_contact_frames / max(1, self.carry_frames) > .90 and
                    self.lift_peak > self.pick_position[2] + .10 and
                    self.object_placed and self.spice_is_placed() and
                    len(self.settle_positions) == 25 and np.max(np.ptp(self.settle_positions, axis=0)) < .001)
                if not self.success:
                    self.error = "Retrieval, placement or drawer closure verification failed"
            elif not self.finished:
                self.phase += 1
                self.phase_time = 0
                self.phase_origin = self.data.site_xpos[self.ee_id].copy()
                self.phase_origin_q = self.data.qpos[self.arm_q].copy()
        return self.snapshot()

    def snapshot(self):
        state = super().snapshot()
        state.update(progress=1.0 if self.finished else (self.phase + min(1, self.phase_time / self.durations[self.phase])) / len(self.labels),
                     observations=list(self.observations), found=self.found,
                     opening_m=[float(self.data.qpos[q]) for q in self.qaddrs],
                     contact_fraction=self.contact_frames / max(1, self.pull_frames),
                     carry_contact_fraction=self.carry_contact_frames / max(1, self.carry_frames),
                     object_placed=self.object_placed, object_position=self.data.xpos[self.spice_id].tolist(),
                     object_location="counter" if self.object_placed else self.drawer_locations[self.target_index],
                     lift_peak=self.lift_peak,
                     memory=dict(self.memory if self.use_memory else self.run_memory),
                     memory_used=self.memory_used, run_number=self.run_number,
                     use_memory=self.use_memory, target_drawer=self.drawer_locations[self.target_index],
                     active_drawer=self.drawer_locations[self.drawer_index],
                     search_result="Herb seasoning: " + self.drawer_locations[self.target_index] if self.found else "Searching")
        return state


if __name__ == "__main__":
    demo = OrganizerSearch()
    for _ in range(4500):
        demo.step()
        if demo.finished:
            break
    first = {**demo.snapshot(), "phases": demo.phase_log}
    assert demo.success, demo.error
    assert demo.object_placed and max(abs(v) for v in first["opening_m"]) < .012
    assert [o["drawer"] for o in demo.observations] == ["top left", "top right", "bottom left"]
    demo.reset()
    assert not demo.memory_used and demo.search_order == (0, 1, 2) and len(demo.memory) == 3
    assert demo.snapshot()["memory"] == {}
    while not demo.observations and not demo.finished:
        demo.step()
    assert demo.snapshot()["memory"] == {"top left": "paprika"}
    demo.use_memory = True
    demo.reset()
    assert demo.memory_used and not demo.observations
    for _ in range(3500):
        demo.step()
        if demo.finished:
            break
    assert demo.success and len(demo.observations) == 1 and demo.frame < first["frame"], demo.error
    report = {"first_search": first, "memory_recall": {**demo.snapshot(), "phases": demo.phase_log}}
    demo.reset()
    for _ in range(3500):
        joints, opening = demo.prepare()
        if demo.action in (11, 12, 13, 14):
            opening = .04
        demo.integrate(joints, opening)
        if demo.finished:
            break
    assert demo.finished and not demo.success and not demo.object_placed
    assert demo.carry_contact_frames == 0
    report["failed_grasp_rejected"] = demo.snapshot()
    (ROOT / "output/organizer-search-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0)
