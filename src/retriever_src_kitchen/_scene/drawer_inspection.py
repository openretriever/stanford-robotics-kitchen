"""Paused scene inspection using the model's articulated drawer joints."""

import time

import mujoco
import numpy as np


class DrawerInspection:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.drawers = {}
        self.moves = {}
        self.active = False
        self.selected = None
        for joint in range(model.njnt):
            name = model.joint(joint).name
            if not name.endswith("_slide"):
                continue
            body = name.removesuffix("_slide")
            if body.startswith("organizer_drawer_"):
                column, row = map(int, body.removeprefix("organizer_drawer_").split("_"))
                group = "Tabletop organizer"
                label = ("Top" if row else "Bottom") + (" left" if column == 0 else " right")
                order = (0, 1 - row, column)
            elif body.startswith(("range_drawers_drawer", "beverage_drawers_drawer")):
                column, row = map(int, body.rsplit("drawer", 1)[1].split("_"))
                group = "Stove cabinets" if body.startswith("range_") else "Beverage cabinet"
                label = ("Left / " if column == 0 else "Right / ") if group == "Stove cabinets" else ""
                label += ("Bottom", "Lower middle", "Upper middle", "Top")[row]
                order = (1 if group == "Stove cabinets" else 2, 3 - row, column)
            else:
                continue
            self.drawers[name] = dict(id=name, body=int(model.jnt_bodyid[joint]),
                joint=joint, q=int(model.jnt_qposadr[joint]), v=int(model.jnt_dofadr[joint]),
                group=group, label=label, limit=float(model.jnt_range[joint, 1]), order=order)
        self.drawers = dict(sorted(self.drawers.items(), key=lambda item: item[1]["order"]))

    def snapshot(self):
        return dict(active=self.active, selected=self.selected, drawers=[
            dict(id=d["id"], group=d["group"], label=d["label"],
                 fraction=float(np.clip(self.data.qpos[d["q"]] / d["limit"], 0, 1)),
                 moving=d["id"] in self.moves)
            for d in self.drawers.values()])

    def clear(self):
        self.active = False
        self.moves.clear()

    def move(self, name, fraction=None, now=None):
        if not isinstance(name, str) or name not in self.drawers:
            raise ValueError("Unknown drawer")
        drawer = self.drawers[name]
        current = float(self.data.qpos[drawer["q"]])
        if fraction is None:
            target = self.moves[name][2] if name in self.moves else current
            fraction = 0 if target > drawer["limit"] / 2 else 1
        if type(fraction) not in (int, float) or not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Opening must be between 0 and 1")
        self.active = True
        self.selected = name
        self.moves[name] = (time.monotonic() if now is None else now, current, fraction * drawer["limit"])

    def update(self, now=None):
        if not self.moves:
            return False
        now = time.monotonic() if now is None else now
        for name, (start, initial, target) in list(self.moves.items()):
            drawer = self.drawers[name]
            t = float(np.clip((now - start) / .75, 0, 1))
            opening = initial + (target - initial) * t * t * (3 - 2 * t)
            delta = opening - self.data.qpos[drawer["q"]]
            if name.startswith("organizer_drawer_"):
                suffix = name.removeprefix("organizer_drawer_").removesuffix("_slide")
                spice = self.model.joint("organizer_spice_" + suffix + "_free")
                q, v = int(self.model.jnt_qposadr[spice.id]), int(self.model.jnt_dofadr[spice.id])
                rotation = self.data.xmat[drawer["body"]].reshape(3, 3)
                local = rotation.T @ (self.data.qpos[q:q + 3] - self.data.xpos[drawer["body"]])
                # Carry only contents still inside the tray; placed objects stay on the counter.
                if abs(local[0]) < .11 and -.28 < local[1] < .04 and -.06 < local[2] < .07:
                    self.data.qpos[q:q + 3] += delta * self.data.xaxis[drawer["joint"]]
                    self.data.qvel[v:v + 6] = 0
            self.data.qpos[drawer["q"]] = opening
            self.data.qvel[drawer["v"]] = 0
            if t == 1:
                del self.moves[name]
        mujoco.mj_forward(self.model, self.data)
        return True
