"""Parametric kitchen reconstruction, in meters, with native MJCF collisions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def numbers(values):
    return " ".join(f"{v:.6g}" for v in values)


class Kitchen:
    def __init__(self, config):
        self.config = config
        self.h = config["counter_height"]
        self.root = ET.Element("mujoco", model="walkthrough_kitchen")
        ET.SubElement(self.root, "compiler", angle="radian", autolimits="true")
        ET.SubElement(self.root, "option", timestep="0.002", integrator="implicitfast")
        visual = ET.SubElement(self.root, "visual")
        ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
        ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
        ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.4 0.4 0.4", specular="0.2 0.2 0.2")
        ET.SubElement(visual, "rgba", haze="0.96 0.97 0.98 1")
        default = ET.SubElement(self.root, "default")
        ET.SubElement(default, "geom", group="1", friction="0.8 0.01 0.001", condim="3", density="500")
        ET.SubElement(default, "joint", damping="4", armature="0.02", frictionloss="0.6")
        assets = ET.SubElement(self.root, "asset")
        self.assets = assets
        palette = {
            "cabinet": (0.78, 0.75, 0.51, 1),
            "cabinet_edge": (0.57, 0.53, 0.30, 1),
            "interior": (0.45, 0.43, 0.29, 1),
            "counter": (0.93, 0.94, 0.93, 1),
            "wall": (0.89, 0.91, 0.91, 1),
            "steel": (0.57, 0.60, 0.61, 1),
            "chrome": (0.71, 0.74, 0.75, 1),
            "dark": (0.045, 0.05, 0.052, 1),
            "glass": (0.65, 0.82, 0.84, 0.12),
            "screen": (0.085, 0.13, 0.16, 1),
            "floor": (0.57, 0.59, 0.58, 1),
            "border": (0.24, 0.26, 0.25, 1),
            "red": (0.55, 0.045, 0.035, 1),
            "blue": (0.08, 0.20, 0.42, 1),
            "green": (0.12, 0.34, 0.20, 1),
            "wood": (0.77, 0.73, 0.55, 1),
            "tracking_led": (0.25, 0.20, 0.95, 1),
        }
        for name, rgba in palette.items():
            ET.SubElement(assets, "material", name=name, rgba=numbers(rgba),
                          specular="0.65" if name in ("steel", "chrome") else "0.18",
                          shininess="0.65" if name in ("steel", "chrome") else "0.12")
        ET.SubElement(assets, "texture", name="sky", type="skybox", builtin="flat", rgb1="0.88 0.91 0.93", width="64", height="64")
        self.world = ET.SubElement(self.root, "worldbody")
        self.sites = {}

    def body(self, parent, name, pos=(0, 0, 0), yaw=0):
        return ET.SubElement(parent, "body", name=name, pos=numbers(pos), euler=f"0 0 {yaw:.16g}")

    def box(self, parent, name, pos, size, material, collision=True, group=1, **extra):
        return ET.SubElement(parent, "geom", name=name, type="box", pos=numbers(pos),
                             size=numbers(tuple(v / 2 for v in size)), material=material,
                             contype="1" if collision else "0", conaffinity="1" if collision else "0",
                             group=str(group), **extra)

    def rod(self, parent, name, a, b, radius=0.008, material="chrome", collision=False, group=1):
        if sum((x - y) ** 2 for x, y in zip(a, b)) < 1e-12:
            return None
        return ET.SubElement(parent, "geom", name=name, type="capsule", fromto=numbers((*a, *b)),
                             size=str(radius), material=material, contype="1" if collision else "0",
                             conaffinity="1" if collision else "0", group=str(group))

    def cylinder(self, parent, name, pos, radius, height, material, collision=True, **extra):
        return ET.SubElement(parent, "geom", name=name, type="cylinder", pos=numbers(pos),
                             size=numbers((radius, height / 2)), material=material,
                             contype="1" if collision else "0", conaffinity="1" if collision else "0", **extra)

    def rounded_panel(self, parent, name, pos, width, height, depth, radius, euler=None):
        points = []
        for cx, cy, start in ((width / 2 - radius, height / 2 - radius, 0),
                               (-width / 2 + radius, height / 2 - radius, 90),
                               (-width / 2 + radius, -height / 2 + radius, 180),
                               (width / 2 - radius, -height / 2 + radius, 270)):
            for step in range(7):
                a = math.radians(start + step * 90 / 6)
                points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        count = len(points)
        vertices = [(x, y, z) for z in (-depth / 2, depth / 2) for x, y in points]
        faces = []
        for i in range(count):
            j = (i + 1) % count
            faces.extend(((i, j, count + j), (i, count + j, count + i)))
        for i in range(1, count - 1):
            faces.extend(((0, i + 1, i), (count, count + i, count + i + 1)))
        ET.SubElement(self.assets, "mesh", name=f"{name}_mesh", vertex=numbers(v for xyz in vertices for v in xyz),
                      face=" ".join(str(v) for face in faces for v in face))
        geom = ET.SubElement(parent, "geom", name=name, type="mesh", mesh=f"{name}_mesh", pos=numbers(pos), material="cabinet")
        if euler:
            geom.set("euler", numbers(euler))
        return geom

    def site(self, name, position, size=(0.015, 0.015, 0.015)):
        ET.SubElement(self.world, "site", name=name, type="box", pos=numbers(position), size=numbers(size),
                      rgba="0.12 0.7 0.5 0.65", group="2")
        self.sites[name] = list(position)

    def pull(self, parent, name, x, y, z, horizontal=False, length=0.13):
        delta = (length / 2, 0, 0) if horizontal else (0, 0, length / 2)
        a = (x - delta[0], y, z - delta[2])
        b = (x + delta[0], y, z + delta[2])
        self.rod(parent, name, a, b, 0.005, collision=True)
        for i, p in enumerate((a, b)):
            self.rod(parent, f"{name}_mount{i}", p, (p[0], p[1] + 0.025, p[2]), 0.004)

    def cabinet(self, name, x, y, width, depth=None, yaw=0, upper=False, drawers=False,
                drawer_columns=1, top_drawers=False, door_count=None):
        depth = depth or self.config["counter_depth"]
        bottom, top = (1.30, 2.03) if upper else (0.085, self.h - 0.025)
        body = self.body(self.world, name, (x, y, 0), yaw)
        height = top - bottom
        for side, sx in (("left", -width / 2 + 0.009), ("right", width / 2 - 0.009)):
            self.box(body, f"{name}_{side}", (sx, 0, (top + bottom) / 2), (0.018, depth, height), "cabinet")
        for part, z in (("base", bottom + 0.009), ("top", top - 0.009)):
            if part == "top" and ("sink" in name or "beverage" in name):
                continue
            self.box(body, f"{name}_{part}", (0, 0, z), (width, depth, 0.018), "cabinet")
        self.box(body, f"{name}_back", (0, depth / 2 - 0.009, (top + bottom) / 2), (width, 0.018, height), "interior")
        if not drawers:
            self.box(body, f"{name}_shelf", (0, 0.02, (top + bottom) / 2), (width - 0.04, depth - 0.07, 0.018), "interior")
        if not upper:
            self.box(body, f"{name}_toe", (0, 0.035, 0.043), (width - 0.04, depth - 0.09, 0.086), "dark")
        if drawers:
            # Walkthrough: three shallow rows above one deeper bottom drawer.
            for column in range(drawer_columns):
                dw = width / drawer_columns
                dx = -width / 2 + dw * (column + 0.5)
                low = bottom
                for i, fraction in enumerate((0.36, 0.22, 0.21, 0.21)):
                    dh = height * fraction
                    key = f"{name}_drawer{column}_{i}"
                    drawer = self.body(body, key, (dx, -depth / 2 - 0.014, low + dh / 2))
                    ET.SubElement(drawer, "joint", name=key + "_slide", type="slide", axis="0 -1 0",
                                  range="0 0.38", damping="4", frictionloss="0.3", armature="0.02")
                    self.box(drawer, key + "_front", (0, 0, 0), (dw - 0.006, 0.021, dh - 0.012), "cabinet")
                    self.box(drawer, key + "_bottom", (0, depth / 2 - 0.04, -dh / 2 + 0.035), (dw - 0.065, depth - 0.08, 0.018), "interior")
                    for side, sx in (("left", -(dw - 0.065) / 2), ("right", (dw - 0.065) / 2)):
                        self.box(drawer, key + "_" + side, (sx, depth / 2 - 0.04, 0.006),
                                 (0.012, depth - 0.08, dh - 0.065), "interior")
                    self.box(drawer, key + "_back", (0, depth - 0.085, 0.006),
                             (dw - 0.065, 0.012, dh - 0.065), "interior")
                    self.pull(drawer, key + "_pull", 0, -0.035, 0, True)
                    ET.SubElement(drawer, "site", name=key + "_handle", pos="0 -0.035 0", size="0.008", group="2")
                    ET.SubElement(drawer, "site", name=key + "_interior", pos=numbers((0, depth / 2 - 0.04, -dh / 2 + 0.054)), size="0.008", group="2")
                    low += dh
        else:
            count = door_count or (2 if width > 0.55 else 1)
            door_width = width / count
            door_height = height - (0.15 if top_drawers else 0)
            for i in range(count):
                direction = 1 if i % 2 == 0 else -1
                hinge_x = -width / 2 + door_width * (i if direction == 1 else i + 1)
                door = self.body(body, f"{name}_door{i}", (hinge_x, -depth / 2, bottom + door_height / 2))
                self.box(door, f"{name}_panel{i}", (direction * door_width / 2, 0, 0),
                         (door_width - 0.006, 0.018, door_height - 0.006), "cabinet")
                self.pull(door, f"{name}_handle{i}", direction * (door_width - 0.055), -0.038, -0.19 if upper else 0.16)
                if top_drawers:
                    dx = -width / 2 + door_width * (i + 0.5)
                    drawer = self.body(body, f"{name}_topdrawer{i}", (dx, -depth / 2, top - 0.075))
                    self.box(drawer, f"{name}_topdrawer{i}_front", (0, 0, 0), (door_width - 0.006, 0.021, 0.144), "cabinet")
                    self.pull(drawer, f"{name}_topdrawer{i}_pull", 0, -0.035, 0, True)
        return body

    def countertop(self, name, x, y, w, d, sink=False, yaw=0, sink_x=0, double=False, faucet_side=1):
        body = self.body(self.world, name, (x, y, self.h), yaw)
        if not sink:
            self.box(body, f"{name}_slab", (0, 0, -0.015), (w, d, 0.03), "counter")
            return body
        sw, sd = (0.78, 0.43) if double else (0.53, 0.42)
        sy = 0.05
        left, right = sink_x - sw / 2, sink_x + sw / 2
        for suffix, cx, cw in (("left", (-w / 2 + left) / 2, left + w / 2), ("right", (w / 2 + right) / 2, w / 2 - right)):
            if cw > 0:
                self.box(body, f"{name}_{suffix}", (cx, 0, -0.015), (cw, d, 0.03), "counter")
        for suffix, cy, cd in (("front", (-d / 2 + sy - sd / 2) / 2, sy - sd / 2 + d / 2),
                               ("rear", (d / 2 + sy + sd / 2) / 2, d / 2 - sy - sd / 2)):
            self.box(body, f"{name}_{suffix}", (sink_x, cy, -0.015), (sw, cd, 0.03), "counter")
        self.box(body, f"{name}_basin", (sink_x, sy, -0.17), (sw, sd, 0.015), "steel")
        for side in (-1, 1):
            self.box(body, f"{name}_sink_x{side}", (sink_x + side * sw / 2, sy, -0.085), (0.012, sd, 0.17), "steel")
            self.box(body, f"{name}_sink_y{side}", (sink_x, sy + side * sd / 2, -0.085), (sw, 0.012, 0.17), "steel")
            self.box(body, f"{name}_rim_x{side}", (sink_x + side * sw / 2, sy, 0.001), (0.018, sd + 0.02, 0.006), "chrome", False)
            self.box(body, f"{name}_rim_y{side}", (sink_x, sy + side * sd / 2, 0.001), (sw, 0.018, 0.006), "chrome", False)
        if double:
            self.box(body, f"{name}_divider", (sink_x, sy, -0.09), (0.018, sd, 0.16), "steel")
        self.cylinder(body, f"{name}_drain", (sink_x - (0.17 if double else 0), sy, -0.16), 0.026, 0.004, "dark", False)
        fy = sy + faucet_side * (sd / 2 + 0.05)
        points = [(sink_x, fy, 0), (sink_x, fy, 0.23)]
        for i in range(13):
            a = math.pi * i / 12
            points.append((sink_x, fy + faucet_side * (-0.09 + 0.09 * math.cos(a)), 0.23 + 0.09 * math.sin(a)))
        points.append((sink_x, fy - faucet_side * 0.18, 0.18))
        for i, (a, b) in enumerate(zip(points, points[1:])):
            self.rod(body, f"{name}_faucet{i}", a, b, 0.011, collision=True)
        self.rod(body, f"{name}_lever", (sink_x + 0.035, fy, 0.03), (sink_x + 0.065, fy, 0.12), 0.007)
        return body

    def fridge(self, x, y):
        name, width, depth, height = "fridge", 0.90, 0.78, self.config["fridge_height"]
        body = self.body(self.world, name, (x, y, 0), math.pi / 2)
        self.box(body, "fridge_case", (0, 0, height / 2), (width, depth, height), "steel")
        self.box(body, "fridge_seal", (0, -depth / 2 - 0.007, height / 2), (width - 0.025, 0.015, height - 0.025), "dark", False)
        for i, sign in enumerate((-1, 1)):
            self.box(body, f"fridge_door{i}", (sign * width / 4, -depth / 2 - 0.026, 0.53 + (height - 0.53) / 2), (width / 2 - 0.012, 0.04, height - 0.55), "steel")
            self.pull(body, f"fridge_handle{i}", sign * 0.055, -depth / 2 - 0.075, 1.1, length=0.53)
        self.box(body, "fridge_freezer", (0, -depth / 2 - 0.027, 0.275), (width - 0.01, 0.042, 0.48), "steel")
        self.pull(body, "freezer_handle", 0, -depth / 2 - 0.077, 0.44, True, 0.62)

    def stove(self, x, y):
        b = self.body(self.world, "range", (x, y, 0))
        self.box(b, "range_body", (0, 0, self.h / 2), (0.76, 0.64, self.h), "steel")
        self.box(b, "oven_door", (0, -0.333, 0.39), (0.65, 0.025, 0.48), "dark")
        self.box(b, "oven_window", (0, -0.35, 0.39), (0.54, 0.01, 0.32), "screen", False)
        self.pull(b, "oven_handle", 0, -0.389, 0.65, True, 0.59)
        self.box(b, "stovetop", (0, 0, self.h + 0.008), (0.78, 0.66, 0.018), "steel")
        for i in range(5):
            self.cylinder(b, f"range_knob{i}", (-0.29 + 0.145 * i, -0.35, self.h - 0.065), 0.024, 0.025, "chrome", False, euler="1.5708 0 0")
        hood = self.body(self.world, "range_hood", (x, y, 0))
        self.box(hood, "hood_lip", (0, -0.035, 1.97), (0.90, 0.71, 0.09), "steel")
        verts = [(-0.45, -0.39, 1.99), (0.45, -0.39, 1.99), (0.45, 0.32, 1.99), (-0.45, 0.32, 1.99),
                 (-0.345, 0.02, 2.27), (0.345, 0.02, 2.27), (0.345, 0.32, 2.27), (-0.345, 0.32, 2.27)]
        faces = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                 (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
        ET.SubElement(self.assets, "mesh", name="hood_taper", vertex=numbers(v for p in verts for v in p),
                      face=" ".join(str(v) for f in faces for v in f))
        ET.SubElement(hood, "geom", name="hood_slope", type="mesh", mesh="hood_taper", material="steel")
        self.box(hood, "hood_chimney", (0, 0.17, 2.62), (0.69, 0.30, 0.92), "steel")

    def dispenser(self, name, x, y, yaw=0, coffee=False):
        b = self.body(self.world, name, (x, y, self.h), yaw)
        w, h = (0.36, 0.48) if coffee else (0.22, 0.31)
        self.box(b, f"{name}_body", (0, 0, h / 2), (w, 0.25, h), "dark")
        self.box(b, f"{name}_panel", (0, -0.129, h * 0.68), (w * 0.72, 0.012, h * 0.43), "screen", False)
        self.box(b, f"{name}_tray", (0, -0.05, 0.03), (w * 0.80, 0.29, 0.025), "steel")
        self.rod(b, f"{name}_spout", (0, -0.15, h * 0.43), (0, -0.15, h * 0.33), 0.014)

    def glass_appliance(self, x, y):
        b = self.body(self.world, "glass_appliance", (x, y, self.h))
        self.box(b, "glass_appliance_body", (0, 0.02, 0.29), (0.41, 0.34, 0.58), "dark")
        self.box(b, "glass_appliance_window", (0, -0.155, 0.30), (0.355, 0.012, 0.48), "screen", False)
        for i, z in enumerate((0.14, 0.29, 0.43)):
            self.box(b, f"glass_appliance_rack{i}", (0, -0.164, z), (0.32, 0.004, 0.008), "steel", False)
            for j in range(3):
                self.box(b, f"glass_appliance_item{i}_{j}", (-0.105 + j * 0.105, -0.166, z + 0.045),
                         (0.055, 0.004, 0.065), "interior", False)
        self.pull(b, "glass_appliance_handle", -0.175, -0.19, 0.30, length=0.20)

    def stool(self, name, x, y):
        b = self.body(self.world, name, (x, y, 0))
        for i, xx in enumerate((-0.20, 0.20)):
            for j, yy in enumerate((-0.20, 0.20)):
                self.rod(b, f"{name}_leg{i}{j}", (xx, yy, 0.025), (xx * 0.83, yy * 0.8, 0.59), 0.011, "dark", True)
            self.rod(b, f"{name}_rail{i}", (xx, -0.20, 0.02), (xx, 0.20, 0.02), 0.011, "dark")
        self.rod(b, f"{name}_footrest", (-0.20, 0.20, 0.25), (0.20, 0.20, 0.25), 0.01, "dark")
        self.rounded_panel(b, f"{name}_seat", (0, 0, 0.59), 0.43, 0.40, 0.035, 0.09).set("material", "wood")
        self.rounded_panel(b, f"{name}_back", (0, -0.18, 0.76), 0.43, 0.32, 0.032, 0.065, (1.41, 0, 0)).set("material", "wood")

    def build(self):
        c, w = self.config, self.world
        rw, rd, ch = c["room_width"], c["room_depth"], c["ceiling_height"]
        self.box(w, "floor_collision", (0, 0, -0.055), (rw, rd, 0.10), "floor")
        # Tile seams and subtle per-tile variation reproduce the light grey vinyl floor.
        nx, ny = math.ceil(rw / 0.6), math.ceil(rd / 0.6)
        for ix in range(nx):
            for iy in range(ny):
                tone = 0.66 + ((ix * 17 + iy * 11) % 7) * 0.007
                self.box(w, f"floor_tile_{ix}_{iy}", (-rw / 2 + (ix + 0.5) * rw / nx, -rd / 2 + (iy + 0.5) * rd / ny, -0.003),
                         (rw / nx - 0.0006, rd / ny - 0.0006, 0.006), "floor", False, rgba=numbers((tone, tone + 0.006, tone - 0.007, 1)))
        for name, pos, size in (("back_wall", (0, rd / 2, ch / 2), (rw, 0.10, ch)),
                                 ("left_wall", (-rw / 2, 0, ch / 2), (0.10, rd, ch))):
            self.box(w, name, pos, size, "wall")
        border_y = rd / 2 - 0.74
        self.box(w, "back_floor_border", (0, border_y, 0.002), (rw, 0.27, 0.005), "border", False)
        # Partition the left strip at the back strip: no coplanar overlapping tops.
        for name, low, high in (("left_floor_border", -rd / 2, border_y - 0.135),
                                ("left_floor_border_rear", border_y + 0.135, rd / 2)):
            self.box(w, name, (-rw / 2 + 0.72, (low + high) / 2, 0.002),
                     (0.26, high - low, 0.005), "border", False)
        self.box(w, "ceiling", (0, 0, ch + 0.025), (rw, rd, 0.05), "counter", False, 5)
        for i in range(7):
            self.box(w, f"ceiling_grid_x{i}", (-rw / 2 + i * rw / 6, 0, ch - 0.003), (0.012, rd, 0.012), "steel", False, 5)
        for i in range(9):
            self.box(w, f"ceiling_grid_y{i}", (0, -rd / 2 + i * rd / 8, ch - 0.003), (rw, 0.012, 0.012), "steel", False, 5)
        for i, x in enumerate((-1.8, 0, 1.8)):
            ET.SubElement(w, "light", name=f"fill{i}", pos=numbers((x, 0.4, 3.05)), dir="0 0 -1", diffuse="0.5 0.5 0.5", castshadow="true")
            self.box(w, f"ceiling_strip{i}", (x, 0, ch - 0.06), (0.055, 4.8, 0.035), "counter", False, 5)
        # Back run faces toward negative Y; left run is the same fixture rotated 90 degrees.
        yback = rd / 2 - c["counter_depth"] / 2 - 0.06
        xleft = -rw / 2 + c["counter_depth"] / 2 + 0.06
        prep_left, prep_right = -0.92 + 0.39 + 0.005, 0.47
        prep_center, prep_width = (prep_left + prep_right) / 2, prep_right - prep_left
        for name, x, width, drawers in (("back_corner", -2.195, 1.25, False),
                                       ("range_drawers", prep_center, prep_width - 0.02, True), ("sink_base", 1.40, 0.94, False),
                                       ("coffee_base", 2.39, 1.00, False)):
            self.cabinet(name, x, yback, width, drawers=drawers, drawer_columns=2,
                         top_drawers=name in ("back_corner", "coffee_base"))
        slots = self.cabinet("prep_base", -1.445, yback, 0.25)
        for child in list(slots):
            if child.tag == "body" or child.get("name") == "prep_base_shelf":
                slots.remove(child)
        for i, dx in enumerate((-0.042, 0.042)):
            self.box(slots, f"prep_tray_divider{i}", (dx, 0, self.h / 2 + 0.03),
                     (0.014, c["counter_depth"] - 0.02, self.h - 0.13), "cabinet")
        corner_start, corner_end = xleft + 0.34, -1.31
        self.countertop("back_left_worktop", (corner_start + corner_end) / 2, yback,
                        corner_end - corner_start, 0.68)
        self.countertop("back_prep_worktop", prep_center, yback, prep_width, 0.68)
        self.countertop("back_sink_worktop", 1.40, yback, 0.94, 0.68, True, double=True)
        self.countertop("coffee_worktop", 2.39, yback, 1.01, 0.68)
        self.stove(-0.92, yback)
        self.box(w, "dishwasher", (0.70, yback, self.h / 2), (0.48, 0.61, self.h - 0.018), "steel")
        self.box(w, "dishwasher_controls", (0.70, yback - 0.315, self.h - 0.065), (0.41, 0.02, 0.052), "dark", False)
        self.pull(w, "dishwasher_handle", 0.70, yback - 0.35, self.h - 0.10, True, 0.32)
        for i, yy in enumerate((1.30, 2.12, 2.94)):
            base = self.cabinet(f"left_base{i}", xleft, yy, 0.82, yaw=math.pi / 2, top_drawers=i != 0)
            if i == 0:
                for child in list(base):
                    if child.tag == "body":
                        base.remove(child)
                self.box(base, "microwave_drawer", (0, -0.31, 0.245), (0.80, 0.02, 0.30), "cabinet")
                for p, px in enumerate((-0.20, 0.20)):
                    self.pull(base, f"microwave_drawer_pull{p}", px, -0.35, 0.30, True)
                self.box(base, "microwave_housing", (0, -0.15, 0.59), (0.75, 0.38, 0.34), "steel")
                self.box(base, "microwave_door", (-0.055, -0.35, 0.59), (0.56, 0.016, 0.25), "screen", False)
                self.box(base, "microwave_controls", (0.29, -0.35, 0.59), (0.10, 0.02, 0.25), "dark", False)
            self.cabinet(f"upper_left{i}", xleft - 0.12, yy, 0.82, 0.34, math.pi / 2, True)
        self.countertop("left_worktop", xleft, 2.08, 2.53, 0.68, yaw=math.pi / 2)
        self.cabinet("upper_corner", -2.10, yback + 0.12, 1.37, 0.34, upper=True)
        self.fridge(xleft + 0.03, 0.28)
        self.cabinet("beverage_base", xleft, -1.50, 1.92, yaw=math.pi / 2, door_count=6)
        self.cabinet("beverage_drawers", xleft, -0.36, 0.36, yaw=math.pi / 2, drawers=True)
        self.countertop("beverage_worktop", xleft, -1.32, 2.31, 0.68, True, math.pi / 2, sink_x=0.23)
        self.dispenser("beverage_dispenser", xleft - 0.08, -1.2, math.pi / 2)
        toaster = self.body(w, "toaster", (xleft, -0.60, self.h), math.pi / 2)
        self.box(toaster, "toaster_body", (0, 0, 0.11), (0.38, 0.27, 0.22), "steel")
        self.box(toaster, "toaster_window", (0, -0.14, 0.11), (0.28, 0.015, 0.14), "dark", False)
        self.glass_appliance(2.42, yback)
        self.dispenser("sink_dispenser", 1.18, yback + 0.19)
        # Island base leaves an overhang on the stool side.
        ix, iy = c["island_center"]
        iw, idepth = c["island_width"], c["island_depth"]
        self.box(w, "island_mat", (ix, iy - 0.20, 0.003), (iw + 0.30, idepth + 0.88, 0.006), "border", False)
        self.cabinet("island_sink_base", ix - 0.62, iy + 0.10, 1.02, 0.74, math.pi)
        self.cabinet("island_storage", ix + 0.53, iy + 0.10, 1.24, 0.74, math.pi)
        self.countertop("island_worktop", ix, iy, iw, idepth, True, sink_x=-0.54, double=True, faucet_side=-1)
        self.box(w, "island_seating_panel", (ix, iy - 0.287, self.h / 2), (iw - 0.06, 0.022, self.h - 0.04), "cabinet")
        for i, sx in enumerate((-0.80, 0, 0.80)):
            self.stool(f"stool{i}", ix + sx, iy - 0.84)
        for i, x in enumerate((-0.78, 0, 0.78)):
            self.rod(w, f"pendant_cord{i}", (ix + x, iy, 2.43), (ix + x, iy, ch), 0.006, "dark")
            vertices = [(radius * math.cos(j * math.tau / 40), radius * math.sin(j * math.tau / 40), z)
                        for radius, z in ((0.135, 0), (0.043, 0.21)) for j in range(40)]
            faces = []
            for j in range(40):
                k = (j + 1) % 40
                faces.extend(((j, k, 40 + k), (j, 40 + k, 40 + j)))
            for j in range(1, 39):
                faces.extend(((0, j + 1, j), (40, 40 + j, 40 + j + 1)))
            ET.SubElement(self.assets, "mesh", name=f"pendant_mesh{i}", vertex=numbers(v for xyz in vertices for v in xyz),
                          face=" ".join(str(v) for face in faces for v in face))
            ET.SubElement(w, "geom", name=f"pendant_shade{i}", type="mesh", mesh=f"pendant_mesh{i}",
                          pos=numbers((ix + x, iy, 2.21)), material="steel", contype="0", conaffinity="0")
            self.cylinder(w, f"pendant_diffuser{i}", (ix + x, iy, 2.203), 0.12, 0.009, "counter", False)
        # Freestanding four-bay waste-sorting island.
        wx, wy = c["waste_island_center"]
        wb = self.body(w, "waste_island", (wx, wy, 0))
        self.box(wb, "waste_counter", (0, 0, self.h - 0.015), (2.45, 0.79, 0.03), "counter")
        self.box(wb, "waste_back", (0, -0.34, self.h / 2 - 0.02), (2.39, 0.03, self.h - 0.04), "cabinet")
        for i in range(5):
            self.box(wb, f"waste_partition{i}", (-1.2 + 0.60 * i, 0, self.h / 2), (0.02, 0.7, self.h - 0.03), "cabinet")
        for i, mat in enumerate(("blue", "blue", "green", "border")):
            x = -0.9 + i * 0.6
            self.box(wb, f"waste_door{i}", (x, 0.36, 0.34), (0.586, 0.018, 0.66), "cabinet")
            self.box(wb, f"waste_bin{i}", (x, 0, 0.55), (0.48, 0.55, 0.08), mat, False)
            self.box(wb, f"waste_label{i}", (x, 0.372, 0.45), (0.24, 0.004, 0.30), mat, False)
            self.box(wb, f"waste_label_paper{i}", (x, 0.376, 0.43), (0.205, 0.002, 0.215), "counter", False)
            for j in range(3):
                self.box(wb, f"waste_label_line{i}_{j}", (x, 0.378, 0.48 - j * 0.05), (0.15, 0.001, 0.018), mat, False)
            self.pull(wb, f"waste_pull{i}", x + 0.23, 0.40, 0.54)
        # Two wall monitors are kept neutral, with no reconstructed screen content.
        for name, pos, width, yaw in (("back_monitor", (1.55, rd / 2 - 0.065, 1.94), 1.12, 0),
                                       ("left_monitor", (-rw / 2 + 0.07, -1.32, 1.98), 1.65, math.pi / 2)):
            b = self.body(w, name, pos, yaw)
            self.box(b, f"{name}_frame", (0, 0, 0), (width, 0.05, 0.63), "dark", False)
            self.box(b, f"{name}_display", (0, -0.028, 0), (width - 0.035, 0.004, 0.595), "screen", False)
        cart = self.body(w, "utility_cart", (wx - 1.53, wy, 0))
        for i, z in enumerate((0.12, 0.40, 0.70)):
            self.box(cart, f"cart_shelf{i}", (0, 0, z), (0.47, 0.68, 0.035), "dark")
        for i, x in enumerate((-0.21, 0.21)):
            for j, y in enumerate((-0.31, 0.31)):
                self.rod(cart, f"cart_post{i}{j}", (x, y, 0.09), (x, y, 0.91), 0.017, "dark", True)
                self.cylinder(cart, f"cart_wheel{i}{j}", (x, y, 0.055), 0.046, 0.025, "dark", euler="1.5708 0 0")
        # Motion-capture rail and cameras are visible architectural landmarks.
        self.rod(w, "tracking_rail_back", (-3.1, rd / 2 - 0.16, 2.91), (3.1, rd / 2 - 0.16, 2.91), 0.014, "counter")
        for i in range(7):
            x = -2.85 + i * 0.92
            self.rod(w, f"tracking_mount{i}", (x, rd / 2 - 0.16, 2.91), (x, rd / 2 - 0.22, 2.78), 0.012, "dark")
            self.box(w, f"tracking_camera{i}", (x, rd / 2 - 0.23, 2.77), (0.105, 0.11, 0.10), "dark", False)
            self.cylinder(w, f"tracking_ring{i}", (x, rd / 2 - 0.29, 2.77), 0.037, 0.012, "tracking_led", False, euler="1.5708 0 0")
            self.cylinder(w, f"tracking_lens{i}", (x, rd / 2 - 0.298, 2.77), 0.027, 0.006, "dark", False, euler="1.5708 0 0")
        self.rod(w, "tracking_rail_left", (-rw / 2 + 0.16, -2.7, 2.91), (-rw / 2 + 0.16, 3.0, 2.91), 0.014, "counter")
        for i, y in enumerate((-2.3, -1.1, 0.1, 1.3, 2.5)):
            x = -rw / 2
            self.rod(w, f"tracking_left_mount{i}", (x + 0.16, y, 2.91), (x + 0.22, y, 2.78), 0.012, "dark")
            self.box(w, f"tracking_left_camera{i}", (x + 0.23, y, 2.77), (0.11, 0.105, 0.10), "dark", False)
            self.cylinder(w, f"tracking_left_ring{i}", (x + 0.29, y, 2.77), 0.037, 0.012, "tracking_led", False, euler="0 1.5708 0")
            self.cylinder(w, f"tracking_left_lens{i}", (x + 0.298, y, 2.77), 0.027, 0.006, "dark", False, euler="0 1.5708 0")
        self.site("island_work_surface", (ix + 0.55, iy, self.h + 0.005), (0.40, 0.37, 0.003))
        self.site("island_sink_target", (ix - 0.54, iy + 0.05, self.h - 0.08))
        self.site("back_sink_target", (1.40, yback + 0.05, self.h - 0.08))
        self.site("stove_target", (-0.92, yback, self.h + 0.035))
        self.site("left_prep_surface", (xleft, 1.6, self.h + 0.005))
        self.site("waste_work_surface", (wx, wy, self.h + 0.005))
        self.site("robot_spawn", c["robot_position"])
        self.site("world_origin", (0, 0, 0))
        return self

    def save(self, output):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        ET.indent(self.root)
        ET.ElementTree(self.root).write(output, encoding="unicode", xml_declaration=True)
        manifest = {"units": "meters", "up_axis": "+Z", "back_wall": "+Y", "left_wall": "-X",
                    "config": self.config, "sites": self.sites,
                    "limitations": ["Approximate dimensions; no metric scan calibration", "Visual reconstruction, not a registered RoboCasa task", "Fixtures are static visual and collision geometry; no appliance task semantics", "Two walls only; other sides intentionally open; ceiling hidden in preview"]}
        output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        return output


def build(config=None, output=None):
    config = config or json.loads((ROOT / "layout.json").read_text())
    return Kitchen(config).build().save(output or ROOT / "output" / "kitchen.xml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "layout.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "kitchen.xml")
    args = parser.parse_args()
    print(build(json.loads(args.config.read_text()), args.output))
