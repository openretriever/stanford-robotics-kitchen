"""Perturbation mechanics with the model stubbed: no Astra call is made, and the flows run on a
paper world except for the two tests that need the real slide joints and jar geoms.

    MUJOCO_GL=cgl ./.venv/bin/python test_perturb.py -v
"""
from __future__ import annotations

import io
import re
import unittest
from contextlib import redirect_stdout

import kitchen
import pipeline
from compare import metrics, summarize, table
from kitchen import Placement
from memory import DrawerMemory, Transcript

D = "range_drawers_drawer"
COLOURS = {"paprika": "reddish-brown jar", "herb seasoning": "green jar"}


class PaperWorld:
    """DrawerWorld's surface without MuJoCo: openable drawers open fully, the rest stop at 0.05 m."""

    def __init__(self, placements, openable=kitchen.OPENABLE):
        self._placements, self.openable, self.jammed = list(placements), set(openable), set()
        self.opening, self.drawer_name, self.finished = {}, None, True
        self.success, self.error = False, ""

    def retarget(self, drawer):
        self.drawer_name, self.finished = drawer, False

    def step(self):
        works = self.drawer_name in self.openable and self.drawer_name not in self.jammed
        self.opening[self.drawer_name] = 0.30 if works else 0.05
        self.success, self.error, self.finished = works, "" if works else "Drawer target not reached", True

    def snapshot(self):
        return {"opening_m": self.opening.get(self.drawer_name, 0.0), "contact_fraction": float(self.success)}

    def openings(self):
        return {d: self.opening.get(d, 0.0) for d in kitchen.DRAWERS}

    def geometry(self):
        return {d: {"handle_height_m": 0.5, "stack": "left"} for d in kitchen.DRAWERS}

    def jpeg(self):
        """The camera frame: what the drawer just pulled shows, or nothing if it stayed shut."""
        if self.opening.get(self.drawer_name, 0.0) < 0.25:
            return b""
        return COLOURS.get(self.ground_truth().get(self.drawer_name), "empty").encode()

    def ground_truth(self):
        return {p.drawer: p.label for p in self._placements}

    def relocate(self, label, drawer):
        next(p for p in self._placements if p.label == label).drawer = drawer

    def jam(self, drawer):
        self.jammed.add(drawer)


class FixedAstra:
    """Planner: scan the candidates in order, report the first reddish sighting in the record.
    Belief: attribute whatever the frame shows to the drawer just pulled, as a late frame would make Astra do."""

    model = "fixed-policy stub"

    def __init__(self, world):
        self.world, self.queue, self.calls = world, list(kitchen.DRAWERS), []

    def decide(self, role, system, text, images=(), **_):
        self.calls.append(role)
        if role == "belief":
            seen = images[0].decode()
            rows = [{"drawer": self.world.drawer_name, "contents": seen, "confidence": 0.9}] if seen else []
            return {"observations": rows, "summary": seen or "shut"}
        record = text.split("Record of this run so far:")[1]
        if "reddish" in record:
            line = next(line for line in record.splitlines() if "reddish" in line)
            return {"action": "report_found", "drawer": re.search(rf"{D}\d_\d", line).group(),
                    "rationale": "seen", "found_item": "reddish-brown jar"}
        if not self.queue:
            return {"action": "finish", "drawer": "", "rationale": "exhausted", "found_item": ""}
        return {"action": "open_drawer", "drawer": self.queue.pop(0), "rationale": "next", "found_item": ""}


class Run:
    """The four flows stepped by hand, the way the engine chains them."""

    def __init__(self, placements, perturb, mode="structured"):
        self.world = PaperWorld(placements)
        self.astra = FixedAstra(self.world)
        self.memory, transcript = DrawerMemory(), Transcript()
        self.planner = pipeline.AstraPlannerFlow(astra=self.astra, memory_ref=self.memory, task="Find the jar.",
                                                 candidates=kitchen.DRAWERS, max_decisions=12)
        robot = pipeline.KitchenFlow(world=self.world, memory=self.memory, transcript=transcript, perturb=perturb)
        belief = pipeline.AstraBeliefFlow(astra=self.astra)
        recall = pipeline.MemoryFlow(memory=self.memory, transcript=transcript, mode=mode,
                                     candidates=kitchen.DRAWERS, planner=self.planner)
        digest = None
        with redirect_stdout(io.StringIO()):
            while (command := self.planner.step(digest)) is not None:
                digest = recall.step(belief.step(robot.step(command)))

    def record(self):
        truth = self.world.ground_truth()
        return {"decisions": self.planner.decisions, "memory_state": self.memory.as_dict(), "ground_truth": truth,
                "score": self.memory.score(truth), "cost": {"total_calls": len(self.astra.calls), "total_usd": 0.0},
                "wall_time_s": 0.0}

    def drawers(self):
        return [d["drawer"] for d in self.planner.decisions]


class Placements(unittest.TestCase):
    LABELS = ["paprika", "herb seasoning"]

    def test_last_and_first_follow_the_scan_order(self):
        for seed in (1, 2, 3):
            last = pipeline.build_placements(seed, kitchen.OPENABLE, self.LABELS, "last")
            first = pipeline.build_placements(seed, kitchen.OPENABLE, self.LABELS, "first")
            self.assertEqual([last[0].drawer, first[0].drawer], [D + "1_3", D + "0_1"])
            self.assertEqual([p.label for p in last], self.LABELS)
            self.assertEqual(len({p.drawer for p in last}), 2)

    def test_random_is_the_september_ablation_draw(self):
        self.assertEqual(pipeline.build_placements(3, kitchen.OPENABLE, self.LABELS),
                         [Placement(D + "0_2", "paprika"), Placement(D + "1_3", "herb seasoning")])


class Perturbations(unittest.TestCase):
    def test_none_scans_to_the_last_drawer(self):
        run = Run([Placement(D + "1_3", "paprika")], pipeline.Perturbation())
        self.assertEqual(run.drawers(), list(kitchen.DRAWERS) + [D + "1_3"])
        self.assertTrue(metrics(run.record())["correct"])

    def test_relocate_moves_the_jar_after_it_was_seen(self):
        perturb = pipeline.Perturbation(kind="relocate", after=2, target="paprika", openable=kitchen.OPENABLE, seed=1)
        run = Run([Placement(D + "0_1", "paprika"), Placement(D + "1_3", "herb seasoning")], perturb)
        [event] = perturb.events
        self.assertEqual((event["pull"], event["kind"], event["moved_from"]), (2, "relocate", D + "0_1"))
        self.assertIn(event["moved_to"], {D + "0_2", D + "0_3", D + "1_2"})     # unopened, and holding nothing
        self.assertEqual(run.world.ground_truth()[event["moved_to"]], "paprika")
        # The record still says the jar is where it was seen, and the planner reports from it.
        self.assertEqual(run.memory.records[D + "0_1"].contents, "reddish-brown jar")
        self.assertEqual(run.planner.decisions[-1]["action"], "report_found")
        m = metrics(run.record())
        self.assertEqual((m["found"], m["correct"], m["colour"]), (True, False, 0))

    def test_stale_frame_files_the_previous_contents_under_the_next_drawer(self):
        perturb = pipeline.Perturbation(kind="stale", after=2)
        run = Run([Placement(D + "1_3", "paprika"), Placement(D + "0_1", "herb seasoning")], perturb)
        self.assertEqual(perturb.events, [{"pull": 3, "kind": "stale", "frame_from_pull": 2}])
        self.assertEqual(run.memory.records[D + "0_2"].contents, "green jar")   # drawer0_2 is empty
        self.assertEqual(run.memory.records[D + "0_3"].contents, "empty")       # only once
        self.assertTrue(metrics(run.record())["correct"])

    def test_jam_sticks_the_next_untried_openable_drawer(self):
        perturb = pipeline.Perturbation(kind="jam", after=1, openable=kitchen.OPENABLE)
        run = Run([Placement(D + "1_3", "paprika")], perturb)
        self.assertEqual(perturb.events, [{"pull": 1, "kind": "jam", "drawer": D + "0_1"}])
        row = run.memory.records[D + "0_1"]
        self.assertEqual((row.opened, len(row.failures)), (False, 1))
        self.assertTrue(run.memory.records[D + "0_2"].opened)

    def test_transcript_arm_sees_the_same_events(self):
        perturb = pipeline.Perturbation(kind="stale", after=2)
        run = Run([Placement(D + "1_3", "paprika"), Placement(D + "0_1", "herb seasoning")], perturb, "transcript")
        self.assertEqual(run.memory.records[D + "0_2"].contents, "green jar")
        self.assertEqual(run.drawers(), list(kitchen.DRAWERS) + [D + "1_3"])


class Summary(unittest.TestCase):
    def test_rows_carry_the_perturbation_column(self):
        m = {"found": True, "correct": False, "supported": True, "horizon": 9, "repeated_visits": 0,
             "repeated_physical": 0, "recovery": 0, "unsupported": 0, "colour": 1, "calls": 17, "usd": 0.2, "wall_s": 200}
        rows = summarize({("structured", "jam", 1): m, ("structured", "jam", 2): {**m, "correct": True, "horizon": 11},
                          ("transcript", "none", 1): m})
        self.assertEqual([(r["arm"], r["perturb"], r["runs"], r["found"], r["correct"], r["horizon"], r["usd"])
                          for r in rows],
                         [("structured", "jam", 2, 2, 1, 10, 0.4), ("transcript", "none", 1, 1, 0, 9, 0.2)])
        md = table(rows)
        self.assertEqual(md[0][:44], "| arm | perturb | runs | found | correct | s")
        self.assertEqual(md[2][:31], "| structured | jam | 2 | 2 | 1 ")
        self.assertEqual(len(md), 4)


class KitchenFaults(unittest.TestCase):
    """The real world: a stuck slide fails DrawerPull's own verification, a moved jar rides its new drawer."""

    @classmethod
    def setUpClass(cls):
        cls.world = kitchen.DrawerWorld([Placement(D + "1_3", "paprika")])

    def test_jam_keeps_a_verified_drawer_under_threshold(self):
        self.world.jam(D + "0_1")
        self.world.retarget(D + "0_1")
        for _ in range(1400):
            if self.world.finished:
                break
            self.world.step()
        self.assertTrue(self.world.finished and not self.world.success)
        self.assertLess(self.world.snapshot()["opening_m"], 0.25)

    def test_relocate_redraws_the_jar_under_another_drawer(self):
        model = self.world.model
        self.world.relocate("paprika", D + "0_2")
        self.assertEqual(self.world.ground_truth(), {D + "0_2": "paprika"})
        self.assertEqual(model.geom(f"target_0_{D}0_2_geom").rgba[3], 1.0)
        self.assertEqual(model.geom(f"target_0_{D}1_3_geom").rgba[3], 0.0)
        self.assertEqual(model.body(f"target_0_{D}0_2").parentid, model.body(D + "0_2").id)


if __name__ == "__main__":
    unittest.main()
