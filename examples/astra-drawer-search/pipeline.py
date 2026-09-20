"""An all-Astra long-horizon drawer search in the SRC kitchen, as a Retriever pipeline.

    Planner ---> Kitchen ---> Belief ---> Memory ---+
       ^                                            |
       +--------------------------------------------+

Every model-backed role is gpt-6-astra: the planner that chooses the next drawer,
and the belief updater that reads what is inside from pixels. Memory is
deliberately NOT a model (see memory.py). The robot's drawer pull is src-kitchen's
own contact-driven primitive, verified by two-finger contact -- not a rail force
and not a teleport.

The planner is CLOCKED, not data-driven. A cycle in this runtime self-clocks: a
two-node loop wired with Latest() on both edges ran 36,161 iterations in three
seconds during development. With Astra on those nodes that is a way to spend real
money by accident, so the loop is paced by Rate() on the planner and bounded a
second time by the spend cap inside astra.Astra.

What the model is never told: which drawer holds what. Ground truth is read only
after the run, to score it.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import retriever
from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io

import astra as astra_mod
import kitchen
from recorder import Recorder
from memory import DrawerMemory

PLANNER_SYSTEM = """You search a kitchen for named items hidden in drawers, using only what you see.

You act through one tool call per turn. You are given a written record of the run
so far: which drawers you attempted, which opened, which failed and why, and what
was seen inside any drawer that opened. Trust that record -- it is measured, not
remembered.

Rules that matter:
- A drawer that has already failed twice is very unlikely to open. Prefer an
  untried drawer over a third attempt.
- A drawer already open needs no second opening; read its contents instead.
- Report an item found only when you have actually seen it in an image.
- If every drawer is either open or repeatedly failed, finish rather than loop."""

BELIEF_SYSTEM = """You report only what is visible in the image of a kitchen drawer bank.

You are told which drawers are currently pulled open and by how far, measured
from the mechanism. That is the only state you are given -- nothing about what is
inside any of them. Attribute what you see to a SPECIFIC named drawer from that
list; several may be open at once, and filing a jar under the wrong drawer is
worse than declining to attribute it.

Name a colour and shape rather than guessing a product: "a reddish-brown
cylindrical jar" is a good answer, "paprika" is a guess unless a label is
legible. If an open drawer is empty, say so for that drawer by name. Never
describe a closed drawer's contents."""


@io
class Command:
    seq: int
    action: str
    drawer: str
    rationale: str


@io
class Observation:
    seq: int
    step: int
    frame_jpeg: bytes
    acted_drawer: str
    success: bool
    opening_m: float
    contact_fraction: float
    error: str
    openings_json: str
    geometry_json: str


@io
class Belief:
    seq: int
    open_drawers_json: str
    contents_json: str
    summary: str


@io
class Digest:
    seq: int
    text: str
    decisions: int
    finished: bool


@dataclass
class AstraPlannerFlow(Flow[Digest, Command]):
    """gpt-6-astra chooses the next drawer. Clocked, and hard-capped."""

    astra: object = None
    memory_ref: object = None
    task: str = ""
    candidates: tuple = ()
    max_decisions: int = 12
    name: str = "AstraPlanner"
    seq: int = 0
    decisions: list = field(default_factory=list)
    finished: bool = False
    _last_digest_seq: int = -1
    _last_text: str = ""

    def step(self, digest: Digest) -> Command | None:
        if self.finished or len(self.decisions) >= self.max_decisions:
            self.finished = True
            return None
        # Act once per new digest. The first turn runs on an empty record.
        incoming = getattr(digest, "seq", None)
        if incoming is not None and incoming == self._last_digest_seq:
            return None
        if incoming is None and self.seq > 0:
            return None
        self._last_digest_seq = incoming if incoming is not None else -1
        record = getattr(digest, "text", None) or "Nothing attempted or observed yet."
        self._last_text = record

        try:
            answer = self.astra.decide(
                "planner", PLANNER_SYSTEM,
                f"Task: {self.task}\n\nCandidate drawers:\n"
                + "\n".join(f"- {c}" for c in self.candidates)
                + f"\n\nRecord of this run so far:\n{record}\n\n"
                f"Decisions used: {len(self.decisions)} of {self.max_decisions}.",
                function="choose_action",
                parameters={"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["open_drawer", "report_found", "finish"]},
                    "drawer": {"type": "string", "description": "Target drawer, or empty for finish."},
                    "rationale": {"type": "string"},
                    "found_item": {"type": "string", "description": "Item named, or empty."},
                }, "required": ["action", "drawer", "rationale", "found_item"],
                    "additionalProperties": False},
            )
        except astra_mod.BudgetExhausted as stop:
            print(f"  [planner] stopping: {stop}")
            self.finished = True
            return None

        self.seq += 1
        self.decisions.append({"seq": self.seq, **answer})
        print(f"  [planner #{self.seq}] {answer['action']} {answer['drawer']} -- {answer['rationale'][:90]}")
        if answer["action"] in ("finish", "report_found"):
            self.finished = True
            if answer["action"] == "report_found":
                item = answer.get("found_item", "")
                supported = self.memory_ref.note_claim(item) if self.memory_ref else None
                print(f"  [planner] reports found: {item!r} "
                      f"(supported by an observation: {supported})")
            return None
        return Command(seq=self.seq, action=answer["action"],
                       drawer=answer["drawer"], rationale=answer["rationale"])


@dataclass
class KitchenFlow(Flow[Command, Observation]):
    """The robot. Executes one contact-driven pull per command, then looks."""

    world: object = None
    memory: object = None
    recorder: object = None
    name: str = "Kitchen"
    step_count: int = 0
    _served: int = -1

    def step(self, command: Command) -> Observation | None:
        seq = getattr(command, "seq", None)
        if seq is None or seq == self._served:
            return None
        self._served = seq
        drawer = command.drawer
        if drawer not in kitchen.DRAWERS:
            print(f"  [kitchen] refusing unknown drawer {drawer!r}")
            return Observation(seq=seq, step=self.step_count, frame_jpeg=self.world.jpeg(),
                               acted_drawer=drawer, success=False, opening_m=0.0,
                               contact_fraction=0.0, error=f"unknown drawer {drawer!r}",
                               openings_json=json.dumps(self.world.openings()),
                               geometry_json=json.dumps(self.world.geometry()))

        self.memory.note_attempt(drawer)
        self.world.retarget(drawer)
        if self.recorder:
            self.recorder.set(decision=f"decision {seq}", action=command.action,
                              drawer=drawer, note=command.rationale[:140])
            self.recorder.hold(self.world, 0.8)
        frames = 0
        while frames < 1400 and not self.world.finished:
            self.world.step()
            frames += 1
            if self.recorder:
                phase = self.world.labels[min(self.world.phase, len(self.world.labels) - 1)]
                self.recorder.set(metric=f"{phase}   opening {self.world.snapshot()['opening_m']:+.3f} m")
                self.recorder.capture(self.world)
        self.step_count += frames
        snap = self.world.snapshot()
        self.memory.note_outcome(drawer, success=bool(self.world.success),
                                 opening_m=snap["opening_m"], error=self.world.error,
                                 step=self.step_count)
        print(f"  [kitchen] pull {drawer}: success={self.world.success} "
              f"opening={snap['opening_m']:.3f}m contact={snap['contact_fraction']:.2f}")
        if self.recorder:
            verdict = "verified grasp" if self.world.success else (self.world.error or "not verified")
            self.recorder.set(metric=f"{verdict}   opening {snap['opening_m']:.3f} m"
                                     f"   contact {snap['contact_fraction']:.2f}")
            self.recorder.hold(self.world, 1.4)
        return Observation(seq=seq, step=self.step_count, frame_jpeg=self.world.jpeg(),
                           acted_drawer=drawer, success=bool(self.world.success),
                           opening_m=float(snap["opening_m"]),
                           contact_fraction=float(snap["contact_fraction"]),
                           error=self.world.error or "",
                           openings_json=json.dumps(self.world.openings()),
                           geometry_json=json.dumps(self.world.geometry()))


@dataclass
class AstraBeliefFlow(Flow[Observation, Belief]):
    """gpt-6-astra reads the drawer bank from pixels. No simulator state reaches it."""

    astra: object = None
    recorder: object = None
    name: str = "AstraBelief"
    _served: int = -1

    def step(self, obs: Observation) -> Belief | None:
        seq = getattr(obs, "seq", None)
        if seq is None or seq == self._served:
            return None
        self._served = seq
        # Mechanism state only: which drawers are out, and by how much. Giving
        # this is what lets the model file a jar under the right drawer; it says
        # nothing about contents, which is the thing being searched for.
        openings = {k: round(v, 3) for k, v in json.loads(obs.openings_json).items() if v > 0.04}
        geom = json.loads(obs.geometry_json)
        # Ordered highest-first, with heights, so "the higher open drawer" in the
        # image can be tied to a name.
        ordered = sorted(openings.items(), key=lambda kv: -geom[kv[0]]["handle_height_m"])
        open_list = "; ".join(
            f"{name} = {geom[name]['stack']} stack, handle {geom[name]['handle_height_m']}m high, "
            f"pulled out {value}m" for name, value in ordered) or "none"
        try:
            answer = self.astra.decide(
                "belief", BELIEF_SYSTEM,
                f"Drawers currently pulled open, highest first:\n{open_list}\n"
                f"The robot has just attempted {obs.acted_drawer}.\n"
                "Report what is inside each open drawer, attributing each "
                "observation to one of the named drawers above.",
                [obs.frame_jpeg],
                function="report_observation",
                parameters={"type": "object", "properties": {
                    "observations": {"type": "array", "description":
                        "One entry per open drawer you can see into.",
                        "items": {"type": "object", "properties": {
                            "drawer": {"type": "string",
                                       "description": "Exact name from the list given."},
                            "contents": {"type": "string",
                                         "description": "Colour and shape, or 'empty'."},
                            "confidence": {"type": "number"},
                        }, "required": ["drawer", "contents", "confidence"],
                            "additionalProperties": False}},
                    "summary": {"type": "string"},
                }, "required": ["observations", "summary"], "additionalProperties": False},
            )
        except astra_mod.BudgetExhausted as stop:
            print(f"  [belief] stopping: {stop}")
            return None
        rows = [r for r in answer.get("observations", []) if r.get("drawer") in kitchen.DRAWERS]
        for row in rows:
            print(f"  [belief #{seq}] {row['drawer']}: {row['contents'][:64]!r} "
                  f"(conf {row['confidence']:.2f})")
        if not rows:
            print(f"  [belief #{seq}] nothing attributable ({len(answer.get('observations', []))} raw)")
        if self.recorder and rows:
            best = max(rows, key=lambda r: r.get("confidence", 0))
            self.recorder.set(note=f"Astra sees -- {best['drawer'].split('_')[-2]}_"
                                   f"{best['drawer'].split('_')[-1]}: {best['contents'][:96]}")
        return Belief(seq=seq, open_drawers_json=json.dumps(openings),
                      contents_json=json.dumps(rows), summary=answer["summary"])


@dataclass
class MemoryFlow(Flow[Belief, Digest]):
    """Folds each observation into the structured record. No model here."""

    memory: object = None
    candidates: tuple = ()
    planner: object = None
    name: str = "Memory"
    _served: int = -1

    def step(self, belief: Belief) -> Digest | None:
        seq = getattr(belief, "seq", None)
        if seq is None or seq == self._served:
            return None
        self._served = seq
        for row in json.loads(belief.contents_json):
            if row.get("contents"):
                self.memory.note_contents(row["drawer"], contents=row["contents"],
                                          confidence=float(row.get("confidence") or 0.0),
                                          step=seq)
        text = self.memory.digest(self.candidates)
        return Digest(seq=seq, text=text,
                      decisions=len(getattr(self.planner, "decisions", [])),
                      finished=bool(getattr(self.planner, "finished", False)))


def build_placements(seed, openable, labels):
    chooser = random.Random(seed)
    drawers = chooser.sample(list(openable), k=min(len(labels), len(openable)))
    return [kitchen.Placement(drawer, label) for drawer, label in zip(drawers, labels)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="Find the reddish-brown spice jar hidden in one of the kitchen drawers.")
    parser.add_argument("--labels", default="paprika,herb seasoning")
    parser.add_argument("--openable", default="", help="Comma list of drawers a pull can open; default: kitchen.OPENABLE (measured).")
    parser.add_argument("--max-decisions", type=int, default=8)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--planner-hz", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--video", default="", help="Write an mp4 of the run to this path.")
    parser.add_argument("--video-stride", type=int, default=2)
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    openable = ([x.strip() for x in args.openable.split(",") if x.strip()]
                or list(kitchen.OPENABLE))
    placements = build_placements(args.seed, openable, labels)
    print("  hiding (never shown to the model):",
          ", ".join(f"{p.label} in {p.drawer}" for p in placements))

    world = kitchen.DrawerWorld(placements)
    memory = DrawerMemory()
    channel = astra_mod.Astra(max_calls=args.max_calls, max_usd=args.budget_usd)

    recorder = Recorder(args.video, stride=args.video_stride) if args.video else None
    planner = AstraPlannerFlow(astra=channel, memory_ref=memory, task=args.task,
                               candidates=kitchen.DRAWERS, max_decisions=args.max_decisions)
    robot = KitchenFlow(world=world, memory=memory, recorder=recorder)
    belief = AstraBeliefFlow(astra=channel, recorder=recorder)
    recall = MemoryFlow(memory=memory, candidates=kitchen.DRAWERS, planner=planner)

    retriever.init(backend="in-process")
    with Pipeline(name="astra drawer search") as pipe:
        p = planner @ Rate(hz=args.planner_hz)      # the only clock in the loop
        k = robot @ Trigger("seq")
        b = belief @ Trigger("seq")
        m = recall @ Trigger("seq")
        pipe.connect(p, k, sync=Latest())
        pipe.connect(k, b, sync=Latest())
        pipe.connect(b, m, sync=Latest())
        pipe.connect(m, p, sync=Latest())           # closes the loop

    started = time.time()
    pipe.run(duration=args.duration)
    wall = time.time() - started

    truth = world.ground_truth()
    record = {
        "task": args.task,
        "model": channel.model,
        "roles_on_astra": ["planner", "belief"],
        "memory": "structured, model-free (memory.py)",
        "candidates": list(kitchen.DRAWERS),
        "unreachable": list(kitchen.UNREACHABLE_DRAWERS),
        "openable_by_primitive": openable,
        "ground_truth": truth,
        "decisions": planner.decisions,
        "memory_state": memory.as_dict(),
        "score": memory.score(truth),
        "cost": channel.ledger(),
        "wall_time_s": round(wall, 1),
        "sim_steps": robot.step_count,
    }
    if recorder:
        recorder.set(decision="run complete", action="", drawer="",
                     metric=f"{len(planner.decisions)} decisions   ${channel.spent_usd:.3f}")
        recorder.hold(world, 2.0)
        n_frames = recorder.close()
        record["video"] = {"path": args.video, "frames": n_frames,
                           "fps": recorder.fps, "stride": args.video_stride}
        print(f"  video: {args.video} ({n_frames} frames)")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"astra_drawer_search_{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    world.render().save(out / f"astra_drawer_search_{stamp}_final.png")

    print("\n  ==== run summary ====")
    print(f"  decisions: {len(planner.decisions)}  sim steps: {robot.step_count}  wall: {wall:.0f}s")
    print(f"  memory: {json.dumps(memory.as_dict()['records'], default=str)[:300]}")
    print(f"  repeated_visits={memory.repeated_visits} unsupported={memory.unsupported_actions} "
          f"recovery={memory.recovery_actions}")
    print(f"  score: colour matched {record['score']['colour_matched']}/{record['score']['of']}"
          f", product named {record['score']['named_product']}/{record['score']['of']}")
    print(f"  cost: ${channel.spent_usd:.4f} over {len(channel.calls)} calls")
    print(f"  record: {path}")


if __name__ == "__main__":
    main()
