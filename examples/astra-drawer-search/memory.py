"""Structured robot memory for a drawer search -- the thing E2 actually compares.

E2's comparison is transcript-only agent state against "a structured robot-memory
capability that records inspected drawers, object hypotheses, completed subgoals,
failed manipulations, and links to supporting evidence". That is this file, and it
holds all five.

Deliberately model-free. If memory were itself a model call, then a wrong memory
and a wrong plan would share a failure mode, and the ablation would no longer
isolate what it claims to isolate. Everything here is arithmetic over observed
events, so the only thing the ablation varies is whether the planner is given
this record or just its own transcript.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class DrawerRecord:
    drawer: str
    attempts: int = 0
    opened: bool = False
    max_opening_m: float = 0.0
    contents: str = ""              # the model's hypothesis, never ground truth
    confidence: float = 0.0
    failures: list = field(default_factory=list)
    evidence_steps: list = field(default_factory=list)


@dataclass
class DrawerMemory:
    """Accumulated, inspectable state for one run."""

    records: dict = field(default_factory=dict)
    subgoals: list = field(default_factory=list)
    # E2 metrics that only a running record can produce:
    repeated_visits: int = 0        # acting on a drawer already known open
    unsupported_actions: int = 0    # acting with no observation behind it
    recovery_actions: int = 0       # acting after a failure on that drawer

    def record(self, drawer):
        return self.records.setdefault(drawer, DrawerRecord(drawer=drawer))

    # -- updates -------------------------------------------------------------

    def note_attempt(self, drawer):
        """Called before a manipulation, so the counters describe intent."""
        row = self.record(drawer)
        if row.opened:
            self.repeated_visits += 1
        if row.failures:
            self.recovery_actions += 1
        row.attempts += 1

    def note_claim(self, item):
        """Score a "found it" claim against what was actually observed.

        E2's unsupported-belief action means acting as though something is true
        with no observation behind it. Opening an untried drawer is not that --
        it is how you get an observation. Naming an item no image ever showed is.
        """
        seen = " ".join(row.contents for row in self.records.values()).lower()
        supported = bool(item) and item.split()[0].lower() in seen
        if not supported:
            self.unsupported_actions += 1
        return supported

    def note_outcome(self, drawer, *, success, opening_m, error, step):
        row = self.record(drawer)
        row.max_opening_m = max(row.max_opening_m, float(opening_m))
        row.evidence_steps.append(int(step))
        if success:
            row.opened = True
            goal = f"opened {drawer}"
            if goal not in self.subgoals:
                self.subgoals.append(goal)
        else:
            row.failures.append({"step": int(step), "reason": error or "unknown"})

    def note_contents(self, drawer, *, contents, confidence, step):
        """A visual hypothesis about one drawer. Overwrites only on more confidence."""
        row = self.record(drawer)
        if confidence >= row.confidence:
            row.contents, row.confidence = contents, float(confidence)
            if step not in row.evidence_steps:
                row.evidence_steps.append(int(step))

    # -- what the planner is allowed to see ----------------------------------

    def digest(self, candidates):
        """A compact, complete account of the run so far. No ground truth here."""
        lines = []
        for drawer in candidates:
            row = self.records.get(drawer)
            if row is None:
                lines.append(f"- {drawer}: never attempted, never observed")
                continue
            state = "OPEN" if row.opened else "still closed"
            bits = [f"attempts={row.attempts}", state, f"max_opening={row.max_opening_m:.3f}m"]
            if row.failures:
                bits.append(f"failed {len(row.failures)}x (last: {row.failures[-1]['reason']})")
            if row.contents:
                bits.append(f"seen inside: {row.contents} (confidence {row.confidence:.2f})")
            lines.append(f"- {drawer}: " + ", ".join(bits))
        return "\n".join(lines)

    def as_dict(self):
        return {
            "records": {k: vars(v) for k, v in self.records.items()},
            "subgoals": list(self.subgoals),
            "repeated_visits": self.repeated_visits,
            "unsupported_actions": self.unsupported_actions,
            "recovery_actions": self.recovery_actions,
        }

    def score(self, ground_truth, colour_words=None):
        """Compare hypotheses with the truth, on a criterion the task allows.

        Scoring on whether the model emitted the word "paprika" is unfair and was
        my first mistake here: the jar is an unlabelled coloured cylinder, so
        "paprika" is not readable from pixels and a model that says it is
        guessing. Astra answered "a reddish-brown cylindrical object", declined to
        name the spice, and scored 0/2 against product names -- a metric failure,
        not a model failure. So the criterion is what the pixels can support:
        did the hypothesis for the right drawer carry the right COLOUR?
        """
        colour_words = colour_words or {
            "paprika": ("reddish", "red", "brown", "maroon", "rust", "orange"),
            "herb seasoning": ("green", "olive"),
            "coarse salt": ("white", "pale", "off-white", "cream"),
        }
        rows = []
        for drawer, label in ground_truth.items():
            row = self.records.get(drawer)
            said = ((row.contents if row else "") or "").lower()
            words = colour_words.get(label, (label.split()[0].lower(),))
            rows.append({
                "drawer": drawer,
                "truth": label,
                "opened": bool(row and row.opened),
                "max_opening_m": round(row.max_opening_m, 4) if row else 0.0,
                "hypothesis": (row.contents if row else "") or "",
                "colour_matched": any(w in said for w in words),
                "named_product": label.split()[0].lower() in said,
            })
        return {"targets": rows,
                "colour_matched": sum(r["colour_matched"] for r in rows),
                "named_product": sum(r["named_product"] for r in rows),
                "of": len(rows)}


@dataclass
class Transcript:
    """E2's control arm: the run as a chronological log, with no aggregation.

    Same facts the structured record is built from -- what was attempted, what
    the pull reported, what the model said it saw -- but presented in the order
    they happened and never folded per drawer. Whether a drawer has already
    failed twice, or is already open, is something the planner must work out
    from the log each time rather than read off a record. That is the whole
    difference the ablation measures, so nothing else may differ.
    """

    lines: list = field(default_factory=list)

    def note_decision(self, seq, action, drawer, rationale):
        self.lines.append(f"{seq}. decided {action} {drawer} -- {rationale[:120]}")

    def note_outcome(self, seq, drawer, *, success, opening_m, contact, error):
        verdict = "verified grasp" if success else f"not verified ({error or 'no error given'})"
        self.lines.append(f"   pull of {drawer}: {verdict}; opening {opening_m:.3f} m; contact {contact:.2f}")

    def note_belief(self, seq, rows, summary):
        if rows:
            for row in rows:
                self.lines.append(f"   seen in {row['drawer']}: {row['contents'][:110]} "
                                  f"(confidence {float(row.get('confidence') or 0):.2f})")
        else:
            self.lines.append(f"   seen: {summary[:140]}")

    def digest(self, candidates):
        if not self.lines:
            return "Nothing attempted or observed yet."
        return "Transcript of this run so far:\n" + "\n".join(self.lines)
