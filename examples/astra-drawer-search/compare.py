"""Run E2's grid: memory arm x mid-task perturbation x seed, the target placed where the scan ends.

    ./.venv/bin/python compare.py --seeds 3 --decisions 12 --budget 0.60
    ./.venv/bin/python compare.py --perturb none --placement random --decisions 8   # the 22 Sept ablation

Each cell is one pipeline.py run in its own process; two run at a time.
The metrics E2 asks for are read back from the run records the pipeline already
writes, so this script adds no measurement of its own -- only the table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("structured", "transcript")
PERTURBS = ("none", "relocate", "stale", "jam")
COUNTED = ("found", "correct", "supported", "usd")     # summed per row; everything else is averaged
COLUMNS = (("arm", "arm", "{}"), ("perturb", "perturb", "{}"), ("runs", "runs", "{}"),
           ("found", "found", "{}"), ("correct", "correct", "{}"), ("supported", "supported", "{}"),
           ("horizon", "horizon", "{:.1f}"), ("repeated_visits", "repeat visits", "{:.1f}"),
           ("repeated_physical", "repeat physical", "{:.1f}"), ("recovery", "recovery", "{:.1f}"),
           ("unsupported", "unsupported", "{:.1f}"), ("colour", "colour/2", "{:.1f}"),
           ("calls", "calls", "{:.1f}"), ("usd", "$ total", "{:.2f}"), ("wall_s", "wall s", "{:.0f}"))


def launch(arm, perturb, seed, args, outdir):
    cmd = [sys.executable, "-u", str(HERE / "pipeline.py"), "--memory", arm, "--seed", str(seed),
           "--placement", args.placement, "--perturb", perturb, "--perturb-after", str(args.perturb_after),
           "--max-decisions", str(args.decisions), "--budget-usd", str(args.budget),
           "--max-calls", str(args.decisions * 2 + 2), "--duration", str(args.duration),
           "--planner-hz", "0.2", "--output", str(outdir)]
    log = open(outdir / f"{arm}_{perturb}_seed{seed}.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "MUJOCO_GL": "cgl"}, cwd=HERE)


def metrics(record):
    dec = record["decisions"]
    acts = [d["action"] for d in dec]
    mem = record["memory_state"]
    attempts = [r["attempts"] for r in mem["records"].values()]
    reported = acts[-1] == "report_found" if acts else False
    target = next(iter(record["ground_truth"]))       # the first label's drawer, as relocated
    return {
        "found": reported,
        "correct": reported and dec[-1]["drawer"] == target,
        "supported": reported and mem["unsupported_actions"] == 0,
        "horizon": len(dec),
        "repeated_visits": mem["repeated_visits"],
        "repeated_physical": sum(a - 1 for a in attempts if a > 1),
        "recovery": mem["recovery_actions"],
        "unsupported": mem["unsupported_actions"],
        "colour": record["score"]["colour_matched"],
        "calls": record["cost"]["total_calls"],
        "usd": record["cost"]["total_usd"],
        "wall_s": record["wall_time_s"],
    }


def summarize(results):
    """One row per (arm, perturbation) over its seeds. results: {(arm, perturb, seed): metrics}."""
    rows = []
    for arm in ARMS:
        for perturb in PERTURBS:
            runs = [m for (a, p, _), m in results.items() if (a, p) == (arm, perturb)]
            if runs:
                rows.append({"arm": arm, "perturb": perturb, "runs": len(runs), **{
                    key: (sum if key in COUNTED else statistics.mean)([m[key] for m in runs])
                    for key in runs[0]}})
    return rows


def table(rows):
    md = ["| " + " | ".join(label for _, label, _ in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        md.append("| " + " | ".join(fmt.format(row[key]) for key, _, fmt in COLUMNS) + " |")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--decisions", type=int, default=12, help="Eight pulls reach the last drawer; leave room to re-plan.")
    ap.add_argument("--budget", type=float, default=0.60)
    ap.add_argument("--duration", type=float, default=720)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--placement", choices=("random", "first", "last"), default="last")
    ap.add_argument("--perturb", default=",".join(PERTURBS), help=f"Comma list from: {', '.join(PERTURBS)}")
    ap.add_argument("--perturb-after", type=int, default=2)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "compare")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    perturbs = [p.strip() for p in args.perturb.split(",") if p.strip()]
    jobs = [(arm, perturb, seed) for seed in range(1, args.seeds + 1) for perturb in perturbs for arm in ARMS]
    est = len(jobs) * args.decisions * 2 * 0.012
    print(f"  {len(jobs)} runs ({args.seeds} seeds x {len(perturbs)} perturbations x {len(ARMS)} arms), "
          f"up to {args.decisions} decisions each; estimated <= ${est:.2f} at ~$0.012/call; "
          f"hard cap ${args.budget}/run")
    running, results = [], {}
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < args.parallel:
            arm, perturb, seed = jobs.pop(0)
            outdir = args.out / f"{arm}_{perturb}_seed{seed}"; outdir.mkdir(exist_ok=True)
            running.append((arm, perturb, seed, outdir, launch(arm, perturb, seed, args, outdir)))
            print(f"  started {arm} {perturb} seed {seed}")
        time.sleep(5)
        for job in list(running):
            arm, perturb, seed, outdir, proc = job
            if proc.poll() is None:
                continue
            running.remove(job)
            recs = sorted(glob.glob(str(outdir / "*.json")))
            if recs:
                results[(arm, perturb, seed)] = m = metrics(json.load(open(recs[-1])))
                print(f"  done    {arm} {perturb} seed {seed}: found={m['found']} correct={m['correct']} "
                      f"horizon={m['horizon']} recovery={m['recovery']} repeat={m['repeated_physical']} ${m['usd']:.3f}")
            else:
                print(f"  FAILED  {arm} {perturb} seed {seed} (no record; see {outdir}/{arm}_{perturb}_seed{seed}.log)")

    rows = summarize(results)
    summary = {"seeds": args.seeds, "decisions_cap": args.decisions, "placement": args.placement,
               "perturb_after": args.perturb_after, "rows": rows,
               "per_run": {f"{a}_{p}_seed{s}": m for (a, p, s), m in results.items()},
               "wall_total_s": round(time.time() - t0)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = table(rows)
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\n  {args.out}/summary.json  (total wall {summary['wall_total_s']}s)")


if __name__ == "__main__":
    main()
