"""Run E2's memory ablation: structured record vs plain transcript, several seeds each.

    ./.venv/bin/python compare.py --seeds 3 --decisions 8 --budget 0.60

Each (arm, seed) is one pipeline.py run in its own process; two run at a time.
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


def launch(arm, seed, args, outdir):
    cmd = [sys.executable, "-u", str(HERE / "pipeline.py"), "--memory", arm, "--seed", str(seed),
           "--max-decisions", str(args.decisions), "--budget-usd", str(args.budget),
           "--max-calls", str(args.decisions * 2 + 2), "--duration", str(args.duration),
           "--planner-hz", "0.2", "--output", str(outdir)]
    log = open(outdir / f"{arm}_seed{seed}.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "MUJOCO_GL": "cgl"}, cwd=HERE)


def metrics(record):
    dec = record["decisions"]
    acts = [d["action"] for d in dec]
    mem = record["memory_state"]
    attempts = [r["attempts"] for r in mem["records"].values()]
    return {
        "found": acts[-1] == "report_found" if acts else False,
        "supported": (acts[-1] == "report_found" and mem["unsupported_actions"] == 0) if acts else False,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--decisions", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.60)
    ap.add_argument("--duration", type=float, default=480)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "compare")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(arm, seed) for seed in range(1, args.seeds + 1) for arm in ARMS]
    est = len(jobs) * args.decisions * 2 * 0.012
    print(f"  {len(jobs)} runs ({args.seeds} seeds x {len(ARMS)} arms), up to {args.decisions} decisions each; "
          f"estimated <= ${est:.2f} at ~$0.012/call; hard cap ${args.budget}/run")
    running, results = [], {}
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < args.parallel:
            arm, seed = jobs.pop(0)
            outdir = args.out / f"{arm}_seed{seed}"; outdir.mkdir(exist_ok=True)
            running.append((arm, seed, outdir, launch(arm, seed, args, outdir)))
            print(f"  started {arm} seed {seed}")
        time.sleep(5)
        for job in list(running):
            arm, seed, outdir, proc = job
            if proc.poll() is None:
                continue
            running.remove(job)
            recs = sorted(glob.glob(str(outdir / "*.json")))
            if recs:
                results[(arm, seed)] = metrics(json.load(open(recs[-1])))
                m = results[(arm, seed)]
                print(f"  done    {arm} seed {seed}: found={m['found']} horizon={m['horizon']} "
                      f"recovery={m['recovery']} repeat={m['repeated_physical']} ${m['usd']:.3f}")
            else:
                print(f"  FAILED  {arm} seed {seed} (no record; see {outdir}/{arm}_seed{seed}.log)")

    def agg(arm, key, how=statistics.mean):
        vals = [m[key] for (a, _), m in results.items() if a == arm]
        return how(vals) if vals else float("nan")

    rows = []
    for arm in ARMS:
        n = sum(1 for (a, _) in results if a == arm)
        rows.append({
            "arm": arm, "runs": n,
            "found": sum(m["found"] for (a, _), m in results.items() if a == arm),
            "supported": sum(m["supported"] for (a, _), m in results.items() if a == arm),
            "horizon": agg(arm, "horizon"), "repeated_visits": agg(arm, "repeated_visits"),
            "repeated_physical": agg(arm, "repeated_physical"), "recovery": agg(arm, "recovery"),
            "unsupported": agg(arm, "unsupported"), "colour": agg(arm, "colour"),
            "calls": agg(arm, "calls"), "usd": agg(arm, "usd", sum), "wall_s": agg(arm, "wall_s"),
        })
    summary = {"seeds": args.seeds, "decisions_cap": args.decisions, "rows": rows,
               "per_run": {f"{a}_seed{s}": m for (a, s), m in results.items()},
               "wall_total_s": round(time.time() - t0)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    hdr = ["arm", "runs", "found", "supported", "horizon", "repeat visits", "repeat physical",
           "recovery", "unsupported", "colour/2", "calls", "$ total", "wall s"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        md.append("| " + " | ".join([
            r["arm"], str(r["runs"]), str(r["found"]), str(r["supported"]), f"{r['horizon']:.1f}",
            f"{r['repeated_visits']:.1f}", f"{r['repeated_physical']:.1f}", f"{r['recovery']:.1f}",
            f"{r['unsupported']:.1f}", f"{r['colour']:.1f}", f"{r['calls']:.1f}", f"{r['usd']:.2f}",
            f"{r['wall_s']:.0f}"]) + " |")
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\n  {args.out}/summary.json  (total wall {summary['wall_total_s']}s)")


if __name__ == "__main__":
    main()
