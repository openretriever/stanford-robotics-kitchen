"""Check an installed scene against an explicit consumer checkout, without a run."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.harness_root.resolve(strict=True)
    for relative in ("src", "examples/plugins/kitchen-sim/src"):
        directory = root / relative
        if not directory.is_dir():
            parser.error("Expected a checkout with the existing kitchen-sim plugin")
        sys.path.insert(0, str(directory))

    from retriever_src_kitchen import scene_root
    from retriever_kitchen_sim.doctor import inspect_setup
    from retriever_kitchen_sim.source import source_fingerprint
    from retriever_kitchen_sim.world import KitchenWorld

    source = scene_root()
    candidate = source_fingerprint(source)
    report = inspect_setup(source, candidate)
    if not report["checks_passed"]:
        raise SystemExit(json.dumps(report, indent=2))
    world = KitchenWorld(source, candidate)
    assert world.scene.model.body("parked_mobile_aloha").id >= 0
    assert world.scene.model.nu == 9
    before = world.scene.data.time
    for _ in range(5):
        world.hold_step()
    assert world.scene.data.time > before
    snapshot = world.viewer_snapshot("compatibility-check", 1)
    assert snapshot["sequence"] == 1
    assert len(world.viewer_model()) > 0
    print(json.dumps({"checks_passed": True, "native_world_constructed": True,
                      "mobile_aloha_in_shared_model": True, "hold_ticks": 5,
                      "candidate_source_digest": candidate, "admitted_runs": 0,
                      "goal_success": "not_evaluated", "live_profiles_modified": False}, indent=2))


if __name__ == "__main__":
    main()
