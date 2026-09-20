"""Create a new operator profile only for an explicitly reviewed scene pin."""

import argparse
import json
from pathlib import Path

from retriever_src_kitchen import harness_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = harness_config(args.expected_source_digest)
    from retriever_kitchen_sim.source import source_fingerprint

    if source_fingerprint(config["source_root"]) != args.expected_source_digest:
        parser.error("Scene differs from the reviewed fingerprint; profile was not written")
    profile = Path(__file__).with_name("profile.template.toml").read_text()
    profile = profile.replace('"REPLACE_WITH_INSTALLED_SCENE_PATH"', json.dumps(config["source_root"]))
    profile = profile.replace("REPLACE_WITH_REVIEWED_SHA256", args.expected_source_digest)
    with args.output.open("x") as output:
        output.write(profile)
    print(args.output)


if __name__ == "__main__":
    main()
