"""Local scene discovery, verification, and standalone preview."""

import argparse
import json
from pathlib import Path

from .api import describe_scene, preview, scene_root
from .resources import verify_resources


def main(argv=None):
    parser = argparse.ArgumentParser(prog="src-kitchen")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("path", help="Print the installed scene source directory")
    commands.add_parser("info", help="Read scene metadata without loading simulation")
    commands.add_parser("verify", help="Check package assets against the recorded hashes")
    viewer = commands.add_parser("preview", help="Start the standalone, scripted local demo")
    viewer.add_argument("--port", type=int, default=8105)
    viewer.add_argument("--viewer-port", type=int, default=8106)
    viewer.add_argument("--task", choices=("search", "cup", "drawer", "seasoning"), default="search")
    viewer.add_argument("--output-dir", type=Path, default=Path.cwd() / "runs")
    args = parser.parse_args(argv)
    if args.command == "path":
        print(scene_root())
    elif args.command == "info":
        print(json.dumps(describe_scene(), indent=2))
    elif args.command == "verify":
        report = verify_resources()
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
    else:
        report = verify_resources()
        if not report["ok"]:
            parser.error("Scene resources are incomplete or changed; run stanford-robotics-kitchen verify (and git lfs pull after cloning)")
        preview(port=args.port, viewer_port=args.viewer_port, task=args.task, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
