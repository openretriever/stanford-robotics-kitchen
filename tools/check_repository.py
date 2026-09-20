"""Check the proposed tree and every reachable Git blob for portability leaks."""

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", "dist", "build", "runs"}
PATTERNS = {
    "machine-specific home": rb"/Users/[a-zA-Z0-9_.-]+/",
    "private temporary directory": rb"/private/(?:tmp|var)/[a-zA-Z0-9]",
    "possible API credential": rb"\bsk-[A-Za-z0-9_-]{24,}\b",
    "credential assignment": rb"(?i)(?:api_key|access_token|password)\s*=\s*[\"'][A-Za-z0-9_-]{20,}[\"']",
}


def main():
    failures = []

    def scan(name, data):
        for label, pattern in PATTERNS.items():
            if re.search(pattern, data):
                failures.append(f"{name}: {label}")

    count = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if set(relative.parts) & SKIP or any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            failures.append(f"{relative}: symlink requires explicit review")
        elif path.is_file():
            if path.suffix in (".mp4", ".webm", ".blend", ".mjb"):
                failures.append(f"{relative}: generated/authoring artifact")
            scan(str(relative), path.read_bytes())
            count += 1

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    history = set()
    if (ROOT / ".git").exists():
        scan("commit messages", git("log", "--all", "--format=%B"))
        for commit in git("rev-list", "--all").decode().splitlines():
            for row in git("ls-tree", "-r", "-z", commit).split(b"\0"):
                if not row:
                    continue
                meta, name = row.split(b"\t", 1)
                _, kind, oid = meta.split()
                if kind == b"blob" and oid not in history:
                    scan(f"history {oid.decode()[:12]}:{name.decode()}", git("cat-file", "blob", oid.decode()))
                    history.add(oid)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {count} proposed files; {len(history)} reachable historical blobs; no flagged patterns")
    print("Automated screening is not publication or asset-license clearance.")


if __name__ == "__main__":
    main()
