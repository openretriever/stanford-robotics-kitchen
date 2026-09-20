# Scope

This repository owns the SRC kitchen scene and its simulation examples.
Keep task authorization, generic execution admission, and task-state ownership
in the consuming host. Imports and Hub discovery must not start a simulator,
viewer, network request, or hardware connection.

# Confidentiality

Keep this repository local/private unless publication is explicitly approved.
Do not copy recordings, user messages, credentials, machine-specific paths,
environments, or unrelated work into it. Before committing or publishing,
scan the proposed tree and full branch history. Keep asset provenance and all
third-party licenses. Do not imply branding or redistribution permission.

# Verification

Use `uv sync --extra preview --extra test`, then `uv run pytest`.
Use `uv run stanford-robotics-kitchen preview --port 8105 --viewer-port 8106` for visual QA.
Runtime outputs belong in `runs/`, never inside the installed scene resources.
Keep the legacy scene-source layout compatible with existing consumers until
they migrate to the package API. Do not silently replace a reviewed scene pin.
