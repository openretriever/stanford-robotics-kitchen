# SRC Kitchen

A standalone, installable kitchen scene for Retriever. It owns the reconstructed
room, portable robot assets, simulation code, and a local demonstration viewer.
It does not own task authorization, a general agent, or real-device access.

The scene includes a controllable Panda and a parked Mobile ALOHA. Mobile ALOHA
is AgileX's simulation variant, not an exact model of the original Stanford
hardware. Its geometry is visual-only in the shared MuJoCo model, so both the
standalone preview and a consuming simulator viewer see the same placement.
It faces the stove with lowered arms and a frying pan on the cooktop. The robot
camera button frames that station. This is static scene staging, not a cooking
policy or a validated manipulation task; the Panda drawer demo is unchanged.
Robot finishes are an approximate visual interpretation of
[AgileX manufacturer photos](https://global.agilex.ai/products/cobot-magic):
dark arms and mounting deck, light body panels, rubber wheels and metal
gripper fingers. They replace the URDF's generic CAD colors, not measured paint
values. The preview preserves their metallic/roughness factors and CAD creases.
There are 12 kitchen drawers and four tabletop organizer drawers. Dimensions
are estimated from a visual reconstruction, not a calibrated survey.

## Run Locally

Python 3.11 and `uv` are required. After cloning, obtain the large assets with
Git LFS before installation:

```sh
git lfs pull
uv sync --extra preview --extra test
uv run stanford-robotics-kitchen verify
uv run stanford-robotics-kitchen preview --port 8105 --viewer-port 8106
```

Open the printed loopback URL. The demo begins paused. Ctrl-C stops its HTTP
server, viewer, and runtime. Recordings go to `runs/` (or `--output-dir`), never
into the installed scene resources. Use unused ports when another preview is
running. The existing source and other consumers are not modified by this repo.

The known simulation stack uses MuJoCo 3.3.1. `mjviser` 0.0.14 declares a newer
MuJoCo requirement; the checked-in `uv` override intentionally selects the
tested version. This override is not a claim of upstream compatibility with
every configuration. The same override is in `tools/simulation-overrides.txt`
for `uv pip install --override ...` workflows.

## Package API

```python
from retriever_src_kitchen import describe_scene, scene_root, create_scene

print(describe_scene())       # metadata only; no native imports or side effects
print(scene_root())           # installed scene-source directory
model, data, spec = create_scene(robot="Panda")  # no clock, policy, or server
```

Install `.[simulation]` for native construction or `.[preview]` for the viewer.
`create_demo("search")` constructs the scripted demonstration separately. That
demo contains target knowledge and is not a task planner or an evaluation of
general object search. The consuming host must use its admitted skill interface,
not the demo's automatic search loop.

The payload retains the established flat scene-source layout for existing
consumers. Do not mix two scene snapshots or import an unrelated `mink` into the
same interpreter; use fresh simulator processes. Package entrypoints restore
the import search path and reject known source-module conflicts.

## Retriever Hub

`pyproject.toml` declares `[tool.retriever.module]` with these lazy exports:
`scene_root`, `describe_scene`, `create_scene`, `create_demo`, `harness_config`,
`describe_pipeline`, `pipeline`, and `preview`. Loading exports does not start
simulation, open sockets, download assets, or acquire hardware. Tests exercise
the installed Hub manifest loader.

The `pipeline` export is a lazy builder for the existing five-Flow execution
graph. It constructs a paused MuJoCo runtime but does not start a clock, viewer,
HTTP server, model request, or hardware connection:

```python
from retriever import hub

build = hub.use("openretriever/stanford-robotics-kitchen:pipeline")
pipeline = build(task="search")

print(pipeline.validate())
pipeline.kitchen_runtime.controls.set_paused(False)
pipeline.step(dt=0.02)
pipeline.close_stepper()
```

Use `hub.use("openretriever/stanford-robotics-kitchen:describe_pipeline")()` to inspect the
nodes and edges without loading Retriever, MuJoCo, or NumPy. The repository is
private; install it explicitly and pin the reviewed release or commit.

## Harness Integration

Install this package into the interpreter selected by the consuming Harness,
alongside its existing `retriever_kitchen_sim` plugin. The scene-source path is
`retriever_src_kitchen.scene_root()`. No Harness source or registry edit is needed.

In that environment, inspect the candidate source fingerprint without running
the simulator:

```sh
python -m retriever_kitchen_sim fingerprint "$(src-kitchen path)"
python -m retriever_kitchen_sim doctor "$(src-kitchen path)" \
  --expected-source-digest REVIEWED_SHA256
```

After review, pass the same digest to `harness_config(REVIEWED_SHA256)` and select
the installed plugin factory in the operator-owned profile. The existing
`source_root` / `expected_source_digest` contract is unchanged. Never silently
substitute this package for an already-reviewed scene or regenerate its pin at
startup. See `examples/harness/` for a profile preparation helper and template.
The prepared live launcher is deliberately not rewritten by this migration.

## Development

```sh
uv run pytest
uv run python tools/check_repository.py
uv build
```

`stanford-robotics-kitchen verify` checks the packaged inventory, including source files and
asset hashes. It does not certify physical behavior, license rights, or native
binary identity. Native tests construct the scene and exercise drawer motion.

The checked-in MJCF and meshes are the runnable snapshot. `build_scene.py`
reconstructs the lightweight room geometry; the textured scene also includes
baked visual meshes. Reproducing those baked textures requires a separate
authoring workflow and is not claimed by this source package.

To rebuild Mobile ALOHA from its pinned public URDF, install `.[assets]`, run the
payload's `build_mobile_aloha.py --fetch`, then `tools/build_shared_visuals.py`.
The latter embeds only static, non-colliding geometry. After reviewing changes,
run `tools/write_asset_manifest.py` and the tests. Updating that inventory is
explicit; it does not update a consuming host's authorization or reviewed pin.

## Distribution

This repository is shared privately through the OpenRetriever organization for
approved collaborators. Clone it with Git LFS enabled and run `git lfs pull` to
obtain the scene assets. Repository access does not grant public redistribution
rights for the reconstruction or brand assets.

Large meshes and images use Git LFS. Recordings, environment directories,
caches, and authoring binaries are excluded. See `LICENSE` and
`THIRD_PARTY.md`; there is no blanket open-source license for the reconstruction
or the brand assets.
