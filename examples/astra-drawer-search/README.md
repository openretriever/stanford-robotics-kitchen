# Astra drawer search in the SRC kitchen

A long-horizon drawer search built as a Retriever pipeline, where **every
model-backed role is `gpt-6-astra`** and memory is deliberately not a model.

```
Planner ---> Kitchen ---> Belief ---> Memory ---+
   ^                                            |
   +--------------------------------------------+
```

| role | what it is | model |
| --- | --- | --- |
| Planner | chooses the next drawer from the written record | gpt-6-astra |
| Kitchen | src-kitchen's contact-driven Panda drawer pull | no model |
| Belief | reads what is inside an opened drawer from pixels | gpt-6-astra |
| Memory | inspected drawers, hypotheses, subgoals, failures, evidence | no model, on purpose |

## Set up

`retriever-core` is on PyPI and conda-forge. The kitchen scene comes from
`stanford-robotics-kitchen` (public at github.com/openretriever/stanford-robotics-kitchen;
not yet on PyPI) -- install it from its own
checkout (a built wheel under its `dist/`, or `uv pip install -e <path>`).

```sh
uv venv -p 3.11 .venv
uv pip install --python .venv/bin/python 'retriever-core[dora,demo]'
uv pip install --python .venv/bin/python --override <(printf 'mujoco==3.3.1\nmink==0.0.13\n') \
    'stanford-robotics-kitchen[preview] @ file:///path/to/stanford-robotics-kitchen/dist/stanford_robotics_kitchen-0.2.0-py3-none-any.whl'
```

`MUJOCO_GL=cgl` is required on macOS (`egl` is Linux-only). The pipeline runs on
retriever-core's **in-process** backend, so no dora daemon and no rerun viewer
are needed; both are optional.

## Run it

```sh
MUJOCO_GL=cgl ./.venv/bin/python -u pipeline.py --max-decisions 6 --budget-usd 1.00
```

The key is read from `~/.config/openai/key` (or `OPENAI_API_KEY`); no key is
stored in this directory. A run writes `runs/astra_drawer_search_<stamp>.json`
(gitignored; a sample is committed at `docs/example-run.json`) with every decision, the full memory
state, the cost ledger, and the score against ground truth.

## Recording a run

```sh
MUJOCO_GL=cgl ./.venv/bin/python -u pipeline.py --max-decisions 6 --video runs/run.mp4
```

Writes 1280x720 h264 / yuv420p at 50 fps, streamed frame by frame -- a
six-decision run is about 4,000 frames, which at 1280x720x3 would be roughly
11 GB if buffered. Each frame carries a caption bar (decision, action, target
drawer, the planner's own rationale, live phase and opening) and insets the
320x320 `drawer_view` the model actually receives, so what Astra sees is visible
beside what happened. `--video-stride` trades frames for time; at the default 4
recording costs about 20 ms per control step.

Two rendering fixes were needed before the video was worth looking at, and the
second one mattered for more than looks:

- **Lighting.** As compiled, the three ceiling lights carry diffuse 0.06 each
  while a camera-mounted headlight does most of the work. A headlight casts no
  shadows and moves with the eye, so everything renders flat and grey. The real
  lights are raised and warmed and the headlight is dropped to a fill.
- **A ten-metre green line through every frame.** It is
  `robot/gripperright_grip_site_cylinder`, robosuite's grip-axis marker --
  `size=[0.005, 10.0, 0.005]`, rgba `(0,1,0,0.30)`, site group 1. Group 2 holds
  the scene's teal annotation markers. Both are now hidden from every render,
  including the model's. This was not cosmetic: asked what it saw, Astra
  reported "a thin green elongated object" inside a drawer.

## What is real and what is not

Honesty about the parts matters more than the demo looking good.

- **The drawer really is opened by the robot.** This reuses src-kitchen's
  `DrawerPull`: seven mink-IK phases, and success requires `opening > 0.25 m`
  **and** two-finger contact held for more than 90% of the pull frames. It is not
  a rail force and not a teleported joint value. Its author's caveat stands --
  the opening is *scripted*; what Astra supplies is the *search*.
- **Astra never sees ground truth.** Which drawer holds which jar is read only
  after the run, to score it.
- **The jars are visual-only** (`contype=0`, `conaffinity=0`, zero mass). A
  collidable jar wedges the drawer and stops the pull at 0.0144 m instead of
  0.2995 m -- measured. They exist to be seen, not manipulated. Nothing here
  picks anything up.
- **5 of the 8 reachable drawers actually open.** Measured by sweeping all eight
  to completion:

  | drawer | opening | verified |
  | --- | --- | --- |
  | `range_drawers_drawer0_1` / `0_2` / `0_3` | 0.298 / 0.298 / 0.299 | yes |
  | `range_drawers_drawer1_2` / `1_3` | 0.301 / 0.299 | yes |
  | `range_drawers_drawer0_0` | 0.067 | no -- "Drawer target not reached" |
  | `range_drawers_drawer1_0` / `1_1` | 0.000 | no |

  The three failures are left in the candidate list on purpose: a search pointed
  only at drawers guaranteed to open would never exercise failure recovery, which
  is half of what the memory is for.

  **Those five are measured in isolation.** During a run they degrade: an open
  drawer physically blocks the approach to the one below it, so later pulls reach
  0.23-0.27 m and fail the two-finger check even though the drawer opens far
  enough to see into. One case went the other way -- re-pulling an
  already-part-open drawer reached 0.380 m. A pull can also shove a
  previously-opened drawer shut. Order matters, and the record reports opening
  and contact separately so a partial open is not mistaken for a clean grasp.
- **The four `beverage_drawers` are unreachable**, at x = -2.5 against a Panda
  based at (0, 1.90, 0) -- about 2.9 m away. They are reported as
  `unreachable` in every run record rather than silently dropped.
- **The robot mount was lowered to 0.30 m.** At `DrawerPull`'s own 0.58 m only
  *one* drawer of four opens, the lower rows failing in "Align with handle" at a
  0.043 m gap that is almost entirely vertical. At 0.30 m it is three of four.

## Two runtime facts worth knowing

**A cycle self-clocks.** A two-node loop wired with `Latest()` on both edges ran
**36,161 iterations in three seconds** during development. With Astra on those
nodes that is a way to spend real money by accident. So the planner carries the
only `Rate()` in the loop, every flow ignores a sequence number it has already
served, and `astra.Astra` enforces a hard call-and-dollar cap as a second line of
defence.

**Astra's tool calls need `/v1/responses`.** Chat completions answers prose
fine, but asking for function tools there is refused: *"Function tools with
reasoning_effort are not supported for gpt-6-astra in /v1/chat/completions."*
Every role here wants a schema-validated answer *and* reasoning, so the client
targets `/v1/responses` -- the same endpoint `robot-sim/agent.py` uses. Note this
means a LiteLLM-based pipeline (as in `retriever-use`) cannot host these roles
unchanged.

## Cost

One Astra call with an image measured **$0.0196** and **11.3 s** at
`reasoning_effort: low` ($12.5/M in, $50/M out). A decision cycle is two calls
(planner + belief), so roughly **$0.04 and 25 s per decision**. A six-decision
run is about **$0.25**. `--budget-usd` is enforced, not advisory.

## An observed run

Six decisions, 11 Astra calls, $0.1362, 230 s wall:

1. `drawer0_0` -- fails at 0.067 m. Belief: dark, "emptiness cannot be confirmed".
2. `drawer0_1` -- **clean pull, 0.299 m, contact 1.00**. Belief: "empty" (correct).
3. `drawer0_2` -- 0.229 m. Belief: "empty" (wrong; too occluded to see the jar).
4. `drawer0_3` -- 0.233 m. Belief: "a partially visible reddish-brown cylindrical
   object at the back", attributed to `drawer0_3` (correct).
5. `drawer0_3` **again** -- the planner: "the partially visible reddish-brown
   cylindrical object could be the spice jar, but it is not confirmed". Opens to
   0.380 m. Belief: "a dark reddish-brown cylindrical jar", confidence 0.94.
6. `report_found` -- "reddish-brown spice jar", supported by an observation.

Step 5 is the point of the whole exercise: an information-seeking revisit driven
by the memory record rather than by a script. `recovery_actions=1`,
`repeated_visits=0`, `unsupported_actions=0`, colour matched 1/2.

Scored on colour rather than product name, deliberately. The jars are unlabelled
coloured cylinders, so "paprika" is not readable from pixels; Astra said
"reddish-brown cylindrical jar" and declined to name the spice, which is the
correct answer. An earlier metric that demanded the product name scored this run
0/2 -- a metric failure, not a model failure.

## E2: the memory ablation

This implements "E2 -- Long-horizon drawer search with memory" from
`docs/icra_experiment_plan_v1.md`, including its central comparison:

```sh
./.venv/bin/python pipeline.py --memory structured   # E2's robot-memory capability
./.venv/bin/python pipeline.py --memory transcript    # the control: a plain chronological log
./.venv/bin/python compare.py --seeds 3 --decisions 8 # both arms, several seeds, one table
```

The two arms are built from the same events -- what was attempted, what the pull
reported, what the model said it saw -- and differ only in what the planner is
handed. `structured` gives it `memory.DrawerMemory.digest()`: one line per
candidate drawer with attempts, open/closed, best opening, failures and their
reasons, and the contents hypothesis with its confidence. `transcript` gives it
`memory.Transcript.digest()`: the same facts in the order they happened, never
folded per drawer, so whether a drawer has already failed twice is something it
must work out from the log each turn.

Measurement is always structured. The `DrawerMemory` record is kept in both arms
because it is how a run is scored; the ablation varies only what the planner
sees. Per run the record carries decisions used (horizon), repeated drawer
visits, repeated physical actions, recovery actions, unsupported-belief claims,
model calls and tokens by role, wall time, and the colour score. `compare.py`
reads those back and tabulates them; it adds no measurement of its own.

Not built: the four injected perturbations (jammed drawer, relocated target,
failed grasp, stale observation) as a separate study. The three drawers the
pull cannot open already supply failed manipulations, but not on demand.

## Running to completion

`pipeline.py` runs the pipeline non-blocking and stops the engine a few seconds
after the planner reports or exhausts its decisions, instead of idling out the
full `--duration`. `--duration` remains the hard ceiling.
