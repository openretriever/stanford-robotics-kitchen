| arm | perturb | runs | found | correct | supported | horizon | repeat visits | repeat physical | recovery | unsupported | colour/2 | calls | $ total | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| structured | relocate | 2 | 2 | 2 | 2 | 5.0 | 0.0 | 0.5 | 0.5 | 0.0 | 2.0 | 9.0 | 0.21 | 53 |
| structured | stale | 2 | 0 | 0 | 0 | 10.0 | 0.0 | 2.0 | 2.0 | 0.0 | 0.5 | 20.0 | 0.55 | 124 |
| structured | jam | 2 | 0 | 0 | 0 | 10.0 | 0.0 | 2.0 | 2.0 | 0.0 | 1.0 | 20.0 | 0.49 | 101 |
| transcript | relocate | 2 | 2 | 2 | 2 | 4.5 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 | 8.0 | 0.19 | 43 |
| transcript | stale | 2 | 0 | 0 | 0 | 10.0 | 0.0 | 2.0 | 2.0 | 0.0 | 0.5 | 20.0 | 0.63 | 129 |
| transcript | jam | 2 | 0 | 0 | 0 | 10.0 | 0.0 | 2.0 | 2.0 | 0.0 | 1.0 | 20.0 | 0.57 | 116 |

### First perturbation batch (22 September): a physics confound, not a memory result

Two seeds x {relocate, stale, jam} x both arms, ten-decision cap, target placed where the
scan ends (`docs/e2-perturbation-2026-09-22.md`, $2.64 in total):

| arm | perturb | found | correct | horizon | recovery | $ total |
|---|---|---|---|---|---|---|
| structured | relocate | 2/2 | 2/2 | 5.0 | 0.5 | 0.21 |
| transcript | relocate | 2/2 | 2/2 | 4.5 | 0.0 | 0.19 |
| structured | stale | 0/2 | 0/2 | 10.0 | 2.0 | 0.55 |
| transcript | stale | 0/2 | 0/2 | 10.0 | 2.0 | 0.63 |
| structured | jam | 0/2 | 0/2 | 10.0 | 2.0 | 0.49 |
| transcript | jam | 0/2 | 0/2 | 10.0 | 2.0 | 0.57 |

The arms are again indistinguishable, and the eight failed runs fail for the same physical
reason: with the target in `drawer1_3`, the scan opens `drawer1_2` first and the pull on
`drawer1_3` then ends `Drawer target not reached` (0.000 m) in every run -- an open drawer
blocks the approach to its column-mate, the same effect measured in the harness's kitchen-sim
plugin the same day. `drawer0_2` fails the same way behind an open `drawer0_1`. Only `relocate`
found the jar, because it moved the jar into `drawer0_3` before the scan got there. So the
perturbations were never the binding constraint: nothing closes a drawer, and after two or
three pulls most of the remaining drawers cannot be opened. Before the memory comparison can
be read, the robot needs a `close_drawer` primitive (or the planner must be told which drawers
an open neighbour blocks), and `relocate` should fire after the target drawer has been
observed, not before.
