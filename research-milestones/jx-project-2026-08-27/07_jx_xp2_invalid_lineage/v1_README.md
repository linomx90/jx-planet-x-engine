# JX-XP2 synthetic one-million-year robustness screen

JX-XP2 is a new, additive, locally frozen synthetic experiment. It does not
modify JX-XP1 and it does not use an observed catalogue, a survey simulator,
private author material, the retired candidate, or any previous trajectory as
an input.

The design contains 128 new deterministic massless tracers in eight 16-tracer
blocks. One control and 24 added-body cells combine the same three public
physical cases used by XP1 with eight fixed, equally weighted orientations.
Every cell is integrated with REBOUND MERCURIUS at 0.125 year and again at
0.0625 year. A single one-million-year trajectory supplies locked landmarks at
250,000, 500,000, and 1,000,000 years, sampled every 50 years.

The official endpoint is the one-million-year change in the sampled `q < 35
AU` fraction. Sampled `q < 30 AU` and `q < 40 AU` are secondary. The change is
an arithmetic paired-design contrast; its 3,072 cell/tracer exposures are not
independent observations. The tracer is the design unit. A fixed unique-tracer
event-support floor prevents a zero-event result from being described as a
small physical response.

A separate SciPy DOP853 implementation, which must not import REBOUND or the
primary runner, integrates a frozen 32-tracer, seven-arm sentinel through the
same one-million-year horizon. It is an independent numerical implementation
of the same declared Newtonian model, not independent physics. Its result is
compared with both MERCURIUS resolutions.

All official integrations use fixed 50,000-year segments. Segment boundaries,
sampling ownership, checkpoint hashes, resume behavior, retry rules, resource
watchdogs, A/B ordering, and fail-closed suppressions are part of the frozen
contract. The package must be content-hash registered before the first
trajectory is produced. A and B use distinct output trees, and B is forbidden
until A has been independently verified. The DOP853 sentinel has one official
locked execution and an independent stored-artifact verification. DOP853 keeps
an immutable checkpoint and parent receipt for every one of its 20 segment
boundaries in every arm; the verifier therefore audits all 140 boundary states,
including the exact 250,000-, 500,000-, and 1,000,000-year endpoints.

This directory is deliberately pre-output until `registration_v1.json` is
created after the contract, runners, verifier, initial-state artifact, and
tests all pass review. No long dynamics may be launched before that
registration exists and both runners accept its exact inventory.

Source-only checks must disable bytecode so the exact package inventory is not
polluted:

```text
python -B build_design.py --contract contract_v1.json --seed-manifest seed_manifest_v1.json --verify initial_states_v1.json
python -B -m unittest -v test_primary.py test_independent.py
```

Use the exact Python and numerical-library build frozen in `contract_v1.json`;
a different shell runtime may skip or reject its runtime-lock test. The
official order after registration is primary A, independent verification of A,
primary B using that A receipt, the one locked DOP853 execution, and final
independent verification. A partial tree never authorizes analysis.

The claim ceiling is
`SYNTHETIC_1MYR_8_ORIENTATION_RESPONSE_WITH_DOP853_SENTINEL_ONLY`. Passing can
describe only this idealized synthetic design. It cannot detect, exclude, or
constrain Planet X; reproduce the authors; establish Solar-System-age
stability; support population or survey inference; or provide evidence or
progress for JX-O2 G0.
