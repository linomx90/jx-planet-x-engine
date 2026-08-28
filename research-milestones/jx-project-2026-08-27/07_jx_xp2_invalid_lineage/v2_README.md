# JX-XP2 synthetic one-million-year robustness screen — protocol v2

This directory is the additive v2 execution protocol for JX-XP2. The stopped
v1 Execution A is permanently invalid because a worker could observe a sibling
segment commit before the coordinator published its PASS, exit, and leave a
FAIL without the permanent receipt promised by the registered contract. The
v1 registration and final invalid-protocol diagnostic are hash-bound here as
read-only lineage; no v1 trajectory, endpoint, or classification is consumed.

V2 reuses the v1 seed manifest, sentinel selection, and initial-state artifact
byte for byte. Thus the 128 tracers, all Cartesian states, scientific matrix,
horizons, gates, and claim ceiling are unchanged. Only the execution protocol
identity and failure/crash bookkeeping change.

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

For MERCURIUS, a child reads an immutable ledger prefix ending at its own exact
START and never audits sibling live commits. The coordinator alone owns global
ledger/commit state. Every terminal FAIL first publishes one deterministic,
fsynced v2 receipt bound to the START, failure class and return code, exact
quarantine inventory, and a stable event digest; the following FAIL row binds
that receipt by filename and SHA-256. Resume reconciles a complete orphan
receipt, a complete pending ledger append, or an uncommitted open START without
inventing scientific output. The verifier requires an exact FAIL/receipt
bijection and rejects omissions, extras, duplicates, or tampering.

The DOP853 coordinator uses the same v2 durability rule. Its deterministic
arm/segment/attempt receipt is published before each
`SEGMENT_ATTEMPT_FAILED` row, and that row binds the receipt filename, hash,
START sequence, event digest, and closed failure class. A receipt-first crash
is reconciled exactly once; partial unpublished bytes become a declared
interrupted attempt, while divergent complete bytes, missing receipts, extras,
or tampering fail closed. Retry exhaustion creates no unpaired fourth receipt.
Both primary and DOP853 attempt ledgers are canonical JSONL published by atomic
whole-file replacement, with exact crash-prefix recovery and torn-tail
rejection.

Both coordinators hold a nonblocking output-tree lock inherited by every child
through final result publication. A surviving child therefore prevents a new
resume coordinator from mutating its tree. This lock and the atomic whole-ledger
publication remove the v1 visibility race and concurrent-resume/torn-tail
failure modes covered by the no-dynamics regression suite.

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
