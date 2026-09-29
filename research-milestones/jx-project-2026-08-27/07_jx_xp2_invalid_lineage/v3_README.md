# JX-XP2 synthetic one-million-year robustness screen — execution protocol v3

This directory is the additive v3 execution protocol for JX-XP2. It repairs
two distinct predecessor-protocol defects without changing the experiment's
science. V1 Execution A is permanently invalid because a worker could observe
a sibling segment commit before the coordinator published its PASS and could
leave a terminal failure without the permanent receipt promised by its
contract. V2 repaired that race, but then incorrectly included the raw REBOUND
binary archive SHA-256 in the scientific replay chain. REBOUND archives contain
process-specific addresses and inert wall-time/random-seed metadata, so two
archives with identical decoded continuation state and sampled scientific
payload acquired different v2 replay identities.

The v1 registration/final diagnostic and a static, hash-bound v2 defect proof
are protected read-only lineage. The v2 proof consumes only declared diagnostic
hashes, M0 segment-0 receipts/commits/states, and independently reproducible
protocol facts. No v1 or v2 scientific outcome, gate result, label, or
classification is used. The evolving remainder and eventual outcome of the v2
B tree are explicitly excluded and non-authorizing.

V3 reuses the v1 seed manifest, sentinel selection, initial-state artifact, and
design builder byte for byte. The 128 tracers, every initial Cartesian state,
scientific matrix, integrators, timesteps, horizons, samples, gates, resource
caps, and claim ceiling are unchanged. V3 A, v3 B, and v3 DOP853 must each start
fresh from the registered v1 initial-state bytes; no v2 checkpoint, ledger,
result, resume state, or A prerequisite may be promoted.

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
sampling ownership, checkpoint integrity, resume behavior, retry rules,
resource watchdogs, A/B ordering, and fail-closed suppressions are part of the
frozen contract. This package must be content-hash registered before the first
**v3** trajectory is produced. A and B use distinct fresh output trees, and B
is forbidden until v3 A has been independently verified. The DOP853 sentinel
has one official locked execution and an independent stored-artifact
verification. DOP853 keeps
an immutable checkpoint and parent receipt for every one of its 20 segment
boundaries in every arm; the verifier therefore audits all 140 boundary states,
including the exact 250,000-, 500,000-, and 1,000,000-year endpoints.

For MERCURIUS, scientific replay identity is now the canonical decoded
continuation state plus the sampled scientific segment payload. It includes
particles, active MERCURIUS/WHFast/IAS15 continuation settings and retained
arrays, parent-pointer binding, and alignment/non-overlap topology, but never
absolute process addresses, wall clocks, or the unused random seed. Every raw
archive remains mandatory integrity evidence: filename, size, and SHA-256 are
checked at each commit and independently accumulated into an ordered 1,000-item
nonsemantic result inventory. A and B must match their complete semantic chains;
their raw-artifact inventories are deliberately not compared.

A child reads an immutable ledger prefix ending at its own exact START and
never audits sibling live commits. The coordinator alone owns global
ledger/commit state. Every terminal FAIL first publishes one deterministic,
fsynced v3 receipt bound to the START, failure class and return code, exact
quarantine inventory, and a stable event digest; the following FAIL row binds
that receipt by filename and SHA-256. If a scientifically complete attempt is
interrupted before commit, its quarantined raw archive and receipt are
integrity-bound and its canonical decoded/scientific digest is recorded in the
failure receipt and FAIL row. Every retry must reproduce that semantic digest,
although its raw archive bytes may differ. Resume reconciles receipt, ledger,
publication, and per-file quarantine crash cuts without inventing scientific
output. The verifier independently decodes quarantined complete attempts and
requires an exact FAIL/receipt/quarantine bijection.

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
through final result publication. They also acquire and hold the exact v2-B
execution lock, so no evolving v2 process can race a v3 execution. A surviving
child prevents a new resume coordinator from mutating its tree. Atomic
whole-ledger publication removes concurrent-resume and torn-tail ambiguity.

V3 also repairs the operational failure that interrupted v2 A: resource-byte
accounting traverses held directory descriptors with `O_DIRECTORY` and
`O_NOFOLLOW`, tolerates only a child file disappearing during atomic
publication, and rejects a missing/replaced directory, symlink, hard link, or
special file. Parent and verifier checkpoint validation explicitly release
decoded Simulation objects and collect them, bounding repeated decode memory
growth. Neither repair changes a numerical step, sample, or scientific metric.

This directory is deliberately pre-v3-output until `registration_v1.json` is
created after the contract, runners, verifier, evidence, initial-state artifact,
and tests pass independent review. The package contains immutable numerical
bytes only as predecessor-defect evidence; `outcomes_generated: false` in the
future registration applies specifically to v3 outputs. No v3 dynamics may be
launched before registration exists and both runners accept its exact inventory.

Source-only checks must disable bytecode so the exact package inventory is not
polluted:

```text
python -B build_design.py --contract contract_v1.json --seed-manifest seed_manifest_v1.json --verify initial_states_v1.json
python -B -m unittest -v test_primary.py
python -B -m unittest -v test_independent.py
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
