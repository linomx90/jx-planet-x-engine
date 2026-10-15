# JX versus REBOUND portfolio v1

Status: prepared charter only. No row is preregistered for execution, no
REBOUND 5.2.2 timing has been run, and no new speed claim is authorized.

## Decision

“Beat REBOUND on everything” is defined here as a finite portfolio of identical
workloads that both engines can actually execute. It does **not** mean that a
specialized JX kernel becomes a general replacement for REBOUND, and it does
not permit us to win by changing the force model, output cadence, tolerance,
worker budget, or timing boundary.

The local evidence audit found several credible historical JX wins and several
large losses. None is a current result: every retained direct race used
REBOUND 5.1.1, while the 100,000-tracer science run used 4.4.11. The required
comparator is now REBOUND 5.2.2. Its CPython 3.14 x86-64 wheel, source
distribution, Git hash, imported module, and loaded `librebound` bytes have
been identified in the frozen protocol.

The machine-readable charter is
[`jx_rebound_portfolio_v1_protocol.json`](../benchmarks/jx_rebound_portfolio_v1_protocol.json).
It is deterministically built and checked by
[`jx_rebound_portfolio_v1.py`](../benchmarks/jx_rebound_portfolio_v1.py).
It freezes a finite row map and the common claim rules. It intentionally does
not yet freeze each row's exact inputs, integrator configuration, output
epochs, accuracy gates, artifact runtime, or timing boundary. Those omissions
are explicit blockers, not details that may be chosen after seeing a timing.

## What the audit recovered

Two preserved research workspaces contain byte-identical copies of the selected
later prototype roster. Several key WH and GR hashes also match the hashes
recorded where surviving protocols or reports provide them. This changes the
engineering assessment:
reimplementation from memory is unnecessary, but promotion and fresh
qualification are still required.

Recovered material includes:

- the WH v8 loop, wrapper, preregistration, and 5.1.1 race runner;
- the later dirty research ledger, retained only as historical disclosure;
- the GR15 v6 two-body core, header, wrapper, preregistration, and compact result;
- the stateful native hybrid and endpoint-synchronization prototypes;
- the raw-C TRACE harnesses and the proof-gated circular-binary FAR kernel,
  preregistration, and TRACE/MERCURIUS protocols;
- the historical fused small-N and long-race runners; and
- the ten-year EIH crossover runner and its historical protocol, including its
  explicit model-concordance caveat and five exact loose dependencies. The two
  protocol-bound wrapper versions also survive byte-exact inside the retained
  checksum-bound rc13 source distribution. A tracked rc13 release manifest
  binds that archive; a read-only verifier checks its exact archive and member
  hashes without extracting it into the project. The archive itself is local
  external custody, not a Git payload; the portable charter records its exact
  filename, byte length, digest, and member receipts without a local path.

These are reconstruction inputs, not release evidence. They are selected exact
bytes from dirty research snapshots. The portfolio binds their exact hashes,
and an executable audit checks the same 57-file selected roster in both
preserved snapshots. The two whole workspaces are not asserted to be
identical, and the selected roster itself contains 55 untracked files, one
tracked modification, and one tracked-clean file. No private result may be
called current until the relevant code is imported into a clean successor,
reviewed, built as the final artifact, and rerun against 5.2.2.
The selected roster is still not a dependency closure: the historical WH build
configuration and the general hybrid closure are unavailable or not yet
frozen. Recovering the EIH wrapper source closes that specific source gap; it
does not establish equation concordance with REBOUNDx `gr_full`. The source
archive itself contains no timing result, but two separately retained and
hash-bound external EIH reports do preserve the historical raw samples.

The audit also recovered a selected 34-artifact external roster from the
backup, including 13 direct comparative reports with numeric raw-sample arrays.
The expansion adds the latest six-artifact outcome-blind translational/EIH
chain, a canonical 100-year GR15-versus-IAS15 accuracy screen, and a one-shot
same-equation integration screen. That one-shot screen is **not solver-speed
evidence**: JX used its fused native path while REBOUND evaluated the same JX
EIH equations through an asymmetric Python `additional_forces` callback.
Other selected artifacts retain exact summary statistics, preregistrations,
science gates, or validation-bundle data. A held-file verifier opens every path
component without following symlinks, checks exact length and hash, strict-JSON
parses the artifact, and rehashes the same inode. The portable receipt omits the
local backup path. These are historical inputs and disclosures only: they
compare against REBOUND 5.1.1, not the required 5.2.2, and some private screens
used unequal timing scopes or non-identical force models.

## Historical performance map

All ratios below are disclosures and priors, not current claims.

| Lane | Historical observation | Evidence limitation |
|---|---:|---|
| Short compiled CPU RKF78, 2/11/32 bodies | JX 1.44x / 3.92x / 4.81x | Exact summary-statistics report recovered, but timed sample arrays are absent; REBOUND 5.1.1 only |
| 256-system fused CUDA, 2/11/32 bodies vs 8 workers | JX 2.01x / 3.56x / 3.60x | Exact summary-statistics report recovered, but timed sample arrays are absent; REBOUND 5.1.1; heterogeneous GPU-system-throughput claim only |
| Public WH, 100 periods, 401 outputs | JX single pass 1.20x | Full raw report recovered; mandatory JX replay lost by 1.68x |
| Private WH v8, same retained outputs | JX pilot 1.1985x; later ledger records 1.20965x | Confirmatory raw report is absent; dirty ledger is disclosure, not proof |
| Public GR15, two-body rows | Tie/output-rich or IAS15 win | Multiple raw reports recovered; REBOUND 5.1.1 |
| Private GR15 v6, two-body rows | JX 1.08–1.55x | JX output copy excluded while REBOUND Python readback included; no raw-C equivalence claim; IAS15 usually more accurate |
| General native encounter loop | TRACE 1.41–2.41x faster | Raw reports survive for selected screens; error levels not accuracy matched |
| Proof-gated circular FAR dispatch | JX 1.38x vs TRACE; 1.21x vs MERCURIUS | Only the proved circular two-body FAR domain; full TRACE/MERCURIUS reports remain absent |
| One 100-year 11-body trajectory | IAS15 23.62x faster | Full raw sample arrays recovered; historical pre-current implementation |
| Historical long smooth hierarchy | WHFast 3331.94x faster | Historical pre-native JX path |
| Historical close scatter | TRACE/MERCURIUS thousands of times faster | Historical Python-heavy JX path |
| 11-body EIH batch | JX CPU 2.17–2.30x in the retained screen | Full three-sample rows recovered; REBOUND 5.1.1/REBOUNDx 5.1.0 only; equation concordance with `gr_full` was not established |
| Latest translational/EIH science chain | EIH-only candidate reduced the fitted-DE440 aggregate error by 99.2135% | Outcome-blind chain and fresh starts recovered, but `SCREENING_ONLY`: DE440 is fitted, not independent physical validation, raw LLR validation, REBOUND speed evidence, or `gr_full` concordance |
| GR15 100-year same-model validation | Historical JX/IAS15 accuracy and accounting screen passed | Canonical report has no raw timing array and is not speed evidence; two timing-only rerun/replication variants were reviewed and excluded as superseded duplicates |
| One-shot 100-year EIH integration screen | Same-equation trajectory comparison retained | One timing per side; asymmetric Python callback versus fused JX; explicitly not solver-speed evidence |

A portable, hash-bound inventory now covers 84 uniquely referenced historical
artifact basenames. Seventy-five basenames are present; 74 are retained artifacts
not known to be the wrong bytes. Sixty-nine exact artifacts match prior
cryptographic report bindings. Six prior-bound artifacts remain absent or have
the wrong bytes, and nine references had no prior cryptographic report binder.
Twenty-four observed files contain numeric raw timing-sample arrays; 21 of
those are exact prior-bound artifacts. The full inventory is
[`jx_rebound_portfolio_v1_recovered_report_inventory.json`](../benchmarks/jx_rebound_portfolio_v1_recovered_report_inventory.json).
It omits local absolute paths while retaining hashes, binder identities, and
receipt pointers. The third verification class, retained-source relationship,
is explicitly non-cryptographic: a companion generator naming an output does
not bind the report bytes or prove the producing execution.

This is a finite comparator-evidence inventory, not every JSON file in retained
storage. Reviewed but excluded categories include lunar/LLR and coupled-model
science without a direct REBOUND/IAS15 comparison, effective-force fits,
deformable/mantle/geodetic/public-lunar chains, GPU-coupled lunar screens, and
superseded/intermediate direct-development variants whose conclusion is
preserved by a selected canonical report. In particular,
`jx_current_newtonian_100y_report.json` and
`jx_gr15_de440_long_validation_report_rerun.json` are excluded as canonical
duplicates, not overlooked evidence. The complete P8 archive is bound
separately rather than flattened into basename rows.

The charter also binds an exact selected 34-artifact execution-audit roster for
the most directly useful evidence, plus two comparative raw-C reports already
tracked in the repository. The original `/tmp` pathnames are not custody. The
private WH-v8 and GR15-v6 confirmatory reports, the exact stateful-near reports,
and the two circular-FAR comparator reports remain absent even though their
expected hashes are known. One older stateful-near file with the same basename
is explicitly recorded as wrong bytes, never as a substitute. Recovered raw
samples strengthen reconstruction and negative-result auditing; they do not
substitute for new current-version runs.

## Useful scientific and engineering data

An operator-side inventory observed roughly 74 GiB of retained scientific and
engineering artifacts across the broader workspaces; its full enumerated
inventory is not part of this charter. Most are not timing data. Their correct
uses are:

- the complete 1,345,188-byte P8 100-year eleven-body archive is retained in
  two hash-identical research snapshots. A held-descriptor verifier checks its
  42-member roster, all 28 checksum-ledger rows, exact input/protocol/report
  bytes, and normal/optimized report identity. It supplies frozen initial bytes
  and accuracy/conservation cases, but contains no speed measurement. Its
  configuration followed an exploratory same-state calibration whose transcript
  is absent, so it is a historical confirmation/reproduction screen—not blind
  preregistered or independent physics validation;
- the reduced-five-source H64/H128/IAS15 preflight supplies a strong,
  independently revalidated source trajectory and second-order convergence
  evidence for later tracer work;
- the 16,800-tracer CUDA acceptance supplies exact initial/end states, routes,
  runtime identity, and a 32.16 million tracer-step/s engineering prior, but no
  REBOUND comparison;
- the 100,000-tracer DE441 population run supplies a realistic roster and
  workload shape, but is a REBOUND 4.4.11 science screen rather than a race;
- multi-gigabyte Stage-1/Stage-1.5 accepted-step and source-node files can drive
  replay, checkpoint, and I/O studies, but cannot be relabeled as solver-speed
  evidence;
- writer qualification artifacts measure durable output behavior, not dynamics.

## Prepared finite row map

The charter has ten lane families and 196 explicit rows. Exactly 151 rows are
marked required for the eventual finite claim. Every row still has
`outcome=NOT_RUN`; required-row success is therefore impossible today. The
families are:

1. one-system, short fully mutual Newtonian CPU RKF78 versus IAS15;
2. matched CPU and fused-CUDA small-N batches versus sequential and six-worker IAS15;
3. equal-output, replay, and endpoint-only WH versus WHFast;
4. public GR15 plus the recovered two-body v6 specialization versus IAS15;
5. general FAR/scatter/pericenter hybrid versus TRACE/MERCURIUS, with the
   circular-FAR specialization reported separately;
6. the eleven-body Newtonian+EIH batch versus IAS15+REBOUNDx `gr_full`;
7. mandatory long loss-sentinel workloads that cannot be deleted after results;
8. the frozen 100-year reproducibility/accuracy suite, with no speed claim;
9. a new native shared-active/passive massless-population lane at 16,800 and
   100,000 tracers, separately testing quiet and routed-encounter populations.
10. an unscored checkpoint, restart, and durable-output service comparison
    against REBOUND SimulationArchive. It reports engineering costs only and
    cannot be called solver speed.

For the massless native comparison, both engines must start from identical
active and passive bytes, use matched equations, steps, and output epochs, and
pass gates against a common oracle. Each engine evolves its own active-source
history. Requiring an identical precomputed source history would instead be a
separate frozen-source replay and would not be a native REBOUND comparison.

The Ryzen 5 5500 has AVX2 but no AVX-512. WHFast512 is therefore recorded as a
host capability gap, not silently omitted and not counted as a JX win.

## What counts as a win

Every row must pass its frozen accuracy, conservation, failure-domain, and
determinism gates before timing is interpreted. Inputs, physics, body order,
units, frame, origin, epoch, horizon, and output epochs must be identical.

Numerical-core and end-to-end times are separate. So are single-system latency
and ensemble throughput. Setup, allocation, host/device transfer,
synchronization, readback, and JX replay are disclosed separately. Every
implementation gets exactly five warmups. P1--P5 use 31 paired repetitions,
P6 and P9 use 15, P7 uses 7, and the unscored P8 accuracy lane uses no timing.
The unscored P10 service rows use 15 descriptive paired repetitions. Pair order
alternates AB/BA by index; raw samples cannot be deleted and early stopping is
forbidden.

A resolved JX speed win requires all gates to pass, a median REBOUND/JX time
ratio of at least 1.05, and a lower 95% paired bootstrap bound above 1.0. The
statistic is the median of paired ratios
`R_i = t_REBOUND,i / t_JX,i`, not a ratio of separately computed medians. Raw
samples and negative results are retained. A lane cannot be removed because it
lost. Portfolio success is non-vacuous: all 151 required row outcomes must be
exactly `JX_WIN`; a tie, loss, invalid row, or unavailable capability blocks it.
A terminal report must contain every one of the 196 charter row IDs exactly
once with the original required flags. Missing, duplicate, or unexpected rows
invalidate the whole result.

## Capability gaps

The known list below is intentionally non-exhaustive. Some REBOUND capabilities
do not yet have an honest same-capability JX race:
general collision/merge/bounce/fragmentation handling, tree/FMM active-N,
variational equations, arbitrary ODE/custom-force breadth, and general
regularization. Boundary-condition breadth, MPI execution, MEGNO/chaos tools,
and integrator breadth are additional gaps. These remain explicit gaps. A fast
specialized kernel is not evidence that those gaps are closed.

## Execution order

1. Seal this charter.
2. Import and independently review the recoverable compiled-force, WH v8, and
   GR15 v6 slices in separate successors.
3. Freeze the exact JX wheel/runtime and an outcome-blind per-row or per-lane
   protocol before each timing run. Comparator tuning must be selected by
   accuracy on non-timed calibration cases and then locked.
4. Rerun the current public small-N and WH rows against 5.2.2 first.
5. Promote/rerun WH v8 and GR15 v6 only after parity and failure-domain tests.
6. Package and accuracy-match the general native hybrid before trying to close
   its measured gap.
7. Build the native massless population solver from a separate preregistration.
8. Keep the long loss sentinel in every release comparison.

Until those steps are complete, the exact claim is: JX has promising narrow
historical results and recoverable optimized prototypes; it has not yet beaten
REBOUND 5.2.2 on any frozen portfolio lane, and it does not beat REBOUND as a
general framework.
