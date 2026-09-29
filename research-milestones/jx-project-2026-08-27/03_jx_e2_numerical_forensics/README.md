# JX-E2 numerical-method forensics

> **Publication note (2026-08-27):** This is a path-neutral public reading
> copy. The exact original README identity is preserved by its SHA-256 in
> `PROVENANCE.md` and the full-tree inventory, but its private ephemeral runtime
> path is not republished. The historical runtime is not bundled here; commands
> below are provenance, not a portable rerun or current execution authority.

`FINAL_REPORT.md` is the clarified publication reading copy.
`FINAL_REPORT_ORIGINAL_HASH_BOUND.md` is the exact report bound by
`closure_v1.json` and is retained only for byte-identity/provenance. Its
unqualified “preregistered” wording means a local pre-output hash lock, not an
independent external timestamp.

JX-E2 is a local, post-failure numerical-method diagnostic. It asks why the
locked JX-E1 50,000-year engineering run exceeded its original momentum gates.
It cannot change that result: JX-E1 remains `ENGINEERING_LONG_INVALID`, and its
execution B remains forbidden.

The design keeps all eight predeclared E1 timestep-audit configurations, not
only the three that exceeded the gates. Each active-body configuration is run
in the original synthetic frame and in an active-body barycentric frame with:

- MERCURIUS at 0.125, 0.0625, and 0.03125 years;
- IAS15 at fixed predeclared tolerances of 1e-10, 1e-12, and 1e-14;
- 50,000 synthetic years, sampled every 10 years;
- exact direct-versus-checkpoint-chained continuation checks;
- two clean executions followed by a separate semantic replay verification.

There are no massless tracers, observed objects, survey files, recovered author
inputs, network calls, or GPU work. The 96 arms isolate the idealized active
system and measure compensated center-of-mass invariants plus Sun-relative
Cartesian state differences. They produce 160 predeclared within-E2 pairwise
comparison records plus 160 separate checkpoint-context comparisons to the
preserved E1 states.

## Interpretation firewall

Execution validity depends only on locked hashes, complete finite output,
setting readback, exact continuation/checkpoint behavior, resource limits, and
clean semantic replay. Numerical magnitudes are classified separately. The old
E1 tolerances are retained only as reference flags and cannot invalidate or
validate JX-E2, rescue E1, rank a model row, or select an angle.

Allowed diagnostic classifications are:

- `FRAME_EFFECT_AND_LINEAR_STEP_SIGNATURE_CONSISTENT`
- `MERCURIUS_REFINEMENT_CONSISTENT`
- `MERCURIUS_REFINEMENT_DIVERGENCE_SUSPECTED`
- `MIXED_OR_INCONCLUSIVE`

None is evidence for or against Planet X.

The classification applies only to states and invariants sampled on the locked
10-year grid. Inter-sample extrema may be missed. The linear-momentum and
center-of-mass angular metrics use new stable scales, so their numerical values
are not called inherited E1 thresholds. The F0 energy flag uses E1's exact
engine-energy formula; the compensated origin-angular flag is only a
mathematically corresponding legacy-value illustration, not an exact replay of
E1's REBOUND angular-momentum evaluator.

The strongest frame/step signature requires both frames to establish the
IAS15 reference, both MERCURIUS refinement sequences to behave consistently,
and separate (non-substitutable) linear-momentum and COM-angular residual
checks above the locked numerical floor. It is still only a pattern consistent
with the registered numerical explanation, not proof of a unique cause.

## Registration and execution

`registration_v1.json` is the local content-hash lock. It binds this README,
the contract, runner, verifier, and tests. The registration has no independent
external timestamp authority; the result records its exact SHA-256.

The runtime lock covers the Python version, REBOUND native-library bytes, and
a deterministic hash of all 29 REBOUND Python source files. The process hashes
those sources before import and disables bytecode-cache use for that import.
The operating system and Python standard library are not content-addressed.

The two pre-registration checks were E2-classification-metric-blind capability
checks; they were not blind to the already known E1 outcome. One
retained only wall time from a 1,000-year active-body run. The other retained
only save/load/continue decoded-state equality booleans over 20 years. No state
values or invariant metrics were retained or inspected.

Use the pinned local Python 3.12 environment. Validation performs no dynamics:

```text
python3 run_numerics.py \
  --contract contract_v1.json \
  --registration registration_v1.json \
  --execution-label E2-A \
  --validate-only
```

Execution A must be provisionally complete before B may begin. The verifier
alone may issue `JX_E2_SEMANTIC_REPLAY_EXACT`. Those labels concern protocol
integrity, not numerical adequacy.

Replay independently checks the exact file/checkpoint inventory, initial-state
reconstruction, decoded checkpoint states, schemas, arithmetic constraints,
classification arithmetic, and A/B semantic equality. Raw 10-year state
samples are intentionally not retained, so sampled maxima are content-bound by
the clean replay but cannot be reconstructed independently from trajectories.
This is not an independent scientific implementation.

## Mandatory nonclaim

This is a post-failure, active-body-only numerical-method diagnostic using an
idealized synthetic benchmark. It does not rerun JX-E1 as a replacement,
alter, validate, or rehabilitate JX-E1; JX-E1 remains
`ENGINEERING_LONG_INVALID` and JX-E1-B remains forbidden. No E2 classification
establishes that E1 was accurate, adequate, or scientifically usable.
Execution completion or semantic replay exactness refers only to protocol
integrity. It uses no observed survey data or recovered author inputs,
does not modify or unblock JX-O2, does not test the existence, detectability,
exclusion, or preferred properties of Planet X, and provides no evidence for
or against Planet X. IAS15 is a numerical comparator within the same REBOUND
build, not ground truth or an independent scientific implementation.
