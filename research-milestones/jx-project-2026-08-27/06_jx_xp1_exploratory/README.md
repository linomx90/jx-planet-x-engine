# JX-XP1 public synthetic response pilot

> **Publication note (2026-08-27):**
> `xp1_synthetic_response_result_v1.md` is the clarified publication reading
> copy. `xp1_synthetic_response_result_original.md` preserves the immutable
> historical wording for hash/provenance only. Its unqualified
> “preregistered” label means a local pre-output content-hash lock with no
> independent external timestamp.

JX-XP1 is a locally frozen, separate synthetic dynamics pilot. It exists so
we can test our own transparent model without waiting indefinitely for the
private `cluster_2` checkpoint, author angles, seeds, or decks.

The pilot uses an idealized Sun and four circular, coplanar giant planets, 64
new deterministic massless tracers, and one matched control. Six M1 arms add
three preselected low/central/high public design cases at two fixed orientation
probes. All common relative construction states are identical. The only
physical arm difference is the declared extra body.

The run lasts 250,000 synthetic years and measures the change in the sampled
fraction reaching `q < 30 AU`. All seven primary arms have predeclared
half-timestep counterparts. The runner recomputes the complete six-arm mixture,
all four analysis-block effects, and the raw exploratory class at both step
sizes. The primary class is emitted only when all per-arm gates pass and the two
raw classes agree exactly. If any required finite orbit metric is unavailable,
both resolution analyses and all timestep comparisons are explicitly
suppressed and the result is `NUMERICALLY_UNRESOLVED`. No observed object,
survey simulator, author checkpoint, retired candidate, or previous trajectory
is an input.

The 64 tracer states are expanded from the locked seed manifest with a portable
SHA-256/Fisher--Yates construction defined byte-for-byte in the contract. Every
arm contains the same 64 pre-translation Sun-relative construction states. A
deterministic active-body barycentric translation can introduce binary64
roundoff in decoded relative components, so the runner records and gates that
roundoff rather than falsely claiming post-translation bitwise identity.

The resource limits are generous safeguards, not a promised runtime. Every
native dynamics arm runs in its own supervised child process, giving its wall
and memory caps a hard parent-enforced boundary. Nonintegrating setup, analysis,
and finalization use synchronous boundary checks rather than an outer hard-kill
supervisor. Two clean executions plus replay verification have no aggregate
time guarantee.

This is outside JX-O2. It cannot complete G0, reproduce the authors' work,
compare characterized-survey fit, detect or exclude Planet X, constrain an
orbit or mass, or establish formation or four-billion-year stability. Even an
exact A/B replay establishes only the response of this frozen synthetic model.
Both executions reuse the same locked REBOUND build, so the replay is not an
independent software implementation or an independent physical validation.
