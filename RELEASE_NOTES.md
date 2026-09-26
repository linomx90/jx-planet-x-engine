# JX General Dynamics 0.6.0rc16

Release date: 24 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc16 turns the retained coupled lunar equations into a supported public engine
component. `integrate_lunar_ephemeris_v1` advances the resolved eleven-body
Solar-System roster, lunar mantle quaternion and angular velocity, and lunar
fluid-core angular velocity through one fixed-step RKF78 checkpoint contract.
Every stage evaluates fully mutual Newtonian gravity, mutual EIH 1PN, reacting
Sun--Moon and Earth--Moon static lunar quadrupoles, reacting fixed-axis Earth
J2, and mantle/core pressure plus viscous coupling. Inputs and outputs are
owned, read-only, provenance-bound, and protected by deterministic ledgers and
content hashes.

The rc16 long-arc decision is governed by the outcome-blind 90/365-day protocol
whose semantic SHA-256 is
`27b1d225d4773b96763407a9acccae5a99bc57a3d7f7b6fbc72545a182ea73b2`.
It uses the untouched day-1200 start, 2,700-second fixed steps, exact endpoint
checkpoints, and a matched no-EIH control. The result is recorded separately
because retained DE440 data and full research evidence are not distribution
inputs.

Every frozen long-arc gate passed. At 365 days, the public candidate's
Earth--Moon position error was 0.061568 km versus 3.729707 km for the no-EIH
control; Sun--Earth error was 0.029261 km versus 56.094548 km. The aggregate
candidate/control RMS ratio was 0.012199 and the worst individual ratio was
0.018686. Exact accounting recorded 11,680 accepted steps and 151,840 force
evaluations, with one binary64 epsilon of maximum quaternion norm error before
projection.

Two independent release-builder invocations, each performing two clean
package-only builds, produced byte-identical artifacts. The CPython 3.14 Linux
wheel SHA-256 is
`f54ac18fb074b25430155dde2bcdd9fdf2d3dc2c96765ba191ed955112fd799f`;
the source archive SHA-256 is
`8f0a9964fca26acf0ba2cc7ab887b63c5544cc45be850fd43bdca8d41722e4b0`.
Fresh installations of both artifacts imported rc16 and exposed the lunar
ephemeris capability as `IMPLEMENTED`. All 17 isolated engine lanes and all 16
reproducible-packaging tests passed. Twine 6.2.0 accepted both metadata
records, the five-lane REBOUND 5.1.1 profile passed, and the 22-test direct
Challenger/REBOUND leapfrog suite passed after bringing its exact current-
source roster forward to include the post-rc15 engine modules.

This component remains NumPy/CPU screening code. It omits delayed tides,
time-variable lunar deformation, lunar degree three and higher, terrestrial
tesserals and higher zonals, minor bodies and planetary satellites, observation
reduction, and parameter fitting. Passing a DE440 screen cannot establish an
independent ephemeris, navigation fitness, raw-LLR validity, or general
superiority over REBOUND.

---

# JX General Dynamics 0.6.0rc15

Release date: 24 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc15 adds `integrate_verified_gr15_eih_1pn`, the first JX propagation boundary
that makes operational state conventions and per-request numerical
confirmation mandatory. Each call declares J2000, TDB seconds past J2000,
system-barycenter origin, kilometre/second units, unique body identities, and
SHA-256 provenance identifiers. A distinct, no-looser `GR15Spec` reruns the
complete trajectory; disagreement at any checkpoint beyond the declared
position or velocity gates fails closed.

The prospective locked portfolio passed forward and backward finite-mass
binaries, an eccentric ten-period binary, and the DE440-derived eleven-body
state propagated for ten Julian years. Every non-timing field repeated exactly
and every primary result was bit-for-bit identical to the pre-existing public
CPU result. On the ten-year case, the primary and confirmation differed by at
most 0.002441 km and 6.913e-9 km/s, within the locked 1 km and 1e-7 km/s gates.

The existing independent REBOUND 5.1.1 IAS15/REBOUNDx 5.1.0 suite also passes,
including the 100-year physical Sun--Earth same-equation comparison. The
10/25/50/100-year DE440 attribution tests pass unchanged. The native GR15 and
EIH force sources are byte-for-byte unchanged from rc14.

Four clean builds across two release-builder invocations produced
byte-identical artifacts. The CPython 3.14 Linux wheel SHA-256 is
`7f00b4636ef07a20b46dc12c96a5ea4971743bc47909e18d4e5580bcbe5401ae`;
the source archive SHA-256 is
`743ea01f32a5c5d320444c3db14ced2b590845557fdaa7262de8033f3336c407`.
Fresh wheel and source-archive installations passed the same verified-
execution smoke digest. Twine 6.2.0 accepted both metadata records, and all
223 isolated live lanes passed, including five CUDA-hardware lanes containing
37 tests with zero skips.

Confirmation deliberately uses the same packaged implementation, so it is a
numerical consistency mechanism rather than an independent reference. Rc15
remains a screening-only mutual point-mass EIH 1PN solver. It does not
authorize navigation, production ephemerides, exact general relativity, DE440
equivalence, or general superiority over REBOUND.

---

# JX General Dynamics 0.6.0rc14

Release date: 23 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc14 promotes the measured CPU/GPU routing work into a supported NumPy batch
component. `integrate_gr15_eih_1pn_cpu_batch` advances independent systems in
native threads; the compiled GR15-EIH1PN call releases the Python GIL for its
complete integration, avoiding subprocess startup and trajectory transport.
Shared and per-system gravitational parameters, forward and backward
integration, exact checkpoint ordering, controller audits, and named
fail-closed system attribution are supported.

`integrate_gr15_eih_1pn_batch` adds a backend-neutral NumPy result. Automatic
routing is intentionally conservative: the bundled profile applies only to
the measured AMD Ryzen 5 5500, NVIDIA GeForce RTX 5060 Ti, eleven-body,
ten-Julian-year workload with up to eight CPU workers. It selects CPU below
128 independent systems and CUDA from 128; values above the largest measured
point are marked as extrapolated. Any unmatched CPU, GPU, duration, body
count, or worker policy stays on CPU. A caller can explicitly force CPU or
CUDA, and an explicit CUDA request never silently falls back.

The locked crossover measured eight-worker JX CPU 2.17--2.30x faster than
eight-worker REBOUND IAS15 plus REBOUNDx `gr_full` at all tested powers of two
from one through 128 systems. CUDA first beat REBOUND at 32 systems and beat
JX CPU at 128 systems. These are one-machine, accuracy-gated engineering
observations, not portable or general performance claims.

The final supported NumPy API was then retimed directly at 64, 96, and 128
systems. CPU won at 64 by 17.5%; CUDA won at 96 by 3.8% and at 128 by 9.9%.
The release keeps the conservative 128-system automatic cutoff rather than
routing on the smaller 96-system margin. Automatic output matched the selected
forced backend bit-for-bit, and every accuracy and agreement gate passed.

Four clean builds across two release-builder invocations produced
byte-identical artifacts. The CPython 3.14 Linux wheel SHA-256 is
`d77400cb7901dde950ba73d9ea3cc5eddaa5a9d31255b3ae92c1b3f00ca6b3f8`;
the source archive SHA-256 is
`032a1185dd683012c2f9950fca1b4b1174c579596048d84dad6bab2258c5eced`.
Fresh artifact installations imported rc14 from their isolated install
directories and passed the same CPU/CUDA smoke digest. The smoke reused the
project's exact CuPy 14.2.0 and CUDA-toolkit dependency set; it was not an
independent dependency installation. Twine 6.2.0 accepted both metadata
records. All 222 isolated live lanes passed, including five CUDA-hardware
lanes containing 37 tests with zero skips.

Rc14 remains a screening-only mutual point-mass EIH 1PN solver. It is not exact
general relativity, a production ephemeris, navigation software, a DE440
replacement, or evidence of general REBOUND superiority. No commit, push,
tag, or publication is authorized by these results.

---

# JX General Dynamics 0.6.0rc13

Release date: 23 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc13 promotes the complete batched CUDA GR15-EIH1PN trajectory path. The new
`integrate_gr15_eih_1pn_cuda_batch` public interface accepts caller-owned CuPy
float64 states for independent 2--32-body systems and shared or system-specific
positive gravitational parameters. One persistent CUDA block per system runs
the predictor, eight-stage corrector, convergence gate, terminal-force rule,
error estimate, transactional rejection, exact checkpoint schedule, and
adaptive step controller. It performs no implicit state transfer.

Trajectory arrays stay on-device. Named status, checkpoint and controller
accounting, numerical metrics, and weak-field diagnostics cross to the host for
a mandatory audit. The CPU GR15-EIH1PN component remains the reference and the
appropriate route for individual small systems.

Eight focused tests pass on the project-owned NVIDIA GeForce RTX 5060 Ti with
CUDA runtime/driver 13.2 and CuPy 14.2.0. They bind CUDA source SHA-256
`2097f0601910739cd6b485e20b32e942f2fca8ae9c69f7265fb1741f94665922`,
require exact binary checkpoint and accounting parity, cover four-system
per-system-GM parity, backward integration, bitwise CUDA replay, the 32-body
boundary, input ownership, and named input, singularity, weak-field,
step-limit, and rejection-limit failures.

In the supported 11-body trajectory benchmark on that host, one system was
about 10.2x slower on CUDA and eight systems remained CPU-faster. CUDA measured
3.46x the sequential CPU throughput at 64 systems and 6.19x at 512 systems.
Across the 512-system case the maximum CPU/CUDA checkpoint differences were
3.56e-15 km and 1.18e-17 km/s; seven systems took a different but valid
floating-point controller path at a convergence boundary. The CPU comparison
is sequential, device inputs are preloaded, and the CUDA timing includes result
construction plus the host audit.

Four clean builds across two release-builder invocations produced
byte-identical artifacts. The CPython 3.14 Linux wheel SHA-256 is
`45abc2fa6d573081c05419a5055ac87fd555996b45646f437bd342096a26ef7a`;
the source archive SHA-256 is
`9285a3ab7136ba76ff41d0e0b001462e626cb6e6499fac7237845a4996a1b555`.
Fresh installations of both passed the same CPU/CUDA public-API smoke digest,
Twine 6.2.0 accepted both metadata records, and all 220 isolated live CPU,
optional-CUDA, and hardware lanes passed.

Rc13 remains a screening-only mutual point-mass EIH 1PN solver. It is not exact
general relativity, a production ephemeris, navigation software, a DE440
replacement, or evidence of portable CPU/GPU or general REBOUND superiority.
No commit, push, tag, or publication is authorized by these results.

---

# JX General Dynamics 0.6.0rc12

Release date: 23 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc12 adds the first supported CUDA component for the exact mutual point-mass
EIH 1PN force equation used by GR15-EIH1PN. The public interface accepts only
caller-owned CuPy float64 device arrays, supports 2--32 bodies and shared or
lane-specific positive gravitational parameters, performs no implicit input
transfer, and returns device-resident total Newtonian-plus-EIH acceleration.
Every call audits named per-lane status and weak-field diagnostics on the host.

The companion `evaluate_gr15_eih_1pn_stages_cuda` interface evaluates all
eight simultaneous force states from one GR15 corrector iteration in one CUDA
kernel launch. It supports multiple independent systems and system-specific
masses. It does not perform prediction, corrector state updates, convergence,
error estimation, rejection, checkpointing, or adaptive step control; the
complete supported GR15 trajectory remains CPU-only.

Nine acceptance tests pass on the project-owned NVIDIA GeForce RTX 5060 Ti
with CUDA runtime/driver 13.2 and CuPy 14.2.0. They bind CUDA source SHA-256
`e5177ef1faf2dfaa17dfd87ae70353eeeb16974eda8a0ceee1e2f2035d0cf4bc`,
compare against both the native C core and independent Python equation, cover
lane-specific masses and the 32-body boundary, require bitwise repeated-launch
determinism, and verify four named failure domains.

The finite-host crossover screen includes the public CUDA call's fail-closed
host audit but excludes transfers and allocation. At 4,096 eleven-body lanes,
CUDA measured 5.28x the throughput of the current native CPU wrapper; at 4,096
thirty-two-body lanes it measured 19.74x. CUDA was slower for one eight-stage
eleven-body system. This is component-level engineering evidence from one
machine, not a complete-GR15, portable, production, or general superiority
claim.

Four clean builds across two independent release-builder invocations produced
byte-identical artifacts. The CPython 3.14 Linux wheel SHA-256 is
`0ae73b214cc11a2adadd405d7b359a1512ff4213fd937a238988ba681f52afe3`;
the source archive SHA-256 is
`7abc3bb3c301267a2eb8609da5529fe14e6b7a56184407fbd79afcf3d568b7a0`.
Fresh installations of both artifacts passed the same native-CPU/CUDA public
API smoke digest. The 219-lane live matrix, two-lane external EIH profile, and
three-lane/28-test hardware profile all pass. Twine 6.2.0 accepts both artifact
metadata records without warnings.

Rc12 remains screening-only EIH 1PN. It is not exact general relativity, a
production ephemeris, navigation software, a DE440 replacement, or evidence of
general superiority over REBOUND. No commit, push, tag, or publication is
authorized by these results.

---

# JX General Dynamics 0.6.0rc11

Release date: 23 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc11 adds the first packaged relativistic GR15 component:
`integrate_gr15_eih_1pn`. It advances 2--32 fully mutual finite-mass point
bodies with Newtonian gravity plus the Einstein--Infeld--Hoffmann first
post-Newtonian correction, reevaluated at every GR15 node and corrector
iteration. The rc10 Newtonian GR15 V3 source remains byte-for-byte unchanged.

The locked scientific matrix passes equation-level native/Python parity,
Newtonian-limit identity, analytic periapsis advance, weak-field failures,
deterministic replay, permutation, convergence, reversal, REBOUNDx `gr_full`
asymptotic comparison, and same-model REBOUND IAS15 parity. A physical
Sun--Earth 100-year comparison differed from IAS15 by about 31 metres and
6.35e-6 m/s at the endpoint, inside the locked 10 km and 0.01 m/s limits.

Against DE440s at 10, 25, 50, and 100 Julian years, the EIH model's aggregate
heliocentric position residual was 0.0882, 0.1282, 0.0987, and 0.1545 times
the Newtonian control. At 100 years the ratio of worst-body residuals was
0.2151, below the locked 0.25 ceiling. This is attribution evidence, not a
claim that the reduced model reproduces DE440.

GR15-EIH1PN is CPU-only and omits extended-body figures, tides, spins, frame
dragging, collisions, signal propagation, and observation reduction. Rc11 is
not exact general relativity, a production ephemeris, a navigation product,
or a general REBOUND superiority claim.

Two clean builds produced byte-identical local artifacts. The CPython 3.14
Linux wheel SHA-256 is
`7ee1a6ba36d448c71f28238f7562e2889c104b93a90b5a6a5abb4d388a7d66e9`;
the source archive SHA-256 is
`d5ec3e7811291c68df9bf2814965eddf99bf180f6c7322df62ee16f79a2f5b0b`.
Fresh installations from both artifacts passed the public GR15-EIH1PN API
smoke. These are local candidate artifacts and have not been published.

## Independent ownership and license

This release retains the proprietary, all-rights-reserved license held by
Lino Avila. Earlier public releases and third-party materials retain their
original notices. AI systems are development tools, not authors or owners.

---

# JX General Dynamics 0.6.0rc10

Release date: 23 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc10 carries the unchanged GR15 V3 numerical core from the unpublished rc9
candidate and fixes a release-engineering defect found during clean-checkout
reproduction. The builder now canonicalizes every staged directory and file
mode before invoking the pinned backend, so wheel ZIP metadata no longer
depends on the source checkout's group-write policy. Two builds in the active
tree and a third build from a clean GitHub-main overlay produced byte-identical
wheel and source archives.

The exact installed rc10 wheel passed the fixed REBOUND 5.1.1 IAS15 matrix and
the complete five-case adversarial portfolio twice on the desktop, with every
non-timing field identical. On that host, the output-rich row remained a tie
inside the declared 5% timing resolution: JX throughput was 1.008x and 1.015x
IAS15. IAS15 remained about 1.19x faster endpoint-only and 1.38--1.39x faster
on the eccentric row. No general speed claim is authorized.

The same exact wheel then passed both correctness portfolios on the RTX 4050
laptop. Laptop timing did not reproduce the desktop tie: IAS15 was 1.152x,
1.377x, and 1.623x faster on the output-rich, endpoint-only, and eccentric
rows. This is useful evidence that the performance result is host-dependent.
The separately scoped five-test CUDA core gate passed with zero skips on both
the RTX 5060 Ti and RTX 4050 Laptop GPU. Common ownership means this is
two-machine reproducibility, not organizationally independent validation or
general GPU qualification.

The exact local Linux wheel remains the acceptance artifact. A separately
hashed `manylinux_2_31_x86_64` wheel was repaired twice with Auditwheel 6.8.2
and Patchelf 0.19.1, reproduced byte-for-byte, passed strict Twine validation,
and passed a fresh native-extension import smoke. It is the PyPI artifact;
both wheel identities and their relationship are recorded explicitly.

GR15 remains limited to 2--32 positive-GM, fully mutual, unsoftened Newtonian
point masses in binary64. Rc10 is not a production ephemeris, a lunar-model
promotion, a scientific qualification, or a general REBOUND superiority
claim. General Dynamics registry v18 remains immutable historical evidence
bound to 0.6.0rc3.

## Independent ownership and license

This release retains the proprietary, all-rights-reserved license held by
Lino Avila. Earlier public releases and third-party materials retain their
original notices. AI systems are development tools, not authors or owners.

---

# JX General Dynamics 0.6.0rc9 (unpublished)

Release date: 23 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc9 promotes the package-owned GR15 V3 CPU component for 2--32 fully mutual,
positive-GM, unsoftened Newtonian point masses. V3 uses a tolerance-aware
corrector threshold and residual-gated terminal-force reuse; the exact V2 path
remains packaged as a private fallback/reference. On three preregistered
holdouts, V3 reduced force evaluations by 24.6%--26.7% and ran 1.234x--1.292x
faster than V2 without increasing rejections or exceeding the locked accuracy,
conservation, reverse-time, residual, or determinism gates.

The exact rc9 wheel passed the direct REBOUND 5.1.1 IAS15 matrix. The
101-output circular row remains a timing tie inside the predeclared 5% band
(1.020x JX throughput in the recorded run and 1.015x on immediate repeat).
IAS15 remained 1.194x and 1.386x faster on the endpoint-only and eccentric
rows; the repeat measured 1.186x and 1.390x. This is not a general solver
ranking.

The sealed correctness portfolio also passed from the exact installed wheel:
an eccentricity-0.99 binary for 10 periods, a 1e12:1 mass-ratio binary for 25
periods, a three-body close scatter, an 11-body hierarchy for 20 inner periods,
and the supported 32-body boundary. Every case passed deterministic replay,
strict IAS15 primary/sensitivity agreement, conservation, forward/backward,
corrector, terminal-force accounting, and retained-separation gates. The
binary cases also passed closed-form Kepler gates. This remains finite,
synthetic, one-host screening evidence.

The wheel and source archive were each reproduced byte-for-byte in two clean
builds. A fresh wheel installation verified all 125 hashed files against the
wheel RECORD before qualification; a fresh source-archive installation passed
the public V3 smoke. Normal and optimized engine matrices, all 191 current-
source lanes, the exact REBOUND profile, and the five-test CUDA core gate pass.
The CUDA gate ran on the NVIDIA GeForce RTX 5060 Ti with CuPy 14.2.0 and CUDA
runtime/driver 13.2.

GR15 remains `SCREENING_ONLY`. It has no arbitrary force callbacks, massless
particles, collision response, event location, regularization, relativity,
dense output, GPU implementation, or production-ephemeris authorization.
General Dynamics registry v18 remains immutable historical evidence bound to
0.6.0rc3; rc9 creates no new registry generation or lunar-model promotion.

## Independent ownership and license

This release retains the proprietary, all-rights-reserved license held by
Lino Avila. Earlier public releases and third-party materials retain their
original notices. AI systems are development tools, not authors or owners.

---

# JX General Dynamics 0.6.0rc8

Release date: 22 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc8 is the evidence-integrity and native-hybrid research candidate. It keeps
the supported rc7 numerical APIs unchanged while adding a deterministic
catalog over the complete local experiment collection, explicit links to
archive copies, lifecycle and family groupings, and a metadata-hygiene queue.
The immutable library authority catalog remains controlling; active workspace
records are never silently promoted.

The preregistered stateful-NEAR RKF78 candidate passed its locked correctness,
classification, work-reduction, accuracy-envelope, and timing gates. It cut
attempted adaptive substeps from 192 to 66 on close scatter and from 354 to 264
at pericenter, producing 1.524x and 1.246x internal speedups. The separate
endpoint-synchronization candidate preserved exact output but missed its
minimum speed threshold and remains a negative result.

The equal-output raw-C screen still measured JX slower than TRACE by 2.158x,
1.415x, and 2.412x on the all-far, close-scatter, and pericenter fixtures. TRACE
retained its own tolerance defaults, the cases are short and synthetic, and
the result is therefore not an accuracy-matched or general solver ranking.
Stateful NEAR remains a source-bound development prototype rather than a
supported installed API.

The exact rc8 wheel separately passed the locked supported-API 100-period
WHFast race. Its single-pass end-to-end median was 9.128 ms versus 10.952 ms
for REBOUND WHFast, producing 1.200x JX throughput on that recorded host and
weak-hierarchy workload. The exact replay lane took 18.440 ms, so REBOUND was
1.684x faster than that lane. Both solvers passed the same preexisting accuracy
and conservation envelope; this result remains single-host, workload-specific,
and `SCREENING_ONLY`.

The release gate now assigns every one of the 258 test modules to exactly one
compatible runtime profile. The 13-lane engine matrix passes 233 tests, and
the new local performance and library-catalog profiles pass their source-bound
checks. General Dynamics registry v18 remains immutable evidence bound to
`0.6.0rc3`; no capability, lunar status, production status, or scientific
claim is promoted.

## Independent ownership and license

This release retains the proprietary, all-rights-reserved license held by
Lino Avila. Earlier public releases and third-party materials retain their
original notices. AI systems are development tools, not authors or owners.

---

# JX General Dynamics 0.6.0rc7

Release date: 22 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc7 advances the supported `jxplanetx.fast_wisdom_holman` CPU interface to the
checkpoint-synchronized v3 native map. The final interaction kick does not
force an unnecessary full coordinate transform: complete position/velocity
transforms occur at retained checkpoints, while the exact position transform
needed by every guard is preserved. The default route executes once and binds
all native-step postconditions, guards, accounting, input/output custody, and
result content in a deterministic certificate. Exact complete replay remains
available by request and is mandatory in release qualification.

The package also contains an implementation-private endpoint-seeded Kepler v4
prototype with observable exact-v2 fallback accounting. Its locked 100-period
development race used zero fallbacks, reduced universal-G evaluations by
24.49%, and was 5.23% faster than v3 in the recorded numerical scope. Against
equal-output tuned REBOUND 5.1.1 WHFast it was 1.99% slower, which is below the
predeclared five-percent timing resolution and is classified as an unresolved
tie. Endpoint-only WHFast remained 4.42x faster but did not return the same
checkpoint cadence. These are one-host, weak-hierarchy development results.
V4 remains private and is not the supported rc7 backend.

The final rc7 wheel and source archive were each reproduced byte-for-byte in
two independent temporary build trees and passed fresh-install supported-API
smokes. The exact wheel then passed public-JX bitwise parity, exact replay,
deterministic-output, universal-G, IAS15-envelope, and conservation gates. On
the locked host/workload its supported single-pass end-to-end median was
9.142 ms versus 10.037 ms for WHFast (1.098x JX throughput); the release-gate
replay lane was 18.222 ms (1.815x REBOUND throughput). This classification is
limited to the exact 100-period weak-hierarchy protocol and does not transfer
to other workloads or machines.

The current-source CUDA core gate also passed all five non-skippable parity,
dtype-rejection, transfer-rejection, tolerance-custody, and device-resident
trajectory tests with zero skips on the NVIDIA GeForce RTX 5060 Ti using CuPy
14.2.0 and CUDA runtime/driver 13.2. This validates the rc7 core on that device;
it does not rerun the earlier two-machine scaling matrix or establish general
GPU qualification.

No portable solver ranking, general REBOUND superiority, production fitness,
or scientific qualification is claimed. The exact public map is restricted to
its guarded 2--32-body hierarchical Newtonian scope. General Dynamics registry
v18 remains immutable evidence historically bound to `0.6.0rc3`; no capability
or lunar status is promoted. Frozen runs, archives, historical evidence, rc6
artifacts, and all earlier manifests are unchanged.

## Independent ownership and license

This release retains the proprietary, all-rights-reserved license held by
Lino Avila. Earlier public releases and third-party materials retain their
original notices. AI systems are development tools, not authors or owners.

---

# JX General Dynamics 0.6.0rc6

Release date: 22 September 2026

Release stage: release candidate

Scientific claim state: `SCREENING_ONLY`

Rc6 adds the opt-in supported `jxplanetx.fast_wisdom_holman` CPU interface.
It moves the complete guarded fixed-step map into one compiled call, uses a
binary64 no-change termination rule for the universal-G series, and preserves
the fixed 64-term rc5 evaluator as the exact fallback. Input and output custody,
failure-domain checks, deterministic replay, and non-production claim controls
remain explicit.

The preregistered 100-period weak-hierarchy race passed every accuracy,
conservation, exact-replay, failure-domain, and public-JX bitwise gate. On the
recorded host, the single-pass v2 lane had 9.358 ms median end-to-end latency
versus 11.013 ms for REBOUND 5.1.1 WHFast, while the default replay lane took
18.748 ms. These are single-host, single-workload screening measurements—not a
portable ranking, scientific qualification, or general victory over REBOUND's
solver portfolio. The rc5 artifacts and manifest remain immutable.

## Independent ownership and license

This release retains the proprietary, all-rights-reserved license held by
Lino Avila. No permission is granted to
copy, modify, publish, distribute, sublicense, sell, commercially exploit, or
create derivative works without prior written authorization. `AUTHORS.md`
records the human author; AI systems are development tools rather than authors
or owners. Earlier public releases and third-party materials retain the
licenses and notices that accompanied them.

This candidate carries the immutable General Dynamics registry v18 evidence
historically bound to `0.6.0rc3`; it does not rewrite that record or create a
new registry generation. The registry remains a capability-and-evidence map, not
a declaration that every listed domain shares one solver or is scientifically
qualified. It explicitly records `production_ready: false` and
`unified_multiphysics_execution: false`.

The CPU engine and optimized-Python engine matrices pass on the preparation
host. NumPy RKF78 results retain disjoint, read-only owned arrays with a full
content digest. CuPy results retain disjoint owning device arrays but remain
caller-mutable and deliberately have no implicit host content hash.

The current core was revalidated on an NVIDIA GeForce RTX 5060 Ti using driver
595.91.07, CUDA 13.2, NumPy 2.3.5, and CuPy 14.2.0. The CUDA-enabled 13-module
engine matrix passed 233 tests with zero skips, and all five GPU-specific
backend/trajectory tests passed again under optimized Python. This is exact-
runtime/device engineering validation only, not cross-device qualification,
performance evidence, production fitness, or scientific validation.

Release packaging now uses an exact setuptools backend, a declared
package-only staging boundary, and canonical source archives. The release tool
builds twice in independent temporary trees and refuses output unless both the
wheel and source archive are byte-identical. It also records exact hashes and
sizes without modifying research evidence.

The candidate compiles the audited 2--32-body point-mass RKF78 C core into an
implementation-private CPython extension with strict floating-point compiler
options. The supported `jxplanetx.smalln` surface routes measured low-lane
NumPy workloads to that core and measured batched CuPy workloads to the fused
CUDA controller without implicit transfers. Native wheels are ABI/platform
specific, and source installs require a C11 toolchain. This engineering API is
still `SCREENING_ONLY`; it is not registered as a scientifically qualified or
production backend.

Rc5 reduces public CPU call overhead without changing the native numerical
core: immutable RKF78 tableau arrays, checkpoint epochs, and read-only
checkpoint views are cached in frozen workspaces and validated before native
execution. The installed-wheel accuracy gate passed with byte-identical state
and ledger behavior. On the recorded RTX 5060 Ti desktop, the public CPU path
had lower individual-system latency than REBOUND IAS15 for the sampled 2-,
11-, and 32-body synthetic workloads. The revised discrete crossover keeps CPU
through eight lanes for bodies 2--31, preserves the nearly tied measured CUDA
choice for 32 bodies at eight lanes, and selects CUDA at 16 or more lanes. A
101-repetition boundary audit is authoritative for the noise-sensitive
28--32-body/eight-lane points and is retained alongside the broader
five-repetition sweep. This is one-host engineering evidence and does not
establish portable routing or general JX superiority.

The exact final rc5 wheel was rerun against eight parallel REBOUND 5.1.1 IAS15
workers on the recorded RTX 5060 Ti desktop. Every 256-system row passed the
shared `1e-10` checkpoint-error gate. JX throughput was 2.01x, 3.56x, and
3.60x REBOUND's eight-worker integration critical path for 2, 11, and 32
bodies. Sequential REBOUND IAS15 still had lower one-lane latency than fused
CUDA. These are short synthetic, one-machine batch results, not a general speed
claim.

For historical cross-hardware context, the corrected rc4 deterministic kit
passed on the separate RTX 4050 Laptop GPU, and the fail-closed two-machine
aggregate passed. On that laptop, JX throughput relative to eight-worker
REBOUND's integration critical path was 1.02x, 1.56x, and 1.27x for 2, 11,
and 32 bodies. The initial kit
correctly failed before timing because one benchmark import was absent; the
rebuilt byte-identical kit includes and hashes the complete import closure.
Both hosts share common ownership, so this remains a finite hardware screen,
not an independent replication or portable ranking. The laptop and aggregate
were not rerun for rc5.

The exact rc5 wheel also passed the registered 10-, 30-, and 100-year
eleven-body mutual-Newtonian point-mass comparison with REBOUND IAS15. At 100
years, the maximum body-position difference was 1.911314 m and the resolved
Earth--Moon relative-position difference was 0.476182 m. This is an accuracy
screen only: the recorded timings are non-comparable, and the model omits the
coupled lunar mantle/core physics required for an ephemeris or lunar-physics
claim.

The rc5 Challenger v2 full portfolio passed its self-validating Leapfrog,
WHFast/IAS15, and MERCURIUS/TRACE/IAS15 workloads, including the registered
limitation witnesses. Its wall times are diagnostic only; it authorizes no
cross-study solver ranking or performance-superiority claim.

A separate prospectively locked three-repetition timing protocol now makes the
long single-trajectory scopes comparable. Every candidate passed its common
workload-specific accuracy envelope. On the recorded desktop, REBOUND IAS15
with an AMD Ryzen 5 5500 and RTX 5060 Ti, REBOUND IAS15 was 23.62x faster than
fused CUDA RKF78 on the 100-year eleven-body point-mass
arc; WHFast was 3,331.94x faster than JX Wisdom--Holman on the 100-period smooth
hierarchy; and MERCURIUS and TRACE were 2,650.18x and 18,819.64x faster than the
JX hybrid on the close-scatter workload. JX timings include the mandatory
public semantic replay. These results show a large current implementation gap
for long single trajectories; they do not negate the distinct 256-system
short-batch CUDA result or establish portable algorithmic rankings.

The initial timing protocol stopped before any measured lane because of a
REBOUND provenance field-name check and produced no report. Its bytes remain
unchanged. A hash-bound v2 overlay changes only that preflight lookup and binds
the successful report. Existing screening outcomes were known before this
protocol and are disclosed; only the new timing sample was prospective.

Additive experimental Moon boundaries now advance translation, lunar mantle
attitude/rate, and lunar fluid-core rate in one RKF78 state. The v1 isolated
Earth--Moon model established common-potential orbit/spin reaction and a
converged numerical chassis. The v2 model simultaneously integrates the Sun,
Earth, and Moon; adds Sun and Earth static-lunar-figure force/torque; and adds a
reaction-balanced Earth-J2 orbital correction around a held J2000 pole. Its
one-day Earth--Moon difference from retained DE440 is 1.251 m, down from v1's
95.99 km, and the Earth-J2 ablation is 5.170 m. This remains unregistered
`SCREENING_ONLY` research: it is not a complete DE440 model, raw LLR
validation, independent physical validation, or a lunar capability promotion.

Two additional opt-in lunar research components now implement the complete
delayed tidal-plus-spin mantle-deformation balance and source-fixed geodetic
transport in the mantle/core relative-frame terms. Their frozen 90/365-day
screens completed 784,368 RKF78 stages each. Neither passed its registered
promotion rule: deformation improved all four metrics but only one cleared the
one-percent threshold, while geodetic transport improved none. Registry v18
therefore keeps the lunar capability `INCONCLUSIVE`; the next clean gate is a
separately preregistered simultaneous translation/rotation model containing
the complete deformation package and common-potential orbital reactions.

No frozen scientific threshold or registry capability was promoted. Frozen
runs, archives, registry v18, and all earlier records remain unchanged.

---

# JX Celestial Dynamics Framework 0.4.0a1

Release date: 6 September 2026
Release stage: alpha
Scientific claim state: `SCREENING_ONLY`

## General-purpose engine surface

- Reframed JX as a general celestial-dynamics framework while retaining the
  `jxplanetx` package name, existing command, and Planet X workflows.
- Added the public `jxplanetx.engine` API with immutable state, backend,
  parameter, force-plan, contribution, checkpoint, and result contracts.
- Implemented direct unsoftened Newtonian point-mass gravity, restricted
  static-central Schwarzschild test-particle 1PN correction, and unshadowed
  isotropic cannonball solar-radiation pressure.
- Implemented an explicit 13-stage adaptive Fehlberg RK7(8) trajectory path
  with a hatted order-eight accepted solution, an ordinary order-seven defect,
  normalized maximum per-component error control, exact checkpoint clipping,
  forward/backward propagation, and velocity-dependent-force compatibility.
- Implemented a standalone full-Cartesian adaptive RKF78 encounter segment for
  NumPy/CPU binary64, all-active positive-GM mutual unsoftened Newtonian
  states. It uses authoritative signed local-duration accounting, independent
  endpoint provenance labels, caller-supplied pair floors, exact simultaneous
  local-IVP clearance certificates, pair-relative and GM-centroid defect
  control, Kahan accepted updates, bounded witness/checksum custody, and one
  separately accounted mandatory semantic replay.
- Implemented an experimental, unqualified ordered-Jacobi second-order
  Wisdom--Holman KDK map for NumPy/CPU, fully mutual active positive-GM
  Newtonian hierarchies. It uses fixed integer map nodes, elliptic
  universal-variable Kepler subflows, strict hierarchy/periapse/interaction/
  step/Hill/path guards, and one mandatory deterministic semantic replay.
- Implemented the specific, unqualified whole-system
  `integrator.hybrid.wisdom_holman_jacobi_rkf78_cartesian.v1` method. On every
  fixed outer interval it transactionally commits a complete typed far probe
  or discards it and redoes the untouched original full interval through the
  private guarded Cartesian encounter execution. One outer replay recomputes
  all mode decisions, work, states, private-child digests, and accounting;
  public child wrappers and nested child replay are not used.
- Added an ordered 46-capability catalog spanning physics, integrators,
  backends, precision, determinism, sensitivity, orbit determination, and
  measurements: exactly twelve rows are `IMPLEMENTED` and 34 are `DECLARED`.
  The implemented set includes the standalone guarded encounter segment and
  the experimental, unqualified Cartesian KDK and ordered-Jacobi
  Wisdom--Holman maps plus the specific whole-step hybrid; broad event-driven
  encounter switching, generic hybrid integration, regularization, and the
  generic split stay declared.
  Every row is `UNQUALIFIED`; declared rows validate and then refuse execution.
- Kept all outputs at `MODEL_OUTPUT` with registry and qualification authority
  fixed false. Units, frame, origin, epoch, parameter validity, provenance,
  float64 dtype, backend/device identity, and summation grouping are explicit.

## Backend and packaging boundary

- Added a NumPy CPU extra, `engine`, constrained to NumPy 2.x.
- Added mutually exclusive `gpu-cuda12` and `gpu-cuda13` extras using the
  corresponding CuPy 14 `[ctk]` wheel bundles; a compatible driver remains
  required.
- Backend selection is explicit. The API performs no implicit host/device
  transfer and never falls back from a requested GPU to CPU.
- The CuPy code path is present but has not been run on project-owned GPU
  hardware or independently qualified. Version 0.4.0a1 makes no GPU
  correctness, cross-device reproducibility, performance, or speedup claim.
- The RKF78 path is unqualified and deliberately narrow: float64 only, no dense
  output, event location, collision response, close-encounter switching, or
  regularization. All other cataloged integrator families remain declared and
  fail closed, including the generic split, implicit velocity-dependent, and
  hybrid close-encounter rows.
- The standalone encounter screen makes only a retained-floor noncollision
  claim for the exact Newtonian local IVP issuing from each accepted numerical
  node on the certified substep. Failure is uncertified rather than collision
  evidence. It does not locate events or minima, respond to collisions,
  regularize, switch integrators, certify trajectory accuracy, or establish
  global/future physical-trajectory clearance. Exact resources and actual
  force calls are capped and reported separately for primary execution and
  the mandatory replay, plus their public totals.
- The Wisdom--Holman path is CPU/all-active/elliptic/non-encounter only. Its
  symplectic and reversible statements concern formal exact subflows, not
  floating-point execution. Sampled energy behavior is not phase accuracy or
  a general long-term stability claim; registry, qualification, production,
  and superiority authority remain false. The implementation and tests were
  independently written from published equations; no external REBOUND source
  was copied or vendored.
- The specific hybrid is NumPy/CPU binary64, all-active positive-GM mutual
  Newtonian, central-first, and limited to 16 bodies on a fixed signed outer
  lattice. A typed near choice discards the full provisional far candidate and
  redoes the original whole interval. There is no hysteresis, latch, partial
  prefix, grouping, dense output, event location, collision response, or
  regularization. Its exact encounter screen proves only accepted-node local-
  IVP retained-floor noncollision, not global trajectory clearance. The
  complete method is not globally symplectic, formally or exactly reversible,
  globally order-qualified, superior, production-ready, or qualified. Hard
  logical-record caps do not replace external process, timeout, memory, and
  concurrency isolation.

## Step 5 Solar-System preparation

- Added an opt-in, unpublished preparation boundary from isolated
  CSPICE/DE440s geometric-state evidence and retained DE440 GM parameters to an
  exact resolved-Earth/Moon eleven-body Newtonian engine input.
- The boundary performs explicit binary64-to-SI projection, operational-GM
  model-barycenter recentering, primary/replay semantic checks, and construction
  of fresh owned read-only NumPy arrays, an engine snapshot, and one direct
  mutual unsoftened Newtonian force plan.
- This is initial-input preparation only. It runs no trajectory and does not
  claim that the reduced Newtonian model is DE440, authenticate or authorize
  source artifacts, establish continuous custody, qualify the roster for
  Wisdom--Holman, or authorize scientific or production use. Every resulting
  engine object remains unqualified `MODEL_OUTPUT`.

## Challenger V1 and post-V5 precision evidence

- Added Challenger V1, a subprocess orchestrator that retains the native
  reports for locked analytic-binary, weak-hierarchy, and close-scatter smoke
  fixtures. Optional comparison lanes require exact REBOUND 5.1.1; the bundle
  defines no cross-study score, ranking, or engine-equivalence rule.
- Added an equal-mass circular-binary precision probe at fixed steps P/256,
  P/512, and P/1024. Its smoke and full profiles cover one and 100 periods,
  respectively, and record analytic-oracle state/phase errors, sampled
  invariants, complete lane replays, and fixture-specific refinement gates for
  JX KDK and optional exact REBOUND 5.1.1 leapfrog.
- These results are narrow numerical regression evidence. They do not establish
  general or theoretical convergence, continuous-time energy bounds,
  equivalence, REBOUND accuracy or speed superiority, timing comparability,
  long-term stability, scientific qualification, or production fitness. All
  reports remain unauthenticated, unqualified `MODEL_OUTPUT`.

## Compatibility and preserved records

The legacy Decimal/Newtonian core, Yoshida and reference integrators, CLI,
previously frozen V4/V5 assets, and prior scientific result sections are
unchanged. The Step 5, Challenger V1, and post-V5 precision assets are
additive. The 0.3.0 release manifest and checksums remain historical records
and were not rewritten for this alpha.

---

# JX N-Body Engine 0.3.0

Release date: 22 August 2026  
Claim state: `SCREENING_ONLY`

## Independent population replication

- Added an independent Newtonian force and SciPy DOP853 population runner that
  does not import or call REBOUND.
- Added independent Kepler element conversion and recovery, annual
  classification, segmented exact-binary64 checkpoint/restart, source
  stability audits, population comparisons, deterministic paired bootstrap,
  and fail-closed `PASSED`, `CONFLICT`, or `INVALID` verdicts.
- Locked Python, NumPy, SciPy, solver-source, coefficient-table, binary,
  initial-state, population, selection, reference-result, and runner hashes.
- Added an `independent` installation extra pinned to NumPy 2.3.5 and SciPy
  1.17.0.
- Expanded the test suite from 70 to 76 tests.

## Independent scientific record

The release includes a compact record of an independent replication of ten
outcome-blind hash-selected 1,000-tracer blocks from the DE441-backed
100,000-tracer experiment.

The original independent attempt is preserved as `INVALID`. It exceeded only
the active endpoint-position consistency gate: 1.27532×10⁻⁶ AU observed
against 1×10⁻⁶ AU locked. All population comparisons passed. A separate locked
diagnostic confirmed adaptive-resolution dependence; no v1 gate was relaxed or
retroactively changed.

A corrective v2 was registered with the same population, physical model,
statistics, and acceptance thresholds. Only DOP853 resolution changed:

- relative tolerance: 1×10⁻¹³;
- absolute tolerance: 1×10⁻¹⁵;
- maximum step: 0.125 year.

V2 returned `PASSED`:

- 10,000 tracers per arm over 10,000 years;
- 433/10,000 sampled injections in both independent arms;
- the same 433 identities in each corresponding REBOUND arm;
- zero injection-identity disagreement;
- 100% final survival in both arms;
- source-minus-control injection fraction 0.0;
- paired-block 95% bootstrap interval `[0.0, 0.0]`;
- maximum active energy drift 8.35832×10⁻¹³;
- maximum active angular-momentum-vector drift 3.02089×10⁻¹³; and
- maximum active endpoint-position disagreement 2.86934×10⁻⁹ AU.

An independent stored-artifact audit returned `AUDIT_PASSED`. It verified 19
locked files, 20 summaries, 20 independent tracer tables, 20 REBOUND reference
tables, and 100 checkpoint state pairs; reconstructed final orbital elements;
and recomputed every statistic, gate, and verdict.

## Scientific boundary

This result strengthens the numerical robustness of one candidate-9118 screen,
but remains `SCREENING_ONLY`. It does not detect or exclude Planet X, validate
candidate 9118, cover the wider candidate space, or substitute for an observed
TNO population and survey selection function. The v2 solver was selected after
the v1 numerical failure and is transparently labeled as a corrective run, not
the original preregistration.

The next scientific gate is an observed-population model plus an explicit
survey-selection likelihood. A longer-horizon hierarchical experiment should
follow only after that gate is defined.

## 0.2.0 foundation

Version 0.3.0 retains the 0.2.0 deterministic ensemble-validation framework,
strict trajectory registration, distribution metrics, claim-control state
machine, ten-year Horizons/DE441 compatibility record, and locked
100,000-tracer-per-arm REBOUND result. It also retains the earlier provenance
corrections that reject non-standard `NaN` and `Infinity` values.

This remains an engine-focused release. Bulk trajectory/checkpoint archives,
observational data, and candidate-search catalogs are excluded; compact
contracts, manifests, source, audits, and scientific reports are included.
