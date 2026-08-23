# JX-O2 public reconstruction design v1

This package records what can and cannot be reconstructed from public sources for a future, physically matched Planet Nine model-family comparison. It is a design and provenance artifact only. It contains no runner, checkpoint, simulated population, observational score, model preference, or scientific result.

## Present state

- JX-O2 remains `BLOCKED` at G0.
- Exact reproduction of the published 2023/2024 simulations remains blocked because the original `cluster_2` checkpoint, full compact-body state, random seeds, modified code, input decks, and machine-readable trajectory or footprint outputs were not recovered in the audited public sources.
- An independent public analogue has not been instantiated and cannot be relabeled as the authors' simulation.
- No synthetic calibration, observed-data analysis, unblinding, or GPU run is authorized.

The existing JX-O2 design, G0 audit, and scoped candidate-retirement records remain byte-identical. This sibling package narrows the safe reconstruction paths without changing any prior receipt.

After this bootstrap registration is merged, a dedicated base-side GitHub workflow compares every protected v1 path with each pull-request head without executing head code. Making that check mandatory and preventing direct pushes still requires a repository ruleset outside these repository-controlled files.

## Public facts recorded

The source inventory binds exact arXiv source-archive bytes for the relevant 2023, 2024, 2026, and 2021 papers. The strongest newly usable input is the nine-row cluster-influenced physical grid published by Bansal et al. (2026). Only its input columns—mass, semimajor axis, eccentricity, and inclination—are recorded. The team has already inspected the published output columns; they are deliberately omitted here, and this protocol forbids using them to tune the future grid or its weights.

Those nine rows are literature support, not an executable M1 family. They supply no probability weights, angular state, epoch, frame, checkpoint, matched no-Planet-Nine arm, seeds, exact code, or machine-readable trajectory or footprint output. The paper's eight “cluster-free” rows still contain Planet Nine and use a different starting population, so they are not M0 controls.

The 2024 benchmark supplies a compact body with 5 Earth masses, semimajor axis 500 AU, eccentricity 0.25, and inclination 20 degrees. Its longitude of ascending node, argument or longitude of perihelion, mean anomaly or true anomaly, epoch, frame, and origin were not recovered in the audited public sources. A convention from a different 2021 simulation suite cannot be imported as if it were the missing 2024 state.

## Reconstruction lanes

There are three checkpoint paths:

1. `AUTHOR_EXACT`: obtain the original checkpoint and associated state, code, seed, license, and semantic manifests. Only this path can support the phrase “exact reproduction,” and only after byte-level replay.
2. `OPEN_REGENERATION`: generate a new cluster-history ensemble from public, fully specified code and assumptions. This would be a new `CLUSTER2_LIKE` model, not a recovery of `cluster_2`.
3. `ANALYTIC_OR_FIGURE_SURROGATE`: use an analytic density or digitized published marginals for engineering tests only. Such a surrogate lacks the six-dimensional correlations needed for scientific replacement and cannot satisfy the matched-model G0 gate.

Within each independent paired realization, M0 and every M1 grid member must begin from the identical content-addressed checkpoint and share all non-compact-body forces, nuisance draws, stellar histories, survey streams, and numerical rules. If multiple cluster histories are modeled, their ensemble and weights must be locked in advance and each history must retain its own paired M0/M1 comparison. Only the registered additional compact body may differ within a pair.

Both the 2024 and 2026 studies introduce Planet Nine after drawing from a parent checkpoint generated without it. A future design must therefore lock the body's emplacement time and full state. Such a run estimates a post-checkpoint intervention and cannot represent a self-consistent primordial Planet Nine formation and survival history. Testing the latter would require a separately registered full-history model beginning from matched primordial conditions.

## Missing angles

Missing angles remain unresolved blockers. A later immutable specification must choose one of these paths before any output exists:

- an author-supplied full state with epoch and coordinate metadata;
- a finite, weighted angular nuisance grid or prior applied identically at every physical grid point; or
- a validated symmetry reduction whose assumptions remain valid with tides, stellar encounters, and the survey footprint.

No angle may be defaulted to zero, selected after inspecting an outcome, or copied from an observation-conditioned catalog without a locked correlation and exposure rule. If an analysis maximizes over angles, the complete maximization must be repeated inside every null-calibration replicate.

## Seeds

The original seeds remain unrecovered. Future JX seeds must inherit the registered derivation:

`SHA256(execution_contract_hash || post_registration_public_randomness_beacon || counter)`

A later contract must freeze the byte encoding, future beacon event and value, counter allocation, digest-to-seed conversion, RNG name and version, stream mapping, and common-random-number pairing. New deterministic seeds may never be described as recovered author seeds.

## What this package permits

Permitted work is limited to public-source acquisition, static design, provenance recording, and non-executing integrity tests. Checkpoint generation, angle sampling, seed realization, dynamics, calibration, observed scoring, unblinding, model preference, and large GPU work remain unauthorized.

Mandatory nonclaim: This design records public reconstruction constraints only. It does not recover or reproduce the authors' original simulation, authorize any run, compare models against observations, or provide evidence for or against Planet X.
