# Provenance and release boundary

This directory is an archival copy of the 14-file XP2 V4/V5 scientific core
selected on 2026-08-27 for public review. Each preserved core file was copied
without byte changes. Its SHA-256 digest is recorded in `SHA256SUMS`.

The JSON contract contains historical relative paths and cryptographic digests
for predecessor registrations, attempts, locks, receipts, and protected trees.
Most of those referenced artifacts are deliberately absent from this public
snapshot. A matching digest string is provenance information, not proof that
the referenced object is present, independently available, valid, or
authorized.

The following are intentionally excluded:

- all XP2 scientific outcomes, checkpoints, trajectories, receipts, ledgers,
  locks, and failure directories;
- historical V1–V3 run trees and registrations, except the two small diagnostic
  evidence records required by the V4 contract;
- host-specific operational-adapter, registration-ceremony, and discovery
  drafts;
- local absolute paths, runtime caches, virtual environments, logs, and private
  machine metadata;
- the REBOUND wheel and every compiled binary;
- observed catalogs and survey-selection data.

The published code is source under review. Because the full historical
dependency and authority chain is absent, this directory must not be described
as a self-contained runnable or reproduction package.

The literal `/private/absolute/secret/input.json` in `test_independent.py` is a
synthetic redaction test fixture. It is not a real path, credential, or secret.
