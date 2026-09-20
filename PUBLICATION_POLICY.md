# Public release boundary

JX uses a public-engine/private-research-vault model. Public source must be
independently inspectable and reproducible without exposing credentials,
licensed or private datasets, unpublished candidate material, or local research
history.

## Public material

The public boundary may include:

- engine source and explicit algorithms under `src/`;
- bounded source tests under `tests/`;
- reproducibility, validation, and publication checks under `tools/`;
- reviewed documentation, audits, and benchmark source;
- compact, reviewed run manifests and summaries whose publication is lawful;
- dependency pins, release metadata, checksums, and scientific claim limits;
- General Dynamics registry metadata whose top-level claim state remains
  `SCREENING_ONLY`.

Public checksums provide integrity only. They are not signatures, proof of
origin, scientific qualification, or permission to distribute the referenced
payload.

## Material that stays private

Never publish:

- passwords, API tokens, cookies, credentials, private or signing keys;
- `.env` files or machine-specific authentication configuration;
- the read-only `library/` mirror unless every selected item passes a separate
  license, provenance, privacy, and size review;
- raw or unpublished observations, candidate catalogs, coordinates, or
  personally identifying information;
- frozen execution payloads, retained environments, proprietary dependencies,
  bulk evidence, archives, build outputs, or virtual environments;
- hostnames, private network details, hardware serial numbers, or absolute
  personal filesystem paths;
- unpublished intermediate Git history from research branches.

Large data is not automatically secret, but it does not belong in the source
repository. A separately reviewed public dataset should use an archival store
and expose a stable citation, license, size, and checksum.

## Publication procedure

1. Start from the public `dev` history in a clean publication checkout.
2. Transfer only explicitly reviewed source-boundary files.
3. Run `python tools/check_publication_boundary.py --root .`.
4. Run the CPU core, optimized-Python, packaging, and applicable hardware gates.
5. Review the complete public diff, including filenames and metadata.
6. Publish a squashed commit whose parent is already public.
7. Never merge the private research branch into `dev`, use `git push --all`, or
   push private tags or refs.

If a credential is exposed, revoke or rotate it immediately before attempting
history cleanup. Removing the latest copy does not remove it from Git history or
from existing clones.

## Scientific boundary

Publication makes source inspectable; it does not promote scientific claims.
Benchmark labels remain restricted to their exact reviewed workloads. JX
General Dynamics `0.6.0rc1` remains `SCREENING_ONLY`, not production-ready, and
not a unified-multiphysics execution qualification.
