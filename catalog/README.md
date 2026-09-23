# JX experiment catalog

This directory is a generated navigation layer. It does not move, rewrite,
merge, promote, or supersede any experiment artifact.

- `JX_EXPERIMENT_CATALOG.json` is the machine-readable consolidated index.
- `JX_EXPERIMENT_CATALOG.csv` is a flat browsing view of the same records.
- `../docs/JX_EXPERIMENT_DATA_INDEX.md` is the human entry point.

The catalog includes stored-byte accounting, links to sibling archives,
explicit loose-file and empty-directory records, lifecycle groupings, and
workspace hygiene flags. Hygiene flags identify missing local metadata; they
do not judge scientific validity.

The immutable repository-root-relative
`../../library/JX_AUTHORITY_CATALOG.json` remains authoritative for library
packages. Active `runs/`, benchmark definitions, and `results/` are cataloged
as workspace records and are not silently promoted to authority.

Regenerate with:

```sh
PYTHONPATH=src:. .venv/bin/python tools/build_experiment_catalog.py
```
