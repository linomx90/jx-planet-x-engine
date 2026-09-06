# Retained Solar 1PN qualification sources

This README is explanatory and nonnormative; the machine package does not use
its prose as scientific input or authority.

These files are retained only to make the frozen, nonauthorizing V5
qualification inputs reproducible. Their presence does not make a parameter
exact, resolve an uncertainty, qualify a force model, or authorize a
trajectory. The machine-readable source records in
`../qualification_inputs_v1.json` bind each file's exact byte size, SHA-256
digest, retrieval time, original URI, provider, version, and equation or table
locator.

The retained set is deliberately narrow:

- `iers_tn36_chapter_10.pdf` provides the restricted Solar Schwarzschild 1PN
  equation and the broader BCRS context that limits its interpretation.
- the 2006 and 2012 IAU resolution PDFs define the coordinate-time and exact
  astronomical-unit conventions used by the transformation fixtures;
- `bipm_si_brochure_9_v4_01.pdf` supplies the exact SI speed-of-light and time
  definitions;
- the JPL astrodynamic-parameter page preserves the chosen high-precision
  DE440 Solar-GM numeral; `gm_de440.tpc` preserves its rounded NAIF kernel
  representation, and the JPL DE440 export page documents the ephemeris
  context.

The DE440 and Horizons systems contain substantially more physics than JX's
restricted static-Sun test-particle equation. Their raw trajectories are
therefore not accepted here as correctness oracles. They may be used only in a
later model-matched comparison with every additional force and transformation
either reproduced or bounded in the preregistered error budget.

No expected holdout output, unblinded result, independently implemented EIH
oracle, or external high-order trajectory oracle is stored in this directory.
Those absences remain execution blockers.
