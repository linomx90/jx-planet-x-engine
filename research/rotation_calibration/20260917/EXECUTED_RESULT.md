# JX rotation-versus-calibration: executed result

Prepared for Lino Avila Maceda. 17 September 2026.
Result ID: JX-LUNAR-ROTATION-CALIBRATION-TEMPLATES-01.
Decision: KINEMATIC_TEMPLATE_SCREEN_EXECUTED__MULTIREFLECTOR_SPLIT_BLOCKED__ACTUAL_JX_SIGNATURE_NOT_TESTED.

## Execution and scope

Local Python/container execution failed and Library listing was rate-limited. The calculation was therefore actually executed through GitHub Actions on the existing isolated branch experiment/llr-public-input-audit-20260917. Main and archived JX dynamics were not modified. No personal contacts, email, outreach, or private archived source/data were used.

Primary source/workflow commit: 7d572532004600980a7067068f8e8952c6b80587.
Primary workflow: .github/workflows/jx-rotation-calibration-public-20260917.yml.
Run 35286444099, job 105419720850.
Post-outcome split-diagnostic commit: 0450c4b95dfe7adc58c39ee268e95ac46ad18e33.
Review run 35286684270, job 105420456103.
Both research jobs completed successfully. Existing repository CI also triggered.

This is a local geometric sensitivity screen using prescribed constant, linear and sinusoidal body-frame rotations. The periods 27.5 days and 365.25 days are engineering test definitions, NOT measured lunar modes. No JX-minus-control dynamic trajectory was generated at the observation epochs. No measured round-trip value or printed range uncertainty was parsed numerically or fitted. This is not observational validation or a new lunar accuracy figure.

Public APOLLO v1 data, DOI 10.5281/zenodo.7818557, were downloaded and their published .np MD5 d8923c930f93aea9dfc9eabeb928295a verified. Source date/time and station/reflector identifiers alone were used. Original NAIF DE440 SPK, lunar PCK, matched lunar frame, dated Earth PCK and leap-second file define the common geometric model; all acquired source hashes are in the result package. CSPICE supplies the states; a new implicit two-way range derivative and SVD/QR projection supply the sensitivity calculation. The nominal station/reflector vectors remain engineering definitions, not an authenticated survey.

## Coverage

All records selected in [2006-04-07,2007-04-07), station code70610:

|Selection|Records|UTC dates|Reflectors|
|---|---:|---:|---|
|Original Apollo15 dates|8|2|Apollo15|
|Full first-year Apollo15|124|40|Apollo15|
|First-year all available sites|162|40|Apollo11:19; Apollo14:15; Apollo15:124; Lunokhod2:4|

There are no Lunokhod1 observations in this selected year. The Apollo15-only endpoint condition number is reproduced as approximately140.19458; the two-date value is approximately13,336,426, within the predeclared cross-implementation tolerances. The all-reflector coordinate condition number360.59227 concerns fifteen rather than six unknown coordinates; it is not directly a better/worse score for the same parameter problem.

## Rotation patterns after permitted calibration adjustments

For nuisance matrix A and rotational template matrix G, nuisance columns are normalized before SVD, and the numerical projection is H=(I-U_A U_A^T)G. Primary rank threshold is1e-12; controls use1e-10 and1e-14. The following percentages are100*||H||_F/||G||_F for the declared equally scaled rotation basis. They are NOT percentages of physical accuracy, statistical information, detectability, or actual JX error. Ratios depend on templates, parameter freedoms, and weights. No physical covariance or parameter priors were supplied.

|Calibration model|Rotation pattern|Apollo15,124 records: equal-record % retained|All four reflectors,162 records: equal-record % retained|
|---|---|---:|---:|
|Coordinates + constant range offset|Linear drift|89.073655|87.567251|
|Coordinates + constant range offset|27.5-day sine/cosine|49.802787|59.089740|
|Coordinates + constant range offset|365.25-day sine/cosine|86.837557|87.747031|
|Coordinates + station velocity + constant range offset|Linear drift|14.953934|49.132418|
|Coordinates + station velocity + constant range offset|27.5-day sine/cosine|35.173426|52.483771|
|Coordinates + station velocity + constant range offset|365.25-day sine/cosine|67.067623|74.826531|
|Coordinates + station velocity + independent offset for each UTC date|Linear drift|0.012756|36.644298|
|Coordinates + station velocity + independent offset for each UTC date|27.5-day sine/cosine|0.111727|25.858844|
|Coordinates + station velocity + independent offset for each UTC date|365.25-day sine/cosine|0.105441|41.588405|

For the monthly basis, equal-total-per-date weighting changes the station-velocity/global-offset values to35.578991% versus47.363767%, and the date-offset values to0.099260% versus22.799159%. The qualitative contrast remains. Those weights are sensitivity choices, not a measured covariance. The inspected rank-threshold controls retain these primary year-scale fractions.

Constant orientation changes vanish to numerical precision when every reflector coordinate can compensate. Linear drift similarly vanishes when free reflector velocities are included. These are expected local geometric nulls, not proof that every time-dependent physical rotation is invisible.

The calculation shows why freely adding calibration offsets can hide a signal under study, and why the tested multiple-reflector schedule retains more of several prescribed time-dependent patterns. It does not justify omitting physically required calibration corrections. It does not measure whether the remaining signal exceeds observational/systematic noise.

## The frozen held-out-date split failed its coverage requirement

The original protocol withheld every fifth chronological UTC date; this was fixed before outcomes and was not subsequently changed. Apollo15 has103 training records and21 withheld; all reflectors have133 training and29 withheld. Both have32 training and8 withheld dates.

For Lunokhod2, only ONE observation remains in training (2007-02-21), with THREE withheld (2006-06-18,2007-01-09,2007-02-10). Consequently, not all its free coordinate components can be estimated from training.

|Model|All-reflector parameter columns|Training rank|
|---|---:|---:|
|Coordinates|15|13|
|Coordinates + global offset|16|14|
|Coordinates + station velocity + global offset|19|17|
|Coordinates + station and reflector velocities + global offset|31|26|

A separate post-outcome diagnostic explicitly propagated the right-nullspace of the training matrix into held-out rows. It is nonzero: distinct equally valid training solutions yield different held-out predictions. Therefore the multi-reflector minimum-norm template-fit extrapolation is NOT a unique calibration prediction and NOT a passed validation test. Even the large apparent residual for a constant frame template on those held-out rows is not physical distinguishability. All initial split outputs, including large extrapolation values, are retained unchanged.

The Apollo15-only training matrices have full column rank for these four models. That numerical fact alone does not qualify their physical calibration. UTC date grouping is not claimed to be the actual instrument/session grouping.

## Verification actually completed

352/352 primary numerical, derivative, projection, null and baseline-reconstruction checks passed. These are related mathematical controls, not352 physical experiments. Finite differences re-solved the moving-endpoint geometry for specified position and rotation perturbations. Pivoted QR independently checked the nuisance-space projections. Twelve named primary text files reproduced byte-for-byte; all saved NPZ arrays reproduced by value. A separate review job verified the exact primary ZIP checksum and all22 delivery-manifest entries. Five split-diagnostic/report files independently replayed byte-for-byte. No claim is made that the multiple-reflector validation split passed.

## Preservation

Primary inner ZIP SHA256: e0fd27f018a31cdc44f1e81bb53fb2c066ef7ea9cd92977f16642e22f3682835.
Final combined inner ZIP: JX_Rotation_Calibration_Screen_Complete_2026-09-17.zip,304251 bytes, SHA25626aba6e245e2c5d40d7ee02694212c3ee8e4f27f2781804621682b5a2f2d3ee1.
Final GitHub artifact wrapper ID10525125331, SHA256412606308bd0bb9c0eb475218e6f8512c557d28d2cedfafe9425eeab66be2af8.
Public-input artifact ID10524506965,46601257 bytes, SHA256efd5118257484a6b9cd4ac859500de1e6735f24cf4431ce1401b99525f7052c4.

Files.manage_library confirmed successful creation/uploads under:
/JX General Dynamics Engine/Checkpoints/2026-09-16 Lunar Rotation Research/Follow-up Results/2026-09-17 Rotation-Calibration Template Screen

Uploaded exact exported artifact snapshots:
- JX_Rotation_Calibration_Screen_Delivery_2026-09-17.zip (Library backing file file_000000001408822f8abac6d8a1fd3ced).
- JX_Rotation_Calibration_Readable_Review_2026-09-17.zip (file_000000000da8822f9d57e4637eba11c6).
- JX_Rotation_Calibration_Public_Inputs_2026-09-17.zip (file_000000002584822f81595794a4e3ad4f).

The final results are nested inside the GitHub delivery ZIP. The readable-review ZIP contains REPORT.md, RESULT_STATE.json and REVIEW_REPLAY.json. The full public-input ZIP preserves all six data/kernel files. Library upload succeeded, but a subsequent Library read returned429 and local container checks failed: this response does NOT claim independent Library readback checksum verification. The checksum checks described above were executed remotely before this Library upload. The existing latest-result pointer and previous scientific results were not overwritten.

## Remaining targeted work

The new screen supplies an executed test of prescribed rotational patterns and a concrete warning about calibration design. The decisive actual-model result is still missing: matched JX and control predictions at the actual observing dates, followed by nuisance projection under a coverage-qualified calibration/validation design and consistent measurement definitions. Neither a large retained template fraction nor a numerically passing routine establishes observational detection. No new annual orientation accuracy, reduced APOLLO residual, or explanation of historical private0.650038% is claimed.

Public sources: https://zenodo.org/records/7818557 ; https://tmurphy.physics.ucsd.edu/apollo/norm_pts.html ; https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/sxform_c.html . Source data attribution: Colmenares et al.(2023), CC BY4.0. Full source versions and hashes are recorded by the executed workflow. The exact workflow commits above are the scientific provenance, not this summary alone.
