# Experimental native lunar rotation port

Prepared for Lino Avila Maceda / JX. Result: **NATIVE_PORT_SUPPORTED** for
`JX-NATIVE-LUNAR-PORT-01`. This is an opt-in C++20 module on a separate branch,
not a production merge or a fully coupled Solar-System model.

## Integration boundary

This module reuses `jx::Vec3` from the existing `native/jx_bm6_types.hpp`.
It does not edit that header, `jx::State`, `bm6_step`, the BM6 executable,
the original Makefile, or the default Python package. `ModelOptions` defaults
to degree-2 only; all static degrees 3 through 6 are explicitly enabled by
`ModelOptions{true}`. The regression executable intentionally runs both lanes.

`torque_over_C(q, sources, options)` accepts simultaneous geometric source
vectors in km and GM in km^3/s^2, returning torque divided by the undistorted
polar moment C, in s^-2. The quaternion is active, scalar-first Hamilton,
body-to-inertial. `derivative` uses body angular velocity in rad/day and carries
inertial torque-impulse and rotational-work integrals. `rk4_step` takes elapsed
TDB days and a caller-supplied geometric source provider. No reference attitude
belongs in this provider. The core header does not require CSPICE; only the
standalone benchmark executable uses it.

DO NOT insert torque/C into the existing translational acceleration buffer.
DO NOT claim mutual orbit-rotation coupling: Earth/Sun geometry is prescribed,
and no reaction force or orbit feedback is applied. Quaternion normalization
is used to construct matrices, not to reset the integrated state.

## Fixed physics and numerical scope

The same undistorted total-inertia proxy and published DE440 coefficients are
retained from the public propagation benchmark. Higher harmonics use the
unnormalized, no-Condon--Shortley convention. Polynomial gradients avoid
latitude-coordinate pole divisions. Deformation, core exchange, geodetic terms,
Earth figure-figure torque, and other sources remain absent.

The native runner integrates four separately initialized 32-day arcs, starting
at J2000 + 0, 32, 64, and 96 TDB days. These dates were already examined in the
Python parent; this is port reproduction, not fresh physical confirmation.
C++ quaternion RK4 uses 32, 64, and 128 steps/day. Main results are CPU binary64
with fast-math forbidden and contraction disabled. Numerical sensitivity
allowances are empirical, not reference-physics confidence intervals.

## Executed comparison

* 272 pointwise torque comparisons passed against parent predicted attitudes.
* All 1,032 primary scored native states agree with saved Python DOP853 states
  within the frozen limits. Maximum orientation separation: 5.10437e-14 rad
  (1.05285e-8 arcsec). Maximum inertial-rate separation: 1.18972e-20 rad/s.
* The original 56 orientation/rate improvement criteria all passed.
* Halving native RK4 steps reduced maximum orientation disagreement from
  1.01909e-11 to 6.45771e-13 to 5.10437e-14 rad.
* O0 and O2 builds had identical compared orientations and inertial rates here.
  A separate process reproduced the four full state files and execution metadata.
* Source/parent manifests remained unchanged; the exact inherited native type
  header matches Git blob 8274391860b7c08aad8dc9b7f46e3d3c09b608d6.

These agreements do not establish absolute lunar accuracy, a speed advantage,
long-term stability, full DE440 reproduction, or the original private 0.650038%.

## Reproduce

Prerequisites: Linux C++20 compiler, CMake, Python with NumPy and SpiceyPy, and
the extracted `JX_Lunar_Rotation_Propagation_Test_2026-09-16.zip` supplied with
the parent benchmark. The parent includes the exact public kernels and a pinned
SpiceyPy 8.0.0 CPython-3.13 Linux wheel. No automatic network download is done.

```sh
python native/lunar_rotation/verify_benchmark.py \
  --parent /path/to/jx_lunar_propagation_2026-09-16 \
  --out /path/to/new-native-replay
```

This builds the native target, verifies input hashes, runs the three step sizes
and a replay, and compares all checkpoints to the retained Python trajectories.
The output path must not exist and must be outside the parent directory.
The fuller supplied port archive also contains the O0/O2, torque, balance,
56-criterion, and invalid-input checks plus raw execution records.

For a direct CMake build:

```sh
cmake -S native/lunar_rotation -B build/lunar \
  -DCMAKE_BUILD_TYPE=Release -DCSPICE_LIBRARY=/full/path/to/libcspice.so
cmake --build build/lunar
ctest --test-dir build/lunar --output-on-failure
build/lunar/jx_lunar_native /path/to/kernels /path/to/new-output 128
```

`native_benchmark.cpp` declares only the small documented CSPICE N0067 C ABI it
uses. It was linked directly to the retained official library in the SpiceyPy
wheel; it does not call Python during propagation. No CSPICE implementation or
font file is vendored in this source directory.

## Sources and provenance

Base native branch: `experiment/bm6-rebound-5.1.1`, commit
`6506a6079adf24bc02a3b11a724c6a7a733a2632`.

Parent archive SHA-256:
`9499188563e74bce849f33543b92023315ea4ad5fdc9a497f28045284af1a996`.

* Park et al. (2021), DE440/DE441, equations 28, 49, 56:
  https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf
* Frozen coefficient source:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440_tech-comments.txt
* Geometric source states and angular-velocity convention:
  https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkpos_c.html
  https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/xf2rav_c.html

An initial CSV fixture preflight rejected CRLF whitespace. Trimming numeric
field whitespace repaired the interface before any native trajectory ran;
physics, constants, schedules, and acceptance criteria were unchanged. The
failed preflight and correction note are retained in the full port archive.
