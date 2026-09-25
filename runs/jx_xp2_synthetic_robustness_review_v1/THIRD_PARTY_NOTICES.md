# Third-party software and method notices

No third-party binary, wheel, source tree, or observational dataset is vendored
in this snapshot. The preserved source names or imports the following external
software and methods.

## REBOUND 4.4.11 and MERCURIUS

The primary path expects REBOUND 4.4.11 and its MERCURIUS hybrid integrator.
REBOUND is developed by Hanno Rein and contributors and is distributed under
the GNU General Public License, version 3 or later. This snapshot does not
redistribute REBOUND.

- Project and license: <https://github.com/hannorein/rebound>
- Rein, H. & Liu, S.-F. (2012), *REBOUND: an open-source multi-purpose N-body
  code for collisional dynamics*, A&A 537, A128.
  <https://doi.org/10.1051/0004-6361/201118085>
- Rein, H. et al. (2019), *Hybrid symplectic integrators for planetary
  dynamics*, MNRAS 485, 5490–5497.
  <https://doi.org/10.1093/mnras/stz769>

## NumPy 2.3.5

The independent numerical path expects NumPy 2.3.5. NumPy is distributed under
a modified BSD license. No NumPy code or binary is included here.

- Project: <https://numpy.org/>
- Versioned license: <https://github.com/numpy/numpy/blob/v2.3.5/LICENSE.txt>

## SciPy 1.17.0 and DOP853

The independent path calls `scipy.integrate.solve_ivp(method="DOP853")` with a
project-specific Newtonian right-hand side. It does not independently
reimplement the DOP853 algorithm. SciPy is distributed under a BSD 3-clause
license. No SciPy code or binary is included here.

- Project: <https://scipy.org/>
- Versioned license: <https://github.com/scipy/scipy/blob/v1.17.0/LICENSE.txt>
- SciPy `solve_ivp` documentation:
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>
- Hairer, E., Norsett, S. P. & Wanner, G., *Solving Ordinary Differential
  Equations I: Nonstiff Problems*, 2nd revised edition, Springer.
  <https://doi.org/10.1007/978-3-540-78862-1>

## MurmurHash3 compatibility routine

`test_primary.py` contains a small Python compatibility implementation of
MurmurHash3 x86-32 constants and operations to check REBOUND string hashes.
MurmurHash3 was created by Austin Appleby and released to the public domain.

- Reference project: <https://github.com/aappleby/smhasher>

