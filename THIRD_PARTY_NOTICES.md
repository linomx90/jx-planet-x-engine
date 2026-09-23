# Third-party notices

Original JX source code and project material in version `0.6.0rc10` is governed
by the proprietary all-rights-reserved notice in the repository root. Earlier
public JX releases retain the licenses and notices that accompanied them.

The complete research archive also preserves REBOUND 4.4.11 source and a
Python 3.12 Linux wheel solely to reproduce the optional IAS15 comparison
path. REBOUND is distributed under the GNU General Public License version 3;
its complete license text is included in both preserved distributions. The
REBOUND source archive is provided alongside the wheel as corresponding
source.

The other preserved ZIP bundles under `imports/` are JX project evidence and
benchmark inputs. They are included for reproducibility and retain their own
embedded provenance and notices. No JX ownership claim is made over third-party
software, data, standards, papers, or licenses preserved for reproducibility.

The optional `llr-screen` dependency group installs PyERFA 2.0.1.5 under its
BSD 3-Clause license and SpiceyPy 8.2.0 under its MIT license. Those packages
are not JX source and retain their upstream copyrights and license terms. The
LLR runner consumes separately supplied IERS, ILRS, JPL/NAIF, and APOLLO data;
JX does not claim ownership of those third-party inputs.

`src/jxplanetx/solar_system/solid_earth_tide.py` is a renamed Python
adaptation of the algorithms and coefficient tables in the IERS Conventions
Chapter 7 `DEHANTTIDEINEL`, `ST1*`, and `STEP2*` routines. It is not official
IERS Conventions software and is neither distributed nor endorsed by the IERS
Conventions Center. The adaptation, its differences, upstream file hashes,
and the complete IERS Conventions Software License notice are preserved in
that source file. Any published work using results from this component must
acknowledge the IERS Conventions software as required by that notice.

`src/jxplanetx/solar_system/solid_pole_tide.py` independently implements the
published equations in the 2018 working version of IERS Conventions Chapter 7;
it is not official IERS software and is not endorsed by IERS. The LLR runner's
FES2014B HARPOS coefficients are a separately supplied NASA International Mass
Loading Service data input. They are hash-bound for reproducibility, remain
outside the JX source tree, and are not owned by JX.
