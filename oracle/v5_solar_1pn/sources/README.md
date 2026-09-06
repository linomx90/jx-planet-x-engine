# Retained RK7(8) method source

This directory retains the primary coefficient source used by the isolated
Solar 1PN oracle candidate. Retention does not establish operational
independence, scientific qualification, or execution authority.

- File: `nasa-tr-r-287-fehlberg.pdf`
- Title: *Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta
  Formulas with Stepsize Control*
- Author: Erwin Fehlberg
- Publisher: National Aeronautics and Space Administration
- Report: NASA TR R-287
- Publication date: October 1968
- Retrieved: 2026-08-29
- URI:
  `https://ntrs.nasa.gov/api/citations/19680027281/downloads/19680027281.pdf`
- Size: 2,625,098 bytes
- SHA-256:
  `5553a2a3eb53785a461762cc2b29428015f1b32c3ad0a5cb57f85a421256a0c8`
- Method locator: Equation (105), printed page 52 (PDF page 59), and Table X,
  printed page 65 (PDF page 72)

Equation (105) distinguishes the seventh-order formula from the hatted
eighth-order formula. Table X supplies the exact 13-stage rational
coefficients. The candidate accepts the hatted eighth-order state; the
embedded seventh-order state supplies the error estimate.
