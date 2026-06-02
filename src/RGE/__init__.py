# src/RGE/__init__.py
"""
RGE sub-package: effective potential, bounce solvers, FOPT utilities, and GW spectra.

Module overview
---------------
VeffFunc_RGE      : effective potential (tree + CW + thermal + daisy) with RGE running
RGEsolver         : one-loop RGE integration for (gD, lambdaS, mS)
BounceSolFull_RGE : bounce solver using the full J_B potential
BounceSolHighT_RGE: bounce solver using the high-T expanded potential
SE_interpolator   : spline interpolator for pre-computed log10(S_E) grids
FOPT_RGE          : phase-transition utilities backed by the S_E spline
FOPT_RGE_real     : phase-transition utilities using direct bounce-solver calls
GW_RGE_spectrum   : generic GW spectrum from macroscopic FOPT parameters
GW_bubbles_LVV22  : GW spectrum from bubble collisions (LVV22 template)
GW_field_LV20     : GW spectrum from field-gradient / runaway walls (LV20 template)
"""
