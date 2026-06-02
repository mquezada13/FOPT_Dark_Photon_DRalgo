# constants.py
"""
Physical and model constants for the FOPT Dark Photon project.

All values are in natural units (GeV) unless otherwise noted.

Model
-----
The model is a dark U(1) gauge theory with a complex scalar S charged under
the dark photon A'. The dark gauge coupling is gD and the scalar sector is
parametrized by (lambdaS, mS).

To adapt this file for a different model, update the sections marked
"RGE initial conditions" and "LV20 spectral presets" accordingly.
"""

# ==============================================================
# Physical constants
# ==============================================================
GF       = 6.7088e-39   # Reduced Newton constant 1/M_Pl^2 [GeV^-2]
MPL      = 2.4e18       # Reduced Planck mass [GeV]
ab       = 5.4076       # Logarithmic constant appearing in the high-T thermal function J_B

# ==============================================================
# Thermodynamic parameters
# ==============================================================
g_dof = 10.75 + 4       # Effective relativistic d.o.f. at the transition (g_SM + 4 dark sector)

# Cosmological redshift factor for GW peak frequency
R = 1.67e-5 * (g_dof / 100) ** (-1 / 3)

# ==============================================================
# Cosmological scales
# ==============================================================
MPC    = 3.0857e22              # 1 Mpc in meters
H_HUND = 1e5 / MPC             # H_0 / (100 km/s/Mpc) in SI units [s^-1]
Hhund  = H_HUND                # Alias used by GW_field_LV20 module
YEAR   = 365.25 * 24 * 3600    # Seconds per year [s]

# ==============================================================
# Detector parameters (LISA)
# ==============================================================
T_LISA  = 3 * YEAR  # LISA mission duration [s]
fmin    = 3e-5      # Minimum frequency in LISA band [Hz]
fmax    = 0.5       # Maximum frequency in LISA band [Hz]
SNR_THR = 10        # SNR detection threshold
SNRthr  = SNR_THR   # Alias used by GW_field_LV20 module

# ==============================================================
# RGE initial conditions  (model-specific)
# ==============================================================
# These are the IR boundary conditions at mu0 = 1 GeV.
# Change these (and the beta functions in RGEsolver.py) when studying a new model.
gD0      = 0.6      # Dark gauge coupling at mu0
lambdaS0 = 1e-10   # Scalar quartic coupling at mu0
mS0      = 1e-10   # Scalar mass parameter at mu0 [GeV]

# ==============================================================
# GW spectral shape presets  (model-specific)
# ==============================================================
# Parameters for the LV20 broken power-law template (arXiv:2007.04967, Table I).
# Each entry is a dict accepted by GW_field_LV20.lv20_shape().
# Keys: A, omegabar_over_beta, a, b, c   (plus d, omegad_over_beta for U(1) gauge).
# Replace with the correct values from Table I of LV20 for your model.
LV20_PRESETS = {
    # U(1) gauge field gradient contribution (T_mu nu ~ R^{-3})
    # Values from Lewicki & Vaskonen (2020), Table I, "Gauge" column.
    # TODO: fill in precise numerical values from the paper.
    "U1": {
        "A":                   0.048,
        "omegabar_over_beta":  0.48,
        "a":                   3.0,
        "b":                   4.0,
        "c":                   1.8,
        "d":                   6.0,
        "omegad_over_beta":    3.6,
    },
}
