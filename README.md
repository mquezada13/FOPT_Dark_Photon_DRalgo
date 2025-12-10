<<<<<<< HEAD


# Effective Potential: Dark Photon Model
### *RGE-improved effective potentials, bounce solvers, and GW predictions for a dark-photon model*

This repository contains a full numerical pipeline for studying first-order phase transitions in a dark-photon model. It combines effective potentials, 1-loop RGEs, bounce equation solvers, nucleation analysis, and gravitational-wave spectra into one coherent (and occasionally stubborn) framework. If you are here, you are probably either:  
1) studying thermal field theory,  
2) debugging your own bounce solver at 3 am,  
or 3) just curious about why the universe insists on changing phase.  
In any case, welcome.
=======

## **Repository Structure**
    FOPT_Dark_Photon_DRAlgo/
    │
    ├── notebooks/
    │   ├── FOPT_RGE_check.ipynb
    │   ├── GW_RGE.ipynb
    │   ├── S_RGE_Full_plot.ipynb
    │   ├── S_RGE_HT_plot.ipynb
    │   └── Veff_RGE_plot.ipynb
    │
    ├── src/
    │   ├── noRGE/
    │   ├── RGE/
    │   │   ├── RGEsolver.py
    │   │   ├── VeffFunc_RGE.py
    │   │   ├── BounceSolFull_RGE.py
    │   │   ├── BounceSolHighT_RGE.py
    │   │   ├── SE_interpolator.py
    │   │   ├── FOPT_RGE.py
    │   │   ├── GW_RGE_spectrum.py
    │   │   └── __init__.py
    │
    ├── utils/
    │   ├── plot_styles.py
    │
    ├── data/
    │   └── VT_integralNumeric.dat
    │
    ├── plots/
    │
    ├── startup.py
    ├── requirements.txt
    └── README.md


---

## **Core Components**

### **1. Effective Potential with RGE**
Implemented in `VeffFunc_RGE.py`.

Features:
- one-loop running of couplings and mass parameters  
- full zero-temperature 1-loop corrections  
- thermal masses and Daisy/DR resummation  
- internal rescaling to stabilize shallow barriers  
- dynamic minimum-finding strategies (especially in High-T)

---

### **2. Renormalization Group Equations**
`RGEsolver.py` evolves:
- \( g_D(\mu) \)  
- \( \lambda_S(\mu) \)  
- \( y(\mu) \)  
- \( m_S^2(\mu) \)

The solver uses log-scale integration for numerical stability so the couplings run smoothly instead of explosively.

---

### **3. Bounce Action Solvers**
Two complementary solvers:

- `BounceSolFull_RGE.py`: full effective potential  
- `BounceSolHighT_RGE.py`: High-T approximation  

Includes:
- overshoot/undershoot methods  
- adaptive radial domain  
- computation of \( S_3(T)/T \) and \( S_4 \)  
- stability routines for tiny thermal barriers

---

### **4. Nucleation and Transition Parameters**
`SE_interpolator.py` and `FOPT_RGE.py` provide:
- interpolation of the action  
- nucleation temperature \( T_n \)  
- decay rate \( \Gamma(T) \)  
- converted volume fraction \( P_f(T) \)  
- transition strength \( \alpha \)  
- duration parameter \( \beta/H \)

Everything is consistently tied to the RGE-improved potential.

---

### **5. Gravitational-Wave Spectrum**
`GW_RGE_spectrum.py` computes contributions from:
- sound waves  
- turbulence  
- bubble collisions (when applicable)

This produces a complete predicted GW spectrum ready for comparison with LISA or PTA observations.

---

## **Installation**

- Python ≥ 3.12  
- Required packages:
numpy
scipy
matplotlib
jupyter


---

## **How to Use**


1. Clone the repository.  
2. Open any notebook under `notebooks/`.  
3. Run the cells to reproduce potential plots, bounce profiles, nucleation parameters, and GW spectra.  
4. Adjust model parameters as needed. The potential will let you know if it disagrees.

---

## **Author**

Developed by **Maura Elizabeth Ramirez-Quezada** as part of ongoing research on first-order phase transitions, thermal field theory, and cosmological signatures. A mix of physics, numerical methods, and the usual persistence needed to make bounce solvers behave.
=======
This project was developed by **Maura Elizabeth Ramirez-Quezada** as part of her research in phase transitions and thermal field theory in the early universe.  
It is also part of her personal portfolio to document numerical tools and workflows used in theoretical physics and data analysis.

