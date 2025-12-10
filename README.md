# Effective_potential_V5: Dark Photon Model

# Effective Potential V5

This repository contains the implementation and analysis of the effective potential related to first-order phase transitions in thermal field theory. It includes numerical tools for checking relevant parameters, solving bounce equations, and comparing different approaches (High-T approximation vs. Full potential).

## Contents

- Jupyter notebooks for parameter checks and comparisons:
  - `Full_checks.ipynb`
  - `HighT_checks.ipynb`
  - `Full_checks-example.ipynb`
- Python scripts for bounce solution:
  - `BounceSolFull.py`, `BounceSolHighT.py`
- Constants and parameter definitions:
  - `constants.py`, `FOPT_params.py`
- Effective potential and helper functions:
  - `VeffFunc.py`, `VeﬀFuncmU_s_zero.py`
- Plots, data files, and intermediate PDFs

## Requirements

- Python 3.12
- Required packages:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `jupyter`

## How to use

1. Clone the repository.
2. Make sure the following core files are available and properly imported in your notebooks:
   - `constants.py`: defines global constants used across the project
   - `FOPT_params.py`: contains model-specific parameters
   - `VeffFunc.py`: defines the effective potential functions
   - `BounceSolFull.py` and `BounceSolHighT.py`: provide numerical bounce solutions
3. Open the Jupyter notebooks in Visual Studio Code or Jupyter Lab.
4. Run the cells to reproduce parameter checks, plots, and comparisons.

## Author

This project was developed by **Maura Elizabeth Ramirez-Quezada** as part of her research in phase transitions and thermal field theory in the early universe.  
It is also part of her personal portfolio to document numerical tools and workflows used in theoretical physics and data analysis.