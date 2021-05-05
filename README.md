# Empirical Fire Modelling

[![License: MIT](https://img.shields.io/badge/License-MIT-blueviolet)](https://github.com/akuhnregnier/empirical-fire-modelling/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![DOI](https://zenodo.org/badge/343888602.svg)](https://zenodo.org/badge/latestdoi/343888602)

## Install

Installation is most easily performed in a two-step process, with installation of the dependencies followed by the installation of the package itself.
Both can easily be done after the repository is cloned.

Clone the repository:

```bash
git clone https://github.com/akuhnregnier/empirical-fire-modelling
```

Install dependencies into a new `conda` environment called 'empirical-fire-modelling':

```bash
cd empirical-fire-modelling
conda env create -f requirements.yaml
```

Activate this new environment before installing the `empirical_fire_modelling` package itself (while remaining in the same directory as above):

```bash
conda activate empirical-fire-modelling
pip install -e .
```

## Structure

The repository contains Python code to carry out essential operations across multiple sets of variables ('experiments').
This is stored in the `src` directory.

Additionally, the `analysis` directory contains code which makes use of the `empirical_fire_modelling` package in order to run and visualise the various analyses.

- The package runs various analyses and stores their results
  - Model fitting
  - Variable importance measures
  - ALE plots
  - 2D ALE plots
  - SHAP values
  - SHAP interaction values
  - SHAP maps
- Storage of results (e.g. trained models, analyses) using Joblib
  - Different input arguments are automatically detected by Joblib and trigger recalculation of results
  - As a consequence of the caching mechanisms, cached functions should not call other cached functions or use imported variables, since this would fail to detect changed results from previous calculations
- Result visualisation
  - Results are not typically visualised interactively
  - Scripts to generate graphs and store them for later viewing are contained in `analysis`
- Organisation:
  - The `empirical_fire_modelling` package contains common functionality
    - Calculation of results or analyses
    - Data/result visualisation
  - The `analysis` folder contains scripts that use the library modules to execute common functionality across all experiments
