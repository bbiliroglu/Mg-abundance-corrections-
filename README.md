# Mg I abundance corrections

This repository provides machine-learning–based abundance corrections  
(1D LTE − 3D NLTE) for selected Mg I spectral lines.

## Supported lines (air wavelengths, nm)
- 416.7
- 457.1 (dedicated model)
- 470.3
- 473.0
- 516.7
- 571.1

## Usage
```bash
python3 main_mg_aberr.py input.csv output.csv

## Input Format

Teff,logg,A(Mg),vmic,lambda
5050,4.0,7.2,1.0,457.1

## Output
The output file contains an additional column:
aberr : abundance correction (dex)
The code automatically selects a dedicated 457.1 nm model or a unified multiline model.
