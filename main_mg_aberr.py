#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mg I abundance correction tool
Computes abundance corrections (1D LTE − 3D NLTE) using pre-trained ML models.

Usage:
    python3 main_mg_aberr.py input.csv output.csv
    python3 main_mg_aberr.py input.xlsx output.csv

Required input columns:
    Teff, logg, A(Mg), vmic, line
"""

import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

MODEL_457 = MODELS / "mlp_pipeline_457nm_aberr.joblib"
MODEL_UNIFIED = MODELS / "unified_mlp_pipeline.joblib"

# =========================
# Constants & physics
# =========================
HC_eVnm = 1239.841984

LINE_PHYSICS = {
    416.7: dict(elo_eV=4.3458, lggf=-0.746, log_gamma_rad=8.69, sigma=5296.0, alpha=0.508),
    457.1: dict(elo_eV=0.0000, lggf=-5.623, log_gamma_rad=7.77, sigma=1000.0, alpha=0.250),
    470.3: dict(elo_eV=4.3458, lggf=-0.456, log_gamma_rad=8.70, sigma=2827.0, alpha=0.264),
    473.0: dict(elo_eV=4.3458, lggf=-2.379, log_gamma_rad=8.68, sigma=5928.0, alpha=0.435),
    516.7: dict(elo_eV=2.7091, lggf=-0.854, log_gamma_rad=8.02, sigma=731.0,  alpha=0.240),
    571.1: dict(elo_eV=4.3458, lggf=-1.742, log_gamma_rad=8.69, sigma=1841.0, alpha=0.120),
}

# =========================
# Helpers
# =========================
def parse_nm(val):
    """Extract numeric wavelength from 'line' column."""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(val))
    if not m:
        raise ValueError(f"Could not parse line wavelength: {val}")
    return float(m.group(1))

def n_air_ciddor(lam_air_nm):
    """Refractive index of dry air."""
    lam_um = lam_air_nm / 1000.0
    s = 1.0 / lam_um
    n_minus_1 = 1e-8 * (
        5792105.0 / (238.0185 - s**2) +
        167917.0 / (57.362 - s**2)
    )
    return 1.0 + n_minus_1

def read_input(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError("Input file must be .csv or .xlsx")

# =========================
# Main logic
# =========================
def main(infile, outfile):
    infile = Path(infile)
    outfile = Path(outfile)

    df = read_input(infile)
    df.columns = df.columns.str.strip()

    required = {"Teff", "logg", "A(Mg)", "vmic", "line"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Parse wavelength
    df["lambda_air"] = df["line"].apply(parse_nm)

    # Choose model
    if np.all(np.abs(df["lambda_air"] - 457.1) < 0.2):
        model = joblib.load(MODEL_457)
        X = df[["Teff", "logg", "A(Mg)", "vmic"]].astype(float)

    else:
        model = joblib.load(MODEL_UNIFIED)

        phys = df["lambda_air"].round(1).map(LINE_PHYSICS)
        if phys.isna().any():
            bad = df.loc[phys.isna(), "lambda_air"].unique()
            raise ValueError(f"No physics defined for lines: {bad}")

        phys_df = pd.DataFrame(list(phys))

        lam_air = df["lambda_air"].values
        lam_vac = lam_air * n_air_ciddor(lam_air)
        deltaE_eV = HC_eVnm / lam_vac
        eup_eV = phys_df["elo_eV"] + deltaE_eV

        X = pd.concat([
            df[["Teff", "logg", "A(Mg)", "vmic"]].astype(float),
            pd.DataFrame({
                "lambda_air": lam_air,
                "lambda_vac": lam_vac,
                "deltaE_eV": deltaE_eV,
                "elo_eV": phys_df["elo_eV"],
                "eup_eV": eup_eV,
                "lggf": phys_df["lggf"],
                "log_gamma_rad": phys_df["log_gamma_rad"],
                "sigma": phys_df["sigma"],
                "alpha": phys_df["alpha"],
            })
        ], axis=1)

    # Predict
    df["aberr"] = model.predict(X)

    df.to_csv(outfile, index=False)
    print(f"Saved output → {outfile}")

# =========================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 main_mg_aberr.py input.(csv|xlsx) output.csv")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
