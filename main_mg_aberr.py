#!/usr/bin/env python3
"""
Mg I abundance corrections (1D LTE − 3D NLTE)

Usage:
    python3 main_mg_aberr.py input.csv output.csv
    python3 main_mg_aberr.py input.xlsx output.csv

Input columns:
    Teff, logg, A(Mg), vmic, line
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODELS_DIR = "models"
MODEL_457 = os.path.join(MODELS_DIR, "mlp_pipeline_457nm_aberr.joblib")
MODEL_UNIFIED = os.path.join(MODELS_DIR, "unified_mlp_pipeline.joblib")


# --------------------------------------------------
# Line physics (INTERNAL — DO NOT REQUIRE USER INPUT)
# --------------------------------------------------

LINE_PHYSICS = {
    416.7: dict(lambda_vac=416.8, elo_eV=2.71, eup_eV=5.69,
                deltaE_eV=2.98, lggf=-0.71, log_gamma_rad=8.52,
                sigma=281.0, alpha=0.23),

    470.3: dict(lambda_vac=470.4, elo_eV=4.35, eup_eV=6.99,
                deltaE_eV=2.64, lggf=-0.44, log_gamma_rad=8.38,
                sigma=327.0, alpha=0.24),

    473.0: dict(lambda_vac=473.1, elo_eV=4.35, eup_eV=6.97,
                deltaE_eV=2.62, lggf=-2.39, log_gamma_rad=8.38,
                sigma=327.0, alpha=0.24),

    516.7: dict(lambda_vac=516.8, elo_eV=2.71, eup_eV=5.11,
                deltaE_eV=2.40, lggf=-0.87, log_gamma_rad=8.49,
                sigma=281.0, alpha=0.23),

    571.1: dict(lambda_vac=571.2, elo_eV=4.35, eup_eV=6.52,
                deltaE_eV=2.17, lggf=-1.83, log_gamma_rad=8.42,
                sigma=327.0, alpha=0.24),
}


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def read_input(fname):
    if fname.lower().endswith(".csv"):
        return pd.read_csv(fname)
    if fname.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(fname)
    raise ValueError("Input file must be CSV or XLSX")


def parse_line(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    return float(str(val).lower().replace("nm", "").strip())


# --------------------------------------------------
# Main
# --------------------------------------------------

def main(infile, outfile):

    df = read_input(infile)

    required = ["Teff", "logg", "A(Mg)", "vmic", "line"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Parse wavelength
    df["lambda_air"] = df["line"].map(parse_line).round(1)

    # Prepare output column
    df["aberr"] = np.nan

    # Load models
    model_457 = joblib.load(MODEL_457)
    model_uni = joblib.load(MODEL_UNIFIED)

    # --------------------------------------------------
    # 457.1 nm dedicated model
    # --------------------------------------------------

    mask_457 = df["lambda_air"] == 457.1
    if mask_457.any():
        X457 = df.loc[mask_457, ["Teff", "logg", "A(Mg)", "vmic"]]
        df.loc[mask_457, "aberr"] = model_457.predict(X457)

    # --------------------------------------------------
    # Unified model
    # --------------------------------------------------

    mask_uni = df["lambda_air"].isin(LINE_PHYSICS.keys())
    mask_uni &= ~mask_457

    if mask_uni.any():
        rows = df.loc[mask_uni].copy()

        # Inject line physics
        for col in ["lambda_vac", "elo_eV", "eup_eV", "deltaE_eV",
                    "lggf", "log_gamma_rad", "sigma", "alpha"]:
            rows[col] = rows["lambda_air"].map(
                lambda l: LINE_PHYSICS[l][col]
            )

        # IMPORTANT: lambda_air MUST be included
        Xuni = rows[[
            "Teff", "logg", "A(Mg)", "vmic",
            "lambda_air", "lambda_vac",
            "elo_eV", "eup_eV", "deltaE_eV",
            "lggf", "log_gamma_rad", "sigma", "alpha"
        ]]

        df.loc[mask_uni, "aberr"] = model_uni.predict(Xuni)

    # --------------------------------------------------
    # Drop unsupported lines
    # --------------------------------------------------

    unsupported = df["aberr"].isna()
    if unsupported.any():
        bad = np.sort(df.loc[unsupported, "lambda_air"].unique())
        print(f"Warning: unsupported Mg I lines dropped: {bad}")

    df = df.loc[~unsupported].drop(columns="lambda_air")
    df.to_csv(outfile, index=False)

    print(f"Saved abundance corrections → {outfile}")


# --------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 main_mg_aberr.py input output")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
