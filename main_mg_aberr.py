#!/usr/bin/env python3
"""
Mg I abundance corrections: 1D LTE – 3D NLTE

Usage:
    python3 main_mg_aberr.py input.csv output.csv
    python3 main_mg_aberr.py input.xlsx output.csv

Input columns (required):
    Teff, logg, A(Mg), vmic, line

Output:
    Same table + column:
        aberr  (dex)
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODELS_DIR = "models"

MODEL_457 = os.path.join(MODELS_DIR, "mlp_pipeline_457nm_aberr.joblib")
MODEL_UNIFIED = os.path.join(MODELS_DIR, "unified_mlp_pipeline.joblib")

SUPPORTED_LINES = np.array([416.7, 457.1, 470.3, 473.0, 516.7, 571.1])


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def read_input(fname):
    if fname.lower().endswith(".csv"):
        return pd.read_csv(fname)
    elif fname.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(fname)
    else:
        raise ValueError("Input file must be .csv or .xlsx")


def parse_line(val):
    """
    Parse wavelength column.
    Accepts:
        457.1
        "457.1"
        "457.1 nm"
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).lower().replace("nm", "").strip()
    return float(s)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main(infile, outfile):

    df = read_input(infile)

    required = ["Teff", "logg", "A(Mg)", "vmic", "line"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse wavelength column
    df["lambda_air"] = df["line"].map(parse_line).round(1)

    # Drop unsupported lines
    mask_supported = np.isin(df["lambda_air"], SUPPORTED_LINES)

    if not mask_supported.all():
        dropped = np.sort(df.loc[~mask_supported, "lambda_air"].unique())
        print(f"Warning: dropping unsupported Mg I lines (nm): {dropped}")

    df = df.loc[mask_supported].reset_index(drop=True)

    if df.empty:
        raise ValueError("No supported Mg I lines left after filtering.")

    # Prepare output column
    df["aberr"] = np.nan

    # Load models
    model_457 = joblib.load(MODEL_457)
    model_unified = joblib.load(MODEL_UNIFIED)

    # Features used by ML models
    Xcols = ["Teff", "logg", "A(Mg)", "vmic"]

    # 457.1 nm (dedicated model)
    mask_457 = df["lambda_air"] == 457.1
    if mask_457.any():
        X = df.loc[mask_457, Xcols]
        df.loc[mask_457, "aberr"] = model_457.predict(X)

    # Other supported lines (unified model)
    mask_other = ~mask_457
    if mask_other.any():
        X = df.loc[mask_other, Xcols].copy()
        X["lambda_air"] = df.loc[mask_other, "lambda_air"].values
        df.loc[mask_other, "aberr"] = model_unified.predict(X)

    # Save output
    df.drop(columns=["lambda_air"]).to_csv(outfile, index=False)
    print(f"Saved abundance corrections to {outfile}")


# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 main_mg_aberr.py input_file output_file")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
