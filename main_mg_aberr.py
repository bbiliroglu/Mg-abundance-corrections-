#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mg I abundance-correction tool
Input: CSV or XLSX file
Output: CSV file with abundance correction (aberr)

Usage:
    python3 main_mg_aberr.py input_file output_file
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# Load trained models
# -------------------------------------------------
MODEL_DIR = Path("models")
MODEL_457 = MODEL_DIR / "mlp_pipeline_457nm_aberr.joblib"
MODEL_MULTI = MODEL_DIR / "mlp_pipeline_multiline_aberr.joblib"

model_457 = joblib.load(MODEL_457)
model_multi = joblib.load(MODEL_MULTI)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_input_table(fname: str) -> pd.DataFrame:
    """Load CSV or XLSX input file"""
    if fname.lower().endswith(".xlsx"):
        df = pd.read_excel(fname)
    elif fname.lower().endswith(".csv"):
        df = pd.read_csv(fname)
    else:
        raise ValueError("Input file must be .csv or .xlsx")
    return df


def validate_columns(df: pd.DataFrame):
    required = {"Teff", "logg", "A(Mg)", "vmic", "line"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# -------------------------------------------------
# Main logic
# -------------------------------------------------
def main(infile: str, outfile: str):

    df = load_input_table(infile)
    validate_columns(df)

    X = df[["Teff", "logg", "A(Mg)", "vmic"]].apply(pd.to_numeric, errors="coerce")
    line = pd.to_numeric(df["line"], errors="coerce")

    aberr = np.zeros(len(df))

    # Dedicated 457.1 nm model
    mask_457 = np.isclose(line, 457.1, atol=0.2)
    if mask_457.any():
        aberr[mask_457] = model_457.predict(X.loc[mask_457])

    # Multiline model
    mask_other = ~mask_457
    if mask_other.any():
        aberr[mask_other] = model_multi.predict(
            pd.concat([X.loc[mask_other], line.loc[mask_other]], axis=1)
        )

    df["aberr"] = aberr
    df.to_csv(outfile, index=False)

    print(f"Saved output → {outfile}")


# -------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 main_mg_aberr.py input_file output_file")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
