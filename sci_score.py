"""
sci_score.py
============
Reads results.csv (produced by train_models.py) and computes a simple
Software Carbon Intensity (SCI) score for every model, then appends it
to results.csv as a new column.

Usage:
    python sci_score.py
"""

from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# WHAT IS SCI? (read this - you'll be asked about it in interviews)
# ------------------------------------------------------------------
# Software Carbon Intensity is a metric published by the Green
# Software Foundation (GSF) and standardised as ISO/IEC 21031. It
# measures how much carbon a piece of software emits *per unit of
# useful work it does* (its "functional unit", R). The full formula is:
#
#     SCI = (E * I) + M     per R
#
#     E = energy consumed by the software (kWh)   <- CodeCarbon measured this
#     I = carbon intensity of the electricity grid (kg CO2 / kWh)
#     M = embodied carbon: emissions from manufacturing the hardware
#         (chips, servers) that ran the software
#     R = the functional unit: one unit of useful output
#
# Here we use a *simplified* version that drops M (embodied carbon),
# which is much harder to measure and needs hardware-manufacturing
# data we don't have:
#
#     SCI = co2_kg / accuracy
#
# CodeCarbon already folded E * I together into co2_kg for us (it
# multiplied the kWh it measured by the grid intensity of the country
# we declared). And we use one "accuracy point" as our functional
# unit R - the useful output of a model is being accurate.
#
# So the score answers: "how many kg of CO2 does one point of accuracy
# cost?" A low (good) score means the model is accurate AND cheap to
# train. A high score means most of its carbon went nowhere useful.
#
# Reference: Green Software Foundation, "Software Carbon Intensity"
# (https://sci.greensoftware.foundation), published as ISO/IEC 21031.
# ------------------------------------------------------------------

csv_path = Path("results.csv")
if not csv_path.exists():
    raise SystemExit("results.csv not found. Run train_models.py first.")

df = pd.read_csv(csv_path)

# accuracy is a fraction like 0.95, so this is "kg CO2 per accuracy
# point". The tiny epsilon only guards against dividing by zero if a
# model ever scores 0.0.
df["sci_score"] = df["co2_kg"] / (df["accuracy"] + 1e-12)

# Round for readable output and save the column back to results.csv.
df["sci_score"] = df["sci_score"].round(10)
df.to_csv(csv_path, index=False)

columns = ["model", "accuracy", "co2_kg", "sci_score"]
print(df[columns].to_string(index=False))
print("\nLower SCI score = more carbon-efficient per accuracy point.\n")
