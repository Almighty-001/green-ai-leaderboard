"""
app.py
======
Streamlit dashboard for the Green AI Leaderboard.

Reads results.csv (produced by train_models.py + sci_score.py) and
shows:
  1. Key numbers at a glance (best accuracy, lowest CO2, best SCI).
  2. A table of every model: accuracy, CO2, energy, SCI.
  3. A scatter plot: accuracy (x-axis) vs CO2 emitted (y-axis), so the
     accuracy-vs-carbon tradeoff is visible at a glance.

Run it with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
csv_path = Path("results.csv")
if not csv_path.exists():
    st.error("results.csv not found. Run `python train_models.py` first.")
    st.stop()

df = pd.read_csv(csv_path)

# sci_score.py normally adds this column. Compute it here just in case
# it wasn't run, so the dashboard always works.
if "sci_score" not in df.columns:
    df["sci_score"] = df["co2_kg"] / (df["accuracy"] + 1e-12)

# ------------------------------------------------------------------
# 2. Page header
# ------------------------------------------------------------------
st.set_page_config(page_title="Green AI Leaderboard", layout="wide")
st.title("🌱 Green AI Leaderboard")
st.markdown(
    "How does each model's **accuracy** compare to the **carbon cost** of "
    "training it? Lower SCI score = more carbon-efficient per accuracy point."
)

# ------------------------------------------------------------------
# 3. Key numbers at a glance
# ------------------------------------------------------------------
best_accuracy = df.loc[df["accuracy"].idxmax()]
lowest_co2 = df.loc[df["co2_kg"].idxmin()]
best_sci = df.loc[df["sci_score"].idxmin()]

c1, c2, c3 = st.columns(3)
c1.metric("Best accuracy", f"{best_accuracy['model']} · {best_accuracy['accuracy']:.3f}")
c2.metric("Lowest CO₂", f"{lowest_co2['model']} · {lowest_co2['co2_kg']:.2e} kg")
c3.metric("Best SCI (carbon-efficient)", f"{best_sci['model']} · {best_sci['sci_score']:.2e}")

# ------------------------------------------------------------------
# 4. Results table
# ------------------------------------------------------------------
st.subheader("Results table")

# Friendly column order and labels for display only.
display_cols = {
    "model": "Model",
    "accuracy": "Accuracy",
    "co2_kg": "CO₂ (kg)",
    "energy_kwh": "Energy (kWh)",
    "training_time_s": "Train time (s)",
    "sci_score": "SCI score",
}
table = df[list(display_cols)].rename(columns=display_cols)
st.dataframe(table, hide_index=True, width="stretch")

# ------------------------------------------------------------------
# 5. Scatter plot: accuracy vs CO2
# ------------------------------------------------------------------
st.subheader("Accuracy vs CO₂ — the tradeoff")
st.markdown(
    "The ideal model lives in the **bottom-right** corner: high accuracy "
    "and low emissions."
)

st.scatter_chart(
    df,
    x="accuracy",
    y="co2_kg",
    color="model",      # one color per model
    size="energy_kwh",  # bubble size = energy used
) 
