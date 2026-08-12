"""
train_models.py
===============
Trains a few classic scikit-learn models on a subset of the MNIST
handwritten-digit dataset and measures the carbon cost of each
training run with CodeCarbon.

For every model we record:
  - test accuracy        (the quality score)
  - training time        (wall-clock seconds)
  - energy consumed      (kWh, estimated by CodeCarbon)
  - CO2 emitted          (kg CO2e, estimated by CodeCarbon)

Results are saved to results.csv so the leaderboard can compare
"how accurate" vs "how dirty" each model is.

Usage:
    python train_models.py
"""

import time
from pathlib import Path

import pandas as pd
from codecarbon import OfflineEmissionsTracker
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------

# ISO 3166-1 alpha-3 code of the country where training runs.
# CodeCarbon multiplies measured energy (kWh) by this country's grid
# carbon intensity to get CO2. Change it to your own country.
# We use the *offline* tracker so no network call is needed at runtime
# and every run is fully reproducible.
COUNTRY_ISO_CODE = "IND"

# How many MNIST images to use. A smaller subset keeps training fast on
# a CPU; a bigger one makes the differences between models larger.
N_SAMPLES = 4000

# ------------------------------------------------------------------
# 2. Load the data (MNIST: 28x28 = 784 pixel grayscale digit images)
# ------------------------------------------------------------------
# First run downloads ~12 MB from OpenML and caches it on disk, so
# later runs are instant.
X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
)
X = X[:N_SAMPLES]
y = y[:N_SAMPLES].astype(int)

# Hold out 25% of the data for testing. random_state fixes the split so
# every model is evaluated on the *same* test set - a fair race.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------------------------------------------
# 3. Define the models to compare.
# ------------------------------------------------------------------
MODELS = {
    # Cheap baseline: a linear model, fast to train, ~90% accurate.
    "logistic_regression": LogisticRegression(max_iter=200),
    # A 100-tree forest. Middle ground on both accuracy and cost.
    "random_forest_100": RandomForestClassifier(n_estimators=100, random_state=42),
    # A 400-tree forest: ~4x the trees, ~4x the energy. Diminishing
    # returns: it buys only a little extra accuracy.
    "random_forest_400": RandomForestClassifier(n_estimators=400, random_state=42),
    # SVM with an RBF kernel: usually the accuracy winner. Famous for
    # being expensive, but at 4,000 samples even SVM is cheap - its
    # quadratic cost only starts to bite on much larger datasets.
    "svm_rbf": SVC(kernel="rbf", gamma="scale"),
}

RESULTS = []

# CodeCarbon writes to an output dir; make sure it exists first.
Path("emissions").mkdir(exist_ok=True)

for name, model in MODELS.items():
    print(f"\n>>> Training {name} ...")

    # The context manager starts tracking on entry and stops on exit
    # (even if fit() raises). We time ONLY the .fit() call, so
    # training_time reflects the model, not CodeCarbon's bookkeeping.
    with OfflineEmissionsTracker(
        output_dir="emissions",
        output_file="emissions.csv",
        project_name=name,
        country_iso_code=COUNTRY_ISO_CODE,
        # Sample power every 0.5s. The default (15s) is far too slow
        # to catch a training run that lasts a few seconds.
        measure_power_secs=0.5,
        log_level=40,  # only show WARNING/ERROR from CodeCarbon
    ) as tracker:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

    # Measure quality on the untouched test set.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Pull the numbers out of the tracker. final_emissions is kg CO2e;
    # final_emissions_data.energy_consumed is kWh.
    co2_kg = tracker.final_emissions
    energy_kwh = tracker.final_emissions_data.energy_consumed

    print(
        f"    accuracy={accuracy:.4f}  time={training_time:.2f}s  "
        f"energy={energy_kwh:.6e} kWh  co2={co2_kg:.3e} kg"
    )

    RESULTS.append(
        {
            "model": name,
            "accuracy": round(accuracy, 4),
            "training_time_s": round(training_time, 3),
            "energy_kwh": round(energy_kwh, 10),
            "co2_kg": round(co2_kg, 10),
        }
    )

# ------------------------------------------------------------------
# 4. Save everything to results.csv (overwrite on each full run)
# ------------------------------------------------------------------
df = pd.DataFrame(RESULTS)
df.to_csv("results.csv", index=False)
print(f"\nSaved {len(df)} rows to results.csv.")
