# 🌱 Green AI Leaderboard

Train a few classic ML models, measure the **carbon cost** of training each one,
and rank them by *accuracy vs carbon efficiency* — like a leaderboard, but for
the planet.

**The core question:** a bigger, more accurate model is usually also a dirtier
one. Is the extra accuracy worth the extra CO₂? This project makes that
tradeoff visible and measurable.

---

## What it does

1. **`train_models.py`** — loads a 4,000-image subset of MNIST (handwritten
   digits) and trains four models of increasing cost: Logistic Regression,
   a 100-tree Random Forest, a 400-tree Random Forest, and an RBF SVM. Every
   training run is wrapped in **CodeCarbon's** `OfflineEmissionsTracker`, which
   samples the machine's power draw, integrates it into energy (kWh), and
   multiplies by the country's grid carbon intensity to estimate CO₂ (kg).
   Results (accuracy, time, kWh, kg CO₂) are written to `results.csv`.
2. **`sci_score.py`** — reads `results.csv` and computes a simplified
   **Software Carbon Intensity** score for each model: `CO₂ / accuracy`, i.e.
   *"kilograms of CO₂ per accuracy point."* Low is good.
3. **`app.py`** — a small Streamlit dashboard showing the leaderboard table
   and an **accuracy vs CO₂ scatter plot**, so the tradeoff is visible at a
   glance.
4. **`.github/workflows/carbon-check.yml`** — a CI job that re-trains the
   models and uploads `results.csv` as an artifact on every push to `main`.

---

## Why software carbon tracking matters

Training a single large deep-learning model can emit as much CO₂ as several
cars over their lifetimes (Strubell et al., 2019). But you don't need a
data-center-sized model to care:

- The **Green Software Foundation (GSF)** — a non-profit backed by Microsoft,
  Google, Accenture, and others — publishes the **Software Carbon Intensity
  (SCI) specification**, standardised as **ISO/IEC 21031**. It measures the
  carbon a piece of software emits *per unit of useful work*, so engineers can
  compare the sustainability of one design vs another the same way they compare
  latency or accuracy.
- The **EU Corporate Sustainability Reporting Directive (CSRD)** now requires
  large companies to report their environmental impact — and **cloud/digital
  emissions are part of that disclosure**. "How much did our ML training
  emit?" is becoming a compliance question, not just a nice-to-have.

This project is the smallest possible version of that idea: track the cost,
make it comparable, and let the data speak.

---

## The SCI metric, explained

The GSF formula is:

```
SCI = (E × I) + M     per R
```

| symbol | meaning | in this project |
|---|---|---|
| `E` | energy used (kWh) | measured by CodeCarbon |
| `I` | grid carbon intensity (kg CO₂/kWh) | the `country_iso_code` you set |
| `M` | embodied carbon (hardware manufacturing) | **omitted** — needs factory data |
| `R` | functional unit of useful work | **one accuracy point** |

CodeCarbon already computes `E × I` for us, so the score simplifies to
`SCI = CO₂ / accuracy`. A model that is both accurate and cheap to train gets a
low (good) score.

> The full explanation lives as a code comment in `sci_score.py` — read it
> before an interview.

---

## How to run it locally

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models and measure carbon (downloads MNIST on first run)
python train_models.py

# 4. Compute SCI scores
python sci_score.py

# 5. Launch the dashboard
streamlit run app.py
```

> **Set your country** — `train_models.py` uses `COUNTRY_ISO_CODE = "IND"` to
> convert kWh into CO₂ (grid intensity depends on where you run). Change it to
> your own [ISO 3166-1 alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3)
> code (e.g. `"USA"`, `"FRA"`, `"DEU"`).

> **GitHub Actions** — push this repo to GitHub with `main` as the default
> branch and the `carbon-check` workflow runs automatically. Download
> `results.csv` from the run's *Artifacts* section.

---

## Results from my test run

Hardware: laptop CPU (8 cores, no GPU) · country grid: `IND` · 4,000 MNIST images
(3,000 train / 1,000 test).

| model | accuracy | time (s) | energy (kWh) | CO₂ (kg) | SCI (CO₂ / acc.) |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.892 | 1.08 | 7.2e-06 | 5.1e-06 | 5.8e-06 |
| random_forest_100 | 0.948 | 1.86 | 9.9e-06 | 7.0e-06 | 7.4e-06 |
| random_forest_400 | 0.950 | 7.70 | 3.8e-05 | 2.7e-05 | 2.8e-05 |
| svm_rbf | **0.957** | 0.92 | 5.7e-06 | 4.1e-06 | **4.3e-06** |

### Findings

- **Accuracy winner:** the **SVM** (95.7%), closely followed by the 400-tree
  forest (95.0%).
- **Carbon-efficiency winner:** the **SVM** again (best SCI) — on *this*
  dataset it's both the most accurate *and* one of the cheapest to train.
- **The stand-out waste:** the **400-tree forest emits ~3.8× the CO₂ of the
  100-tree forest for only +0.2% accuracy**. That is exactly the kind of
  "is it worth it?" question this project is designed to expose.
- **A pleasant surprise:** the SVM's reputation for being expensive doesn't
  show here — its quadratic cost only dominates on much larger datasets. At
  4,000 samples it trains in under a second. (Try raising `N_SAMPLES` in
  `train_models.py` and watch the SVM take over as the carbon hog.)

### Caveats

- Absolute emissions are **tiny** (micrograms of CO₂) because these are small
  models trained for seconds on a laptop. The value is in the **relative**
  comparison, not the absolute number.
- CO₂ is an **estimate**: CodeCarbon samples power, doesn't measure every
  electron. The estimator used, the sampling rate, the hardware, and the grid
  intensity all affect the numbers. Results will differ on your machine and in
  CI — treat them as a comparison tool, not a billing meter.
- Embodied carbon (manufacturing the hardware) is **excluded** — the full SCI
  spec includes it; we simplified.

---

## Project structure

```
green-ai-leaderboard/
├── train_models.py          # trains models, measures carbon, writes results.csv
├── sci_score.py             # computes the SCI score and appends it
├── app.py                   # Streamlit dashboard
├── results.csv              # committed snapshot from my run
├── requirements.txt
├── .github/workflows/
│   └── carbon-check.yml     # CI: re-train + upload results on every push
└── emissions/               # CodeCarbon per-run logs (git-ignored)
```

---

## Ideas to extend it

- Add a deep-learning model (PyTorch) and see its carbon footprint vs classic ML.
- Run on a GPU (Colab/Kaggle free tier) and compare CPU vs GPU training.
- Report the CI artifact back into the dashboard so the leaderboard updates
  automatically on every push.
- Try the full [SCI spec](https://sci.greensoftware.foundation) including
  embodied carbon.
