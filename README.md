# Jane Street Real-Time Market Data Forecasting (Kaggle Competition Project) 

This project investigates short-horizon financial prediction using machine learning and strict forward-looking validation on the Jane Street Real-Time Market Data Kaggle dataset.

The objective is to predict `responder_6` from anonymized market features while emphasizing robust validation design, leakage-aware modeling, and out-of-sample evaluation stability.

---

# Project Overview

This project was developed as a research-oriented financial machine learning workflow focused on predictive modeling under non-stationary time-series conditions.

The workflow focuses on:

* strict forward-looking validation
* leakage-aware feature engineering
* weighted financial evaluation metrics
* LightGBM baseline modeling
* robustness diagnostics
* temporal stability analysis
* lag-feature experimentation
* out-of-sample validation analysis

The main goal of the project is not to maximize leaderboard performance, but to study how methodological decisions affect predictive robustness in noisy financial datasets.

---

# Dataset

Dataset source:

Jane Street Real-Time Market Data Forecasting Competition (Kaggle)

https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting

Dataset characteristics:

* ~25 million observations
* 90+ anonymized market features
* multiple financial instruments (`symbol_id`)
* intraday timestamps (`date_id`, `time_id`)
* weighted evaluation metric

The dataset itself is NOT included in this repository because of size and licensing constraints.

Because the dataset is fully anonymized, the project focuses on statistical predictive structure rather than direct economic interpretation of the features.

---

# Research Pipeline

## 1. Data Diagnostics

Initial exploratory analysis included:

* missing-value analysis
* target distribution inspection
* feature sparsity analysis
* temporal coverage verification

---

## 2. Strict Time-Series Validation

Validation was performed using a strictly forward-looking split:

* training on earlier dates
* validation on future dates
* no random shuffling

This avoids temporal leakage and produces more realistic out-of-sample evaluation.

---

## 3. Baseline LightGBM Model

Main modeling choices:

* LightGBM regression
* weighted training
* competition-aligned weighted zero-mean R²
* native missing-value handling
* feature clipping for heavy-tailed distributions
* sparse-feature filtering

Validation performance:

| Model | Weighted R² |
|---|---|
| Initial baseline | ~0.0045 |
| Improved baseline | ~0.0086 |

The improvement mainly resulted from methodological corrections involving temporal sampling, validation design, leakage control, and feature preprocessing rather than increasing model complexity alone.

---

## 4. Robustness Diagnostics

Several robustness checks were implemented:

* leakage checks
* harder out-of-sample validation splits
* shuffled-target testing
* validation stability across dates
* overfitting diagnostics

Main observations:

* no evidence of major leakage
* positive out-of-sample predictive signal
* expected overfitting structure typical of financial ML problems
* reduced performance under harder validation regimes

---

## 5. Lag Feature Experiment

Additional lagged features were tested to evaluate short-term temporal persistence effects.

Lag features were constructed using:

* grouped symbol history
* strictly past observations only
* multiple lag horizons

This experiment evaluated whether short-term memory effects improved predictive performance.

---

# Key Results

Main observations obtained during validation:

| Metric | Value |
|---|---|
| Initial weighted R² | ~0.0045 |
| Improved weighted R² | ~0.0086 |
| Harder OOS weighted R² | ~0.0031 |
| Positive validation dates | ~85.7% |

The experiments highlighted the importance of validation methodology, temporal sampling design, and leakage control in financial prediction tasks.

---

# Research Observations

Several experiments illustrated the difficulty of extracting stable predictive structure from noisy and non-stationary financial datasets.

For example:

* stronger regularization slightly reduced predictive performance
* validation performance varied across time regimes
* harder out-of-sample splits significantly reduced weighted R²
* methodological choices impacted performance more than model complexity alone

These observations reinforced the importance of robust validation and careful research design in financial machine learning workflows.

---

# Research Limitations

This project uses a fully anonymized dataset, which prevents direct economic interpretation of the predictive structure.

As a result, the project focuses on predictive modeling methodology and statistical validation rather than realistic trading strategy construction or execution modeling.

---

# Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* LightGBM
* Matplotlib
* Scikit-learn

---

# Repository Structure

```text
jane-street-financial-forecasting-research/
│
├── notebooks/
├── models/
├── artifacts/
├── figures/
├── README.md
├── requirements.txt
└── .gitignore
