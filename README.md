# Football Player Value Maximization

An exploratory football analytics and machine-learning project focused on understanding **what contributes to a player's development and market value**, and what a player may need to improve to move toward a **€100 million market valuation**.

The project brings together player-season statistics, valuation history, age and position information, and club/league context. It aims to turn those data into interpretable insights about player development rather than simply ranking players by current value.

> **Project status:** This README is based on the available project handoff notes. The recorded workspace audit dates from September 2026, and the current contents of the local `C:\football` folder have not been re-audited for this README. Treat model metrics and run commands as historical until reproduced locally.

## Objectives

- Build a longitudinal dataset of player-season records.
- Study relationships between performance, age, position, playing time, club/league context, and market value.
- Engineer comparable features such as per-90 statistics, age groups, season-to-season changes, and environmental indicators.
- Train and evaluate models for future market value, the probability of reaching a €100M valuation, and expected playing time.
- Explain model outputs so the results can help identify areas for player development.
- Provide an interactive dashboard for exploring players, predictions, and explanations.

## What the project is intended to answer

- Which performance and development indicators are associated with higher future market value?
- How does a player's value and performance change across seasons?
- Which areas could be worth investigating for a player's further development?
- How uncertain is a prediction, and which inputs most influence it?

A €100M valuation is an **ambitious modelling target, not a guaranteed outcome**. Market value is an estimate and is not the same as a transfer fee. Predictions should be interpreted alongside data quality, playing time, position, age, competition level, injuries and other context that may not be fully represented in the data.

## Data

The handoff notes describe these inputs under `data/`:

- `laliga_2022_2025.zip` — a ZIP containing player, season, career, team, valuation and player-season statistics tables in Parquet format.
- `SD-DB(PLAYERS) (1).csv` — player database snapshot.
- `player_valuations_full (1).csv` — valuation history.
- `team_full_dossier.csv` — player-level valuation/risk-related metrics.
- `club_financial_metrics.csv` — club-level financial indicators.
- `club_transfers (1).csv` — transfer records.
- `real_madrid_la_fabrica_deep_analysis_expanded_2006_2026 (1).xlsx` — academy-related analysis.
- `soccersolver_export_investigacion_2023-24_a_2025-26.sql` — a large SQL dump noted in the handoff as **not integrated into the recorded ingestion pipeline**.

Generated datasets noted in the handoff include:

- `data/trajectories_raw.parquet`
- `data/trajectories_engineered.parquet`
- `data/processed/trajectory_matrix.parquet`

The recorded engineered dataset contained **2,223 player-season rows and 442 columns**, covering 1,265 unique player IDs. These are historical audit figures and should be rechecked against the current files before being treated as current results.

**Data rights and privacy:** Verify the licence and permitted use of each dataset before redistributing data. Do not commit credentials, private data, or large raw database dumps to a public repository.

## Technical overview

The project is primarily written in Python. The handoff describes the following modules and responsibilities:

| File / module | Purpose described in the handoff |
|---|---|
| `run_pipeline.py` | Orchestrates ingestion, feature engineering, and output generation. |
| `src/data_loader.py` | Loads source tables and assembles player-season trajectories. |
| `src/feature_engineering.py` | Creates position, age, per-90, environment, and trajectory-momentum features. |
| `src/build_trajectory_matrix.py` | Adds target and feature aliases for the trajectory modelling dataset. |
| `src/models.py` | Contains model-training/evaluation logic for value, high-value classification, and minutes prediction. |
| `src/simulator.py` | Legacy counterfactual pathway simulation and scoring logic. |
| `app.py` | Streamlit dashboard for exploring players and model outputs. |
| `workspace_audit_script.py` | Audits files in the workspace. |

The intended modelling stages include:

1. **Future-value regression** — estimate a future market-value target.
2. **High-value classification** — estimate the probability of reaching a defined valuation threshold.
3. **Playing-time prediction** — estimate expected minutes or minutes share.
4. **Interpretability and exploration** — use feature explanations and dashboard views to investigate predictions.

The pathway simulator is part of the recorded legacy implementation. The current project emphasis is player performance and development toward higher market value; pathway simulations should be treated as supporting/legacy functionality until their assumptions and outputs are validated.

## Repository layout

The exact layout may differ in your current working copy. The handoff describes a structure similar to:

```text
football/
├── app.py
├── run_pipeline.py
├── workspace_audit_script.py
├── data/
│   ├── *.csv
│   ├── *.zip
│   ├── *.xlsx
│   ├── trajectories_raw.parquet
│   ├── trajectories_engineered.parquet
│   └── processed/
│       └── trajectory_matrix.parquet
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── build_trajectory_matrix.py
│   ├── models.py
│   └── simulator.py
├── models/
├── models_saved/
└── README.md
```

Model files (`.pkl`), generated datasets, audit logs, and raw source data may exist in the workspace but are not all required to explain the project. Check their provenance and whether they can be regenerated before publishing them.

## Setup

### 1. Create and activate a virtual environment

On Windows PowerShell, from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install dependencies

The handoff notes indicate usage of packages including `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `shap`, `streamlit`, `plotly`, `scipy`, and `openpyxl`. The notes did not identify a verified `requirements.txt` or `pyproject.toml`, so this list is **indicative, not a tested installation specification**.

If the repository contains a maintained dependency file, prefer it:

```powershell
pip install -r requirements.txt
```

If it does not, inspect the imports and project environment before installing dependencies. Once the project runs reproducibly, save the tested dependencies in a `requirements.txt` file.

## Running the project

The handoff describes `run_pipeline.py` as the pipeline entry point and `app.py` as the Streamlit dashboard. From the project root, the expected commands are:

```powershell
python run_pipeline.py
streamlit run app.py
```

These commands are based on the recorded project structure and **have not been tested against the current folder in this README**. If either command fails, check the current module names, dependencies, required data files, and relative paths before proceeding.

## Modelling and validation notes

- Use player-aware train/test splits or cross-validation so records from the same player do not leak across splits.
- Use only information available at the prediction date; exclude future values and future-derived features from model inputs.
- Report regression metrics such as MAE, RMSE and R², and classification metrics such as PR-AUC, ROC-AUC, F1 and calibration where appropriate.
- Compare against simple baselines and inspect performance across positions, age groups, seasons and value ranges.
- Check missing targets, class imbalance, data drift and unusual validation folds.
- Treat feature importance as evidence of model association, not proof that changing a feature will cause a player's value to rise.
- Validate recommendations with football context and domain expertise before drawing conclusions.

The handoff records promising historical model results but also notes an outlier validation fold, inconsistent saved model artefacts, incomplete pipeline rebuilding, and a demonstration that may use placeholder growth. Re-run the pipeline and evaluation before reporting any scores as current or final.

## Current limitations to verify

- The SQL dump was recorded as unused by the ingestion code at the time of the audit.
- No verified dependency lockfile was noted.
- Saved model artefacts may not all correspond to the same code/data version.
- Some target horizons have many missing values because future valuation observations are not available for every player-season.
- Historical market-value data and club/league context may be incomplete or inconsistent.
- Counterfactual pathway results require careful validation and should not be presented as causal predictions.

## Suggested next steps

1. Audit the current repository and confirm which files are active.
2. Make the pipeline reproducible from documented raw inputs.
3. Add and test a dependency file and clear run instructions.
4. Verify target construction, time-based leakage controls, player-aware validation, and baseline comparisons.
5. Reproduce model metrics and document them with the dataset/model version.
6. Make the dashboard clearly distinguish observed statistics, model estimates, and development suggestions.
7. Add a `.gitignore` for virtual environments, caches, logs, large raw files, generated datasets and model artefacts where appropriate.

## Disclaimer

This project is for research and educational analysis. It does not guarantee a player's future performance, career progression, or market valuation. Outputs depend on the quality, coverage and assumptions of the data and models.
