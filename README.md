# Indian Rainfall Prediction - HackSphere, Aayam XI

Predicting next-day rainfall (mm) for Indian weather stations from today's readings.
Problem 1 of HackSphere (DSAI Club, NIT Agartala). Metric: RMSE (lower is better).
Only the dataset provided by the organisers is used.

## Files

| File | What it is |
|---|---|
| `rainfall_prediction.ipynb` | Full pipeline: EDA, features, validation, blending, error analysis, final predictions |
| `submission.csv` | Predictions for every row of `test.csv` (`id, rainfall_next_day_mm`) |
| `validation_results.csv` | RMSE of each model on the 2022 and 2023 validation years |
| `experiments/tune.py` | Searches model settings on both validation years and writes the best ones into the notebook |
| `experiments/experiments.csv` | Every configuration tried, with its validation RMSE |
| `requirements.txt` | Python packages |

The data files are not in the repo. Put `train.csv`, `test.csv` and `sample_submission.csv` in a `data/` folder to run the notebook.

## What we found in the data

- **Train is 2015-2023, test is Jan 2024 - Feb 2025.** This is a forecasting problem, so we validate on later years than we train on, never with a random split.
- **In test, each station appears only every second day.** The day after a test row is never present, so every station-level lag feature uses offsets that exist in test too (2, 4, 6 days back).
- **About 400 stations report on every test date**, so rain at nearby stations on the same day can be used.
- **The target is heavy-tailed:** half of all days are dry, but some days get several hundred mm. More than half of the squared error comes from days with more than 50 mm of rain, so getting heavy-rain days right matters most.

## Approach

**Features**
- Today's weather: temperatures, temperature range, wind, pressure, today's rain.
- Calendar: day of year (sin/cos), month, season, state, station.
- Station history: rain 2, 4 and 6 days back, 14-day rolling mean, days since last rain, pressure and temperature change.
- Neighbouring stations: for each day, the 6 nearest stations within 250 km (haversine distance). Distance-weighted mean rain, max rain, share of wet neighbours, their pressure change, and the same values 2 days back.
- Regional rain today: state-wide and all-India mean rain and share of wet stations.
- Climatology: smoothed average next-day rain and rain probability per station, station-month and state-month, computed only from the training part of each split.

**Models** (blended)
- LightGBM with squared error
- LightGBM with Tweedie loss (suited to a target with many zeros and a long tail)
- CatBoost
- Two-stage model: probability of rain x expected amount on wet days

Blend weights are fitted with non-negative least squares on the validation predictions, followed by one overall scale factor.
Model settings (tree size, leaf size, column sampling, regularisation, Tweedie power, CatBoost depth) and feature groups to drop are chosen by `experiments/tune.py`.

**Validation**
Two folds: train on everything before 2022 and predict 2022, then train on everything before 2023 and predict 2023.
Validation rows are thinned to every second day per station, the same way test is sampled.
A change is kept only if it lowers RMSE pooled over both years.

## Results

Validation RMSE in mm (lower is better). Each year is predicted by models trained only on earlier years.

| Model | 2022 | 2023 | Both years |
|---|---|---|---|
| baseline: tomorrow = today | 11.623 | 11.309 | 11.466 |
| l2 | 9.439 | 8.901 | 9.172 |
| tweedie | 9.510 | 8.908 | 9.212 |
| cat | 9.497 | 8.902 | 9.202 |
| twostage | 9.744 | 8.985 | 9.370 |
| blend | 9.435 | 8.867 | 9.154 |
| **final (submitted blend)** | **9.434** | **8.865** | **9.152** |

Model settings were chosen from 44 configurations tried on both validation years (`experiments/tune.py`, results in `experiments/experiments.csv`).

## How to run

```
pip install -r requirements.txt
python experiments/tune.py        # optional: re-run the settings search (20-60 min)
jupyter notebook rainfall_prediction.ipynb
```
Run all cells. A full notebook run takes roughly 10-30 minutes on a laptop CPU and writes `submission.csv`.
