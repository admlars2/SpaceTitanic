## Spaceship Titanic (Kaggle)

End-to-end solution for the Kaggle **Spaceship Titanic** classification problem: predict whether each passenger was **Transported**.

This repo focuses on:
- **Leakage-safe cross-validation** using passenger groups derived from `PassengerId`
- Feature engineering around **Cabin**, **spend behavior**, and **group consistency**
- Two strong tabular baselines: **CatBoost** and **HistGradientBoosting**

### Approach (high level)

- **Goal**: predict `Transported` (binary) from passenger demographics, cabin assignment, and onboard spending.
- **Core idea**: passengers share a **group id** (the `PassengerId` prefix before `_`). Those groups correlate strongly with outcomes, so we:
  - validate with **GroupKFold** (groups never split across folds)
  - use group-level **modes** to impute categorical fields (HomePlanet, Destination, Cabin deck/side)

### Feature engineering

Feature engineering lives in `src/features.py` and is applied consistently to train/test.

- **Passenger groups**
  - `GroupId`: parsed from `PassengerId` (`####_##` → group prefix)
  - `GroupSize`, `GroupSizeCapped`, `IsGroupSize6Plus`
- **Cabin parsing**
  - `CabinDeck`, `CabinNum`, `CabinSide` from `Cabin` (`Deck/Num/Side`)
  - `CabinMissing`
  - `CabinNum_bin`: binned cabin number (bin edges computed from combined train+test)
- **Group-aware imputation**
  - For `HomePlanet`, `Destination`, `CabinDeck`, `CabinSide`:
    - fill missing values with the **mode within GroupId** when available
    - add `*_GroupMode` and `*_GroupImputed` indicators
- **Age features**
  - `AgeMissing`, simple boolean flags (`IsNewborn`, `IsToddler`, `IsAgeLt18`)
  - `AgeBin` (bucketed with `src/settings.py:AGE_BINS`)
- **Spending features**
  - Fill missing spend columns with 0 for aggregations
  - `TotalExpenses`, `TotalExpenses_log`
  - `NumAmenitiesUsed`, `NoSpend`, `SpendMissingCount`
  - per-amenity `*_log` features (`log1p`)
- **CryoSleep imputation from spend**
  - If `CryoSleep` is missing:
    - set to `"False"` when total spend > 0
    - set to `"True"` when total spend == 0
  - `CryoSleepImputed` flag

### CV strategy (important)

Validation uses **GroupKFold** where the group is the `PassengerId` prefix (`GroupId`). This avoids leakage where members of the same booking/group appear in both train and validation folds.

- **Splitter**: `sklearn.model_selection.GroupKFold`
- **Metric**: accuracy (matches Kaggle evaluation)
- **Config**: `src/settings.py` (`N_SPLITS`, `RANDOM_SEED`)

### Models included

Two runnable training scripts are provided:

- **CatBoost** (`src/model_catboost.py`)
  - Handles categorical features directly via CatBoost `Pool(cat_features=...)`
  - Hyperparameters in `src/settings.py:CATBOOST_PARAMS`
  - Also produces model-based feature importance plots

- **HistGradientBoostingClassifier** (`src/model_gradientboost.py`)
  - Uses integer-encoded categorical columns (see `encode_features`)
  - Hyperparameters in `src/settings.py:GRADIENTBOOST_PARAMS`
  - Produces **permutation importance** plots (since HGB doesn’t expose built-in importances)

### Results

- **Local CV**: the scripts print fold accuracies and mean ± std across folds.
- **Kaggle public LB**: _fill in your best submission score here_
- **Kaggle rank**: _fill in your rank here_

If you want, tell me your best Kaggle score/rank and I’ll wire it into this section cleanly.

### What I learned (takeaways)

- **Group leakage is real**: splitting passengers from the same booking across folds inflates CV and leads to LB disappointment. GroupKFold made validation trustworthy.
- **Spend ↔ CryoSleep is the strongest signal**: deriving `NoSpend`, `TotalExpenses_log`, and imputing `CryoSleep` from spend improved stability and accuracy.
- **Simple, robust transforms beat fancy tricks**: `log1p` on spending and coarse bins for age/cabin numbers were high ROI.
- **CatBoost is a great tabular baseline**: strong performance with minimal preprocessing and good behavior on mixed types.

### Repository layout

- `data/`
  - `train.csv`, `test.csv`, `sample_submission.csv`
- `src/`
  - `data.py`: loading + group/cabin bin helpers
  - `features.py`: feature engineering + encoding helper
  - `model_catboost.py`: CatBoost training, GroupKFold CV, submission writer
  - `model_gradientboost.py`: HGB training, GroupKFold CV, submission writer
  - `model_viz.py`: feature importance plotting utilities
  - `settings.py`: constants + hyperparameters
- `artifacts/feature_importance/`: saved feature importance plots
- `submission.csv`: latest generated submission

### Setup

This project uses **Poetry** and Python **3.11+** (see `pyproject.toml`).

```bash
poetry install
```

### Data

Download the competition data from Kaggle and place files under `data/`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### Run (generate `submission.csv`)

CatBoost:

```bash
poetry run python src/model_catboost.py
```

HistGradientBoosting:

```bash
poetry run python src/model_gradientboost.py
```

Both scripts will:
- print **GroupKFold CV** fold accuracies + mean ± std
- write `submission.csv` in the repo root
- write feature-importance plots to `artifacts/feature_importance/`

### Notes / next improvements

- Add calibrated probabilities + threshold tuning (accuracy objective) on grouped CV.
- Try light ensembling (CatBoost + HGB average vote).
- Evaluate whether group-mode imputation helps/hurts per column (ablation).
