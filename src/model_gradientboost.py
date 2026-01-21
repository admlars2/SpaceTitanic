import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import (
    compute_cabin_bins,
    compute_group_modes,
    compute_group_sizes,
    load_data,
)
from src.features import encode_features, get_categorical_cols, make_features
from src.model_viz import visualize_importance_from_importances
from src.settings import GROUP_MODE_COLS, GRADIENTBOOST_PARAMS, N_SPLITS, RANDOM_SEED


def build_model(categorical_idx: list[int]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        **GRADIENTBOOST_PARAMS,
        categorical_features=categorical_idx,
        random_state=RANDOM_SEED,
    )


def run_cv(
    X_train: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    categorical_idx: list[int],
) -> tuple[list[float], list[np.ndarray]]:
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_scores: list[float] = []
    fold_importances: list[np.ndarray] = []
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X_train, y, groups=groups), start=1
    ):
        model = build_model(categorical_idx)
        model.fit(X_train.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        acc = accuracy_score(y.iloc[val_idx], preds)
        fold_scores.append(acc)
        perm = permutation_importance(
            model,
            X_train.iloc[val_idx],
            y.iloc[val_idx],
            scoring="accuracy",
            n_repeats=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        fold_importances.append(perm.importances_mean.astype(float))
        print(f"Fold {fold} accuracy: {acc:.5f}")
    return fold_scores, fold_importances


def train_and_predict(
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_idx: list[int],
) -> np.ndarray:
    model = build_model(categorical_idx)
    model.fit(X_train, y)
    return model.predict(X_test).astype(bool)


def main() -> None:
    train_df, test_df, project_dir = load_data()

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    group_sizes = compute_group_sizes(combined)
    group_modes = compute_group_modes(combined, GROUP_MODE_COLS)
    cabin_bins = compute_cabin_bins(combined)

    train_features = make_features(train_df, group_sizes, group_modes, cabin_bins)
    test_features = make_features(test_df, group_sizes, group_modes, cabin_bins)

    y = train_df["Transported"].astype(int)
    groups = train_features["GroupId"]

    X_train = train_features.drop(columns=["GroupId"])
    X_test = test_features.drop(columns=["GroupId"])

    categorical_cols = get_categorical_cols()
    X_train_enc, X_test_enc, categorical_idx = encode_features(
        X_train, X_test, categorical_cols
    )

    fold_scores, fold_importances = run_cv(X_train_enc, y, groups, categorical_idx)
    if fold_scores:
        mean_acc = float(np.mean(fold_scores))
        std_acc = float(np.std(fold_scores))
        print(f"Mean CV accuracy: {mean_acc:.5f} +/- {std_acc:.5f}")

    if fold_importances:
        viz_path = visualize_importance_from_importances(
            fold_importances,
            list(X_train_enc.columns),
            title="HistGradientBoosting (Permutation)",
            top=20,
        )
        print(f"Wrote: {viz_path}")

    test_preds = train_and_predict(X_train_enc, y, X_test_enc, categorical_idx)
    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_preds}
    )
    out_path = project_dir / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
