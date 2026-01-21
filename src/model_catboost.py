import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
from src.features import get_categorical_cols, make_features
from src.model_viz import visualize_importance
from src.settings import CATBOOST_PARAMS, GROUP_MODE_COLS, N_SPLITS


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(**CATBOOST_PARAMS)


def main() -> None:
    train_df, test_df, project_dir = load_data()

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    group_sizes = compute_group_sizes(combined)
    group_modes = compute_group_modes(combined, GROUP_MODE_COLS)
    cabin_bins = compute_cabin_bins(combined)

    train_features = make_features(train_df, group_sizes, group_modes, cabin_bins)
    test_features = make_features(test_df, group_sizes, group_modes, cabin_bins)

    y = train_df["Transported"].astype(int).to_numpy()
    groups = train_features["GroupId"]

    X_train = train_features.drop(columns=["GroupId"])
    X_test = test_features.drop(columns=["GroupId"])

    categorical_cols = get_categorical_cols()
    categorical_idx = [X_train.columns.get_loc(col) for col in categorical_cols]

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("string").fillna("Missing")
        X_test[col] = X_test[col].astype("string").fillna("Missing")

    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
    for col in numeric_cols:
        if X_train[col].isna().any():
            median = X_train[col].median()
            X_train[col] = X_train[col].fillna(median)
            X_test[col] = X_test[col].fillna(median)

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_scores: list[float] = []
    fold_models: list[CatBoostClassifier] = []
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X_train, y, groups=groups), start=1
    ):
        model = build_model()
        train_pool = Pool(
            X_train.iloc[train_idx],
            y[train_idx],
            cat_features=categorical_idx,
        )
        val_pool = Pool(
            X_train.iloc[val_idx],
            y[val_idx],
            cat_features=categorical_idx,
        )
        model.fit(train_pool)
        preds = model.predict(val_pool).astype(int)
        acc = accuracy_score(y[val_idx], preds)
        fold_scores.append(acc)
        fold_models.append(model)
        print(f"Fold {fold} accuracy: {acc:.5f}")

    if fold_scores:
        mean_acc = float(np.mean(fold_scores))
        std_acc = float(np.std(fold_scores))
        print(f"Mean CV accuracy: {mean_acc:.5f} +/- {std_acc:.5f}")

    if fold_models:
        viz_path = visualize_importance(
            fold_models,
            list(X_train.columns),
            title="CatBoost",
            top=20,
        )
        print(f"Wrote: {viz_path}")

    final_model = build_model()
    train_pool = Pool(X_train, y, cat_features=categorical_idx)
    final_model.fit(train_pool)
    test_pool = Pool(X_test, cat_features=categorical_idx)
    test_preds = final_model.predict(test_pool).astype(bool)

    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_preds}
    )
    out_path = project_dir / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
