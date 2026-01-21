from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

from src.data import find_project_root


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "plot"


def _get_feature_importances(model) -> np.ndarray:
    """
    Extract feature importance for a fitted model.

    Supported:
    - CatBoost: model.get_feature_importance() or model.feature_importances_
    - sklearn-style: model.feature_importances_

    Raises:
        AttributeError if importance is not available.
    """
    if hasattr(model, "get_feature_importance"):
        imp = model.get_feature_importance()
        return np.asarray(imp, dtype=float)

    if hasattr(model, "feature_importances_"):
        imp = getattr(model, "feature_importances_")
        return np.asarray(imp, dtype=float)

    raise AttributeError(
        "Model does not expose feature importances. "
        "Expected `feature_importances_` (sklearn-style) or "
        "`get_feature_importance()` (CatBoost)."
    )


def visualize_importance(
    models: Sequence[object],
    feature_cols: Sequence[str],
    title: str,
    top: int = 20,
    out_dir: str | Path | None = None,
    filename: str | None = None,
    dpi: int = 200,
) -> Path:
    """
    Save a barplot of feature importance across CV folds.

    Args:
        models: Sequence of fitted fold models.
        feature_cols: Feature column names in the same order as training matrix.
        title: Plot title (also used for default filename).
        top: Number of top features (by importance) to display per fold.
        out_dir: Output directory. Defaults to <repo>/artifacts/feature_importance.
        filename: Optional output filename (e.g. "catboost.png"). If omitted,
            uses a slugified version of `title`.
        dpi: Figure DPI.

    Returns:
        Path to the saved image.
    """
    # Local imports so this module can be imported without plotting installed.
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
        raise TypeError("`models` must be a sequence of fitted models (e.g. list).")

    feature_importance = pd.DataFrame()
    for i, model in enumerate(models):
        importances = _get_feature_importances(model)
        if len(importances) != len(feature_cols):
            raise ValueError(
                f"Importance length mismatch for fold {i}: "
                f"got {len(importances)} importances, expected {len(feature_cols)} "
                "to match `feature_cols`."
            )

        _df = pd.DataFrame({"importance": importances, "feature": list(feature_cols)})
        _df["fold"] = i
        _df = _df.sort_values("importance", ascending=False).head(int(top))
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)

    feature_importance = feature_importance.sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance,
        color="skyblue",
        errorbar="sd",
    )
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.title(f"{title} Feature Importance [Top {top}]", fontsize=18)
    plt.grid(True, axis="x")

    project_dir = find_project_root()
    out_dir_path = Path(out_dir) if out_dir is not None else project_dir / "artifacts" / "feature_importance"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    out_name = filename if filename is not None else f"{_slugify(title)}_top{int(top)}.png"
    out_path = out_dir_path / out_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close()
    return out_path


def visualize_importance_from_importances(
    importances_per_fold: Sequence[np.ndarray],
    feature_cols: Sequence[str],
    title: str,
    top: int = 20,
    out_dir: str | Path | None = None,
    filename: str | None = None,
    dpi: int = 200,
) -> Path:
    """
    Save a barplot of feature importance across CV folds from precomputed importances.

    This is useful for models that don't expose `feature_importances_`
    (e.g. HistGradientBoostingClassifier), where you may compute permutation
    importance per fold.
    """
    # Local imports so this module can be imported without plotting installed.
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_importance = pd.DataFrame()
    for i, importances in enumerate(importances_per_fold):
        importances = np.asarray(importances, dtype=float)
        if len(importances) != len(feature_cols):
            raise ValueError(
                f"Importance length mismatch for fold {i}: "
                f"got {len(importances)} importances, expected {len(feature_cols)} "
                "to match `feature_cols`."
            )

        _df = pd.DataFrame({"importance": importances, "feature": list(feature_cols)})
        _df["fold"] = i
        _df = _df.sort_values("importance", ascending=False).head(int(top))
        feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)

    feature_importance = feature_importance.sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=feature_importance,
        color="skyblue",
        errorbar="sd",
    )
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.title(f"{title} Feature Importance [Top {top}]", fontsize=18)
    plt.grid(True, axis="x")

    project_dir = find_project_root()
    out_dir_path = (
        Path(out_dir)
        if out_dir is not None
        else project_dir / "artifacts" / "feature_importance"
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    out_name = filename if filename is not None else f"{_slugify(title)}_top{int(top)}.png"
    out_path = out_dir_path / out_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close()
    return out_path

