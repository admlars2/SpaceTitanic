import numpy as np
import pandas as pd

from .settings import AGE_BINS


def make_features(
    df: pd.DataFrame,
    group_sizes: pd.Series,
    group_modes: dict[str, pd.Series],
    cabin_bins: np.ndarray,
) -> pd.DataFrame:
    df = df.copy()

    group_str = df["PassengerId"].astype("string").str.split("_").str[0]
    df["GroupId"] = pd.to_numeric(group_str, errors="coerce").astype("Int64")
    df["GroupSize"] = df["GroupId"].map(group_sizes).astype("Int64")
    df["GroupSize"] = df["GroupSize"].fillna(1).astype(int)
    df["GroupSizeCapped"] = df["GroupSize"].clip(upper=6)
    df["IsGroupSize6Plus"] = df["GroupSizeCapped"].gt(5).astype(int)

    cabin_parts = df["Cabin"].astype("string").str.split("/", expand=True)
    df["CabinMissing"] = df["Cabin"].isna().astype(int)
    df["CabinDeck"] = cabin_parts.get(0).astype("string")
    df["CabinSide"] = cabin_parts.get(2).astype("string")
    cabin_num = pd.to_numeric(cabin_parts.get(1), errors="coerce")
    df["CabinNum"] = cabin_num
    df["CabinNum_bin"] = pd.cut(
        cabin_num,
        bins=cabin_bins,
        right=False,
        include_lowest=True,
    ).astype("string")

    for col in [
        "CryoSleep",
        "VIP",
        "HomePlanet",
        "Destination",
        "CabinDeck",
        "CabinSide",
        "CabinNum_bin",
    ]:
        df[col] = df[col].astype("string").fillna("Missing")

    for col, mode_series in group_modes.items():
        group_mode = df["GroupId"].map(mode_series)
        group_mode = group_mode.astype("string")
        missing_mask = df[col].eq("Missing") | df[col].isna()
        df.loc[missing_mask, col] = group_mode
        df[f"{col}_GroupMode"] = group_mode.fillna("Missing")
        df[f"{col}_GroupImputed"] = (missing_mask & group_mode.notna()).astype(int)

    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["IsNewborn"] = df["Age"].le(1).fillna(False).astype(int)
    df["IsToddler"] = df["Age"].le(5).fillna(False).astype(int)
    df["IsAgeLt18"] = df["Age"].le(18).fillna(False).astype(int)
    df["AgeBin"] = (
        pd.cut(df["Age"], bins=AGE_BINS, right=False, include_lowest=True)
        .astype("string")
        .fillna("Missing")
    )

    expense_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["SpendMissingCount"] = df[expense_cols].isna().sum(axis=1).astype(int)
    filled = df[expense_cols].fillna(0)
    total_spend = filled.sum(axis=1)
    cryo_missing = df["CryoSleep"].isna() | df["CryoSleep"].eq("Missing")
    df.loc[cryo_missing & total_spend.gt(0), "CryoSleep"] = "False"
    df.loc[cryo_missing & total_spend.eq(0), "CryoSleep"] = "True"
    df["CryoSleepImputed"] = cryo_missing.astype(int)
    df["TotalExpenses"] = total_spend
    df["TotalExpenses_log"] = np.log1p(df["TotalExpenses"])
    df["NumAmenitiesUsed"] = filled.gt(0).sum(axis=1).astype(int)
    df["NoSpend"] = (df["TotalExpenses"] == 0).astype(int)
    for col in expense_cols:
        df[f"{col}_log"] = np.log1p(filled[col])

    # df["HomePlanet_Destination"] = (
    #     df["HomePlanet"] + "_" + df["Destination"]
    # ).fillna("Missing")
    # df["CabinDeck_Side"] = (df["CabinDeck"] + "_" + df["CabinSide"]).fillna(
    #     "Missing"
    # )
    # df["CryoSleep_VIP"] = (df["CryoSleep"] + "_" + df["VIP"]).fillna("Missing")
    # df["CryoSleep_NoSpend"] = (
    #     df["CryoSleep"] + "_" + df["NoSpend"].astype("string")
    # ).fillna("Missing")

    keep = [
        "CryoSleep",
        "VIP",
        "HomePlanet",
        "Destination",
        "CabinMissing",
        "CabinDeck",
        "CabinSide",
        "CabinNum",
        "CabinNum_bin",
        # "HomePlanet_Destination",
        # "CabinDeck_Side",
        # "CryoSleep_VIP",
        # "CryoSleep_NoSpend",
        "Age",
        "AgeMissing",
        "IsNewborn",
        "IsToddler",
        "IsAgeLt18",
        "AgeBin",
        "HomePlanet_GroupMode",
        "Destination_GroupMode",
        "CabinDeck_GroupMode",
        "CabinSide_GroupMode",
        "HomePlanet_GroupImputed",
        "Destination_GroupImputed",
        "CabinDeck_GroupImputed",
        "CabinSide_GroupImputed",
        "GroupSizeCapped",
        "IsGroupSize6Plus",
        "NoSpend",
        "SpendMissingCount",
        "CryoSleepImputed",
        "NumAmenitiesUsed",
        "TotalExpenses_log",
        "RoomService_log",
        "FoodCourt_log",
        "ShoppingMall_log",
        "Spa_log",
        "VRDeck_log",
        "GroupId",
    ]
    return df[keep]


def get_categorical_cols() -> list[str]:
    return [
        "CryoSleep",
        "VIP",
        "HomePlanet",
        "Destination",
        "CabinDeck",
        "CabinSide",
        "CabinNum_bin",
        # "HomePlanet_Destination",
        # "CabinDeck_Side",
        # "CryoSleep_VIP",
        # "CryoSleep_NoSpend",
        "AgeBin",
        "HomePlanet_GroupMode",
        "Destination_GroupMode",
        "CabinDeck_GroupMode",
        "CabinSide_GroupMode",
    ]


def encode_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

    for col in categorical_cols:
        combined[col] = combined[col].astype("category")
        if "Missing" not in combined[col].cat.categories:
            combined[col] = combined[col].cat.add_categories(["Missing"])
        combined[col] = combined[col].fillna("Missing").cat.codes

    numeric_cols = [col for col in combined.columns if col not in categorical_cols]
    for col in numeric_cols:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())
        combined[col] = combined[col].astype(float)

    train_encoded = combined.iloc[: len(train_features)].copy()
    test_encoded = combined.iloc[len(train_features) :].copy()
    categorical_idx = [train_encoded.columns.get_loc(col) for col in categorical_cols]
    return train_encoded, test_encoded, categorical_idx
