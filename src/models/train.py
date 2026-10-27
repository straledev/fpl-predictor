import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import feature_columns

warnings.filterwarnings("ignore", category=ConstantInputWarning)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

TARGET = "total_points"
TRAIN_SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20",
                 "2020-21", "2021-22", "2022-23", "2023-24"]
VAL_SEASON = "2024-25"  # 2025-26 cuvamo za zavrsni test


def evaluate(df: pd.DataFrame, pred_col: str) -> dict:
    df = df.dropna(subset=[pred_col])
    played = df[df["minutes"] > 0]
    spearman = (
        df.groupby("round")[[TARGET, pred_col]]
          .apply(lambda g: g[TARGET].corr(g[pred_col], method="spearman"))
          .mean()
    )
    return {
        "MAE": mean_absolute_error(df[TARGET], df[pred_col]),
        "RMSE": np.sqrt(mean_squared_error(df[TARGET], df[pred_col])),
        "Spearman po kolu": spearman,
        "MAE (samo igrali)": mean_absolute_error(played[TARGET], played[pred_col]),
        "Redova": len(df),
    }


def main() -> None:
    df = pd.read_csv(PROCESSED / "features.csv", low_memory=False)
    features = feature_columns(df)

    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"] == VAL_SEASON].copy()
    print(f"Trening: {len(train)} redova, validacija: {len(val)} redova, feature-a: {len(features)}")

    # 1. Baseline: prosek poslednjih 5 meceva (prvi mec sezone: prosek pozicije)
    pos_mean = train.groupby("position")[TARGET].mean()
    val["pred_baseline"] = val["total_points_roll5"].fillna(val["position"].map(pos_mean))

    # 2. Zvanicna FPL procena
    val["pred_fpl_xp"] = val["xP"]

    # 3. Ridge regresija
    ridge = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    ridge.fit(train[features], train[TARGET])
    val["pred_ridge"] = ridge.predict(val[features])

    # 4. LightGBM (sam obradjuje prazne vrednosti, nema potrebe za imputacijom)
    lgbm = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
    )
    lgbm.fit(
        train[features], train[TARGET],
        eval_set=[(val[features], val[TARGET])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    val["pred_lgbm"] = lgbm.predict(val[features])

    importance = pd.Series(lgbm.feature_importances_, index=features)
    print("\nNajvazniji feature-i:")
    print(importance.sort_values(ascending=False).head(15))

    results = {
        "Baseline (roll5)": evaluate(val, "pred_baseline"),
        "FPL xP (curenje, nije posteno)": evaluate(val, "pred_fpl_xp"),
        "Ridge": evaluate(val, "pred_ridge"),
        "LightGBM": evaluate(val, "pred_lgbm"),
    }
    print()
    print(pd.DataFrame(results).T.round(3))


if __name__ == "__main__":
    main()