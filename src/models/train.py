import warnings
from pathlib import Path

import joblib
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
MODELS = ROOT / "models"

TARGET = "total_points"
TRAIN_SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20",
                 "2020-21", "2021-22", "2022-23", "2023-24"]
VAL_SEASON = "2024-25"
TEST_SEASON = "2025-26"  # ne dira se do zavrsnog testa

LGBM_PARAMS = dict(
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


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


def save_model(df: pd.DataFrame, seasons: list[str], features: list[str],
               n_estimators: int, name: str) -> None:
    """Trenira LightGBM na datim sezonama i cuva ga zajedno sa spiskom feature-a."""
    data = df[df["season"].isin(seasons)]
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **LGBM_PARAMS)
    model.fit(data[features], data[TARGET])

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "seasons": seasons}, MODELS / name)
    print(f"Sacuvan {name}: {len(data)} redova, {n_estimators} stabala")


def main() -> None:
    df = pd.read_csv(PROCESSED / "features.csv", low_memory=False)
    features = feature_columns(df)

    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"] == VAL_SEASON].copy()
    print(f"Trening: {len(train)} redova, validacija: {len(val)} redova, feature-a: {len(features)}")

    # 1. Baseline: prosek poslednjih 5 meceva (prvi mec sezone: prosek pozicije)
    pos_mean = train.groupby("position")[TARGET].mean()
    val["pred_baseline"] = val["total_points_roll5"].fillna(val["position"].map(pos_mean))

    # 2. FPL xP: skupljen POSLE kola, pa sadrzi buduce informacije
    val["pred_fpl_xp"] = val["xP"]

    # 3. Ridge regresija
    ridge = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    ridge.fit(train[features], train[TARGET])
    val["pred_ridge"] = ridge.predict(val[features])

    # 4. LightGBM sa early stopping-om na validaciji
    lgbm = lgb.LGBMRegressor(n_estimators=3000, **LGBM_PARAMS)
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

    # 5. Cuvanje modela sa brojem stabala koji je nasao early stopping
    best_iter = lgbm.best_iteration_
    print()
    save_model(df, TRAIN_SEASONS + [VAL_SEASON], features, best_iter,
               "lgbm_backtest.joblib")
    save_model(df, TRAIN_SEASONS + [VAL_SEASON, TEST_SEASON], features, best_iter,
               "lgbm_final.joblib")


if __name__ == "__main__":
    main()