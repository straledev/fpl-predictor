from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

SEASON = "2025-26"
SAMPLE_SIZE = 5000
DEPENDENCE = ["minutes_roll3", "price", "opp_ga_roll5"]


def save(name: str) -> None:
    plt.gcf().savefig(REPORTS / name, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Sacuvano: reports/{name}")


def main() -> None:
    df = pd.read_csv(PROCESSED / "features.csv", low_memory=False)
    bundle = joblib.load(MODELS / "lgbm_backtest.joblib")
    model, features = bundle["model"], bundle["features"]

    season = df[(df["season"] == SEASON) & (df["minutes_roll5"] > 0)]
    sample = season.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    X = sample[features]

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    REPORTS.mkdir(exist_ok=True)

    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("What drives predicted FPL points (2025-26)")
    save("shap_summary.png")

    importance = (
        pd.Series(np.abs(sv.values).mean(axis=0), index=features, name="mean_abs_shap")
          .sort_values(ascending=False)
    )
    importance.to_csv(REPORTS / "shap_importance.csv")
    print("\nProsecan uticaj na predikciju (poeni):")
    print(importance.head(15).round(3).to_string())

    for col in DEPENDENCE:
        shap.plots.scatter(sv[:, col], show=False)
        plt.title(f"Effect of {col} on predicted points")
        save(f"shap_dependence_{col}.png")

    i = int(np.argmax(model.predict(X)))
    row = sample.loc[i]
    print(f"\nWaterfall za: {row['name']}, kolo {row['round']}, "
          f"predikcija {model.predict(X.iloc[[i]])[0]:.2f}, stvarno {row['total_points']}")
    shap.plots.waterfall(sv[i], max_display=12, show=False)
    save("shap_waterfall_top_player.png")

    print(f"\nProsecna predikcija (polazna tacka): {sv.base_values[0]:.2f} poena")


if __name__ == "__main__":
    main()