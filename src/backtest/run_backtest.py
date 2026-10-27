from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.models.train import evaluate
from src.optimize.squad import optimize_squad

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEASON = "2025-26"


def gameweek_table(season_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Jedan red po igracu po kolu (dupli krug: zbir oba meca)."""
    agg = (
        season_df.groupby(["round", "element"])
        .agg(
            name=("name", "first"),
            position=("position", "first"),
            team=("team", "first"),
            value=("value", "first"),
            ep=(pred_col, "sum"),
            points=("total_points", "sum"),
            minutes=("minutes", "sum"),
        )
        .reset_index()
    )
    agg["team_name"] = agg["team"].astype(str)
    return agg


def score_squad(squad: pd.DataFrame) -> float:
    starters = squad[squad["starter"]]
    return starters["points"].sum() + starters.loc[starters["captain"], "points"].sum()


def run_strategy(table: pd.DataFrame, label: str) -> pd.Series:
    scores, blanks = {}, {}
    for gw, players in table.groupby("round"):
        squad = optimize_squad(players)
        scores[gw] = score_squad(squad)
        starters = squad[squad["starter"]]
        blanks[gw] = (starters["minutes"] == 0).sum()
    print(f"{label}: prosecno {pd.Series(blanks).mean():.1f} od 11 startera bez minuta po kolu")
    return pd.Series(scores, name=label)


def main() -> None:
    df = pd.read_csv(PROCESSED / "features.csv", low_memory=False)
    season = df[df["season"] == SEASON].copy()

    # predikcije modela koji NIJE video 2025-26
    bundle = joblib.load(MODELS / "lgbm_backtest.joblib")
    season["pred_model"] = bundle["model"].predict(season[bundle["features"]])

    pos_mean = df[df["season"] != SEASON].groupby("position")["total_points"].mean()
    season["pred_baseline"] = season["total_points_roll5"].fillna(
        season["position"].map(pos_mean)
    )

    # zavrsni test modela na sezoni koju nikad nije video
    print("Zavrsni test na sezoni 2025-26:")
    test = {"Baseline (roll5)": evaluate(season, "pred_baseline"),
            "LightGBM": evaluate(season, "pred_model")}
    print(pd.DataFrame(test).T.round(3), "\n")

    model_table = gameweek_table(season, "pred_model")
    base_table = gameweek_table(season, "pred_baseline")
    oracle_table = model_table.assign(ep=model_table["points"])

    results = pd.concat([
        run_strategy(model_table, "Model"),
        run_strategy(base_table, "Baseline"),
        run_strategy(oracle_table, "Savrsen tim"),
    ], axis=1)
    results.index.name = "round"

    REPORTS.mkdir(exist_ok=True)
    results.to_csv(REPORTS / "backtest_2025-26.csv")

    print("\nUkupno poena u sezoni 2025-26:")
    print(results.sum().round(0))
    print("\nProsek po kolu:")
    print(results.mean().round(1))

    fig, ax = plt.subplots(figsize=(10, 5))
    results.cumsum().plot(ax=ax)
    ax.set_title("Backtest 2025-26: kumulativni poeni po strategiji")
    ax.set_xlabel("Kolo")
    ax.set_ylabel("Ukupno poena")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "backtest_2025-26.png", dpi=150)
    print(f"\nGrafik sacuvan u {REPORTS / 'backtest_2025-26.png'}")


if __name__ == "__main__":
    main()