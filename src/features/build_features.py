from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

PLAYER_KEYS = ["code"]               # isti igrac kroz sve sezone
SEASON_KEYS = ["season", "element"]  # igrac unutar jedne sezone
TEAM_KEYS = ["season", "team"]
WINDOWS = [3, 5]
PLAYER_STATS = [
    "total_points", "minutes", "played", "played60",
    "goals_scored", "assists", "bonus", "bps", "ict_index",
    "clean_sheets", "saves",
    "expected_goals", "expected_assists", "expected_goal_involvements",
]
TEAM_STATS = ["gf", "ga", "xgf", "xga"]


def prev_rolling_mean(df: pd.DataFrame, keys: list[str], col: str, window: int) -> pd.Series:
    """Prosek poslednjih `window` meceva PRE tekuceg (shift(1) sprecava curenje)."""
    groups = [df[k] for k in keys]
    prev = df[col].groupby(groups).shift(1)
    return (
        prev.groupby(groups)
            .rolling(window, min_periods=1).mean()
            .reset_index(level=list(range(len(keys))), drop=True)
    )


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(PLAYER_KEYS + ["kickoff_time"]).copy()
    df["played"] = (df["minutes"] > 0).astype(int)
    df["played60"] = (df["minutes"] >= 60).astype(int)

    # forma kroz sezone: prvi mec sezone vidi kraj prosle sezone
    for col in PLAYER_STATS:
        for w in WINDOWS:
            df[f"{col}_roll{w}"] = prev_rolling_mean(df, PLAYER_KEYS, col, w)

    df["matches_before"] = df.groupby(SEASON_KEYS).cumcount()
    df["minutes_last"] = df.groupby(PLAYER_KEYS)["minutes"].shift(1)
    df["days_since_last"] = df.groupby(PLAYER_KEYS)["kickoff_time"].diff().dt.days
    return df


def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Jedan red po timu po mecu, sa rolling formom tima."""
    cols = ["season", "fixture", "team", "opponent_team", "kickoff_time",
            "was_home", "team_h_score", "team_a_score"]
    t = df.drop_duplicates(["season", "fixture", "team"])[cols].copy()
    t["gf"] = np.where(t["was_home"], t["team_h_score"], t["team_a_score"])
    t["ga"] = np.where(t["was_home"], t["team_a_score"], t["team_h_score"])

    xg = (
        df.groupby(["season", "fixture", "team"])["expected_goals"]
          .sum(min_count=1).rename("xgf").reset_index()
    )
    t = t.merge(xg, on=["season", "fixture", "team"], how="left")
    xga = xg.rename(columns={"team": "opponent_team", "xgf": "xga"})
    t = t.merge(xga, on=["season", "fixture", "opponent_team"], how="left")

    t = t.sort_values(TEAM_KEYS + ["kickoff_time"]).reset_index(drop=True)
    roll_cols = []
    for col in TEAM_STATS:
        name = f"team_{col}_roll5"
        t[name] = prev_rolling_mean(t, TEAM_KEYS, col, 5)
        roll_cols.append(name)

    return t[["season", "fixture", "team"] + roll_cols]


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    team = build_team_features(df)
    df = df.merge(team, on=["season", "fixture", "team"], how="left")

    opp = team.rename(columns={"team": "opponent_team"})
    opp = opp.rename(columns={c: c.replace("team_", "opp_", 1)
                              for c in opp.columns if c.startswith("team_")})
    return df.merge(opp, on=["season", "fixture", "opponent_team"], how="left")


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_home"] = df["was_home"].astype(int)
    df["price"] = df["value"] / 10
    df["season_year"] = df["season"].str[:4].astype(int)
    positions = pd.get_dummies(df["position"], prefix="pos", dtype=int)
    return pd.concat([df, positions], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Spisak kolona koje model sme da koristi (samo informacije pre meca)."""
    suffixes = ("_roll3", "_roll5")
    singles = ["matches_before", "minutes_last", "days_since_last",
               "is_home", "price", "season_year"]
    return [
        c for c in df.columns
        if c.endswith(suffixes) or c.startswith("pos_") or c in singles
    ]


def main() -> None:
    df = pd.read_csv(PROCESSED / "all_seasons_gw.csv", low_memory=False)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)

    df = add_player_features(df)
    df = add_team_features(df)
    df = add_context_features(df)

    features = feature_columns(df)
    out = PROCESSED / "features.csv"
    df.to_csv(out, index=False)

    print(f"Sacuvano {len(df)} redova u {out}")
    print(f"Broj feature-a: {len(features)}")
    print("\nUdeo praznih vrednosti (top 10):")
    print(df[features].isna().mean().sort_values(ascending=False).head(10).round(3))


if __name__ == "__main__":
    main()