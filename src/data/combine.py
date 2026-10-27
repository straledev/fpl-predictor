from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAASTAV = ROOT / "data" / "raw" / "vaastav"
PROCESSED = ROOT / "data" / "processed"

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
BASE_COLS = [
    "element", "name", "fixture", "round", "kickoff_time", "was_home",
    "opponent_team", "team_h_score", "team_a_score", "minutes", "total_points",
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
    "penalties_missed", "penalties_saved", "saves", "yellow_cards", "red_cards",
    "bonus", "bps", "influence", "creativity", "threat", "ict_index",
    "value", "selected", "transfers_in", "transfers_out", "transfers_balance",
]
EXTRA_COLS = [
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "starts", "xP",
]


def add_match_team(df: pd.DataFrame) -> pd.DataFrame:
    """Tim igraca u svakom mecu, izveden iz protivnika (radi i za transfere)."""
    opp = (
        df.groupby(["season", "fixture", "was_home"])["opponent_team"]
          .first().unstack()
          .rename(columns={True: "away_team", False: "home_team"})
          .reset_index()
    )
    df = df.merge(opp, on=["season", "fixture"], how="left")
    df["team"] = np.where(df["was_home"], df["home_team"], df["away_team"])
    return df.drop(columns=["home_team", "away_team"])


def load_season(season: str) -> pd.DataFrame:
    gw = pd.read_csv(VAASTAV / f"{season}.csv", low_memory=False)
    cols = [c for c in BASE_COLS + EXTRA_COLS if c in gw.columns]
    gw = gw[cols].copy()

    players = pd.read_csv(VAASTAV / "players" / f"{season}.csv", low_memory=False)
    players = players[["id", "code", "element_type"]].rename(columns={"id": "element"})
    gw = gw.merge(players, on="element", how="left")

    gw["position"] = gw["element_type"].map(POSITIONS)
    gw["season"] = season
    return gw[gw["position"].notna()]


def main() -> None:
    seasons = sorted(p.stem for p in VAASTAV.glob("*.csv"))
    df = pd.concat([load_season(s) for s in seasons], ignore_index=True)

    df = add_match_team(df)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df = df.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "all_seasons_gw.csv"
    df.to_csv(out, index=False)

    print(f"Sacuvano {len(df)} redova, {df.shape[1]} kolona u {out}")
    print("Redova bez tima:", df["team"].isna().sum())
    print("Redova bez code:", df["code"].isna().sum())
    print("Razlicitih igraca kroz sve sezone:", df["code"].nunique())
    print(df.groupby("season")["team"].nunique())


if __name__ == "__main__":
    main()