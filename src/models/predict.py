from pathlib import Path

import joblib
import pandas as pd

from src.data.build import NUMERIC_COLS, POSITIONS, latest_raw_dir, load_json
from src.data.combine import add_match_team
from src.features.build_features import (
    add_context_features, add_player_features, add_team_features,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
SEASON = "2026-27"
PREV_SEASON = "2025-26"
REPORTS = ROOT / "reports"


def players_table(bootstrap: dict) -> pd.DataFrame:
    players = pd.DataFrame(bootstrap["elements"])[
        ["id", "code", "web_name", "element_type", "team", "now_cost",
         "status", "chance_of_playing_next_round"]
    ].rename(columns={"id": "element", "now_cost": "value"})
    players["position"] = players["element_type"].map(POSITIONS)

    teams = pd.DataFrame(bootstrap["teams"])[["id", "short_name"]]
    teams = teams.rename(columns={"id": "team", "short_name": "team_name"})
    players = players.merge(teams, on="team")
    return players[players["position"].notna()]


def history_rows(summaries: dict, players: pd.DataFrame) -> pd.DataFrame:
    rows = [r for s in summaries.values() for r in s["history"]]
    hist = pd.DataFrame(rows)
    hist = hist.merge(players[["element", "code", "position"]], on="element", how="inner")
    hist["is_future"] = False
    return hist


def future_rows(fixtures: list, players: pd.DataFrame, gw: int) -> pd.DataFrame:
    fx = pd.DataFrame([f for f in fixtures if f["event"] == gw])
    fx = fx[["id", "event", "team_h", "team_a", "kickoff_time"]].rename(
        columns={"id": "fixture", "event": "round"}
    )
    cols = ["element", "code", "team", "value", "position"]

    home = fx.merge(players[cols], left_on="team_h", right_on="team")
    home["was_home"], home["opponent_team"] = True, home["team_a"]

    away = fx.merge(players[cols], left_on="team_a", right_on="team")
    away["was_home"], away["opponent_team"] = False, away["team_h"]

    fut = pd.concat([home, away], ignore_index=True)
    fut = fut.drop(columns=["team_h", "team_a"])
    fut["is_future"] = True
    return fut


def previous_season_rows() -> pd.DataFrame:
    """Mecevi prosle sezone, da forma ne pocinje od nule u prvim kolima."""
    past = pd.read_csv(PROCESSED / "all_seasons_gw.csv", low_memory=False)
    past = past[past["season"] == PREV_SEASON].copy()
    past["is_future"] = False
    return past


def main() -> None:
    folder = latest_raw_dir()
    bootstrap = load_json(folder, "bootstrap_static")
    fixtures = load_json(folder, "fixtures")
    summaries = load_json(folder, "element_summaries")

    gw = next(e["id"] for e in bootstrap["events"] if e["is_next"])
    print(f"Podaci iz {folder.name}, predikcije za kolo {gw}")

    players = players_table(bootstrap)
    current = pd.concat([history_rows(summaries, players),
                         future_rows(fixtures, players, gw)], ignore_index=True)
    current["season"] = SEASON

    df = pd.concat([previous_season_rows(), current], ignore_index=True)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, format="ISO8601")
    df["is_future"] = df["is_future"].astype(bool)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # isti koraci kao u treningu
    df = add_match_team(df)
    df = add_player_features(df)
    df = add_team_features(df)
    df = add_context_features(df)

    bundle = joblib.load(MODELS / "lgbm_final.joblib")
    model, features = bundle["model"], bundle["features"]

    fut = df[df["is_future"]].copy()
    fut["ep_raw"] = model.predict(fut.reindex(columns=features))

    # dupli krug: saberi predikcije po igracu
    pred = fut.groupby("element", as_index=False)["ep_raw"].sum()
    pred = pred.merge(players, on="element")

    # povrede i suspenzije: FPL procenat sanse da igrac igra
    chance = pred["chance_of_playing_next_round"].fillna(100) / 100
    pred["ep"] = pred["ep_raw"] * chance
    pred["price"] = pred["value"] / 10

    pred = pred.sort_values("ep", ascending=False)
    out = PROCESSED / f"predictions_gw{gw}.csv"
    pred.to_csv(out, index=False)

    cols = ["web_name", "team_name", "position", "price", "status", "ep"]
    print(f"\nSacuvano u {out}\n")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"--- {pos} ---")
        print(pred[pred["position"] == pos][cols].head(5).round(2).to_string(index=False))
    REPORTS.mkdir(exist_ok=True)
    keep = ["element", "web_name", "team_name", "position", "price", "value",
            "status", "chance_of_playing_next_round", "ep"]
    pred[keep].assign(gw=gw).to_csv(REPORTS / "predictions_latest.csv", index=False)
    print(f"Predikcije za dashboard: {REPORTS / 'predictions_latest.csv'}")

if __name__ == "__main__":
    main()