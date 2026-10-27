import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
NUMERIC_COLS = [
    "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
    "influence", "creativity", "threat", "ict_index",
]


def latest_raw_dir() -> Path:
    """Najnoviji folder sa datumom u data/raw."""
    dated = [p for p in RAW_ROOT.iterdir() if p.is_dir() and p.name[0].isdigit()]
    return max(dated)


def load_json(folder: Path, name: str):
    return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))


def build_current_season() -> pd.DataFrame:
    folder = latest_raw_dir()
    bootstrap = load_json(folder, "bootstrap_static")
    summaries = load_json(folder, "element_summaries")

    players = pd.DataFrame(bootstrap["elements"])[
        ["id", "web_name", "element_type", "team", "now_cost"]
    ]
    players["position"] = players["element_type"].map(POSITIONS)
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name"]].rename(
        columns={"id": "team", "name": "team_name"}
    )
    players = players.merge(teams, on="team")

    rows = [row for s in summaries.values() for row in s["history"]]
    gw = pd.DataFrame(rows)
    gw = gw.merge(players, left_on="element", right_on="id", how="left")

    for col in NUMERIC_COLS:
        if col in gw.columns:
            gw[col] = pd.to_numeric(gw[col], errors="coerce")

    return gw.sort_values(["element", "round"]).reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gw = build_current_season()
    out = PROCESSED_DIR / "current_season_gw.csv"
    gw.to_csv(out, index=False)
    print(f"Sacuvano {len(gw)} redova u {out}")
    print(gw[["web_name", "position", "team_name", "round", "minutes", "total_points"]].head())


if __name__ == "__main__":
    main()