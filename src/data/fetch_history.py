from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "vaastav"
PLAYERS_DIR = OUT_DIR / "players"


def read_csv_url(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(url, encoding="latin-1", low_memory=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAYERS_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        try:
            gw = read_csv_url(f"{BASE_URL}/{season}/gws/merged_gw.csv")
            players = read_csv_url(f"{BASE_URL}/{season}/players_raw.csv")
        except Exception as e:
            print(f"{season}: GRESKA ({e})")
            continue
        gw.to_csv(OUT_DIR / f"{season}.csv", index=False)
        players.to_csv(PLAYERS_DIR / f"{season}.csv", index=False)
        print(f"{season}: {len(gw)} redova, {len(players)} igraca")


if __name__ == "__main__":
    main()