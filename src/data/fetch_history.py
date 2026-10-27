from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "vaastav"


def fetch_season(season: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
    try:
        return pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(url, encoding="latin-1")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        try:
            df = fetch_season(season)
        except Exception as e:
            print(f"{season}: GRESKA ({e})")
            continue
        df.to_csv(OUT_DIR / f"{season}.csv", index=False)
        print(f"{season}: {len(df)} redova, {df.shape[1]} kolona")


if __name__ == "__main__":
    main()