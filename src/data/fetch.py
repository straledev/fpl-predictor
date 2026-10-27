import json
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0"}

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / date.today().isoformat()

def get_json(endpoint: str) -> dict | list:
    """Preuzima se JSON sa FPL API-ja."""
    response = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()
def save_json(data, name: str) -> None:
    """Cuvaju se podaci u data/raw/<danasnji datum>/<name>.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Sacuvano: {path}")

def main() -> None:
    bootstrap = get_json("bootstrap-static/")
    save_json(bootstrap, "bootstrap_static")

    fixtures = get_json("fixtures/")
    save_json(fixtures, "fixtures")

    print(
        f"Igraca: {len(bootstrap['elements'])}, "
        f"timova: {len(bootstrap['teams'])}, "
        f"utakmica: {len(fixtures)}"
    )


if __name__ == "__main__":
    main()
