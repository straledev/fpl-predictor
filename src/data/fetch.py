import json
import time
from datetime import date
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0"}

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / date.today().isoformat()


def make_session() -> requests.Session:
    """Sesija koja sama ponavlja zahtev ako server ne odgovori ili je preopterecen."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = make_session()


def get_json(endpoint: str) -> dict | list:
    """Preuzima JSON sa FPL API-ja."""
    response = SESSION.get(f"{BASE_URL}/{endpoint}", timeout=30)
    response.raise_for_status()
    return response.json()


def save_json(data, name: str) -> None:
    """Cuva podatke u data/raw/<danasnji datum>/<name>.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Sacuvano: {path}")


def fetch_player_histories(player_ids: list[int], pause: float = 0.5) -> dict:
    """Preuzima istoriju po kolima za svakog igraca (element-summary)."""
    histories = {}
    for i, player_id in enumerate(player_ids, start=1):
        histories[player_id] = get_json(f"element-summary/{player_id}/")
        if i % 50 == 0:
            print(f"{i}/{len(player_ids)} igraca")
        time.sleep(pause)
    return histories


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
    player_ids = [p["id"] for p in bootstrap["elements"]]
    histories = fetch_player_histories(player_ids)
    save_json(histories, "element_summaries")


if __name__ == "__main__":
    main()