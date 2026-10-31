import argparse
import requests

import pandas as pd
import pulp

from src.data.build import latest_raw_dir, load_json
from src.data.fetch import get_json
from src.models.predict import players_table
from src.optimize.squad import (
    BENCH_WEIGHT, MAX_PER_TEAM, SQUAD_SIZE, START_MAX, START_MIN,
    latest_predictions, print_squad,
)

HIT_COST = 4


def load_current_team(entry_id: int, event: int) -> tuple[list[int], int, str | None]:
    """Igraci, banka (u desetinama miliona) i cip iz datog kola."""
    try:
        data = get_json(f"entry/{entry_id}/event/{event}/picks/")
    except requests.exceptions.RetryError:
        raise SystemExit("FPL API trenutno nije dostupan (verovatno se igra azurira "
                         "posle roka). Pokusaj ponovo za sat-dva.")
    ids = [p["element"] for p in data["picks"]]
    return ids, data["entry_history"]["bank"], data.get("active_chip")


def build_pool() -> pd.DataFrame:
    """Svi igraci sa predikcijom; igraci bez meca u kolu dobijaju 0."""
    _, pred = latest_predictions()
    bootstrap = load_json(latest_raw_dir(), "bootstrap_static")
    pool = players_table(bootstrap).merge(pred[["element", "ep"]], on="element", how="left")
    pool["ep"] = pool["ep"].fillna(0.0)
    pool["price"] = pool["value"] / 10
    return pool.reset_index(drop=True)


def optimize_transfers(pool: pd.DataFrame, current_ids: list[int], bank: int,
                       free_transfers: int, max_transfers: int) -> tuple[pd.DataFrame, int]:
    ids = list(pool.index)
    owned = pool["element"].isin(current_ids)
    ep = pool["ep"]

    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)
    x = prob.add_variable_dicts("squad", ids, cat=pulp.LpBinary)
    s = prob.add_variable_dicts("start", ids, cat=pulp.LpBinary)
    c = prob.add_variable_dicts("captain", ids, cat=pulp.LpBinary)
    hits = prob.add_variable("hits", lowBound=0, cat=pulp.LpInteger)

    n_transfers = pulp.lpSum(x[i] for i in ids if not owned[i])

    prob += (
        pulp.lpSum(ep[i] * (s[i] + c[i] + BENCH_WEIGHT * (x[i] - s[i])) for i in ids)
        - HIT_COST * hits
    )
    prob += hits >= n_transfers - free_transfers
    prob += n_transfers <= max_transfers

    budget = int(bank + pool.loc[owned, "value"].sum())
    prob += pulp.lpSum(int(pool.loc[i, "value"]) * x[i] for i in ids) <= budget

    for pos, n in SQUAD_SIZE.items():
        idx = pool.index[pool["position"] == pos]
        prob += pulp.lpSum(x[i] for i in idx) == n
        prob += pulp.lpSum(s[i] for i in idx) >= START_MIN[pos]
        prob += pulp.lpSum(s[i] for i in idx) <= START_MAX[pos]
    prob += pulp.lpSum(s[i] for i in ids) == 11
    prob += pulp.lpSum(c[i] for i in ids) == 1

    for team in pool["team_name"].unique():
        idx = pool.index[pool["team_name"] == team]
        prob += pulp.lpSum(x[i] for i in idx) <= MAX_PER_TEAM

    for i in ids:
        prob += s[i] <= x[i]
        prob += c[i] <= s[i]

    stats = prob.solve(pulp.COIN_CMD(msg=False))
    if not stats.has_solution:
        raise RuntimeError(f"Optimizacija nije uspela: {stats.status_str}")

    result = pool.copy()
    result["in_squad"] = [x[i].value() > 0.5 for i in ids]
    result["starter"] = [s[i].value() > 0.5 for i in ids]
    result["captain"] = [c[i].value() > 0.5 for i in ids]
    return result[result["in_squad"]], int(round(hits.value()))


def team_score(squad: pd.DataFrame, hits: int) -> float:
    starters = squad[squad["starter"]]
    captain = starters.loc[starters["captain"], "ep"].sum()
    return starters["ep"].sum() + captain - HIT_COST * hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Predlog transfera za FPL tim")
    parser.add_argument("--entry", type=int, required=True, help="FPL ID tvog tima")
    parser.add_argument("--ft", type=int, default=1, help="broj besplatnih transfera")
    parser.add_argument("--max-transfers", type=int, default=2)
    args = parser.parse_args()

    gw, _ = latest_predictions()
    pool = build_pool()

    ids, bank, chip = load_current_team(args.entry, gw - 1)
    if chip == "freehit":
        ids, bank, _ = load_current_team(args.entry, gw - 2)

    missing = set(ids) - set(pool["element"])
    if missing:
        print(f"UPOZORENJE: {len(missing)} igraca iz tvog tima nema u bazi: {missing}")

    hold, _ = optimize_transfers(pool, ids, bank, args.ft, max_transfers=0)
    best, hits = optimize_transfers(pool, ids, bank, args.ft, args.max_transfers)

    out = pool[pool["element"].isin(ids) & ~pool["element"].isin(best["element"])]
    inn = best[~best["element"].isin(ids)]

    print(f"Kolo {gw} | banka: {bank / 10:.1f}m | besplatnih transfera: {args.ft}\n")
    cols = ["web_name", "team_name", "position", "price", "ep"]
    if inn.empty:
        print("PREDLOG: bez transfera (sacuvaj transfer za sledece kolo)\n")
    else:
        print("PRODAJ:")
        print(out[cols].round(2).to_string(index=False))
        print("\nKUPI:")
        print(inn[cols].round(2).to_string(index=False))
        print(f"\nKazna: -{HIT_COST * hits} poena\n")

    print_squad(best)

    gain = team_score(best, hits) - team_score(hold, 0)
    print(f"\nOcekivani dobitak u odnosu na tim bez transfera: {gain:+.2f} poena")
    print("Napomena: budzet koristi trenutne cene, a ne tvoje prodajne cene. "
          "Proveri cenu na FPL sajtu pre potvrde.")


if __name__ == "__main__":
    main()