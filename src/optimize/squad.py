from pathlib import Path

import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

BUDGET = 1000  # u desetinama miliona (100.0m), da izbegnemo greske sa decimalama
SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
START_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
START_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
BENCH_WEIGHT = 0.1  # klupa vredi malo, ali ne nula (zamene kad neko ne igra)


def optimize_squad(pred: pd.DataFrame, budget: int = BUDGET,
                   bench_weight: float = BENCH_WEIGHT) -> pd.DataFrame:
    """Bira 15 igraca, startnih 11 i kapitena tako da maksimizuje ocekivane poene."""
    pred = pred.reset_index(drop=True).copy()
    ids = list(pred.index)
    ep = pred["ep"]

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = prob.add_variable_dicts("squad", ids, cat=pulp.LpBinary)
    s = prob.add_variable_dicts("start", ids, cat=pulp.LpBinary)
    c = prob.add_variable_dicts("captain", ids, cat=pulp.LpBinary)

    # cilj: startni + dupli poeni kapitena + mala vrednost klupe
    prob += pulp.lpSum(ep[i] * (s[i] + c[i] + bench_weight * (x[i] - s[i])) for i in ids)

    # budzet
    prob += pulp.lpSum(pred.loc[i, "value"] * x[i] for i in ids) <= budget

    # sastav po pozicijama i formacija
    for pos, n in SQUAD_SIZE.items():
        idx = pred.index[pred["position"] == pos]
        prob += pulp.lpSum(x[i] for i in idx) == n
        prob += pulp.lpSum(s[i] for i in idx) >= START_MIN[pos]
        prob += pulp.lpSum(s[i] for i in idx) <= START_MAX[pos]

    prob += pulp.lpSum(s[i] for i in ids) == 11
    prob += pulp.lpSum(c[i] for i in ids) == 1

    # najvise 3 igraca po klubu
    for team in pred["team_name"].unique():
        idx = pred.index[pred["team_name"] == team]
        prob += pulp.lpSum(x[i] for i in idx) <= MAX_PER_TEAM

    # start samo iz tima, kapiten samo iz starta
    for i in ids:
        prob += s[i] <= x[i]
        prob += c[i] <= s[i]

    stats = prob.solve(pulp.COIN_CMD(msg=False))
    if stats.status != pulp.LpSolveStatus.Optimal:
        raise RuntimeError(f"Optimizacija nije uspela: {stats.status_str}")

    pred["in_squad"] = [x[i].value() > 0.5 for i in ids]
    pred["starter"] = [s[i].value() > 0.5 for i in ids]
    pred["captain"] = [c[i].value() > 0.5 for i in ids]
    return pred[pred["in_squad"]]


def latest_predictions() -> tuple[int, pd.DataFrame]:
    files = list(PROCESSED.glob("predictions_gw*.csv"))
    gw = max(int(f.stem.replace("predictions_gw", "")) for f in files)
    return gw, pd.read_csv(PROCESSED / f"predictions_gw{gw}.csv")


def print_squad(squad: pd.DataFrame) -> None:
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad = squad.assign(pos_order=squad["position"].map(order)) \
                 .sort_values(["starter", "pos_order", "ep"], ascending=[False, True, False])
    squad["C"] = squad["captain"].map({True: "(C)", False: ""})

    cols = ["web_name", "team_name", "position", "price", "ep", "C"]
    print("STARTNIH 11:")
    print(squad[squad["starter"]][cols].round(2).to_string(index=False))
    print("\nKLUPA:")
    print(squad[~squad["starter"]][cols].round(2).to_string(index=False))

    starters = squad[squad["starter"]]
    total_ep = starters["ep"].sum() + starters.loc[starters["captain"], "ep"].sum()
    formation = "-".join(str((starters["position"] == p).sum()) for p in ["DEF", "MID", "FWD"])
    print(f"\nFormacija: {formation}")
    print(f"Cena tima: {squad['price'].sum():.1f}m")
    print(f"Ocekivani poeni (sa kapitenom): {total_ep:.2f}")


def main() -> None:
    gw, pred = latest_predictions()
    print(f"Optimalni tim za kolo {gw}\n")
    squad = optimize_squad(pred)
    print_squad(squad)


if __name__ == "__main__":
    main()