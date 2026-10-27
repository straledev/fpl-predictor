import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # da bi import iz src radio i na Streamlit Cloud-u

from src.optimize.squad import optimize_squad  # noqa: E402

REPORTS = ROOT / "reports"
POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

st.set_page_config(page_title="FPL Predictor", page_icon="⚽", layout="wide")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "predictions_latest.csv")


@st.cache_data
def load_backtest() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "backtest_2025-26.csv", index_col="round")


@st.cache_data
def best_squad(pred: pd.DataFrame, budget: float) -> pd.DataFrame:
    squad = optimize_squad(pred, budget=int(round(budget * 10)))
    squad["pos_order"] = squad["position"].map(POSITION_ORDER)
    return squad.sort_values(["starter", "pos_order", "ep"], ascending=[False, True, False])


pred = load_predictions()
gw = int(pred["gw"].iloc[0])

st.title("FPL Predictor")
st.caption(
    f"Expected points for gameweek {gw}, from a LightGBM model trained on 10 seasons "
    "of Premier League data. Injured or suspended players are scaled down by FPL's "
    "chance of playing."
)

tab_pred, tab_squad, tab_backtest = st.tabs(["Predictions", "Optimal squad", "Backtest"])

money = st.column_config.NumberColumn("Price", format="£%.1fm")
points = st.column_config.NumberColumn("xPts", format="%.2f")

# ---------- Predictions ----------
with tab_pred:
    c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
    positions = c1.multiselect("Position", ["GK", "DEF", "MID", "FWD"],
                               default=["GK", "DEF", "MID", "FWD"])
    teams = c2.multiselect("Team", sorted(pred["team_name"].unique()))
    max_price = c3.slider("Max price (£m)", 4.0, float(pred["price"].max()),
                          float(pred["price"].max()), 0.5)
    search = c4.text_input("Search player")

    view = pred[pred["position"].isin(positions) & (pred["price"] <= max_price)]
    if teams:
        view = view[view["team_name"].isin(teams)]
    if search:
        view = view[view["web_name"].str.contains(search, case=False)]

    st.dataframe(
        view[["web_name", "team_name", "position", "price", "status", "ep"]],
        column_config={"web_name": "Player", "team_name": "Team",
                       "position": "Pos", "status": "Status",
                       "price": money, "ep": points},
        hide_index=True, use_container_width=True, height=560,
    )

# ---------- Optimal squad ----------
with tab_squad:
    budget = st.slider("Budget (£m)", 80.0, 110.0, 100.0, 0.5)
    squad = best_squad(pred, budget)
    starters = squad[squad["starter"]]
    bench = squad[~squad["starter"]]

    captain = starters.loc[starters["captain"], "web_name"].iloc[0]
    total = starters["ep"].sum() + starters.loc[starters["captain"], "ep"].sum()
    formation = "-".join(str((starters["position"] == p).sum()) for p in ["DEF", "MID", "FWD"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected points", f"{total:.1f}")
    m2.metric("Squad cost", f"£{squad['price'].sum():.1f}m")
    m3.metric("Formation", formation)
    m4.metric("Captain", captain)

    cols = ["web_name", "team_name", "position", "price", "ep"]
    config = {"web_name": "Player", "team_name": "Team", "position": "Pos",
              "price": money, "ep": points}
    st.subheader("Starting XI")
    st.dataframe(starters[cols], column_config=config, hide_index=True,
                 use_container_width=True)
    st.subheader("Bench")
    st.dataframe(bench[cols], column_config=config, hide_index=True,
                 use_container_width=True)

# ---------- Backtest ----------
with tab_backtest:
    bt = load_backtest()
    st.markdown(
        "Every gameweek of the **2025-26 season** (never seen in training), the optimizer "
        "picks a fresh squad from each strategy's predictions, scored with real points. "
        "*Perfect hindsight* knows the actual points and is an upper bound."
    )
    totals = bt.sum()
    b1, b2, b3 = st.columns(3)
    b1.metric("Model", f"{totals['Model']:,.0f} pts",
              f"{(totals['Model'] / totals['Baseline'] - 1):+.1%} vs baseline")
    b2.metric("Baseline (last 5 matches)", f"{totals['Baseline']:,.0f} pts")
    b3.metric("Perfect hindsight", f"{totals['Savrsen tim']:,.0f} pts")

    st.line_chart(bt.cumsum().rename(columns={"Savrsen tim": "Perfect hindsight"}),
                  x_label="Gameweek", y_label="Cumulative points")