import pandas as pd

from src.features.build_features import feature_columns, prev_rolling_mean


def test_prev_rolling_mean_uses_only_past_matches():
    df = pd.DataFrame({
        "season": ["2023-24"] * 4 + ["2024-25"] * 2,
        "element": [1] * 6,
        "points": [2, 6, 10, 1, 8, 3],
    })
    result = prev_rolling_mean(df, ["season", "element"], "points", 3).sort_index()

    # prvi mec sezone nema proslost, ostali vide samo prethodne meceve
    assert result.isna().tolist() == [True, False, False, False, True, False]
    assert result.dropna().tolist() == [2.0, 4.0, 6.0, 8.0]


def test_feature_columns_exclude_match_outcomes():
    df = pd.DataFrame(columns=[
        "total_points", "minutes", "goals_scored", "bonus", "clean_sheets",
        "expected_goals", "team_h_score", "team_a_score",
        "total_points_roll3", "team_gf_roll5", "opp_ga_roll5",
        "pos_MID", "is_home", "price",
    ])
    features = set(feature_columns(df))

    outcomes = {"total_points", "minutes", "goals_scored", "bonus", "clean_sheets",
                "expected_goals", "team_h_score", "team_a_score"}
    assert not outcomes & features
    assert {"total_points_roll3", "team_gf_roll5", "opp_ga_roll5",
            "pos_MID", "is_home", "price"} <= features