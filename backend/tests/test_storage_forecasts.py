def test_forecasts_and_feature_names_columns(db):
    cols_f = {r[1] for r in db.execute("PRAGMA table_info('forecasts')").fetchall()}
    assert {
        "plant_id", "horizon_hours", "issue_time", "valid_time",
        "p10", "p50", "p90", "strategy_trial_id",
    } <= cols_f
    cols_m = {r[1] for r in db.execute("PRAGMA table_info('models')").fetchall()}
    assert "feature_names" in cols_m
