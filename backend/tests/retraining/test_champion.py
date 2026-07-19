import datetime as dt
import numpy as np
import polars as pl
from openenergy.assets.repository import Plant, create_plant
from openenergy.providers.base import ApiRole, WeatherSeries
from openenergy.ingestion.weather_writer import write_weather_series
from openenergy.ingestion.production import write_production
from openenergy.experiments.runner import run_experiment
from openenergy.retraining.champion import retrain


def _populate(db, n=240):
    """Build realistic wind plant dataset."""
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    base = dt.datetime(2024, 1, 1)
    vts = [base + dt.timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(0)
    ws = 5 + 3 * np.sin(np.arange(n) / 12.0) + rng.normal(0, 0.5, n)
    rows = {"valid_time": [], "issue_time": [], "lead_hours": [], "variable": [], "value": []}
    for i, vt in enumerate(vts):
        for var, val in [("wind_speed_100m", ws[i]), ("wind_speed_10m", ws[i] * 0.6),
                         ("temperature_2m", 10.0), ("surface_pressure", 1013.0)]:
            rows["valid_time"].append(vt)
            rows["issue_time"].append(vt - dt.timedelta(hours=24))
            rows["lead_hours"].append(24)
            rows["variable"].append(var)
            rows["value"].append(float(val))
    frame = pl.DataFrame(rows, schema_overrides={"valid_time": pl.Datetime("us"), "issue_time": pl.Datetime("us"),
                                                 "lead_hours": pl.Int32, "value": pl.Float64})
    write_weather_series(db, WeatherSeries(point_id=7, role=ApiRole.PREVIOUS_RUNS.value, model="best_match", frame=frame))
    power = np.clip((ws ** 3) / 1000.0, 0, 10)
    prod = pl.DataFrame({"ts": vts, "power_mw": power.astype(float)},
                        schema_overrides={"ts": pl.Datetime("us"), "power_mw": pl.Float64})
    write_production(db, "wf1", prod)


def test_first_retrain_promotes(db, tmp_path):
    """First retrain (no incumbent) always promotes."""
    _populate(db)
    result = retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
                     capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
                     n_trials=3, models_dir=tmp_path, seed=42)
    assert result[24]["promoted"] is True
    assert result[24]["old_nrmse"] is None
    assert result[24]["new_nrmse"] is not None


def test_second_retrain_does_not_promote_when_incumbent_perfect(db, tmp_path):
    """If incumbent test_nrmse is 0.0 (unbeatable), new retrain must NOT promote."""
    _populate(db)
    # First retrain establishes incumbent champion
    retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
            capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
            n_trials=3, models_dir=tmp_path / "r1", seed=42)
    # Set incumbent's test_nrmse to 0.0 (unbeatable)
    db.execute(
        "UPDATE experiment_trials SET test_nrmse=0.0 WHERE is_champion=true AND horizon_hours=24"
    )
    # Also update models metrics to reflect
    # Second retrain — should NOT promote (can't beat 0.0)
    result = retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
                     capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
                     n_trials=3, models_dir=tmp_path / "r2", seed=99)
    assert result[24]["promoted"] is False
    # Incumbent champion trial should still be is_champion=true (the 0.0 one)
    champ_nrmse = db.execute(
        "SELECT test_nrmse FROM experiment_trials WHERE is_champion=true AND horizon_hours=24 ORDER BY trial_id ASC LIMIT 1"
    ).fetchone()
    assert champ_nrmse is not None and champ_nrmse[0] == 0.0


import pytest
from openenergy.retraining import champion as champion_mod


def test_significance_gate_noisy_equal_skill_not_promoted():
    """A challenger that only differs from the incumbent by zero-mean noise has
    no significant skill edge and must NOT clear the gate."""
    rng = np.random.default_rng(0)
    y = rng.normal(5.0, 2.0, 200)
    incumbent = y + rng.normal(0.0, 1.0, 200)
    # Challenger = incumbent perturbed by fresh zero-mean noise: same skill.
    challenger = incumbent + rng.normal(0.0, 0.3, 200)
    passed, p_value, stat = champion_mod.significance_gate(
        y, incumbent, challenger, horizon=1
    )
    assert passed is False
    assert 0.0 <= p_value <= 1.0


def test_significance_gate_genuinely_better_is_promoted():
    """A challenger with systematically smaller errors clears the gate."""
    rng = np.random.default_rng(1)
    y = rng.normal(5.0, 2.0, 200)
    incumbent = y + rng.normal(0.0, 1.5, 200)   # large errors
    challenger = y + rng.normal(0.0, 0.3, 200)  # much smaller errors
    passed, p_value, stat = champion_mod.significance_gate(
        y, incumbent, challenger, horizon=1
    )
    assert passed is True
    assert p_value < 0.05
    assert stat > 0


def test_significance_gate_worse_challenger_not_promoted():
    rng = np.random.default_rng(2)
    y = rng.normal(5.0, 2.0, 200)
    incumbent = y + rng.normal(0.0, 0.3, 200)
    challenger = y + rng.normal(0.0, 1.5, 200)
    passed, p_value, stat = champion_mod.significance_gate(
        y, incumbent, challenger, horizon=1
    )
    assert passed is False


def test_significance_gate_requires_horizon():
    y = np.zeros(10)
    with pytest.raises(TypeError):
        champion_mod.significance_gate(y, np.ones(10), np.zeros(10))


def test_dm_gate_h_widens_variance():
    """Positive serial correlation must widen the h-step DM uncertainty band."""
    rng = np.random.default_rng(42)
    n = 300
    innovations = rng.normal(0.0, 0.35, n)
    loss_diff = np.empty(n)
    loss_diff[0] = innovations[0]
    for idx in range(1, n):
        loss_diff[idx] = 0.85 * loss_diff[idx - 1] + innovations[idx]
    loss_diff += 0.15

    # Encode the desired loss differential as two valid squared-error series.
    y = np.zeros(n)
    base_loss = float(np.max(np.abs(loss_diff))) + 1.0
    incumbent = np.sqrt(base_loss + loss_diff / 2.0)
    challenger = np.sqrt(base_loss - loss_diff / 2.0)

    short_passed, short_p, short_stat = champion_mod.significance_gate(
        y, incumbent, challenger, horizon=1
    )
    long_passed, long_p, long_stat = champion_mod.significance_gate(
        y, incumbent, challenger, horizon=12
    )

    assert short_passed is True
    assert long_passed is False
    assert abs(long_stat) < abs(short_stat)
    assert long_p > short_p


def test_retrain_outcome_carries_p_value(db, tmp_path):
    """Every promotion decision records a p_value key (None when no incumbent)."""
    _populate(db)
    first = retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
                    capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
                    n_trials=3, models_dir=tmp_path / "r1", seed=42)
    assert "p_value" in first[24]
    assert first[24]["p_value"] is None  # no incumbent on first run
    second = retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
                     capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
                     n_trials=3, models_dir=tmp_path / "r2", seed=7)
    assert "p_value" in second[24]


def test_retrain_records_scenario_outcome_promote(db, tmp_path):
    """First retrain promotes → its scenario_runs row for the new experiment
    carries is_champion=True (operational reality)."""
    _populate(db)
    retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
            capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
            n_trials=3, models_dir=tmp_path, seed=42)
    row = db.execute(
        "SELECT is_champion FROM scenario_runs WHERE plant_id='wf1' AND horizon_hours=24 "
        "ORDER BY experiment_id DESC, scenario_run_id DESC LIMIT 1"
    ).fetchone()
    assert row is not None and bool(row[0]) is True


def test_retrain_records_scenario_outcome_defend(db, tmp_path):
    """A defended retrain (unbeatable incumbent) marks the challenger's scenario
    row is_champion=False so the leaderboard reflects the defeat, not the offline
    experiment's optimistic champion flag."""
    _populate(db)
    retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
            capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
            n_trials=3, models_dir=tmp_path / "r1", seed=42)
    db.execute(
        "UPDATE experiment_trials SET test_nrmse=0.0 WHERE is_champion=true AND horizon_hours=24"
    )
    result = retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
                     capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
                     n_trials=3, models_dir=tmp_path / "r2", seed=99)
    assert result[24]["promoted"] is False
    # The most recent (challenger) scenario row must record the defeat.
    row = db.execute(
        "SELECT is_champion FROM scenario_runs WHERE plant_id='wf1' AND horizon_hours=24 "
        "ORDER BY experiment_id DESC, scenario_run_id DESC LIMIT 1"
    ).fetchone()
    assert row is not None and bool(row[0]) is False
    # The leaderboard must not double-count the challenger as a win: exactly one
    # champion row survives for this plant+horizon.
    wins = db.execute(
        "SELECT count(*) FROM scenario_runs WHERE plant_id='wf1' AND horizon_hours=24 "
        "AND is_champion=true"
    ).fetchone()[0]
    assert wins == 1


def test_pointer_is_single_source_of_truth(db, tmp_path):
    """T-05: the served strategy champion is resolved through champion_pointer —
    exactly one current row per plant+horizon, and get_champion_strategy_model
    returns exactly the trial it names."""
    from openenergy.experiments.persistence import get_champion_strategy_model

    _populate(db)
    retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
            capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
            n_trials=3, models_dir=tmp_path, seed=42)
    n = db.execute(
        "SELECT count(*) FROM champion_pointer WHERE plant_id='wf1' AND horizon_hours=24"
    ).fetchone()[0]
    assert n == 1
    pointer_id = db.execute(
        "SELECT strategy_trial_id FROM champion_pointer "
        "WHERE plant_id='wf1' AND horizon_hours=24"
    ).fetchone()[0]
    served = get_champion_strategy_model(db, "wf1", 24)
    assert served is not None and served["strategy_trial_id"] == pointer_id


def test_defend_keeps_incumbent_on_serving_path(db, tmp_path):
    """T-05: a rejected challenger must NOT reach the serving path. After a defend
    (unbeatable incumbent), the served strategy champion stays the incumbent — the
    Diebold-Mariano gate is no longer a no-op on the forecast path."""
    from openenergy.experiments.persistence import get_champion_strategy_model

    _populate(db)
    retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
            capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
            n_trials=3, models_dir=tmp_path / "r1", seed=42)
    incumbent = get_champion_strategy_model(db, "wf1", 24)
    assert incumbent is not None
    incumbent_id = incumbent["strategy_trial_id"]

    # Make the incumbent unbeatable so the second run is a rejected challenger.
    db.execute(
        "UPDATE experiment_trials SET test_nrmse=0.0 WHERE is_champion=true AND horizon_hours=24"
    )
    result = retrain(db, plant_id="wf1", point_id=7, horizon_hours_list=[24],
                     capacity_mw=10.0, kind="wind", nwp_sources=["icon"],
                     n_trials=3, models_dir=tmp_path / "r2", seed=99)
    assert result[24]["promoted"] is False
    served = get_champion_strategy_model(db, "wf1", 24)
    assert served is not None and served["strategy_trial_id"] == incumbent_id, (
        "rejected challenger is being served — DM gate is a no-op on the forecast path")
