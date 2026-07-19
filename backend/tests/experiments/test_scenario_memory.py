from openenergy.experiments.scenario_memory import (
    ScenarioRun,
    feature_signature,
    model_signature,
    plant_strategy_profile,
    policy_leaderboard,
    record_scenario_run,
    refresh_scenario_leaderboard,
    scenario_leaderboard,
    strategy_family,
    target_leaderboard,
)


def test_signatures_are_stable_and_deduplicated():
    assert feature_signature(["b", "a", "a"]) == "a+b"
    assert model_signature(["ridge", "lightgbm", "ridge"]) == "lightgbm+ridge"


def test_strategy_family_folds_point_and_source_policies():
    # Trivial single-point/single-source regime stays compact (backward compatible).
    base = strategy_family(asset_kind="wind", ensemble_policy="mean")
    assert base == "wind_topk_mean"
    # Non-trivial spatial/source regimes yield distinct family strings.
    spatial = strategy_family(
        asset_kind="wind",
        ensemble_policy="mean",
        point_policy="spatial_mean",
        source_policy="icon+gfs",
    )
    assert spatial != base
    assert "spatial_mean" in spatial
    assert "icon+gfs" in spatial
    # Distinct spatial regimes distinguish, and source-only differs from point-only.
    other = strategy_family(
        asset_kind="wind", ensemble_policy="mean", point_policy="upwind_bias"
    )
    assert other != spatial and other != base


def test_strategy_family_folds_target_policy():
    # Legacy capacity_norm target stays compact (backward compatible).
    base = strategy_family(asset_kind="solar", ensemble_policy="mean")
    assert base == "solar_topk_mean"
    assert strategy_family(
        asset_kind="solar", ensemble_policy="mean", target_policy="capacity_norm"
    ) == base
    # kpv target yields a distinct family string.
    kpv = strategy_family(asset_kind="solar", ensemble_policy="mean", target_policy="kpv")
    assert kpv != base
    assert "kpv" in kpv


def test_recorded_run_carries_target_policy(db):
    record_scenario_run(
        db,
        ScenarioRun(
            experiment_id=1,
            plant_id="sp1",
            asset_kind="solar",
            horizon_hours=24,
            strategy_family="solar_topk_mean_kpv",
            target_policy="kpv",
            is_champion=True,
        ),
    )
    row = db.execute("SELECT target_policy FROM scenario_runs").fetchone()
    assert row == ("kpv",)


def test_target_leaderboard_distinguishes_kpv_vs_capacity_norm(db):
    # kpv: 2 champions of 2, better (lower) peak_mae and higher skill.
    for i, (skill, peak_mae) in enumerate([(0.5, 0.10), (0.6, 0.08)]):
        record_scenario_run(
            db,
            ScenarioRun(
                experiment_id=i + 1, plant_id="sp1", asset_kind="solar", horizon_hours=24,
                strategy_family=strategy_family(asset_kind="solar", ensemble_policy="mean",
                                                target_policy="kpv"),
                target_policy="kpv", skill_score=skill, peak_mae=peak_mae, is_champion=True,
            ),
        )
    # capacity_norm: worse peak_mae, lower skill.
    for i, (skill, peak_mae) in enumerate([(0.2, 0.30), (0.1, 0.35)]):
        record_scenario_run(
            db,
            ScenarioRun(
                experiment_id=i + 10, plant_id="sp1", asset_kind="solar", horizon_hours=24,
                strategy_family=strategy_family(asset_kind="solar", ensemble_policy="mean",
                                                target_policy="capacity_norm"),
                target_policy="capacity_norm", skill_score=skill, peak_mae=peak_mae,
                is_champion=False,
            ),
        )
    rows = target_leaderboard(db, asset_kind="solar", horizon_hours=24)
    assert len(rows) == 2
    top = rows[0]
    assert top["target_policy"] == "kpv"
    assert top["trials"] == 2 and top["wins"] == 2
    assert abs(top["median_peak_mae"] - 0.09) < 1e-9
    bottom = rows[1]
    assert bottom["target_policy"] == "capacity_norm"
    assert bottom["median_peak_mae"] > top["median_peak_mae"]


def test_recorded_run_carries_source_policy(db):
    record_scenario_run(
        db,
        ScenarioRun(
            experiment_id=1,
            plant_id="wf1",
            asset_kind="wind",
            horizon_hours=24,
            strategy_family="wind_topk_mean_spatial_mean_icon+gfs",
            point_policy="spatial_mean",
            source_policy="icon+gfs",
            is_champion=True,
        ),
    )
    row = db.execute(
        "SELECT point_policy, source_policy FROM scenario_runs"
    ).fetchone()
    assert row == ("spatial_mean", "icon+gfs")


def test_record_scenario_run_and_refresh_leaderboard(db):
    for i in range(10):
        record_scenario_run(
            db,
            ScenarioRun(
                experiment_id=i + 1,
                plant_id="wf1",
                asset_kind="wind",
                horizon_hours=24,
                strategy_family="wind_topk_weighted_by_cv",
                feature_signature="wind_power+air_density",
                model_signature="lightgbm+ridge",
                train_policy="rolling_180d",
                point_policy="single_point",
                ensemble_policy="weighted_by_cv",
                quantile_policy="native_or_none",
                nrmse=0.2 + i * 0.01,
                skill_score=0.3,
                is_champion=i < 6,
            ),
        )

    refresh_scenario_leaderboard(db)
    rows = scenario_leaderboard(db, asset_kind="wind", horizon_hours=24)
    assert len(rows) == 1
    row = rows[0]
    assert row["trials"] == 10
    assert row["wins"] == 6
    assert row["win_rate"] == 0.6
    assert row["median_nrmse"] is not None


def test_plant_strategy_profile_groups_by_horizon_and_family(db):
    record_scenario_run(
        db,
        ScenarioRun(
            experiment_id=1,
            plant_id="solar-a",
            asset_kind="solar",
            horizon_hours=48,
            strategy_family="solar_clear_sky",
            nrmse=0.15,
            skill_score=0.4,
            is_champion=True,
        ),
    )
    profile = plant_strategy_profile(db, "solar-a")
    assert len(profile) == 1
    assert profile[0]["horizon_hours"] == 48
    assert profile[0]["strategy_family"] == "solar_clear_sky"
    assert profile[0]["wins"] == 1


def _seed_policy_run(db, *, exp, point, source, skill, nrmse, champion, plant="wf1"):
    record_scenario_run(
        db,
        ScenarioRun(
            experiment_id=exp,
            plant_id=plant,
            asset_kind="wind",
            horizon_hours=24,
            strategy_family=strategy_family(
                asset_kind="wind",
                ensemble_policy="mean",
                point_policy=point,
                source_policy=source,
            ),
            point_policy=point,
            source_policy=source,
            skill_score=skill,
            nrmse=nrmse,
            is_champion=champion,
        ),
    )


def test_policy_leaderboard_groups_by_point_and_source(db):
    # Combo A: spatial_mean x icon+gfs — 2 champions of 3 (win_rate 2/3), skills [0.5,0.6,0.4]
    _seed_policy_run(db, exp=1, point="spatial_mean", source="icon+gfs", skill=0.5, nrmse=0.20, champion=True)
    _seed_policy_run(db, exp=2, point="spatial_mean", source="icon+gfs", skill=0.6, nrmse=0.18, champion=True)
    _seed_policy_run(db, exp=3, point="spatial_mean", source="icon+gfs", skill=0.4, nrmse=0.22, champion=False)
    # Combo B: single_point x single_source — 0 champions of 2, skills [0.1,0.3]
    _seed_policy_run(db, exp=4, point="single_point", source="single_source", skill=0.1, nrmse=0.30, champion=False)
    _seed_policy_run(db, exp=5, point="single_point", source="single_source", skill=0.3, nrmse=0.28, champion=False)

    rows = policy_leaderboard(db, asset_kind="wind", horizon_hours=24)
    assert len(rows) == 2
    # Best win_rate first.
    top = rows[0]
    assert top["point_policy"] == "spatial_mean"
    assert top["source_policy"] == "icon+gfs"
    assert top["trials"] == 3
    assert top["wins"] == 2
    assert abs(top["win_rate"] - 2 / 3) < 1e-9
    assert abs(top["median_skill_score"] - 0.5) < 1e-9
    assert abs(top["median_nrmse"] - 0.20) < 1e-9

    bottom = rows[1]
    assert bottom["point_policy"] == "single_point"
    assert bottom["wins"] == 0
    assert bottom["win_rate"] == 0.0
    assert abs(bottom["median_skill_score"] - 0.2) < 1e-9


def test_policy_leaderboard_filters_by_plant_and_horizon(db):
    _seed_policy_run(db, exp=1, point="spatial_mean", source="icon+gfs", skill=0.5, nrmse=0.2, champion=True, plant="wf1")
    _seed_policy_run(db, exp=2, point="upwind_bias", source="icon+gfs", skill=0.3, nrmse=0.4, champion=True, plant="wf2")

    only_wf1 = policy_leaderboard(db, plant_id="wf1")
    assert len(only_wf1) == 1
    assert only_wf1[0]["point_policy"] == "spatial_mean"

    # Ties broken so a stable ordering exists; empty filter returns everything.
    all_rows = policy_leaderboard(db)
    assert len(all_rows) == 2
