import json
import pytest
from openenergy.assets.repository import Plant, create_plant
from openenergy.experiments.persistence import (
    create_experiment, record_trial, mark_champion, best_trials,
    get_champion_feature_blocks, get_champion_model, record_strategy_trial,
)
from openenergy.experiments.search_space import PipelineConfig
from openenergy.experiments.strategy import StrategyConfig


def _cfg():
    return PipelineConfig(model_family="ridge", nwp_source="icon", feature_blocks=["wind_power"], params={"alpha": 1.0})


def test_experiment_and_trial_roundtrip(db):
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    eid = create_experiment(db, "wf1", [1, 24])
    tid = record_trial(db, experiment_id=eid, horizon_hours=24, pipeline_config=_cfg(),
                       cv_nrmse=0.12, test_metrics={"nrmse": 0.13, "nmae": 0.09, "bias": 0.0, "pinball": 0.05},
                       skill_score=0.4)
    mark_champion(db, eid, 24, tid)
    champ = db.execute("SELECT is_champion FROM experiment_trials WHERE trial_id=?", [tid]).fetchone()[0]
    assert champ is True
    bests = best_trials(db, eid)
    assert any(b["horizon_hours"] == 24 and b["test_nrmse"] == 0.13 for b in bests)


def test_clean_metric_and_quality_counts_roundtrip(db):
    create_plant(db, Plant(plant_id="wf_qc", name="WF QC", kind="wind", capacity_mw=10))
    eid = create_experiment(db, "wf_qc", [24])
    tid = record_trial(
        db, experiment_id=eid, horizon_hours=24, pipeline_config=_cfg(),
        cv_nrmse=0.12, test_metrics={"nrmse": 0.30}, skill_score=0.1,
        clean_metrics={"nrmse": 0.20}, clean_skill_score=0.25,
        quality_audit={
            "train_all_count": 100, "train_clean_count": 96,
            "eval_all_count": 20, "eval_clean_count": 18,
        },
    )
    stored = db.execute(
        "SELECT test_nrmse, test_clean_nrmse, skill_score, clean_skill_score, "
        "train_all_count, train_clean_count, eval_all_count, eval_clean_count "
        "FROM experiment_trials WHERE trial_id=?",
        [tid],
    ).fetchone()
    assert stored == (0.30, 0.20, 0.1, 0.25, 100, 96, 20, 18)


def test_strategy_uncertainty_diagnostics_roundtrip(db):
    create_plant(db, Plant(plant_id="wf_ci", name="WF CI", kind="wind", capacity_mw=10))
    eid = create_experiment(db, "wf_ci", [24])
    audit = {
        "champion_metric": "nrmse",
        "champion_metric_lower": 0.10,
        "champion_metric_upper": 0.16,
        "metric_ci_bootstrap": "moving",
        "metric_ci_seed": 42,
        "mcs_members": ["mean", "weighted_by_cv"],
    }
    strategy_id = record_strategy_trial(
        db,
        experiment_id=eid,
        horizon_hours=24,
        strategy_config=StrategyConfig(
            candidates=[_cfg()], ensemble_method="mean", top_k=1
        ),
        cv_nrmse=0.12,
        test_metrics={"nrmse": 0.13},
        skill_score=0.4,
        artifact_path=None,
        uncertainty_diagnostics=audit,
    )
    stored, metric_name, lower, upper, members = db.execute(
        "SELECT uncertainty_diagnostics, champion_metric_name, "
        "champion_metric_lower, champion_metric_upper, mcs_members FROM strategy_trials "
        "WHERE strategy_trial_id=?",
        [strategy_id],
    ).fetchone()
    assert json.loads(stored) == audit
    assert (metric_name, lower, upper) == ("nrmse", 0.10, 0.16)
    assert json.loads(members) == ["mean", "weighted_by_cv"]


def test_strategy_evaluation_evidence_roundtrip_and_best_trials_surface(db):
    create_plant(db, Plant(plant_id="wf_grade", name="WF Grade", kind="wind", capacity_mw=10))
    eid = create_experiment(db, "wf_grade", [24])
    strategy_id = record_strategy_trial(
        db,
        experiment_id=eid,
        horizon_hours=24,
        strategy_config=StrategyConfig(
            candidates=[_cfg()], ensemble_method="mean", top_k=1
        ),
        cv_nrmse=0.12,
        test_metrics={"nrmse": 0.13},
        skill_score=0.4,
        artifact_path=None,
        evaluation_grade="limited",
        evaluation_n_days=120,
        evaluation_origins=3,
        evaluation_seasons=2,
    )
    stored = db.execute(
        "SELECT evaluation_grade, evaluation_n_days, evaluation_origins, "
        "evaluation_seasons FROM strategy_trials WHERE strategy_trial_id=?",
        [strategy_id],
    ).fetchone()
    assert stored == ("limited", 120, 3, 2)
    best = best_trials(db, eid)[0]
    assert (
        best["evaluation_grade"],
        best["evaluation_n_days"],
        best["evaluation_origins"],
        best["evaluation_seasons"],
    ) == stored


def test_get_champion_feature_blocks_raises_when_no_champion(db):
    """I3: get_champion_feature_blocks raises ValueError when no champion trial exists."""
    create_plant(db, Plant(plant_id="wf2", name="WF2", kind="solar", capacity_mw=5))
    with pytest.raises(ValueError, match="champion"):
        get_champion_feature_blocks(db, "wf2", 24)


def test_get_champion_feature_blocks_raises_for_unknown_plant(db):
    """I3: also raises when plant has no experiments at all."""
    with pytest.raises(ValueError, match="champion"):
        get_champion_feature_blocks(db, "no_such_plant", 24)


def test_legacy_model_resolvers_are_scoped_by_served_pointer(db):
    """T-05: boolean champion flags cannot override the serving pointer.

    Both experiments deliberately retain optimistic ``is_champion`` flags.  The
    newer row would win the old global lookup, but every serving resolver must
    stay within the experiment selected by ``champion_pointer``.
    """
    create_plant(db, Plant(plant_id="wf1", name="WF", kind="wind", capacity_mw=10))
    incumbent_eid = create_experiment(db, "wf1", [24])
    challenger_eid = create_experiment(db, "wf1", [24])

    for eid, blocks, artifact in (
        (incumbent_eid, ["wind_power"], "incumbent.pkl"),
        (challenger_eid, ["raw"], "challenger.pkl"),
    ):
        db.execute(
            "INSERT INTO experiment_trials "
            "(experiment_id, horizon_hours, feature_blocks, is_champion) "
            "VALUES (?,?,?,true)",
            [eid, 24, json.dumps(blocks)],
        )
        db.execute(
            "INSERT INTO models "
            "(experiment_id, horizon_hours, artifact_path, feature_names, metrics, is_champion) "
            "VALUES (?,?,?,?,?,true)",
            [eid, 24, artifact, "[]", "{}",],
        )
        strategy_id = db.execute(
            "INSERT INTO strategy_trials "
            "(experiment_id, horizon_hours, ensemble_method, strategy_config, is_champion) "
            "VALUES (?,24,'none','{}',true) RETURNING strategy_trial_id",
            [eid],
        ).fetchone()[0]
        if eid == incumbent_eid:
            db.execute(
                "INSERT INTO champion_pointer "
                "(plant_id, horizon_hours, strategy_trial_id) VALUES ('wf1',24,?)",
                [strategy_id],
            )

    model = get_champion_model(db, "wf1", 24)
    assert model is not None
    assert model["experiment_id"] == incumbent_eid
    assert model["artifact_path"] == "incumbent.pkl"
    assert get_champion_feature_blocks(db, "wf1", 24) == ["wind_power"]
