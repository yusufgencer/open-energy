import optuna
from openenergy.experiments.search_space import (
    suggest, build_model, PipelineConfig, source_policies_for,
)
from openenergy.models.catboost_model import CatBoostModel
from openenergy.models.lightgbm_model import LightGBMModel
from openenergy.models.quantile_gbm_model import QuantileGBMModel
from openenergy.models.random_forest_model import RandomForestModel
from openenergy.models.seed_bagging import SeedBaggingModel
from openenergy.models.stacking_model import StackingModel
from openenergy.models.xgboost_model import XGBoostModel


def test_suggest_with_fixed_trial_lightgbm():
    # nwp_source is no longer a search axis — it is fixed to nwp_sources[0].
    # FixedTrial params do NOT include nwp_source since suggest() no longer calls
    # trial.suggest_categorical("nwp_source", ...).
    trial = optuna.trial.FixedTrial({
        "model_family": "lightgbm",
        "train_policy": "expanding_all",
        "feature_family": "wind_physical_core",
        "blk_wind_power": True, "blk_air_density": False, "blk_wind_direction": False,
        "blk_wind_shear": False, "blk_wind_gust": False, "blk_cyclical_time": True,
        "blk_nwp_trajectory": False, "blk_wind_physics_ext": False,
        "n_estimators": 100, "num_leaves": 15, "learning_rate": 0.05,
    })
    cfg = suggest(trial, "wind", ["icon", "gfs"])
    assert cfg.model_family == "lightgbm"
    assert "wind_power" in cfg.feature_blocks and "cyclical_time" in cfg.feature_blocks
    # with no source_policies passed, source stays the single default (nwp_sources[0])
    assert cfg.nwp_source == "icon"
    assert cfg.source_policy == "icon"
    assert isinstance(build_model(cfg), LightGBMModel)


def test_suggest_guarantees_at_least_one_block():
    trial = optuna.trial.FixedTrial({
        "model_family": "ridge",
        "train_policy": "expanding_all",
        "feature_family": "wind_physical_core",
        "blk_wind_power": False, "blk_air_density": False, "blk_wind_direction": False,
        "blk_wind_shear": False, "blk_wind_gust": False, "blk_cyclical_time": False,
        "blk_nwp_trajectory": False, "blk_wind_physics_ext": False,
        "alpha": 1.0,
    })
    cfg = suggest(trial, "wind", ["icon"])
    assert len(cfg.feature_blocks) >= 1
    # nwp_source is always the first element — not searched
    assert cfg.nwp_source == "icon"


def test_build_model_supports_expanded_families():
    cases = [
        ("random_forest", RandomForestModel, {"n_estimators": 10}),
        ("xgboost", XGBoostModel, {"n_estimators": 10}),
        ("catboost", CatBoostModel, {"iterations": 10, "verbose": False}),
        ("stacking", StackingModel, {
            "rf_n_estimators": 10,
            "gb_n_estimators": 10,
            "gb_learning_rate": 0.05,
            "final_alpha": 1.0,
        }),
        ("quantile_gbm", QuantileGBMModel, {"n_estimators": 10}),
    ]
    for family, cls, params in cases:
        cfg = PipelineConfig(
            model_family=family,
            nwp_source="icon",
            feature_blocks=["wind_power"],
            params=params,
            train_policy="expanding_all",
        )
        assert isinstance(build_model(cfg), cls)


def test_suggest_seed_bagged_lightgbm_builds_wrapper():
    trial = optuna.trial.FixedTrial({
        "model_family": "seed_bagged_lightgbm",
        "train_policy": "expanding_all",
        "feature_family": "wind_physical_core",
        "blk_wind_power": True, "blk_air_density": False, "blk_wind_direction": False,
        "blk_wind_shear": False, "blk_wind_gust": False, "blk_cyclical_time": False,
        "blk_nwp_trajectory": False, "blk_wind_physics_ext": False,
        "sb_n_estimators": 120, "sb_num_leaves": 15, "sb_learning_rate": 0.05,
        "sb_n_seeds": 4,
    })
    cfg = suggest(trial, "wind", ["icon"])
    assert cfg.model_family == "seed_bagged_lightgbm"
    model = build_model(cfg)
    assert isinstance(model, SeedBaggingModel)
    assert model.n_seeds == 4
    # n_seeds must not leak into the underlying LightGBM params.
    assert "n_seeds" not in model.params


def test_target_policy_defaults_to_capacity_norm():
    cfg = PipelineConfig(model_family="ridge", nwp_source="icon",
                         feature_blocks=["wind_power"], params={"alpha": 1.0})
    assert cfg.target_policy == "capacity_norm"


def test_suggest_wind_has_no_target_policy_axis():
    # target_policy is solar-only: wind trials never add the parameter and keep
    # the legacy capacity_norm target (byte-identical FixedTrial param set).
    trial = optuna.trial.FixedTrial({
        "model_family": "ridge",
        "train_policy": "expanding_all",
        "feature_family": "wind_physical_core",
        "blk_wind_power": True, "blk_air_density": False, "blk_wind_direction": False,
        "blk_wind_shear": False, "blk_wind_gust": False, "blk_cyclical_time": False,
        "blk_nwp_trajectory": False, "blk_wind_physics_ext": False,
        "alpha": 1.0,
    })
    cfg = suggest(trial, "wind", ["icon"])
    assert cfg.target_policy == "capacity_norm"


def test_suggest_solar_searches_target_policy():
    from openenergy.features.registry import available_blocks
    params = {
        "model_family": "ridge",
        "target_policy": "kpv",
        "train_policy": "expanding_all",
        "feature_family": "solar_clear_sky",
        "alpha": 1.0,
    }
    for blk in available_blocks("solar"):
        params[f"blk_{blk}"] = (blk == "cyclical_time")
    trial = optuna.trial.FixedTrial(params)
    cfg = suggest(trial, "solar", ["ecmwf"])
    assert cfg.target_policy == "kpv"


def test_source_policies_for_filters_to_available():
    # wind full set, only what's available survives
    assert source_policies_for("wind", ["ecmwf", "icon_eu"]) == ["ecmwf", "icon_eu", "ecmwf+icon_eu"]
    assert source_policies_for("wind", ["ecmwf"]) == ["ecmwf"]
    # gfs available adds the gfs single-source policy
    assert "gfs" in source_policies_for("wind", ["ecmwf", "icon_eu", "gfs"])
    # solar is deprioritized: no gfs / no combo
    assert source_policies_for("solar", ["ecmwf", "icon_eu", "gfs"]) == ["ecmwf", "icon_eu"]


def test_suggest_searches_source_policy_when_multiple_available():
    trial = optuna.trial.FixedTrial({
        "model_family": "ridge",
        "source_policy": "ecmwf+icon_eu",
        "train_policy": "expanding_all",
        "feature_family": "wind_physical_core",
        "blk_wind_power": True, "blk_air_density": False, "blk_wind_direction": False,
        "blk_wind_shear": False, "blk_wind_gust": False, "blk_cyclical_time": False,
        "blk_nwp_trajectory": False, "blk_wind_physics_ext": False,
        "alpha": 1.0,
    })
    cfg = suggest(trial, "wind", ["ecmwf", "icon_eu"],
                  source_policies=["ecmwf", "icon_eu", "ecmwf+icon_eu"])
    # the chosen policy is honestly recorded on both fields
    assert cfg.source_policy == "ecmwf+icon_eu"
    assert cfg.nwp_source == "ecmwf+icon_eu"


def test_suggest_no_source_axis_when_single_policy():
    # a single-element source_policies list adds NO trial parameter (byte-identical)
    trial = optuna.trial.FixedTrial({
        "model_family": "ridge",
        "train_policy": "expanding_all",
        "feature_family": "wind_physical_core",
        "blk_wind_power": True, "blk_air_density": False, "blk_wind_direction": False,
        "blk_wind_shear": False, "blk_wind_gust": False, "blk_cyclical_time": False,
        "blk_nwp_trajectory": False, "blk_wind_physics_ext": False,
        "alpha": 1.0,
    })
    cfg = suggest(trial, "wind", ["ecmwf"], source_policies=["ecmwf"])
    assert cfg.source_policy == "ecmwf"
    assert cfg.nwp_source == "ecmwf"
