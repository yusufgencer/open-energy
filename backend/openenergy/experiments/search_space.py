from __future__ import annotations

from pydantic import BaseModel

from openenergy.experiments.train_policy import POLICY_NAMES, SOLAR_POLICY_NAMES
from openenergy.features.families import available_families, blocks_for_family
from openenergy.features.registry import available_blocks
from openenergy.models.base import ModelWrapper
from openenergy.models.catboost_model import CatBoostModel
from openenergy.models.lightgbm_model import LightGBMModel
from openenergy.models.linear_model import RidgeModel
from openenergy.models.quantile_gbm_model import QuantileGBMModel
from openenergy.models.random_forest_model import RandomForestModel
from openenergy.models.seed_bagging import SeedBaggingModel
from openenergy.models.stacking_model import StackingModel
from openenergy.models.xgboost_model import XGBoostModel


# Offline conformal calibration axis (D2-t2/T-21). Adaptive conformal is
# deliberately absent: adaptation needs realised serving feedback and therefore
# cannot honestly be selected/frozen on an outer evaluation window. Stateful
# online adaptation lives only in ``calibration.online``.
QUANTILE_POLICIES = ["native", "cqr"]

# Lead-pooling axis (F1-t2). Unlike the per-candidate axes, this is a top-level
# pool member: "per_horizon" is the classic one-model-per-lead search, "pooled"
# trains ONE GBM across all leads with lead_time_hours as a feature and D2
# calibration applied per lead bucket. The pooled competitor enters the pool via
# :func:`openenergy.experiments.runner.run_pooled_competitor` and is remembered
# under the ``{kind}_pooled_gbm`` family so scenario memory can rank it against
# the per-horizon champions. "per_horizon" is the default (byte-identical) path.
POOLING_POLICIES = ["per_horizon", "pooled"]


class PipelineConfig(BaseModel):
    model_family: str
    nwp_source: str
    feature_blocks: list[str]
    params: dict
    train_policy: str = "expanding_all"
    point_policy: str = "single_point"
    source_policy: str = "single_source"
    # kPV target transform (C1-t3). Solar-only search axis; wind stays capacity_norm.
    target_policy: str = "capacity_norm"


# Per-kind NWP source policies, in preference order. A "+"-joined value is a
# combo: its weather columns are namespaced per source so the model sees both.
# Solar is deprioritized per research (ecmwf/icon_eu only, no gfs, no combos).
_SOURCE_POLICY_SETS: dict[str, list[str]] = {
    "wind": ["ecmwf", "icon_eu", "gfs", "ecmwf+icon_eu"],
    "solar": ["ecmwf", "icon_eu"],
}


def source_policies_for(kind: str, available_sources: list[str]) -> list[str]:
    """The applicable source policies for a kind, filtered to available sources.

    A policy survives only when every source it names is present in
    ``available_sources`` (so a combo needs both models actually ingested).
    """
    available = set(available_sources)
    return [
        pol for pol in _SOURCE_POLICY_SETS.get(kind, [])
        if all(src in available for src in pol.split("+"))
    ]


def suggest(trial, kind: str, nwp_sources: list[str],
            point_policies: list[str] | None = None,
            source_policies: list[str] | None = None) -> PipelineConfig:
    # point_policy is a real search axis only when a grid offers >1 policy;
    # otherwise it stays "single_point" and adds no trial parameter.
    if point_policies and len(point_policies) > 1:
        point_policy = trial.suggest_categorical("point_policy", point_policies)
    else:
        point_policy = point_policies[0] if point_policies else "single_point"
    # source_policy is a real search axis only when >1 NWP source is available;
    # with a single source it stays fixed and adds no trial parameter.
    if source_policies and len(source_policies) > 1:
        source_policy = trial.suggest_categorical("source_policy", source_policies)
    elif source_policies:
        source_policy = source_policies[0]
    else:
        source_policy = nwp_sources[0]
    family = trial.suggest_categorical(
        "model_family",
        ["lightgbm", "ridge", "random_forest", "xgboost", "catboost", "stacking",
         "quantile_gbm", "seed_bagged_lightgbm"],
    )
    # pcs_weighted is a solar-only weight axis (needs P_cs); wind keeps the base
    # policy set so its FixedTrial param space is byte-identical.
    policy_names = SOLAR_POLICY_NAMES if kind == "solar" else POLICY_NAMES
    train_policy = trial.suggest_categorical("train_policy", policy_names)
    # target_policy is a real axis for solar only (kPV needs the clear-sky
    # envelope). Wind never adds the parameter and stays capacity_norm — the
    # legacy path is byte-identical.
    if kind == "solar":
        target_policy = trial.suggest_categorical("target_policy", ["capacity_norm", "kpv"])
    else:
        target_policy = "capacity_norm"
    family_name = trial.suggest_categorical("feature_family", available_families(kind))
    family_blocks = blocks_for_family(family_name)
    # nwp_source now truthfully mirrors the searched source_policy (single source or
    # a "+"-joined combo) — the data path filters/namespaces weather_raw.model to match.
    nwp = source_policy
    blocks = []
    for name in available_blocks(kind):
        if trial.suggest_categorical(f"blk_{name}", [True, False]):
            blocks.append(name)
    if not blocks:
        blocks = [family_blocks[0] if family_blocks else available_blocks(kind)[0]]
    if family == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "verbose": -1,
        }
    elif family == "ridge":
        params = {"alpha": trial.suggest_float("alpha", 0.01, 10.0)}
    elif family == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 50, 300),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 18),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        }
    elif family == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 400),
            "max_depth": trial.suggest_int("xgb_max_depth", 2, 8),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            "objective": "reg:squarederror",
        }
    elif family == "catboost":
        params = {
            "iterations": trial.suggest_int("cat_iterations", 50, 400),
            "depth": trial.suggest_int("cat_depth", 3, 8),
            "learning_rate": trial.suggest_float("cat_learning_rate", 0.01, 0.2),
            "loss_function": "RMSE",
            "verbose": False,
        }
    elif family == "stacking":
        params = {
            "rf_n_estimators": trial.suggest_int("stack_rf_n_estimators", 50, 200),
            "gb_n_estimators": trial.suggest_int("stack_gb_n_estimators", 50, 200),
            "gb_learning_rate": trial.suggest_float("stack_gb_learning_rate", 0.01, 0.2),
            "final_alpha": trial.suggest_float("stack_final_alpha", 0.01, 10.0),
        }
    elif family == "quantile_gbm":
        params = {
            "n_estimators": trial.suggest_int("qgbm_n_estimators", 50, 300),
            "max_depth": trial.suggest_int("qgbm_max_depth", 2, 6),
            "learning_rate": trial.suggest_float("qgbm_learning_rate", 0.01, 0.2),
        }
    elif family == "seed_bagged_lightgbm":
        params = {
            "n_estimators": trial.suggest_int("sb_n_estimators", 50, 400),
            "num_leaves": trial.suggest_int("sb_num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("sb_learning_rate", 0.01, 0.2),
            "verbose": -1,
            "n_seeds": trial.suggest_int("sb_n_seeds", 3, 5),
        }
    else:
        raise ValueError(f"bilinmeyen model_family: {family}")
    return PipelineConfig(
        model_family=family,
        nwp_source=nwp,
        feature_blocks=blocks,
        params=params,
        train_policy=train_policy,
        point_policy=point_policy,
        source_policy=source_policy,
        target_policy=target_policy,
    )


def build_model(config: PipelineConfig) -> ModelWrapper:
    if config.model_family == "lightgbm":
        return LightGBMModel(params=config.params)
    if config.model_family == "ridge":
        return RidgeModel(alpha=config.params.get("alpha", 1.0))
    if config.model_family == "random_forest":
        return RandomForestModel(params=config.params)
    if config.model_family == "xgboost":
        return XGBoostModel(params=config.params)
    if config.model_family == "catboost":
        return CatBoostModel(params=config.params, quantiles=(0.1, 0.9))
    if config.model_family == "stacking":
        return StackingModel(params=config.params)
    if config.model_family == "quantile_gbm":
        return QuantileGBMModel(params=config.params)
    if config.model_family == "seed_bagged_lightgbm":
        params = dict(config.params)
        n_seeds = params.pop("n_seeds", 5)
        return SeedBaggingModel(params=params, quantiles=(0.1, 0.9), n_seeds=n_seeds)
    raise ValueError(f"bilinmeyen model_family: {config.model_family}")
