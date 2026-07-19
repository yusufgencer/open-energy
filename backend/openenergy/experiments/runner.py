from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from openenergy.calibration.conformal import CQRCalibrator
from openenergy.evaluation import baselines, metrics
from openenergy.evaluation.splits import (
    blocked_cv_splits,
    blocked_oof_splits,
    group_time_split,
    rolling_origin_splits,
    walk_forward_split,
)
from openenergy.datasets.horizon import build_multipoint_dataset
from openenergy.experiments.assembly import (
    AssembledDataset,
    PooledDataset,
    assemble_pooled,
    assemble_raw,
    featurize,
)
from openenergy.features.point_policy import POINT_POLICIES
from openenergy.features.registry import available_blocks
from openenergy.models.lightgbm_model import LightGBMModel
from openenergy.experiments.persistence import (
    create_experiment,
    get_pointer_strategy_trial,
    mark_champion,
    mark_strategy_champion,
    promote_strategy_champion,
    record_strategy_candidate,
    record_strategy_trial,
    record_trial,
)
from openenergy.experiments.scenario_memory import (
    ScenarioRun,
    feature_signature,
    model_signature,
    pooled_strategy_family,
    record_scenario_run,
    refresh_scenario_leaderboard,
    strategy_family,
)
from openenergy.experiments.search_space import (
    PipelineConfig,
    build_model,
    source_policies_for,
    suggest,
)
from openenergy.experiments.strategy import (
    ENSEMBLE_METHODS,
    FittedCandidate,
    StrategyArtifact,
    StrategyConfig,
    QUANTILE_LEVELS,
    combine_bands,
    linear_pool_quantiles,
    optimize_weights,
    select_perquantile_ensemble,
    select_quantile_policy,
    p50_regime_bin_edges,
    weights_from_cv,
)
from openenergy.experiments.train_policy import sample_weights, select_train_indices
from openenergy.features import FeatureConfig
from openenergy.features.target import PlantGeometry, kpv_inverse
from openenergy.models.base import Prediction, enforce_quantile_order

optuna.logging.set_verbosity(optuna.logging.WARNING)

# F1-t1: horizons beyond the leakage-safe day-2 previous-run (24/48h) come from
# previous_day3..7 reforecasts (lead 72..168h). They are only allowed to serve a
# champion once the search PROVES skill against the honest baselines (A2). Legacy
# horizons (≤48h) keep their byte-identical promotion path.
EXTENDED_HORIZON_MIN_HOURS = 72
TARGET_BAND_COVERAGE = 0.80
MIN_BAND_GATE_SAMPLES = 10


def _horizon_may_serve(horizon_hours: int, skill: float) -> bool:
    """Skill gate for extended horizons: ≤48h always serve; ≥72h need skill > 0
    (i.e. beat the best baseline, which subsumes climatology)."""
    if horizon_hours < EXTENDED_HORIZON_MIN_HOURS:
        return True
    return skill > 0.0


def selection_band_diagnostics(
    y_true: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    *,
    target_coverage: float = TARGET_BAND_COVERAGE,
    min_samples: int = MIN_BAND_GATE_SAMPLES,
    confidence: float = 0.95,
) -> dict:
    """Evaluate the validation-selection interval and return its gate verdict.

    Status is deliberately fail-closed without a band, ``unknown`` when fewer
    than ``min_samples`` finite rows are available, and otherwise ``pass`` only
    when the exact Clopper-Pearson interval contains ``target_coverage``.
    """
    if lower is None or upper is None:
        return {
            "band_status": "fail",
            "sel_coverage": None,
            "sel_coverage_lower": None,
            "sel_coverage_upper": None,
            "sel_winkler": None,
            "sel_kupiec_pvalue": None,
            "sel_independence_pvalue": None,
            "sel_conditional_coverage_pvalue": None,
            "sel_n": 0,
        }

    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("selection truth ve bant dizileri aynı shape'te olmalı")
    finite = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[finite], lo[finite], hi[finite]
    n = int(y.size)
    if n == 0:
        return {
            "band_status": "unknown",
            "sel_coverage": None,
            "sel_coverage_lower": None,
            "sel_coverage_upper": None,
            "sel_winkler": None,
            "sel_kupiec_pvalue": None,
            "sel_independence_pvalue": None,
            "sel_conditional_coverage_pvalue": None,
            "sel_n": 0,
        }

    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    hits = (y >= lo) & (y <= hi)
    covered = int(np.sum(hits))
    observed = covered / n
    cp_lower, cp_upper = metrics.clopper_pearson_interval(
        covered, n, confidence=confidence
    )
    result = {
        "band_status": (
            "unknown"
            if n < min_samples
            else "pass"
            if cp_lower <= target_coverage <= cp_upper
            else "fail"
        ),
        "sel_coverage": observed,
        "sel_coverage_lower": cp_lower,
        "sel_coverage_upper": cp_upper,
        "sel_winkler": metrics.winkler_score(
            y, lo, hi, alpha=1.0 - target_coverage
        ),
        "sel_kupiec_pvalue": None,
        "sel_independence_pvalue": None,
        "sel_conditional_coverage_pvalue": None,
        "sel_n": n,
    }
    if n >= 2:
        _, result["sel_kupiec_pvalue"] = metrics.kupiec_pof_test(
            y, lo, hi, target_coverage=target_coverage
        )
        _, result["sel_independence_pvalue"] = (
            metrics.christoffersen_independence_test(y, lo, hi)
        )
        _, result["sel_conditional_coverage_pvalue"] = (
            metrics.christoffersen_conditional_coverage_test(
                y, lo, hi, target_coverage=target_coverage
            )
        )
    return result


def _plant_geometry(con: duckdb.DuckDBPyConnection, point_id: int, kind: str) -> PlantGeometry | None:
    """Fixed-tilt PV geometry for a solar plant's weather point (C1-t7).

    Needed to rebuild the clear-sky envelope P_cs for the kPV target transform.
    Returns None for wind (which never uses kpv) or when the point has no stored
    coordinates. Missing tilt/azimuth fall back to a due-south 30° default.
    """
    if kind != "solar":
        return None
    row = con.execute(
        "SELECT latitude, longitude, tilt, azimuth FROM plant_points WHERE point_id=?",
        [point_id],
    ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        return None
    lat, lon, tilt, azimuth = row
    return PlantGeometry(
        lat=float(lat), lon=float(lon),
        tilt=float(tilt) if tilt is not None else 30.0,
        azimuth=float(azimuth) if azimuth is not None else 180.0,
    )


def _norm_p50(p50: np.ndarray, config: PipelineConfig, p_cs: np.ndarray | None,
              capacity_mw: float) -> np.ndarray:
    """Map a candidate p50 into normalized-power space so every candidate is
    compared on the same scale. kpv predictions live in clear-sky-ratio space;
    invert (ŷ' × P_cs, clamped to nameplate) then divide by capacity. capacity_norm
    predictions are already normalized power — returned unchanged (byte-identical)."""
    if config.target_policy == "kpv" and p_cs is not None:
        return kpv_inverse(p50, p_cs, capacity_mw) / capacity_mw
    return p50


def _norm_prediction(pred, config: PipelineConfig, p_cs: np.ndarray | None,
                     capacity_mw: float):
    """Map a full :class:`Prediction` (all quantiles) into normalized-power space.

    kpv bands are inverted per quantile through ``ŷ' × P_cs`` then normalized;
    the clamped inverse is monotone so p10 ≤ p50 ≤ p90 ordering is preserved.
    capacity_norm predictions pass through unchanged (byte-identical legacy path)."""
    if config.target_policy == "kpv" and p_cs is not None:
        def inv(a):
            return kpv_inverse(a, p_cs, capacity_mw) / capacity_mw
        return Prediction(
            p50=inv(pred.p50),
            p10=inv(pred.p10) if pred.p10 is not None else None,
            p90=inv(pred.p90) if pred.p90 is not None else None,
        )
    return pred


def evaluate_pipeline(ds: AssembledDataset, config: PipelineConfig, *, n_splits: int, embargo: int,
                      eval_y: np.ndarray | None = None, p_cs: np.ndarray | None = None,
                      capacity_mw: float | None = None) -> float:
    """Inner-CV nRMSE of a candidate.

    ``ds`` carries the candidate's *training* target (clear-sky ratio for kpv,
    normalized power otherwise). When ``eval_y``/``p_cs`` are supplied (kpv) the
    fold predictions are inverted to normalized power via :func:`_norm_p50` and
    scored against ``eval_y`` — so kpv and capacity_norm candidates are ranked on
    the same scale. With both None the legacy capacity_norm path is byte-identical.
    """
    n = ds.X.shape[0]
    if n < (n_splits + 1) * (embargo + 5):
        return 1e6
    errs = []
    try:
        folds = blocked_cv_splits(n, n_splits=n_splits, embargo=embargo)
    except ValueError:
        return 1e6
    for tr, te in folds:
        tr_policy = select_train_indices(ds.valid_time, tr, config.train_policy)
        if ds.clean_mask is not None:
            # QC is applied only inside the already-created fold.  It can alter
            # fit membership, never the row universe or fold boundaries.
            tr_policy = tr_policy[np.asarray(ds.clean_mask[tr_policy], dtype=bool)]
        if len(tr_policy) == 0:
            return 1e6
        weights = sample_weights(ds.valid_time, tr_policy, config.train_policy)
        model = build_model(config)
        model.fit(ds.X[tr_policy], ds.y[tr_policy], sample_weight=weights)
        pred = model.predict(ds.X[te])
        p50 = _norm_p50(pred.p50, config, p_cs[te] if p_cs is not None else None, capacity_mw)
        truth = eval_y[te] if eval_y is not None else ds.y[te]
        errs.append(metrics.nrmse(truth, p50))
    return float(np.mean(errs)) if errs else 1e6


def _peak_mask(y_true: np.ndarray, kind: str) -> np.ndarray | None:
    """Boolean mask of peak-production rows, for solar only.

    Phase-A proxy (no clear-sky model yet): daytime top-band production,
    y_true >= 0.8 * max(y_true). Returns None for wind, or when there is no
    positive production, so peak metrics are reported only where meaningful.
    """
    if kind != "solar" or y_true.size == 0:
        return None
    mx = float(np.max(y_true))
    if mx <= 0:
        return None
    return y_true >= 0.8 * mx


def _test_metrics(y_true: np.ndarray, pred, peak_mask: np.ndarray | None = None) -> dict:
    has_band = pred.p10 is not None and pred.p90 is not None
    return {
        "nrmse": metrics.nrmse(y_true, pred.p50),
        "nmae": metrics.nmae(y_true, pred.p50),
        "bias": metrics.bias(y_true, pred.p50),
        "pinball": metrics.pinball_loss(y_true, pred.p10, 0.1) if pred.p10 is not None else None,
        "coverage": metrics.coverage(y_true, pred.p10, pred.p90) if has_band else None,
        "interval_width": metrics.interval_width(pred.p10, pred.p90) if has_band else None,
        "crps": metrics.crps_approx(y_true, pred.p50, pred.p10, pred.p90),
        "peak_bias": metrics.peak_bias(y_true, pred.p50, peak_mask) if peak_mask is not None else None,
        "peak_mae": metrics.peak_mae(y_true, pred.p50, peak_mask) if peak_mask is not None else None,
    }


def _masked_test_metrics(
    y_true: np.ndarray,
    pred: Prediction,
    mask: np.ndarray,
    *,
    kind: str,
) -> dict:
    """Return metrics on a post-split subset without changing all-row scoring."""
    keep = np.asarray(mask, dtype=bool)
    if keep.shape != np.asarray(y_true).shape:
        raise ValueError("quality mask ve evaluation truth aynı shape'te olmalı")
    if not np.any(keep):
        return {name: None for name in _test_metrics(
            np.asarray([0.0]), Prediction(p50=np.asarray([0.0]))
        )}
    clean_pred = Prediction(
        p50=np.asarray(pred.p50)[keep],
        p10=np.asarray(pred.p10)[keep] if pred.p10 is not None else None,
        p90=np.asarray(pred.p90)[keep] if pred.p90 is not None else None,
    )
    clean_y = np.asarray(y_true)[keep]
    return _test_metrics(clean_y, clean_pred, peak_mask=_peak_mask(clean_y, kind))


def _fit_candidate(
    raw,
    cfg: PipelineConfig,
    *,
    kind: str,
    capacity_mw: float,
    train_idx: np.ndarray,
    point_ids: list[int] | None = None,
    geometry: PlantGeometry | None = None,
):
    """Fit one candidate in its own target space.

    Returns ``(ds, model, y_norm, p_cs)`` where ``ds`` holds the *training* target
    (clear-sky ratio for kpv). ``y_norm`` is the normalized-power truth aligned
    row-for-row (for kpv this is a second capacity_norm featurization of the same
    raw/blocks — X, row order and count are identical, only the target differs),
    and ``p_cs`` is the clear-sky envelope for inverting kpv predictions. For
    capacity_norm both extras collapse to ``ds.y`` / ``None`` — byte-identical.
    """
    use_kpv = cfg.target_policy == "kpv"
    ds = featurize(
        raw,
        kind=kind,
        feature_config=FeatureConfig(blocks=cfg.feature_blocks),
        capacity_mw=capacity_mw,
        point_policy=cfg.point_policy,
        point_ids=point_ids,
        target_policy=cfg.target_policy,
        geometry=geometry if use_kpv else None,
    )
    if use_kpv:
        ds_norm = featurize(
            raw,
            kind=kind,
            feature_config=FeatureConfig(blocks=cfg.feature_blocks),
            capacity_mw=capacity_mw,
            point_policy=cfg.point_policy,
            point_ids=point_ids,
        )
        y_norm = ds_norm.y
        p_cs = ds.p_cs
    else:
        y_norm = ds.y
        p_cs = None
    tr_policy = select_train_indices(ds.valid_time, train_idx, cfg.train_policy)
    if ds.clean_mask is not None:
        tr_policy = tr_policy[np.asarray(ds.clean_mask[tr_policy], dtype=bool)]
    if len(tr_policy) == 0:
        raise ValueError("quality mask sonrası boş train dilimi")
    weights = sample_weights(ds.valid_time, tr_policy, cfg.train_policy, pcs=p_cs)
    model = build_model(cfg)
    model.fit(ds.X[tr_policy], ds.y[tr_policy], sample_weight=weights)
    return ds, model, y_norm, p_cs


def _candidate_validation_predictions(
    raw,
    candidates: list[tuple[PipelineConfig, float]],
    *,
    kind: str,
    capacity_mw: float,
    train_idx: np.ndarray,
    val_size: int,
    point_ids: list[int] | None = None,
    geometry: PlantGeometry | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-quantile candidate predictions on the inner validation window.

    Returns ``(val_p10s, val_p50s, val_p90s, y_val)``. Each ``val_pNN`` is an
    ``(n_candidates × n_val)`` matrix in the shared normalized-power space
    (kpv predictions inverted via P_cs), post :func:`enforce_quantile_order`,
    so a per-quantile combiner learns weights/meta consistently. Candidates
    without quantile bands are masked with NaN in their p10/p90 rows — never
    zero-filled — so band aggregation can skip them while still using their p50.
    """
    sub_train, val_idx = walk_forward_split(len(train_idx), test_size=val_size)
    sub_train_idx = train_idx[sub_train]
    val_abs_idx = train_idx[val_idx]
    n_val = len(val_abs_idx)
    p10s, p50s, p90s = [], [], []
    y_val = None
    for cfg, _ in candidates:
        ds, model, y_norm, p_cs = _fit_candidate(
            raw, cfg, kind=kind, capacity_mw=capacity_mw, train_idx=sub_train_idx,
            point_ids=point_ids, geometry=geometry,
        )
        pred = model.predict(ds.X[val_abs_idx])
        norm = enforce_quantile_order(_norm_prediction(
            pred, cfg, p_cs[val_abs_idx] if p_cs is not None else None, capacity_mw))
        p50s.append(norm.p50)
        p10s.append(norm.p10 if norm.p10 is not None else np.full(n_val, np.nan))
        p90s.append(norm.p90 if norm.p90 is not None else np.full(n_val, np.nan))
        y_val = y_norm[val_abs_idx]
    return np.vstack(p10s), np.vstack(p50s), np.vstack(p90s), y_val


@dataclass(frozen=True)
class OOFCalibrationSet:
    """Candidate predictions and split provenance for conformal calibration."""

    p10s: np.ndarray
    p50s: np.ndarray
    p90s: np.ndarray
    y: np.ndarray
    source_indices: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]


def _candidate_oof_predictions(
    raw,
    candidates: list[tuple[PipelineConfig, float]],
    *,
    kind: str,
    capacity_mw: float,
    train_idx: np.ndarray,
    n_splits: int,
    embargo: int,
    horizon: int,
    min_coverage_count: int,
    point_ids: list[int] | None = None,
    geometry: PlantGeometry | None = None,
) -> OOFCalibrationSet:
    """Build blocked, expanding-window OOF predictions for every candidate.

    A fresh candidate model is fitted for each fold. Its predictions are emitted
    only for that fold's later validation block, which is absent from the model's
    fit indices and separated by at least ``max(embargo, horizon)`` positions.
    Returned fold indices are absolute raw-row positions so model disjointness is
    externally auditable instead of being an implicit implementation detail.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    effective_embargo = max(int(embargo), int(horizon))
    local_folds = blocked_oof_splits(
        len(train_idx),
        n_splits=n_splits,
        embargo=effective_embargo,
        min_coverage_count=min_coverage_count,
    )
    folds = tuple(
        (train_idx[fold_train], train_idx[fold_test])
        for fold_train, fold_test in local_folds
    )
    source_indices = np.concatenate([fold_test for _, fold_test in folds])
    n_oof = len(source_indices)
    candidate_p10s: list[np.ndarray] = []
    candidate_p50s: list[np.ndarray] = []
    candidate_p90s: list[np.ndarray] = []
    y_oof: np.ndarray | None = None

    for cfg, _ in candidates:
        fold_p10s: list[np.ndarray] = []
        fold_p50s: list[np.ndarray] = []
        fold_p90s: list[np.ndarray] = []
        fold_truth: list[np.ndarray] = []
        for fold_train, fold_test in folds:
            ds, model, y_norm, p_cs = _fit_candidate(
                raw,
                cfg,
                kind=kind,
                capacity_mw=capacity_mw,
                train_idx=fold_train,
                point_ids=point_ids,
                geometry=geometry,
            )
            pred = enforce_quantile_order(
                _norm_prediction(
                    model.predict(ds.X[fold_test]),
                    cfg,
                    p_cs[fold_test] if p_cs is not None else None,
                    capacity_mw,
                )
            )
            fold_p50s.append(pred.p50)
            fold_p10s.append(
                pred.p10
                if pred.p10 is not None
                else np.full(len(fold_test), np.nan)
            )
            fold_p90s.append(
                pred.p90
                if pred.p90 is not None
                else np.full(len(fold_test), np.nan)
            )
            fold_truth.append(y_norm[fold_test])
        candidate_p10s.append(np.concatenate(fold_p10s))
        candidate_p50s.append(np.concatenate(fold_p50s))
        candidate_p90s.append(np.concatenate(fold_p90s))
        candidate_truth = np.concatenate(fold_truth)
        if y_oof is None:
            y_oof = candidate_truth
        else:
            np.testing.assert_allclose(y_oof, candidate_truth)

    assert y_oof is not None
    assert len(y_oof) == n_oof
    return OOFCalibrationSet(
        p10s=np.vstack(candidate_p10s),
        p50s=np.vstack(candidate_p50s),
        p90s=np.vstack(candidate_p90s),
        y=y_oof,
        source_indices=source_indices,
        folds=folds,
    )


def _strategy_prediction(
    method: str,
    p50s: np.ndarray,
    cv_scores: list[float],
    val_p50s: np.ndarray,
    y_val: np.ndarray,
    val_p10s: np.ndarray | None = None,
    val_p90s: np.ndarray | None = None,
    test_p10s: np.ndarray | None = None,
    test_p90s: np.ndarray | None = None,
) -> tuple[
    Prediction, np.ndarray | None, Ridge | None, StandardScaler | None,
    dict[float, np.ndarray] | None,
]:
    """Combine candidate test predictions into one non-crossing ensemble Prediction.

    Returns ``(pred, weights, meta, scaler, quantile_weights)``. ``pred`` carries
    the combined p50 AND — via :func:`combine_bands` — the combined p10/p90 bands
    (uniform mean over the banded members for legacy methods; learned per-quantile
    weights for vincentization/qra_meta), passed through :func:`enforce_quantile_order`
    so ensemble trials record real coverage/interval_width/crps instead of dropping
    the bands (D1-t3). The remaining fields are the artifact state for ``method``.
    """
    weights: np.ndarray | None = None
    meta: Ridge | None = None
    scaler: StandardScaler | None = None
    quantile_weights: dict[float, np.ndarray] | None = None
    if method == "none":
        p50, weights = p50s[0], np.ones(1)
    elif method == "mean":
        weights = np.ones(p50s.shape[0]) / p50s.shape[0]
        p50 = weights @ p50s
    elif method == "weighted_by_cv":
        weights = weights_from_cv(cv_scores)
        p50 = weights @ p50s
    elif method == "optuna_weights":
        weights = optimize_weights(val_p50s, y_val)
        p50 = weights @ p50s
    elif method == "stacking_meta_ridge":
        scaler = StandardScaler().fit(val_p50s.T)
        meta = Ridge(alpha=1.0).fit(scaler.transform(val_p50s.T), y_val)
        p50 = meta.predict(scaler.transform(p50s.T))
    elif method in ("vincentization", "qra_meta"):
        # Per-quantile combiner (D1-t2): learn a weight vector per quantile from
        # the validation bands, kept only if it beats the plain mean out-of-sample.
        q_mats = {0.1: val_p10s, 0.5: val_p50s, 0.9: val_p90s}
        quantile_weights, _chose = select_perquantile_ensemble(method, q_mats, y_val, QUANTILE_LEVELS)
        p50 = quantile_weights[0.5] @ p50s
    elif method == "linear_pool":
        weights = np.ones(p50s.shape[0], dtype=float) / p50s.shape[0]
        if test_p10s is not None and test_p90s is not None:
            pred = linear_pool_quantiles(
                test_p10s, p50s, test_p90s, weights=weights
            )
            return pred, weights, meta, scaler, quantile_weights
        p50 = weights @ p50s
    else:
        raise ValueError(f"bilinmeyen ensemble method: {method}")
    # D1-t3: combine the candidate test bands per quantile so ensemble coverage /
    # interval_width / crps are recorded, then enforce non-crossing ordering.
    if test_p10s is not None and test_p90s is not None:
        p10, p90 = combine_bands(
            method, quantile_weights, test_p10s, test_p90s,
            p50s=p50s, weights=weights,
        )
    else:
        p10 = p90 = None
    pred = enforce_quantile_order(Prediction(p50=p50, p10=p10, p90=p90))
    return pred, weights, meta, scaler, quantile_weights


def select_ensemble_method(
    val_p50s: np.ndarray,
    y_val: np.ndarray,
    val_p10s: np.ndarray,
    val_p90s: np.ndarray,
    *,
    cv_scores: list[float],
    method_limit: int,
    horizon: int,
    return_diagnostics: bool = False,
) -> str | tuple[str, dict]:
    """Choose the ensemble combination method HONESTLY — on a held-out slice of the
    VALIDATION window, never the test window (T-08).

    The validation predictions are split chronologically into a fit slice (fits each
    combiner) and a disjoint, later select slice (scores it out-of-sample). The
    method with the lowest select-slice CRPS wins; ``mean`` — a hard-to-beat
    baseline — takes ties via ENSEMBLE_METHODS ordering and is the fallback when the
    validation window is too small to split (so selection is never made in-sample).

    Carries NO y_test / test-band parameter by design: the reporting window cannot
    leak into method selection.
    """
    # A single candidate cannot be combined — 'none' is the only option.
    if method_limit < 2:
        result = ("none", {"mcs_members": ["none"], "mcs_loss_rows": 0})
        return result if return_diagnostics else result[0]
    n_val = val_p50s.shape[1]
    half = n_val // 2
    if half < 1 or (n_val - half) < 1:
        result = ("mean", {"mcs_members": ["mean"], "mcs_loss_rows": n_val})
        return result if return_diagnostics else result[0]
    fit_sl = slice(0, half)
    sel_sl = slice(half, n_val)
    y_sel = y_val[sel_sl]

    # 'mean' is the default (the forecast-combination-puzzle baseline). A learned or
    # alternative combiner may only replace it when it beats the equal-weight average
    # by a STATISTICALLY SIGNIFICANT margin on the out-of-sample select slice
    # (Diebold-Mariano on paired CRPS differentials) — noise alone must not
    # earn a departure from the average. Among the significant challengers the one
    # with the lowest select-slice CRPS wins. A method without a complete band is
    # scored as a point mass at p50 (CRPS == MAE), the same explicit fallback for
    # every method; it cannot gain an arbitrary advantage from a missing interval.
    def _sel_prediction(method: str) -> Prediction:
        pred_sel, *_ = _strategy_prediction(
            method,
            val_p50s[:, sel_sl], cv_scores,
            val_p50s[:, fit_sl], y_val[fit_sl],
            val_p10s[:, fit_sl], val_p90s[:, fit_sl],
            val_p10s[:, sel_sl], val_p90s[:, sel_sl],
        )
        return pred_sel

    # Parsimony order is explicit and stable. MCS first removes methods that are
    # demonstrably inferior; the earliest surviving method then wins. This keeps
    # the plain mean for statistically equivalent alternatives.
    parsimony_order = [m for m in ENSEMBLE_METHODS if m != "none"]
    method_names: list[str] = []
    method_losses: list[np.ndarray] = []
    predictions: dict[str, Prediction] = {}
    for method in parsimony_order:
        try:
            predictions[method] = _sel_prediction(method)
        except Exception:
            continue
        pred = predictions[method]
        loss = metrics.crps_values(y_sel, pred.p50, pred.p10, pred.p90)
        if np.all(np.isfinite(loss)):
            method_names.append(method)
            method_losses.append(loss)

    if not method_names:
        result = ("mean", {"mcs_members": ["mean"], "mcs_loss_rows": len(y_sel)})
        return result if return_diagnostics else result[0]
    loss_matrix = np.column_stack(method_losses)
    block_length = max(1, min(len(y_sel), int(round(len(y_sel) ** (1.0 / 3.0)))))
    member_indices = metrics.model_confidence_set(
        loss_matrix,
        alpha=0.05,
        n_resamples=1_000,
        block_length=block_length,
        seed=0,
    )
    member_names = [method_names[i] for i in member_indices]
    best_method = next(name for name in parsimony_order if name in member_names)
    diagnostics = {
        "mcs_members": member_names,
        "mcs_loss_rows": int(len(y_sel)),
        "mcs_alpha": 0.05,
        "mcs_bootstrap": "moving",
        "mcs_block_length": block_length,
        "mcs_seed": 0,
        "mcs_resamples": 1_000,
    }
    return (best_method, diagnostics) if return_diagnostics else best_method


def run_experiment(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
                   horizon_hours_list: list[int], capacity_mw: float, kind: str,
                   nwp_sources: list[str], n_trials: int, n_splits: int = 4, embargo: int = 24,
                   test_size: int | None = None, seed: int = 42, models_dir: str | Path,
                   top_k: int = 10, point_ids: list[int] | None = None,
                   source: str | None = None, outer_splits: int = 3) -> int:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = create_experiment(con, plant_id, horizon_hours_list)

    # point_policy is a real search axis only when a grid of >1 point is supplied.
    multipoint = bool(point_ids and len(point_ids) > 1)
    policies = POINT_POLICIES if multipoint else ["single_point"]
    # ``source`` pins the data path, so it must also pin the Optuna/config axis.
    # Without an explicit pin, expose only policies that the caller says are
    # available. Keep arbitrary legacy provider names usable when they are not in
    # the curated policy set (for example ``icon`` / ``best_match``).
    source_policies = [source] if source else source_policies_for(kind, nwp_sources)
    if not source_policies:
        source_policies = list(dict.fromkeys(nwp_sources))
    # Solar plants may search the kPV target transform, which needs the plant's
    # clear-sky geometry to rebuild P_cs. None for wind (kpv is never proposed).
    geometry = _plant_geometry(con, point_id, kind)

    for horizon in horizon_hours_list:
        # C1: assemble raw ONCE per horizon — fixed row universe (valid_time sorted)
        if multipoint:
            raw = build_multipoint_dataset(con, plant_id=plant_id, point_ids=point_ids,
                                           horizon_hours=horizon, source_policy=source)
        else:
            raw = assemble_raw(con, plant_id=plant_id, point_id=point_id,
                               horizon_hours=horizon, source=source)
        n = raw.height
        if n < 30:
            continue
        # T-22: use several fixed-window rolling origins when the history can
        # support both the outer folds and the existing inner-CV contract.  All
        # model/ensemble/quantile selection below uses the *first* (therefore
        # common-to-all) outer training window.  No outer-test label can influence
        # a selected policy.  Small histories retain the legacy one-origin split.
        #
        # ``test_size`` remains backwards compatible: when explicitly supplied it
        # is the per-origin test width.  The rolling default is deliberately half
        # the legacy 20% holdout, yielding three useful origins without making the
        # fixed training window needlessly small.
        outer_test_size = test_size or max(1, n // 10)
        min_selection_train = (n_splits + 1) * (embargo + 5)
        outer_folds: list[tuple[np.ndarray, np.ndarray]]
        try:
            candidate_folds = rolling_origin_splits(
                raw["valid_time"].to_numpy(),
                n_splits=outer_splits,
                test_size=outer_test_size,
                embargo=embargo,
            )
            if len(candidate_folds[0][0]) < min_selection_train:
                raise ValueError("rolling-origin train window inner CV için kısa")
            outer_folds = candidate_folds
        except ValueError:
            legacy_test_size = test_size or max(1, n // 5)
            outer_folds = [walk_forward_split(n, test_size=legacy_test_size)]

        valid_times = raw["valid_time"].to_list()
        evaluation_n_days = len({value.date() for value in valid_times})
        evaluation_origins = len(outer_folds)
        evaluation_seasons = len(
            {(value.month % 12) // 3 for value in valid_times}
        )
        evaluation_grade_value = metrics.evaluation_grade(
            n_days=evaluation_n_days,
            origins=evaluation_origins,
            seasons=evaluation_seasons,
        )

        # Selection/calibration universe is contained in every outer fold's train
        # side.  ``te_idx`` concatenates reporting-only origins for aggregate
        # champion metrics; it is never captured by the Optuna objective.
        tr_idx = outer_folds[0][0]
        te_idx = np.concatenate([fold_test for _, fold_test in outer_folds])

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial, _raw=raw, _tr_idx=tr_idx, _geom=geometry):
            cfg = suggest(
                trial, kind, nwp_sources, policies, source_policies=source_policies
            )
            use_kpv = cfg.target_policy == "kpv"
            # C1: featurize with per-trial blocks + point_policy — search axis is real.
            # target_policy selects the training target space (kpv needs geometry).
            ds_cfg = featurize(_raw, kind=kind,
                               feature_config=FeatureConfig(blocks=cfg.feature_blocks),
                               capacity_mw=capacity_mw,
                               point_policy=cfg.point_policy, point_ids=point_ids,
                               target_policy=cfg.target_policy,
                               geometry=_geom if use_kpv else None)
            train_ds = AssembledDataset(
                ds_cfg.X[_tr_idx], ds_cfg.y[_tr_idx],
                ds_cfg.feature_names, ds_cfg.valid_time[_tr_idx],
                clean_mask=(
                    ds_cfg.clean_mask[_tr_idx]
                    if ds_cfg.clean_mask is not None else None
                ),
            )
            # kpv is scored in normalized-power space (shared with capacity_norm) so
            # the search ranks target policies fairly: supply the norm truth + P_cs.
            eval_y = p_cs = None
            if use_kpv:
                ds_norm = featurize(_raw, kind=kind,
                                    feature_config=FeatureConfig(blocks=cfg.feature_blocks),
                                    capacity_mw=capacity_mw,
                                    point_policy=cfg.point_policy, point_ids=point_ids)
                eval_y = ds_norm.y[_tr_idx]
                p_cs = ds_cfg.p_cs[_tr_idx] if ds_cfg.p_cs is not None else None
            # Outer te_idx MUST NOT appear in the objective
            return evaluate_pipeline(train_ds, cfg, n_splits=n_splits, embargo=embargo,
                                     eval_y=eval_y, p_cs=p_cs, capacity_mw=capacity_mw)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        completed = []
        for trial in study.trials:
            if trial.value is None:
                continue
            cfg = suggest(
                optuna.trial.FixedTrial(trial.params), kind, nwp_sources, policies,
                source_policies=source_policies,
            )
            completed.append((cfg, float(trial.value)))
        completed.sort(key=lambda item: item[1])
        top_candidates = [(cfg, cv) for cfg, cv in completed[:top_k] if cv < 1e6]

        if not top_candidates:
            best_cfg = suggest(
                optuna.trial.FixedTrial(study.best_params), kind, nwp_sources, policies,
                source_policies=source_policies,
            )
            record_trial(con, experiment_id=experiment_id, horizon_hours=horizon,
                         pipeline_config=best_cfg, cv_nrmse=None,
                         test_metrics={"nrmse": None, "nmae": None, "bias": None, "pinball": None},
                         skill_score=0.0)
            continue

        fitted: list[FittedCandidate] = []
        test_p50s = []
        test_p10s: list[np.ndarray] = []
        test_p90s: list[np.ndarray] = []
        cv_scores = []
        y_test = None
        test_clean_mask = None
        ref_y_train = None
        for cfg, cv_score in top_candidates:
            fold_p50: list[np.ndarray] = []
            fold_p10: list[np.ndarray] = []
            fold_p90: list[np.ndarray] = []
            fold_truth: list[np.ndarray] = []
            fold_clean: list[np.ndarray] = []
            # Refit at every origin.  The last fitted model is retained in the
            # serving artifact, while all fold predictions feed aggregate
            # reporting/champion selection.
            for fold_train, fold_test in outer_folds:
                ds_cand, model, y_norm, p_cs_cand = _fit_candidate(
                    raw, cfg, kind=kind, capacity_mw=capacity_mw,
                    train_idx=fold_train, point_ids=point_ids, geometry=geometry,
                )
                pred_fold = model.predict(ds_cand.X[fold_test])
                pred_fold = _norm_prediction(
                    pred_fold, cfg,
                    p_cs_cand[fold_test] if p_cs_cand is not None else None,
                    capacity_mw,
                )
                fold_p50.append(np.asarray(pred_fold.p50))
                fold_p10.append(
                    np.asarray(pred_fold.p10)
                    if pred_fold.p10 is not None
                    else np.full(len(fold_test), np.nan)
                )
                fold_p90.append(
                    np.asarray(pred_fold.p90)
                    if pred_fold.p90 is not None
                    else np.full(len(fold_test), np.nan)
                )
                fold_truth.append(np.asarray(y_norm[fold_test]))
                fold_clean.append(
                    np.asarray(ds_cand.clean_mask[fold_test], dtype=bool)
                    if ds_cand.clean_mask is not None
                    else np.ones(len(fold_test), dtype=bool)
                )

            aggregate_p10 = np.concatenate(fold_p10)
            aggregate_p90 = np.concatenate(fold_p90)
            pred = Prediction(
                p50=np.concatenate(fold_p50),
                p10=None if np.isnan(aggregate_p10).all() else aggregate_p10,
                p90=None if np.isnan(aggregate_p90).all() else aggregate_p90,
            )
            y_te_norm = np.concatenate(fold_truth)
            candidate_clean_mask = np.concatenate(fold_clean)
            tm = _test_metrics(y_te_norm, pred, peak_mask=_peak_mask(y_te_norm, kind))
            record_trial(con, experiment_id=experiment_id, horizon_hours=horizon,
                         pipeline_config=cfg, cv_nrmse=cv_score,
                         test_metrics=tm, skill_score=0.0)
            fitted.append(FittedCandidate(
                config=cfg,
                model=model,
                feature_blocks=cfg.feature_blocks,
                feature_names=ds_cand.feature_names,
                cv_nrmse=cv_score,
                test_nrmse=tm["nrmse"],
            ))
            test_p50s.append(pred.p50)
            # Mask bandless candidates with NaN (never zero-fill) so band
            # aggregation skips them while still combining the rest (D1-t3).
            n_pred = len(pred.p50)
            test_p10s.append(pred.p10 if pred.p10 is not None else np.full(n_pred, np.nan))
            test_p90s.append(pred.p90 if pred.p90 is not None else np.full(n_pred, np.nan))
            cv_scores.append(cv_score)
            y_test = y_te_norm
            test_clean_mask = candidate_clean_mask
            ref_y_train = y_norm[tr_idx]

        assert y_test is not None and ref_y_train is not None and test_clean_mask is not None
        val_size = max(1, min(len(tr_idx) // 5, len(tr_idx) - 1))
        val_p10s, val_p50s, val_p90s, y_val = _candidate_validation_predictions(
            raw, top_candidates, kind=kind, capacity_mw=capacity_mw,
            train_idx=tr_idx, val_size=val_size, point_ids=point_ids, geometry=geometry,
        )

        p50s = np.vstack(test_p50s)
        test_p10s_mat = np.vstack(test_p10s)
        test_p90s_mat = np.vstack(test_p90s)
        # Select the ensemble method on a held-out VALIDATION slice — never on the
        # test window (T-08). The test window is scored once below for reporting only.
        method, mcs_diagnostics = select_ensemble_method(
            val_p50s, y_val, val_p10s, val_p90s,
            cv_scores=cv_scores, method_limit=len(fitted), horizon=horizon,
            return_diagnostics=True,
        )
        best_pred, weights, meta, scaler, quantile_weights = _strategy_prediction(
            method, p50s, cv_scores, val_p50s, y_val, val_p10s, val_p90s,
            test_p10s_mat, test_p90s_mat,
        )
        best_metrics = _test_metrics(y_test, best_pred, peak_mask=_peak_mask(y_test, kind))
        clean_metrics = _masked_test_metrics(
            y_test, best_pred, test_clean_mask, kind=kind
        )

        strategy_artifact = StrategyArtifact(
            candidates=fitted,
            ensemble_method=method,
            weights=weights,
            meta_model=meta,
            meta_scaler=scaler,
            # Per-quantile combination weights (D1-t2); None for legacy methods.
            quantile_weights=quantile_weights,
            # Serving parity (B2-t5): persist the grid point set + source policy
            # used to assemble this horizon's raw frame so predict_serving can
            # rebuild the exact multipoint transform. None for single-point runs.
            point_ids=list(point_ids) if multipoint else None,
            source_policy=source if multipoint else None,
            # kPV serving parity (C1-t3): persist the target policy + geometry so a
            # pure-kpv champion inverts ŷ' × P_cs at serving. Only homogeneous
            # kpv ensembles set "kpv" (the single artifact inverse must match every
            # member); mixed / capacity_norm ensembles stay on the legacy path.
            target_policy=(
                "kpv" if all(c.config.target_policy == "kpv" for c in fitted) else "capacity_norm"
            ),
            geometry=geometry,
        )

        # T-09/T-18: policy selection stays on a trailing validation-only holdout,
        # while calibrator fitting uses a much larger blocked OOF set. Every OOF
        # row comes from a fresh model trained strictly before that row with a
        # horizon-safe embargo. The outer test remains reporting-only.
        ens_val_p10, ens_val_p90 = combine_bands(
            method, quantile_weights, val_p10s, val_p90s,
            p50s=val_p50s, weights=weights,
        )
        if method == "linear_pool":
            val_pool = linear_pool_quantiles(
                val_p10s, val_p50s, val_p90s, weights=weights
            )
            val_p50_comb = val_pool.p50
        else:
            val_p50_comb = strategy_artifact.predict_from_candidate_arrays(val_p50s)
        if ens_val_p10 is not None and ens_val_p90 is not None:
            val_band = enforce_quantile_order(
                Prediction(p50=val_p50_comb, p10=ens_val_p10, p90=ens_val_p90))
            ens_val_p10, ens_val_p90 = val_band.p10, val_band.p90
        quantile_split = len(y_val) // 2
        if quantile_split == 0 or quantile_split == len(y_val):
            quantile_policy, calibrator = "native", None
        else:
            # Exclude the entire ensemble validation window from OOF generation.
            # Learned ensemble weights/meta-models consume that window, so letting
            # it contribute calibration residuals would break full-pipeline
            # model-disjointness even though each base model were fold-disjoint.
            oof_train_idx = tr_idx[:-len(y_val)]
            min_oof_count = int(np.ceil(0.60 * len(tr_idx)))
            try:
                oof = _candidate_oof_predictions(
                    raw,
                    top_candidates,
                    kind=kind,
                    capacity_mw=capacity_mw,
                    train_idx=oof_train_idx,
                    n_splits=n_splits,
                    embargo=embargo,
                    horizon=horizon,
                    min_coverage_count=min_oof_count,
                    point_ids=point_ids,
                    geometry=geometry,
                )
            except ValueError:
                # Very short histories cannot simultaneously provide a
                # horizon-safe embargo, >=60% OOF coverage, and a disjoint
                # selection holdout. Keep the honest native policy in that case.
                quantile_policy, calibrator = "native", None
                oof = None

        if quantile_split not in (0, len(y_val)) and oof is not None:
            oof_p10, oof_p90 = combine_bands(
                method, quantile_weights, oof.p10s, oof.p90s,
                p50s=oof.p50s, weights=weights,
            )
            if method == "linear_pool":
                oof_p50 = linear_pool_quantiles(
                    oof.p10s, oof.p50s, oof.p90s, weights=weights
                ).p50
            else:
                oof_p50 = strategy_artifact.predict_from_candidate_arrays(oof.p50s)
            if oof_p10 is not None and oof_p90 is not None:
                oof_band = enforce_quantile_order(
                    Prediction(p50=oof_p50, p10=oof_p10, p90=oof_p90)
                )
                oof_p10, oof_p90 = oof_band.p10, oof_band.p90
            selection_p50 = val_p50_comb[quantile_split:]
            regime_edges = p50_regime_bin_edges(oof_p50)
            quantile_policy, calibrator = select_quantile_policy(
                calibration_lower=oof_p10,
                calibration_upper=oof_p90,
                y_calibration=oof.y,
                selection_lower=(
                    ens_val_p10[quantile_split:] if ens_val_p10 is not None else None
                ),
                selection_upper=(
                    ens_val_p90[quantile_split:] if ens_val_p90 is not None else None
                ),
                y_selection=y_val[quantile_split:],
                calibration_regime=oof_p50,
                selection_regime=selection_p50,
                bin_edges=regime_edges,
            )
        strategy_artifact.quantile_policy = quantile_policy
        strategy_artifact.calibrator = calibrator
        selection_lower = (
            ens_val_p10[quantile_split:]
            if ens_val_p10 is not None
            else None
        )
        selection_upper = (
            ens_val_p90[quantile_split:]
            if ens_val_p90 is not None
            else None
        )
        selection_truth = y_val[quantile_split:]
        if calibrator is not None and selection_lower is not None and selection_upper is not None:
            selection_lower, selection_upper = calibrator.adjust(
                selection_lower,
                selection_upper,
                regime=(
                    val_p50_comb[quantile_split:]
                    if getattr(calibrator, "bin_edges", None) is not None
                    else None
                ),
            )
        band_diagnostics = selection_band_diagnostics(
            selection_truth, selection_lower, selection_upper
        )
        if calibrator is not None:
            cal_p10, cal_p90 = calibrator.adjust(
                best_pred.p10,
                best_pred.p90,
                regime=(
                    best_pred.p50
                    if getattr(calibrator, "bin_edges", None) is not None
                    else None
                ),
            )
            best_pred = enforce_quantile_order(
                Prediction(p50=best_pred.p50, p10=cal_p10, p90=cal_p90))
            best_metrics = _test_metrics(y_test, best_pred, peak_mask=_peak_mask(y_test, kind))
            clean_metrics = _masked_test_metrics(
                y_test, best_pred, test_clean_mask, kind=kind
            )

        # Reporting uncertainty is allowed to use the outer rolling-origin
        # window, but it must never flow back into selection above.
        metric_ci_block = max(1, int(round(len(y_test) ** (1.0 / 3.0))))
        metric_ci_lower, metric_ci_upper = metrics.block_bootstrap_ci(
            np.square(np.asarray(y_test) - np.asarray(best_pred.p50)),
            statistic=lambda squared: float(np.sqrt(np.mean(squared))),
            confidence=0.95,
            n_resamples=2_000,
            block_length=metric_ci_block,
            seed=seed,
            method="moving",
        )
        uncertainty_diagnostics = {
            **mcs_diagnostics,
            "champion_metric": "nrmse",
            "champion_metric_lower": metric_ci_lower,
            "champion_metric_upper": metric_ci_upper,
            "metric_ci_confidence": 0.95,
            "metric_ci_bootstrap": "moving",
            "metric_ci_block_length": metric_ci_block,
            "metric_ci_seed": seed,
            "metric_ci_resamples": 2_000,
        }

        # Baselines use the observations available before each origin, including
        # the embargo gap.  The embargo constrains model fitting/selection; it must
        # not erase already-observed history from persistence references.  In
        # particular, diurnal persistence needs t-24 observations at the origin.
        # Fold scores are then averaged on the same outer test blocks as the model.
        ref_by_fold = []
        clean_ref_by_fold = []
        for _fold_train, fold_test in outer_folds:
            history_idx = np.arange(0, int(np.min(fold_test)), dtype=int)
            ref_by_fold.append(
                baselines.baseline_errors(
                    y_norm[history_idx],
                    y_norm[fold_test],
                    metrics.nrmse,
                    vt_train=ds_cand.valid_time[history_idx],
                    vt_test=ds_cand.valid_time[fold_test],
                )
            )
            clean_ref_by_fold.append(
                baselines.baseline_errors(
                    y_norm[history_idx],
                    y_norm[fold_test],
                    metrics.nrmse,
                    vt_train=ds_cand.valid_time[history_idx],
                    vt_test=ds_cand.valid_time[fold_test],
                    train_mask=(
                        ds_cand.clean_mask[history_idx]
                        if ds_cand.clean_mask is not None else None
                    ),
                    test_mask=(
                        ds_cand.clean_mask[fold_test]
                        if ds_cand.clean_mask is not None else None
                    ),
                )
            )
        ref = {
            name: float(np.mean([fold_ref[name] for fold_ref in ref_by_fold]))
            for name in ref_by_fold[0]
        }
        skill = metrics.skill_score(best_metrics["nrmse"], min(ref.values()))
        usable_clean_refs = [fold for fold in clean_ref_by_fold if fold]
        clean_ref = (
            {
                name: float(np.mean([fold_ref[name] for fold_ref in usable_clean_refs]))
                for name in usable_clean_refs[0]
            }
            if usable_clean_refs else {}
        )
        clean_skill = (
            metrics.skill_score(clean_metrics["nrmse"], min(clean_ref.values()))
            if clean_metrics.get("nrmse") is not None and clean_ref else None
        )
        train_clean_mask = (
            np.asarray(ds_cand.clean_mask[tr_idx], dtype=bool)
            if ds_cand.clean_mask is not None
            else np.ones(len(tr_idx), dtype=bool)
        )
        quality_audit = {
            "train_all_count": int(len(tr_idx)),
            "train_clean_count": int(np.sum(train_clean_mask)),
            "eval_all_count": int(len(y_test)),
            "eval_clean_count": int(np.sum(test_clean_mask)),
        }
        # F1-t1 skill gate: an extended horizon (≥72h) may only crown a champion
        # once it beats the baselines. When it doesn't, the trial + scenario_run
        # are still recorded (an explicit "no skill" verdict), but nothing is
        # promoted to champion, so it never serves and any prior champion stays.
        serves = (
            _horizon_may_serve(horizon, skill)
            and band_diagnostics["band_status"] == "pass"
        )

        artifact = models_dir / f"exp{experiment_id}_h{horizon}_strategy.pkl"
        strategy_artifact.save(artifact)
        strategy_config = StrategyConfig(
            candidates=[cand.config for cand in fitted],
            ensemble_method=method,
            top_k=len(fitted),
        )
        strategy_trial_id = record_strategy_trial(
            con, experiment_id=experiment_id, horizon_hours=horizon,
            strategy_config=strategy_config, cv_nrmse=float(np.mean(cv_scores)),
            test_metrics=best_metrics, skill_score=skill, artifact_path=str(artifact),
            clean_metrics=clean_metrics, clean_skill_score=clean_skill,
            quality_audit=quality_audit,
            selection_diagnostics=band_diagnostics,
            uncertainty_diagnostics=uncertainty_diagnostics,
            evaluation_grade=evaluation_grade_value,
            evaluation_n_days=evaluation_n_days,
            evaluation_origins=evaluation_origins,
            evaluation_seasons=evaluation_seasons,
        )
        for rank, cand in enumerate(fitted, start=1):
            record_strategy_candidate(
                con, strategy_trial_id=strategy_trial_id, rank=rank,
                candidate_config=cand.config, cv_nrmse=cand.cv_nrmse,
                test_nrmse=cand.test_nrmse, artifact_path=None,
                feature_names=cand.feature_names,
            )
        if serves:
            mark_strategy_champion(con, experiment_id, horizon, strategy_trial_id)
            con.execute(
                """
                UPDATE strategy_trials SET is_champion=false
                WHERE horizon_hours=? AND is_champion=true AND strategy_trial_id != ?
                AND experiment_id IN (SELECT experiment_id FROM experiments WHERE plant_id=?)
                AND experiment_id != ?
                """,
                [horizon, strategy_trial_id, plant_id, experiment_id],
            )
            # Bootstrap the served champion pointer on the FIRST champion for this
            # plant+horizon. Later runs are challengers: they do not move the served
            # pointer here — only a DM-gated retrain promotion may (T-05).
            if get_pointer_strategy_trial(con, plant_id, horizon) is None:
                promote_strategy_champion(con, plant_id, horizon, strategy_trial_id)
        feature_sig = feature_signature([
            block for cand in fitted for block in cand.feature_blocks
        ])
        model_sig = model_signature([cand.config.model_family for cand in fitted])
        train_policies = sorted({cand.config.train_policy for cand in fitted})
        point_policy_str = "+".join(sorted({cand.config.point_policy for cand in fitted}))
        source_policy_str = "+".join(sorted({cand.config.source_policy for cand in fitted}))
        target_policy_str = "+".join(sorted({cand.config.target_policy for cand in fitted}))
        has_nwp_spread = "nwp_spread" in feature_sig.split("+")
        family = strategy_family(
            asset_kind=kind,
            ensemble_policy=method,
            has_nwp_spread=has_nwp_spread,
            point_policy=point_policy_str,
            source_policy=source_policy_str,
            target_policy=target_policy_str,
        )
        record_scenario_run(
            con,
            ScenarioRun(
                experiment_id=experiment_id,
                plant_id=plant_id,
                asset_kind=kind,
                horizon_hours=horizon,
                strategy_trial_id=strategy_trial_id,
                strategy_family=family,
                feature_signature=feature_sig,
                model_signature=model_sig,
                train_policy="+".join(train_policies),
                point_policy=point_policy_str,
                source_policy=source_policy_str,
                target_policy=target_policy_str,
                ensemble_policy=method,
                quantile_policy=quantile_policy,
                nrmse=best_metrics.get("nrmse"),
                nmae=best_metrics.get("nmae"),
                bias=best_metrics.get("bias"),
                pinball=best_metrics.get("pinball"),
                coverage=best_metrics.get("coverage"),
                crps=best_metrics.get("crps"),
                peak_bias=best_metrics.get("peak_bias"),
                peak_mae=best_metrics.get("peak_mae"),
                skill_score=skill,
                evaluation_grade=evaluation_grade_value,
                evaluation_n_days=evaluation_n_days,
                evaluation_origins=evaluation_origins,
                evaluation_seasons=evaluation_seasons,
                is_champion=serves,
            ),
        )
        refresh_scenario_leaderboard(con)

        best_cfg = fitted[0].config
        trial_id = record_trial(con, experiment_id=experiment_id, horizon_hours=horizon,
                                pipeline_config=best_cfg, cv_nrmse=cv_scores[0],
                                test_metrics=best_metrics, skill_score=skill,
                                clean_metrics=clean_metrics,
                                clean_skill_score=clean_skill,
                                quality_audit=quality_audit)
        if not serves:
            # Extended horizon without proven skill → record the verdict but do not
            # promote, demote, or write a champion model. Serving stays unchanged.
            continue
        mark_champion(con, experiment_id, horizon, trial_id)

        # M2: globally demote prior champions for same plant+horizon from other experiments,
        # leaving exactly one champion per plant+horizon across all experiments.
        con.execute(
            """
            UPDATE experiment_trials SET is_champion=false
            WHERE horizon_hours=? AND is_champion=true AND trial_id != ?
            AND experiment_id IN (SELECT experiment_id FROM experiments WHERE plant_id=?)
            AND experiment_id != ?
            """,
            [horizon, trial_id, plant_id, experiment_id],
        )

        # Demote prior champion models for same plant+horizon before inserting new one
        con.execute(
            """
            UPDATE models SET is_champion=false
            WHERE horizon_hours=?
            AND experiment_id IN (SELECT experiment_id FROM experiments WHERE plant_id=?)
            AND experiment_id != ?
            """,
            [horizon, plant_id, experiment_id],
        )
        con.execute(
            "INSERT INTO models (experiment_id, horizon_hours, model_family, artifact_path, metrics, is_champion, feature_names) "
            "VALUES (?,?,?,?,?,true,?)",
            [experiment_id, horizon, f"strategy:{method}", str(artifact),
             json.dumps(best_metrics), json.dumps(fitted[0].feature_names)])

    con.execute("UPDATE experiments SET status='done' WHERE experiment_id=?", [experiment_id])
    return experiment_id


# ---------------------------------------------------------------------------
# F1-t2: pooled-GBM competitor (one model across all leads + per-lead calibration)
# ---------------------------------------------------------------------------


@dataclass
class PooledResult:
    """Outcome of the pooled-GBM competitor (F1-t2).

    ``per_lead`` maps each lead (hours, as float) to its test-window metric dict —
    the pooled model's per-lead skill, judged on the same metrics as the
    per-horizon champions. ``train_idx`` / ``test_idx`` are the positional split
    into the pooled dataset; they are exposed so callers (and tests) can verify the
    split partitions by valid_time (no cross-lead leakage).
    """

    per_lead: dict[float, dict]
    train_idx: np.ndarray
    test_idx: np.ndarray


def _lead_bin_edges(leads: np.ndarray) -> np.ndarray | None:
    """Interior bucket boundaries (midpoints) between distinct leads, or None when
    a single lead makes bucketing meaningless (calibrator falls back to global)."""
    uniq = np.unique(np.asarray(leads, dtype=float))
    if uniq.size < 2:
        return None
    return (uniq[:-1] + uniq[1:]) / 2.0


def evaluate_pooled(pooled: PooledDataset, *, params: dict | None = None,
                    alpha: float = 0.20, test_fraction: float = 0.20,
                    cal_fraction: float = 0.25) -> PooledResult:
    """Train ONE GBM across all leads and return per-lead test metrics (F1-t2).

    ``lead_time_hours`` is already a feature column of ``pooled.X``. The
    train/test split is by valid_time (:func:`group_time_split`) so a valid_time's
    rows at different leads never straddle the boundary — no cross-lead leakage.
    A D2 conformal calibrator is fit *per lead bucket* on a held-out calibration
    window carved (again by valid_time) from the train portion, then applied to
    the test bands so each lead's coverage is corrected independently — short
    leads naturally get tighter bands. Metrics are then sliced per lead.
    """
    tr, te = group_time_split(pooled.valid_time, test_fraction=test_fraction)

    fit_idx = tr
    cal_idx = np.array([], dtype=int)
    try:
        sub_fit, sub_cal = group_time_split(pooled.valid_time[tr], test_fraction=cal_fraction)
        if sub_fit.size > 0 and sub_cal.size > 0:
            fit_idx, cal_idx = tr[sub_fit], tr[sub_cal]
    except ValueError:
        pass

    model = LightGBMModel(params=params or {})
    model.fit(pooled.X[fit_idx], pooled.y[fit_idx])

    edges = _lead_bin_edges(pooled.lead_time_hours)
    calibrator: CQRCalibrator | None = None
    if cal_idx.size > 0:
        cal_pred = model.predict(pooled.X[cal_idx])
        calibrator = CQRCalibrator(alpha=alpha, bin_edges=edges)
        calibrator.fit(
            pooled.y[cal_idx], cal_pred.p10, cal_pred.p90,
            regime=pooled.lead_time_hours[cal_idx] if edges is not None else None,
        )

    te_pred = model.predict(pooled.X[te])
    if calibrator is not None:
        lo, hi = calibrator.adjust(
            te_pred.p10, te_pred.p90,
            regime=pooled.lead_time_hours[te] if edges is not None else None,
        )
        te_pred = enforce_quantile_order(Prediction(p50=te_pred.p50, p10=lo, p90=hi))

    te_leads = pooled.lead_time_hours[te]
    per_lead: dict[float, dict] = {}
    for lead in np.unique(te_leads):
        m = te_leads == lead
        lead_pred = Prediction(
            p50=te_pred.p50[m],
            p10=te_pred.p10[m] if te_pred.p10 is not None else None,
            p90=te_pred.p90[m] if te_pred.p90 is not None else None,
        )
        per_lead[float(lead)] = _test_metrics(pooled.y[te][m], lead_pred)
    return PooledResult(per_lead=per_lead, train_idx=tr, test_idx=te)


def run_pooled_competitor(con: duckdb.DuckDBPyConnection, *, plant_id: str, point_id: int,
                          horizon_hours_list: list[int], capacity_mw: float, kind: str,
                          feature_blocks: list[str] | None = None, params: dict | None = None,
                          source: str | None = None, test_fraction: float = 0.20,
                          alpha: float = 0.20) -> PooledResult:
    """Enter the pooled-GBM into the pool and record its per-lead verdict (F1-t2).

    Trains the pooled competitor across ``horizon_hours_list`` and, for each lead,
    competes it against that horizon's current per-horizon champion (lower nRMSE
    wins). The scenario_run is written under the ``{kind}_pooled_gbm`` family and
    marked ``is_champion`` iff the pooled model beats the incumbent — so the
    leaderboard (grouped by asset_kind × horizon × family) shows the
    pooled-vs-per-horizon verdict per plant. Never demotes the per-horizon
    champion model itself; this is a scenario-memory competitor, judged by the
    same memory as every other pool member.
    """
    blocks = feature_blocks or available_blocks(kind)
    pooled = assemble_pooled(
        con, plant_id=plant_id, point_id=point_id, horizon_hours_list=horizon_hours_list,
        capacity_mw=capacity_mw, kind=kind, feature_config=FeatureConfig(blocks=blocks),
        source=source,
    )
    empty = np.array([], dtype=int)
    if pooled.X.shape[0] == 0:
        return PooledResult(per_lead={}, train_idx=empty, test_idx=empty)

    result = evaluate_pooled(pooled, params=params, alpha=alpha, test_fraction=test_fraction)
    if not result.per_lead:
        return result

    experiment_id = create_experiment(con, plant_id, sorted(dict.fromkeys(horizon_hours_list)))
    family = pooled_strategy_family(kind)
    feat_sig = feature_signature(list(blocks) + ["lead_time_hours"])
    model_sig = model_signature(["lightgbm"])
    for lead, tm in result.per_lead.items():
        horizon = int(lead)
        pooled_nrmse = tm.get("nrmse")
        # Incumbent per-horizon champion for this lead (any family but the pooled one).
        champ_nrmse = con.execute(
            "SELECT min(nrmse) FROM scenario_runs "
            "WHERE plant_id=? AND horizon_hours=? AND is_champion AND strategy_family != ?",
            [plant_id, horizon, family],
        ).fetchone()[0]
        wins = pooled_nrmse is not None and (champ_nrmse is None or pooled_nrmse < champ_nrmse)
        record_scenario_run(
            con,
            ScenarioRun(
                experiment_id=experiment_id,
                plant_id=plant_id,
                asset_kind=kind,
                horizon_hours=horizon,
                strategy_family=family,
                feature_signature=feat_sig,
                model_signature=model_sig,
                train_policy="pooled",
                point_policy="single_point",
                source_policy="single_source",
                target_policy="capacity_norm",
                ensemble_policy="none",
                quantile_policy="cqr",
                nrmse=pooled_nrmse,
                nmae=tm.get("nmae"),
                bias=tm.get("bias"),
                pinball=tm.get("pinball"),
                coverage=tm.get("coverage"),
                crps=tm.get("crps"),
                peak_bias=tm.get("peak_bias"),
                peak_mae=tm.get("peak_mae"),
                skill_score=None,
                is_champion=bool(wins),
            ),
        )
    con.execute("UPDATE experiments SET status='done' WHERE experiment_id=?", [experiment_id])
    refresh_scenario_leaderboard(con)
    return result
