CREATE SEQUENCE IF NOT EXISTS seq_point START 1;
CREATE SEQUENCE IF NOT EXISTS seq_experiment START 1;
CREATE SEQUENCE IF NOT EXISTS seq_trial START 1;
CREATE SEQUENCE IF NOT EXISTS seq_model START 1;
CREATE SEQUENCE IF NOT EXISTS seq_job START 1;
CREATE SEQUENCE IF NOT EXISTS seq_strategy_trial START 1;
CREATE SEQUENCE IF NOT EXISTS seq_strategy_candidate START 1;
CREATE SEQUENCE IF NOT EXISTS seq_scenario_run START 1;

CREATE TABLE IF NOT EXISTS plants (
    plant_id     VARCHAR PRIMARY KEY,
    name         VARCHAR NOT NULL,
    kind         VARCHAR NOT NULL CHECK (kind IN ('wind','solar')),
    capacity_mw  DOUBLE  NOT NULL,
    timezone     VARCHAR NOT NULL DEFAULT 'UTC'
);

CREATE TABLE IF NOT EXISTS plant_points (
    point_id    BIGINT PRIMARY KEY DEFAULT nextval('seq_point'),
    plant_id    VARCHAR NOT NULL REFERENCES plants(plant_id),
    point_type  VARCHAR NOT NULL CHECK (point_type IN ('weather','turbine','panel_array')),
    latitude    DOUBLE NOT NULL,
    longitude   DOUBLE NOT NULL,
    hub_height_m DOUBLE,
    tilt        DOUBLE,
    azimuth     DOUBLE,
    grid_id     VARCHAR,
    grid_row    INTEGER,
    grid_col    INTEGER
);

CREATE TABLE IF NOT EXISTS production (
    plant_id  VARCHAR NOT NULL REFERENCES plants(plant_id),
    ts        TIMESTAMP NOT NULL,
    power_mw  DOUBLE NOT NULL,
    PRIMARY KEY (plant_id, ts)
);

-- T-25: ingest-time quality annotations.  The raw production value remains in
-- ``production`` unchanged ("flag, don't delete").  Rows describe the current
-- revision of an observation; re-ingest replaces flags for the affected
-- timestamp, so consumers never see a stale flag from an older revision.
CREATE TABLE IF NOT EXISTS production_quality (
    plant_id          VARCHAR NOT NULL REFERENCES plants(plant_id),
    ts                TIMESTAMP NOT NULL,
    flag_code         VARCHAR NOT NULL,
    severity          VARCHAR NOT NULL CHECK (severity IN ('info','warning','error')),
    details           VARCHAR NOT NULL,
    source            VARCHAR NOT NULL,
    revision          VARCHAR NOT NULL,
    observed_power_mw DOUBLE NOT NULL,
    flagged_at        TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (plant_id, ts, flag_code)
);

-- T-27: operator-native availability and production-plan signals.  These are
-- deliberately separate from metered production: AIC describes available
-- capacity and DPP describes the day-ahead plan, neither is an observation.
CREATE TABLE IF NOT EXISTS epias_plant_mapping (
    plant_id         VARCHAR PRIMARY KEY REFERENCES plants(plant_id),
    power_plant_id   BIGINT NOT NULL,
    power_plant_name VARCHAR,
    updated_at       TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS epias_aic (
    plant_id   VARCHAR NOT NULL REFERENCES plants(plant_id),
    ts         TIMESTAMP NOT NULL,
    aic_mw     DOUBLE NOT NULL CHECK (aic_mw >= 0),
    ingested_at TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (plant_id, ts)
);

CREATE TABLE IF NOT EXISTS epias_dpp (
    plant_id   VARCHAR NOT NULL REFERENCES plants(plant_id),
    ts         TIMESTAMP NOT NULL,
    dpp_mw     DOUBLE NOT NULL CHECK (dpp_mw >= 0),
    ingested_at TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (plant_id, ts)
);

CREATE TABLE IF NOT EXISTS weather_raw (
    point_id    BIGINT NOT NULL,
    role        VARCHAR NOT NULL,
    model       VARCHAR NOT NULL,
    valid_time  TIMESTAMP NOT NULL,
    issue_time  TIMESTAMP,
    lead_hours  INTEGER,
    variable    VARCHAR NOT NULL,
    value       DOUBLE
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id       BIGINT PRIMARY KEY DEFAULT nextval('seq_job'),
    kind         VARCHAR NOT NULL,
    status       VARCHAR NOT NULL DEFAULT 'pending',
    progress     DOUBLE DEFAULT 0.0,
    detail       VARCHAR,
    payload      VARCHAR,
    dedupe_key   VARCHAR,
    attempt      INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    available_at TIMESTAMP NOT NULL DEFAULT now(),
    lease_owner  VARCHAR,
    lease_until  TIMESTAMP,
    heartbeat_at TIMESTAMP,
    last_error   VARCHAR,
    started_at   TIMESTAMP,
    completed_at TIMESTAMP,
    updated_at   TIMESTAMP NOT NULL DEFAULT now(),
    created_at   TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id BIGINT PRIMARY KEY DEFAULT nextval('seq_experiment'),
    plant_id      VARCHAR,
    created_at    TIMESTAMP DEFAULT now(),
    horizons      VARCHAR,
    objective     VARCHAR DEFAULT 'nrmse',
    status        VARCHAR DEFAULT 'pending',
    best_summary  VARCHAR
);

CREATE TABLE IF NOT EXISTS experiment_trials (
    trial_id       BIGINT PRIMARY KEY DEFAULT nextval('seq_trial'),
    experiment_id  BIGINT,
    horizon_hours  INTEGER,
    model_family   VARCHAR,
    nwp_source     VARCHAR,
    feature_blocks VARCHAR,
    params         VARCHAR,
    cv_nrmse       DOUBLE,
    test_nrmse     DOUBLE,
    test_nmae      DOUBLE,
    test_bias      DOUBLE,
    test_pinball   DOUBLE,
    test_coverage       DOUBLE,
    test_interval_width DOUBLE,
    test_crps           DOUBLE,
    test_peak_bias      DOUBLE,
    test_peak_mae       DOUBLE,
    skill_score    DOUBLE,
    test_clean_nrmse DOUBLE,
    clean_skill_score DOUBLE,
    train_all_count INTEGER,
    train_clean_count INTEGER,
    eval_all_count INTEGER,
    eval_clean_count INTEGER,
    is_champion    BOOLEAN DEFAULT false
);

CREATE TABLE IF NOT EXISTS models (
    model_id      BIGINT PRIMARY KEY DEFAULT nextval('seq_model'),
    experiment_id BIGINT,
    horizon_hours INTEGER,
    model_family  VARCHAR,
    artifact_path VARCHAR,
    metrics       VARCHAR,
    created_at    TIMESTAMP DEFAULT now(),
    is_champion   BOOLEAN DEFAULT false,
    feature_names VARCHAR
);

CREATE TABLE IF NOT EXISTS forecasts (
    plant_id      VARCHAR NOT NULL,
    point_id      BIGINT NOT NULL,
    horizon_hours INTEGER NOT NULL,
    issue_time    TIMESTAMP NOT NULL,
    valid_time    TIMESTAMP NOT NULL,
    p10           DOUBLE,
    p50           DOUBLE NOT NULL,
    p90           DOUBLE,
    model_id      BIGINT,
    strategy_trial_id BIGINT,
    created_at    TIMESTAMP DEFAULT now(),
    UNIQUE (plant_id, point_id, horizon_hours, issue_time, valid_time)
);

CREATE TABLE IF NOT EXISTS strategy_trials (
    strategy_trial_id BIGINT PRIMARY KEY DEFAULT nextval('seq_strategy_trial'),
    experiment_id     BIGINT NOT NULL,
    horizon_hours     INTEGER NOT NULL,
    ensemble_method   VARCHAR NOT NULL,
    strategy_config   VARCHAR NOT NULL,
    cv_nrmse          DOUBLE,
    test_nrmse        DOUBLE,
    test_nmae         DOUBLE,
    test_bias         DOUBLE,
    test_pinball      DOUBLE,
    test_coverage       DOUBLE,
    test_interval_width DOUBLE,
    test_crps           DOUBLE,
    test_peak_bias      DOUBLE,
    test_peak_mae       DOUBLE,
    band_status         VARCHAR NOT NULL DEFAULT 'unknown',
    sel_coverage        DOUBLE,
    sel_coverage_lower  DOUBLE,
    sel_coverage_upper  DOUBLE,
    sel_winkler         DOUBLE,
    sel_kupiec_pvalue   DOUBLE,
    sel_independence_pvalue DOUBLE,
    sel_conditional_coverage_pvalue DOUBLE,
    sel_n               INTEGER,
    champion_metric_name  VARCHAR,
    champion_metric_lower DOUBLE,
    champion_metric_upper DOUBLE,
    mcs_members           VARCHAR,
    uncertainty_diagnostics VARCHAR,
    evaluation_grade     VARCHAR NOT NULL DEFAULT 'exploratory',
    evaluation_n_days    INTEGER,
    evaluation_origins   INTEGER,
    evaluation_seasons   INTEGER,
    skill_score       DOUBLE,
    test_clean_nrmse  DOUBLE,
    clean_skill_score DOUBLE,
    train_all_count   INTEGER,
    train_clean_count INTEGER,
    eval_all_count    INTEGER,
    eval_clean_count  INTEGER,
    artifact_path     VARCHAR,
    is_champion       BOOLEAN DEFAULT false,
    created_at        TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strategy_candidates (
    strategy_candidate_id BIGINT PRIMARY KEY DEFAULT nextval('seq_strategy_candidate'),
    strategy_trial_id     BIGINT NOT NULL,
    rank                  INTEGER NOT NULL,
    candidate_config      VARCHAR NOT NULL,
    cv_nrmse              DOUBLE,
    test_nrmse            DOUBLE,
    artifact_path         VARCHAR,
    feature_names         VARCHAR
);

-- Served strategy champion pointer (T-05). Exactly one current row per
-- (plant_id, horizon_hours): the ONLY source of truth for which strategy_trial
-- the forecast path serves. Moved solely by an explicit promotion (bootstrap on
-- first champion, or a DM-gated retrain promotion) — never by run_experiment's
-- optimistic per-run is_champion marker, so a rejected challenger cannot be served.
CREATE TABLE IF NOT EXISTS champion_pointer (
    plant_id          VARCHAR NOT NULL,
    horizon_hours     INTEGER NOT NULL,
    strategy_trial_id BIGINT NOT NULL,
    promoted_at       TIMESTAMP DEFAULT now(),
    PRIMARY KEY (plant_id, horizon_hours)
);

CREATE TABLE IF NOT EXISTS scenario_runs (
    scenario_run_id BIGINT PRIMARY KEY DEFAULT nextval('seq_scenario_run'),
    experiment_id     BIGINT NOT NULL,
    plant_id          VARCHAR NOT NULL,
    asset_kind        VARCHAR NOT NULL,
    horizon_hours     INTEGER NOT NULL,
    strategy_trial_id BIGINT,
    strategy_family   VARCHAR NOT NULL,
    feature_signature VARCHAR,
    model_signature   VARCHAR,
    train_policy      VARCHAR,
    point_policy      VARCHAR,
    source_policy     VARCHAR,
    target_policy     VARCHAR,
    ensemble_policy   VARCHAR,
    quantile_policy   VARCHAR,
    nrmse             DOUBLE,
    nmae              DOUBLE,
    bias              DOUBLE,
    pinball           DOUBLE,
    coverage          DOUBLE,
    crps              DOUBLE,
    peak_bias         DOUBLE,
    peak_mae          DOUBLE,
    skill_score       DOUBLE,
    evaluation_grade   VARCHAR NOT NULL DEFAULT 'exploratory',
    evaluation_n_days  INTEGER,
    evaluation_origins INTEGER,
    evaluation_seasons INTEGER,
    runtime_seconds   DOUBLE,
    is_champion       BOOLEAN DEFAULT false,
    created_at        TIMESTAMP DEFAULT now()
);

-- E1-t3: daily LASSO polynomial quantile post-processor coefficients, one row per
-- (plant, horizon, quantile). Refit daily on recent (forecast, actual) pairs; absent
-- rows mean identity (no correction). CREATE ... IF NOT EXISTS is itself the
-- idempotent migration for DBs created before this table existed.
CREATE TABLE IF NOT EXISTS postprocessor_coefficients (
    plant_id      VARCHAR NOT NULL,
    horizon_hours INTEGER NOT NULL,
    quantile      VARCHAR NOT NULL,
    degree        INTEGER NOT NULL,
    coefficients  VARCHAR NOT NULL,
    n_samples     INTEGER NOT NULL,
    fitted_at     TIMESTAMP DEFAULT now(),
    PRIMARY KEY (plant_id, horizon_hours, quantile)
);

-- T-21: serving-only dynamic conformal state. Model/search artifacts never carry
-- this state; it advances solely from realised forecasts strictly before the
-- current issue time. ``online_conformal_feedback`` is the idempotency ledger and
-- records the exact state version/order in which each valid_time was consumed.
CREATE TABLE IF NOT EXISTS online_conformal_state (
    plant_id                VARCHAR NOT NULL,
    horizon_hours           INTEGER NOT NULL,
    target_alpha            DOUBLE NOT NULL,
    alpha_t                 DOUBLE NOT NULL,
    gamma                   DOUBLE NOT NULL,
    score_window            INTEGER NOT NULL,
    scores                  VARCHAR NOT NULL,
    feedback_count          BIGINT NOT NULL DEFAULT 0,
    last_feedback_valid_time TIMESTAMP,
    updated_at              TIMESTAMP DEFAULT now(),
    PRIMARY KEY (plant_id, horizon_hours)
);

CREATE TABLE IF NOT EXISTS online_conformal_issued (
    plant_id          VARCHAR NOT NULL,
    horizon_hours     INTEGER NOT NULL,
    valid_time        TIMESTAMP NOT NULL,
    issue_time        TIMESTAMP NOT NULL,
    point_id          BIGINT NOT NULL,
    base_lower        DOUBLE NOT NULL,
    base_upper        DOUBLE NOT NULL,
    served_lower      DOUBLE NOT NULL,
    served_upper      DOUBLE NOT NULL,
    model_id          BIGINT,
    strategy_trial_id BIGINT,
    state_version     BIGINT NOT NULL,
    created_at        TIMESTAMP DEFAULT now(),
    PRIMARY KEY (plant_id, horizon_hours, valid_time)
);

CREATE TABLE IF NOT EXISTS online_conformal_feedback (
    plant_id      VARCHAR NOT NULL,
    horizon_hours INTEGER NOT NULL,
    valid_time    TIMESTAMP NOT NULL,
    actual        DOUBLE NOT NULL,
    score         DOUBLE NOT NULL,
    error         DOUBLE NOT NULL,
    state_version BIGINT NOT NULL,
    processed_at  TIMESTAMP DEFAULT now(),
    PRIMARY KEY (plant_id, horizon_hours, valid_time)
);

CREATE TABLE IF NOT EXISTS drift_state (
    plant_id             VARCHAR NOT NULL,
    horizon_hours        INTEGER NOT NULL,
    consecutive_breaches INTEGER NOT NULL DEFAULT 0,
    fired                BOOLEAN NOT NULL DEFAULT false,
    recent_nrmse         DOUBLE,
    champion_nrmse       DOUBLE,
    ratio                DOUBLE,
    last_job_id          BIGINT,
    last_window          TIMESTAMP,
    updated_at           TIMESTAMP DEFAULT now(),
    PRIMARY KEY (plant_id, horizon_hours)
);

-- T-33: auditable per-feature covariate/freshness detector outcomes.  The
-- evaluation window is part of the key, making repeated checks of unchanged
-- data an upsert rather than an ever-growing stream of duplicate events.
CREATE TABLE IF NOT EXISTS drift_events (
    plant_id          VARCHAR NOT NULL,
    horizon_hours     INTEGER NOT NULL,
    source            VARCHAR NOT NULL,
    feature_name      VARCHAR NOT NULL,
    window_end        TIMESTAMP NOT NULL,
    detector_status   VARCHAR NOT NULL,
    psi               DOUBLE,
    psi_threshold     DOUBLE,
    ks_statistic      DOUBLE,
    ks_pvalue         DOUBLE,
    ks_alpha          DOUBLE,
    reference_start   TIMESTAMP,
    reference_end     TIMESTAMP,
    current_start     TIMESTAMP,
    current_end       TIMESTAMP,
    reference_samples INTEGER NOT NULL DEFAULT 0,
    current_samples   INTEGER NOT NULL DEFAULT 0,
    reference_season  VARCHAR,
    freshness_hours   DOUBLE,
    freshness_limit_hours DOUBLE,
    metadata          VARCHAR,
    evaluated_at      TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (plant_id, horizon_hours, source, feature_name, window_end)
);

CREATE TABLE IF NOT EXISTS scenario_leaderboard (
    asset_kind          VARCHAR NOT NULL,
    horizon_hours      INTEGER NOT NULL,
    strategy_family    VARCHAR NOT NULL,
    trials             BIGINT NOT NULL,
    wins               BIGINT NOT NULL,
    win_rate           DOUBLE NOT NULL,
    median_skill_score DOUBLE,
    median_nrmse       DOUBLE,
    median_pinball     DOUBLE,
    last_updated_at    TIMESTAMP DEFAULT now(),
    PRIMARY KEY (asset_kind, horizon_hours, strategy_family)
);
