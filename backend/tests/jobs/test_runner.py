from datetime import datetime, timedelta
from concurrent.futures import Future
import os


def _echo_handler(con, payload, models_dir):
    assert con.execute("SELECT 1").fetchone() == (1,)
    return payload["value"]


class _ImmediateExecutor:
    def __init__(self, _workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_job_lifecycle_done(db):
    from openenergy.jobs.runner import create_job, run_job, get_job
    jid = create_job(db, "experiment", "test")
    run_job(db, jid, lambda: None)
    assert get_job(db, jid)["status"] == "done"


def test_job_failure_recorded(db):
    from openenergy.jobs.runner import create_job, run_job, get_job
    jid = create_job(db, "experiment")
    def boom(): raise RuntimeError("patladı")
    run_job(db, jid, boom)
    j = get_job(db, jid)
    assert j["status"] == "failed" and "patladı" in (j["detail"] or "")


def test_job_progress_on_success(db):
    from openenergy.jobs.runner import create_job, run_job, get_job
    jid = create_job(db, "retrain")
    run_job(db, jid, lambda: None)
    j = get_job(db, jid)
    assert j["progress"] == 1.0


def test_get_job_returns_none_for_unknown(db):
    from openenergy.jobs.runner import get_job
    assert get_job(db, 99999) is None


def test_list_jobs_orders_desc_and_limits(db):
    from openenergy.jobs.runner import create_job, list_jobs

    ids = [create_job(db, f"kind-{index}") for index in range(4)]

    jobs = list_jobs(db, limit=2)

    assert [job["job_id"] for job in jobs] == list(reversed(ids[-2:]))


def test_job_return_value_stored_in_detail(db):
    from openenergy.jobs.runner import create_job, run_job, get_job
    jid = create_job(db, "experiment", "initial")
    run_job(db, jid, lambda: 42)
    j = get_job(db, jid)
    assert j["status"] == "done"
    assert j["detail"] == "42"


def test_expired_lease_is_reclaimed(db):
    from openenergy.jobs.runner import claim_job, create_job

    jid = create_job(db, "experiment")
    first = claim_job(db, worker_id="worker-a", lease_seconds=30)
    assert first is not None and first["job_id"] == jid

    expired_at = datetime.now() - timedelta(seconds=1)
    db.execute("UPDATE jobs SET lease_until=? WHERE job_id=?", [expired_at, jid])
    second = claim_job(db, worker_id="worker-b", lease_seconds=30)

    assert second is not None
    assert second["job_id"] == jid
    assert second["lease_owner"] == "worker-b"
    assert second["attempt"] == 2


def test_dedupe_key_returns_existing(db):
    from openenergy.jobs.runner import create_job, deterministic_dedupe_key

    key = deterministic_dedupe_key("forecast", {"plant_id": "wf1", "horizon": 24})
    first = create_job(db, "forecast", "first", dedupe_key=key)
    second = create_job(db, "forecast", "second", dedupe_key=key)

    assert second == first
    assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
    # An idempotent retry must not mutate the original job payload.
    assert db.execute("SELECT detail FROM jobs WHERE job_id=?", [first]).fetchone()[0] == "first"


def test_max_attempts_moves_to_dead(db):
    from openenergy.jobs.runner import create_job, get_job, run_pending

    jid = create_job(db, "always_fails", max_attempts=2)

    def boom():
        raise RuntimeError("persistent failure")

    results = run_pending(
        db,
        {"always_fails": boom},
        limit=2,
        retry_backoff_seconds=0,
    )

    job = get_job(db, jid)
    assert len(results) == 2
    assert all(not result["ok"] for result in results)
    assert job["attempt"] == 2
    assert job["status"] == "dead"
    assert "persistent failure" in job["last_error"]
    assert job["lease_owner"] is None


def test_stale_running_job_requeued_on_startup(db):
    from openenergy.jobs.runner import claim_job, create_job, get_job, recover_stale_jobs

    jid = create_job(db, "experiment")
    assert claim_job(db, worker_id="crashed-worker") is not None
    db.execute(
        "UPDATE jobs SET lease_until=? WHERE job_id=?",
        [datetime.now() - timedelta(minutes=1), jid],
    )

    recovered = recover_stale_jobs(db)
    job = get_job(db, jid)

    assert recovered == {"requeued": 1, "dead": 0}
    assert job["status"] == "pending"
    assert job["lease_owner"] is None
    assert job["lease_until"] is None


def test_worker_runs_claimed_job(tmp_path):
    from openenergy.jobs.runner import create_job, get_job
    from openenergy.jobs.worker import run_worker
    from openenergy.storage import connect, init_schema

    db_path = tmp_path / "worker.duckdb"
    con = connect(db_path)
    init_schema(con)
    jid = create_job(con, "echo", payload={"value": "completed in child"})
    con.close()

    results = run_worker(
        db_path,
        handlers={"echo": _echo_handler},
        max_workers=1,
        once=True,
        executor_factory=(
            None if os.environ.get("OPENENERGY_TEST_REAL_POOL") else _ImmediateExecutor
        ),
    )

    con = connect(db_path)
    try:
        assert results[0]["ok"] is True
        job = get_job(con, jid)
        assert job["status"] == "completed"
        assert job["detail"] == "completed in child"
    finally:
        con.close()
