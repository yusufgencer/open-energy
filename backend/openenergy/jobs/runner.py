from __future__ import annotations

import hashlib
import inspect
import json
import logging
import socket
import time
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

import duckdb


logger = logging.getLogger(__name__)

# Job kind for drift-triggered emergency retrains (E1-t4) — distinct from the
# scheduled monthly "retrain" kind so the two are separable in the jobs surface.
EMERGENCY_RETRAIN_KIND = "emergency_retrain"

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LEASE_SECONDS = 300
DEFAULT_BACKOFF_SECONDS = 5.0

_JOB_COLUMNS = """
    job_id, kind, status, progress, detail, payload, dedupe_key, attempt, max_attempts,
    available_at, lease_owner, lease_until, heartbeat_at, last_error,
    started_at, completed_at, updated_at, created_at
"""


def _utcnow() -> datetime:
    # DuckDB's timezone-naive TIMESTAMP ``now()`` follows the connection's local
    # timezone, so use the matching naive wall clock for bound comparisons.
    return datetime.now()


def _default_worker_id() -> str:
    return f"{socket.gethostname()}:{uuid.uuid4().hex}"


def deterministic_dedupe_key(kind: str, payload: Any) -> str:
    """Build a stable, compact idempotency key from a job kind and JSON payload."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(f"{kind}\0{canonical}".encode()).hexdigest()
    return f"{kind}:{digest}"


def enqueue_emergency_retrain(
    con: duckdb.DuckDBPyConnection,
    plant_id: str,
    horizon_hours: int,
    detail: str = "",
    *,
    payload: Mapping[str, Any] | None = None,
) -> int:
    """Idempotently enqueue one emergency retrain per plant/horizon."""
    d = detail or f"plant={plant_id} horizon={horizon_hours} (drift)"
    key = deterministic_dedupe_key(
        EMERGENCY_RETRAIN_KIND,
        {"plant_id": plant_id, "horizon_hours": int(horizon_hours)},
    )
    return create_job(con, EMERGENCY_RETRAIN_KIND, d, payload=payload, dedupe_key=key)


def create_job(
    con: duckdb.DuckDBPyConnection,
    kind: str,
    detail: str = "",
    *,
    payload: Mapping[str, Any] | None = None,
    dedupe_key: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> int:
    """Create a pending job, returning the existing id for a duplicate key.

    ``dedupe_key=None`` intentionally preserves the legacy create-on-every-call
    behaviour. Producers that need idempotency should pass an explicit key, often
    generated with :func:`deterministic_dedupe_key`.
    """
    if not kind:
        raise ValueError("kind must not be empty")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    serialized_payload = json.dumps(
        payload or {},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    params = [kind, detail or None, serialized_payload, dedupe_key, int(max_attempts)]
    if dedupe_key is None:
        row = con.execute(
            """
            INSERT INTO jobs
                (kind, status, progress, detail, payload, dedupe_key, attempt,
                 max_attempts, available_at, updated_at)
            VALUES (?, 'pending', 0.0, ?, ?, ?, 0, ?, now(), now())
            RETURNING job_id
            """,
            params,
        ).fetchone()
    else:
        # One statement is important for DuckDB: a select-then-insert sequence
        # allows two producers to observe absence concurrently. The unique index
        # plus ON CONFLICT makes enqueue idempotent at the storage boundary.
        try:
            row = con.execute(
                """
                INSERT INTO jobs
                    (kind, status, progress, detail, payload, dedupe_key, attempt,
                     max_attempts, available_at, updated_at)
                VALUES (?, 'pending', 0.0, ?, ?, ?, 0, ?, now(), now())
                ON CONFLICT (dedupe_key) DO UPDATE
                    SET dedupe_key = excluded.dedupe_key
                RETURNING job_id
                """,
                params,
            ).fetchone()
        except (duckdb.ConstraintException, duckdb.TransactionException):
            # DuckDB's ON CONFLICT cannot see an uncommitted row inserted by a
            # concurrent connection. The losing autocommit may therefore fail
            # at commit; after the winner becomes visible, return its id.
            row = None
            for conflict_number in range(5):
                row = con.execute(
                    "SELECT job_id FROM jobs WHERE dedupe_key=?",
                    [dedupe_key],
                ).fetchone()
                if row is not None:
                    break
                time.sleep(0.002 * (2**conflict_number))
            if row is None:
                raise
    if row is None:  # defensive; INSERT ... RETURNING must yield one row
        raise RuntimeError("job enqueue did not return a job_id")
    return int(row[0])


def enqueue(
    con: duckdb.DuckDBPyConnection,
    kind: str,
    detail: str = "",
    *,
    payload: Mapping[str, Any] | None = None,
    dedupe_key: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> int:
    """Queue API alias retained for callers that use ``enqueue`` terminology."""
    return create_job(
        con,
        kind,
        detail,
        payload=payload,
        dedupe_key=dedupe_key,
        max_attempts=max_attempts,
    )


def recover_stale_jobs(
    con: duckdb.DuckDBPyConnection, *, now: datetime | None = None
) -> dict[str, int]:
    """Requeue running jobs whose worker lease expired (startup crash recovery)."""
    at = now or _utcnow()
    dead_rows = con.execute(
        """
        UPDATE jobs
        SET status='dead',
            lease_owner=NULL,
            lease_until=NULL,
            heartbeat_at=NULL,
            completed_at=?,
            updated_at=?,
            last_error=coalesce(last_error, 'worker lease expired at startup')
        WHERE status='running'
          AND (lease_until IS NULL OR lease_until <= ?)
          AND attempt >= max_attempts
        RETURNING job_id
        """,
        [at, at, at],
    ).fetchall()
    pending_rows = con.execute(
        """
        UPDATE jobs
        SET status='pending',
            available_at=?,
            lease_owner=NULL,
            lease_until=NULL,
            heartbeat_at=NULL,
            updated_at=?,
            last_error=coalesce(last_error, 'worker lease expired at startup')
        WHERE status='running'
          AND (lease_until IS NULL OR lease_until <= ?)
          AND attempt < max_attempts
        RETURNING job_id
        """,
        [at, at, at],
    ).fetchall()
    return {"requeued": len(pending_rows), "dead": len(dead_rows)}


# Explicit startup-oriented name for application/worker boot hooks.
requeue_stale_jobs_on_startup = recover_stale_jobs


def _row_to_job(row: tuple[Any, ...]) -> dict[str, Any]:
    job = dict(zip((c.strip() for c in _JOB_COLUMNS.split(",")), row, strict=True))
    raw_payload = job.get("payload")
    if isinstance(raw_payload, str):
        job["payload"] = json.loads(raw_payload)
    elif raw_payload is None:
        job["payload"] = {}
    return job


def claim_job(
    con: duckdb.DuckDBPyConnection,
    *,
    worker_id: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    job_id: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Atomically claim one available job, including an expired running lease.

    A worker must use its own DuckDB connection. The single UPDATE statement is
    the compare-and-set boundary; concurrent writers cannot both successfully
    claim the same row.
    """
    if not worker_id:
        raise ValueError("worker_id must not be empty")
    if lease_seconds < 1:
        raise ValueError("lease_seconds must be at least 1")

    at = now or _utcnow()
    lease_until = at + timedelta(seconds=lease_seconds)
    id_filter = "" if job_id is None else "AND job_id = ?"

    # Eligibility appears both in the candidate subquery and outer predicate.
    # The outer check prevents a stale candidate from being overwritten if
    # another connection commits a claim between planning and update.
    statement = f"""
        UPDATE jobs
        SET status='running',
            attempt=attempt + 1,
            lease_owner=?,
            lease_until=?,
            heartbeat_at=?,
            started_at=coalesce(started_at, ?),
            updated_at=?
        WHERE job_id = (
            SELECT job_id
            FROM jobs
            WHERE attempt < max_attempts
              AND (
                    (status IN ('pending', 'failed') AND coalesce(available_at, ?) <= ?)
                 OR (status='running' AND (lease_until IS NULL OR lease_until <= ?))
              )
              {id_filter}
            ORDER BY coalesce(available_at, created_at), created_at, job_id
            LIMIT 1
        )
          AND attempt < max_attempts
          AND (
                (status IN ('pending', 'failed') AND coalesce(available_at, ?) <= ?)
             OR (status='running' AND (lease_until IS NULL OR lease_until <= ?))
          )
        RETURNING {_JOB_COLUMNS}
        """
    params = [
        worker_id,
        lease_until,
        at,
        at,
        at,
        at,
        at,
        at,
        *([int(job_id)] if job_id is not None else []),
        at,
        at,
        at,
    ]
    # DuckDB uses optimistic concurrency control. Two connections may select the
    # same candidate and one will receive a transaction conflict; retrying the
    # complete atomic statement makes the loser observe the winner's lease and
    # return None (or claim the next job), never duplicate ownership.
    for conflict_number in range(5):
        try:
            row = con.execute(statement, params).fetchone()
            break
        except duckdb.TransactionException:
            if conflict_number == 4:
                raise
            time.sleep(0.002 * (2**conflict_number))
    return None if row is None else _row_to_job(row)


def heartbeat_job(
    con: duckdb.DuckDBPyConnection,
    job_id: int,
    worker_id: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: datetime | None = None,
) -> bool:
    """Extend a claimed job lease; false means ownership was lost."""
    at = now or _utcnow()
    row = con.execute(
        """
        UPDATE jobs
        SET heartbeat_at=?, lease_until=?, updated_at=?
        WHERE job_id=? AND status='running' AND lease_owner=?
        RETURNING job_id
        """,
        [at, at + timedelta(seconds=lease_seconds), at, int(job_id), worker_id],
    ).fetchone()
    return row is not None


def _retry_delay(attempt: int, base_seconds: float) -> timedelta:
    # attempt is 1-based; first failure waits base, then doubles.
    seconds = max(0.0, base_seconds) * (2 ** max(0, attempt - 1))
    return timedelta(seconds=seconds)


def _finish_success(
    con: duckdb.DuckDBPyConnection,
    claimed: Mapping[str, Any],
    result: Any,
    *,
    legacy_done_status: bool = False,
) -> dict[str, Any]:
    at = _utcnow()
    status = "done" if legacy_done_status else "completed"
    row = con.execute(
        f"""
        UPDATE jobs
        SET status=?,
            progress=1.0,
            detail=CASE WHEN ? IS NULL THEN detail ELSE ? END,
            lease_owner=NULL,
            lease_until=NULL,
            heartbeat_at=NULL,
            last_error=NULL,
            completed_at=?,
            updated_at=?
        WHERE job_id=? AND status='running' AND lease_owner=?
        RETURNING {_JOB_COLUMNS}
        """,
        [
            status,
            None if result is None else str(result),
            None if result is None else str(result),
            at,
            at,
            int(claimed["job_id"]),
            claimed["lease_owner"],
        ],
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"lease ownership lost while completing job {claimed['job_id']}"
        )
    return _row_to_job(row)


def _finish_failure(
    con: duckdb.DuckDBPyConnection,
    claimed: Mapping[str, Any],
    error: BaseException,
    *,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    at = _utcnow()
    formatted = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    terminal = int(claimed["attempt"]) >= int(claimed["max_attempts"])
    status = "dead" if terminal else "failed"
    available_at = at if terminal else at + _retry_delay(
        int(claimed["attempt"]), retry_backoff_seconds
    )
    row = con.execute(
        f"""
        UPDATE jobs
        SET status=?,
            detail=?,
            last_error=?,
            available_at=?,
            lease_owner=NULL,
            lease_until=NULL,
            heartbeat_at=NULL,
            completed_at=CASE WHEN ? THEN ? ELSE NULL END,
            updated_at=?
        WHERE job_id=? AND status='running' AND lease_owner=?
        RETURNING {_JOB_COLUMNS}
        """,
        [
            status,
            formatted,
            formatted,
            available_at,
            terminal,
            at,
            at,
            int(claimed["job_id"]),
            claimed["lease_owner"],
        ],
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"lease ownership lost while failing job {claimed['job_id']}"
        ) from error
    logger.error(
        "job %s (%s) failed on attempt %s/%s; status=%s",
        claimed["job_id"],
        claimed["kind"],
        claimed["attempt"],
        claimed["max_attempts"],
        status,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _row_to_job(row)


def _invoke(handler: Callable[..., Any], job: Mapping[str, Any]) -> Any:
    """Support both legacy zero-argument handlers and queue-aware handlers."""
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return handler()
    required_positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]
    return handler(job) if required_positional else handler()


def _execute_claimed(
    con: duckdb.DuckDBPyConnection,
    claimed: Mapping[str, Any],
    handler: Callable[..., Any],
    *,
    retry_backoff_seconds: float,
    legacy_done_status: bool = False,
) -> dict[str, Any]:
    try:
        result = _invoke(handler, claimed)
    except Exception as exc:
        failed = _finish_failure(
            con,
            claimed,
            exc,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        return {"job": failed, "ok": False, "error": exc}
    completed = _finish_success(
        con,
        claimed,
        result,
        legacy_done_status=legacy_done_status,
    )
    return {"job": completed, "ok": True, "result": result}


def complete_claimed_job(
    con: duckdb.DuckDBPyConnection,
    claimed: Mapping[str, Any],
    result: Any,
) -> dict[str, Any]:
    """Mark a leased job completed after execution in an isolated process."""
    return _finish_success(con, claimed, result)


def fail_claimed_job(
    con: duckdb.DuckDBPyConnection,
    claimed: Mapping[str, Any],
    error: BaseException,
    *,
    retry_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Persist an isolated worker failure while preserving retry semantics."""
    return _finish_failure(
        con,
        claimed,
        error,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def run_job(
    con: duckdb.DuckDBPyConnection,
    job_id: int,
    fn: Callable[[], Any],
    *,
    worker_id: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    retry_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Run one job synchronously and return an observable success/failure result.

    This compatibility entry point keeps the historical ``done`` success status.
    Durable workers should use :func:`run_pending`, whose success state is
    ``completed``. Exceptions are persisted, logged with their traceback, and
    returned in the result instead of disappearing in a bare ``except`` block.
    """
    owner = worker_id or _default_worker_id()
    claimed = claim_job(
        con,
        worker_id=owner,
        lease_seconds=lease_seconds,
        job_id=job_id,
    )
    if claimed is None:
        raise RuntimeError(f"job {job_id} is not available for claim")
    return _execute_claimed(
        con,
        claimed,
        fn,
        retry_backoff_seconds=retry_backoff_seconds,
        legacy_done_status=True,
    )


def run_pending(
    con: duckdb.DuckDBPyConnection,
    handlers: Mapping[str, Callable[..., Any]],
    *,
    worker_id: str | None = None,
    limit: int | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    retry_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    recover_stale: bool = True,
) -> list[dict[str, Any]]:
    """Claim and run available jobs, returning one observable result per job."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    if recover_stale:
        recover_stale_jobs(con)

    owner = worker_id or _default_worker_id()
    results: list[dict[str, Any]] = []
    while limit is None or len(results) < limit:
        claimed = claim_job(
            con,
            worker_id=owner,
            lease_seconds=lease_seconds,
        )
        if claimed is None:
            break
        handler = handlers.get(str(claimed["kind"]))
        if handler is None:
            exc = LookupError(f"no handler registered for job kind {claimed['kind']!r}")
            failed = _finish_failure(
                con,
                claimed,
                exc,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            results.append({"job": failed, "ok": False, "error": exc})
            continue
        results.append(
            _execute_claimed(
                con,
                claimed,
                handler,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        )
    return results


def get_job(con: duckdb.DuckDBPyConnection, job_id: int) -> dict[str, Any] | None:
    """Return the full durable job record, or ``None`` when it does not exist."""
    row = con.execute(
        f"SELECT {_JOB_COLUMNS} FROM jobs WHERE job_id=?",
        [int(job_id)],
    ).fetchone()
    if row is None:
        return None
    job = _row_to_job(row)
    for key in (
        "available_at",
        "lease_until",
        "heartbeat_at",
        "started_at",
        "completed_at",
        "updated_at",
        "created_at",
    ):
        job[key] = None if job[key] is None else str(job[key])
    return job
