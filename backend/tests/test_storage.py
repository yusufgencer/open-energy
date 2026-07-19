def test_schema_creates_core_tables(db):
    rows = db.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert {"plants", "plant_points", "production", "weather_raw", "jobs"} <= names


def test_init_schema_is_idempotent(db):
    # ikinci kez çağırmak hata vermemeli
    from openenergy.storage import init_schema

    init_schema(db)
    db.execute(
        "INSERT INTO plants VALUES ('p1','Test','wind',10.0,'UTC')"
    )
    assert db.execute("SELECT count(*) FROM plants").fetchone()[0] == 1
