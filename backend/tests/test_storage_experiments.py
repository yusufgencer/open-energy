def test_experiment_tables_exist(db):
    rows = db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
    names = {r[0] for r in rows}
    assert {"experiments", "experiment_trials", "models"} <= names

def test_insert_experiment_autoincrements(db):
    db.execute("INSERT INTO plants VALUES ('p1','T','wind',10.0,'UTC')")
    eid = db.execute("INSERT INTO experiments (plant_id, horizons) VALUES ('p1','1,24') RETURNING experiment_id").fetchone()[0]
    assert eid >= 1
