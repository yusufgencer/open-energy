import pytest

from openenergy.storage import connect, init_schema


@pytest.fixture
def db():
    con = connect(None)
    init_schema(con)
    yield con
    con.close()
