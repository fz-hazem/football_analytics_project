import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def test_database_connection():
    db_url = os.getenv("DATABASE_URL")
    assert db_url is not None, "DATABASE_URL non définie dans le .env"
    
    engine = create_engine(db_url)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT 1;")).fetchone()
        assert res[0] == 1