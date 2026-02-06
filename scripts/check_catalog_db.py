from pathlib import Path
from sqlalchemy import text
from odonto_rag.catalog.db import make_engine

if __name__ == "__main__":
    db_path = Path("data/catalog.db")
    engine = make_engine(db_path)

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")).fetchall()
        print("TABELLE:")
        for r in rows:
            print("-", r[0])
