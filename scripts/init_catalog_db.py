from pathlib import Path
from odonto_rag.catalog.init_db import init_db

if __name__ == "__main__":
    db_path = Path("data/catalog.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    print(f"OK: creato {db_path}")
