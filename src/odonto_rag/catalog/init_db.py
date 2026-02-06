from __future__ import annotations
from pathlib import Path
from .db import make_engine, Base
from . import models  # noqa: F401

def init_db(db_path: str | Path):
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    return engine
