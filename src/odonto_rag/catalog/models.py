from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    String,
    Integer,
    DateTime,
    JSON,
    UniqueConstraint,
    Index,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)

    # NEW: type of source (future-proof for video)
    doc_type: Mapped[str] = mapped_column(String, nullable=False, default="pdf", index=True)

    # where the original source lives (optional)
    gcs_raw_path: Mapped[str | None] = mapped_column(String, nullable=True)

    # where parsed outputs live (prefix/folder)
    gcs_parsed_prefix: Mapped[str | None] = mapped_column(String, nullable=True)

    active_version: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="new")
    last_error: Mapped[str | None] = mapped_column(String, nullable=True)

    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("doc_id", "version", name="uq_doc_version"),
        Index("ix_docver_docid_version", "doc_id", "version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String, nullable=False)

    # Back-compat with old pipeline (chunks.json)
    gcs_chunks_path: Mapped[str | None] = mapped_column(String, nullable=True)
    n_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # NEW canonical outputs
    gcs_items_path: Mapped[str | None] = mapped_column(String, nullable=True)
    gcs_assets_path: Mapped[str | None] = mapped_column(String, nullable=True)
    n_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_assets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    ingest_status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    last_error: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ContentItem(Base):
    """
    Admin/query index of items stored on GCS (items.jsonl is source-of-truth).
    Stores preview + metadata (locator/tags) for fast UI/admin.
    """
    __tablename__ = "content_items"
    __table_args__ = (
        Index("ix_items_docid_version", "doc_id", "version"),
        Index("ix_items_doc_type", "doc_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    item_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String, nullable=False, index=True)

    doc_type: Mapped[str] = mapped_column(String, nullable=False, default="pdf")
    item_type: Mapped[str] = mapped_column(String, nullable=False, default="chunk")

    text_preview: Mapped[str | None] = mapped_column(Text, nullable=True)

    locator_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tags_json: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Pointer back to GCS storage (items.jsonl) + optional hint
    gcs_uri: Mapped[str | None] = mapped_column(String, nullable=True)

    meta_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class Asset(Base):
    """
    Admin/query index of assets stored on GCS (assets.jsonl + files is source-of-truth).
    """
    __tablename__ = "assets"
    __table_args__ = (
        Index("ix_assets_docid_version", "doc_id", "version"),
        Index("ix_assets_type", "asset_type"),
        Index("ix_assets_doc_type", "doc_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    asset_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String, nullable=False, index=True)

    doc_type: Mapped[str] = mapped_column(String, nullable=False, default="pdf")
    asset_type: Mapped[str] = mapped_column(String, nullable=False)  # figure|table|image|chart|keyframe

    caption_preview: Mapped[str | None] = mapped_column(Text, nullable=True)

    locator_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tags_json: Mapped[list | None] = mapped_column(JSON, nullable=True)

    files_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # image_uri/table_uri etc.
    meta_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
