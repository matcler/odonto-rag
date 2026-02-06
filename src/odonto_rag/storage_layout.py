from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedLayout:
    """
    Canonical layout for parsed artifacts on GCS.

    parsed/{doc_id}/{version_id}/
      items.jsonl
      assets.jsonl
      assets/images/{asset_id}.png
      assets/tables/{asset_id}.json
      raw/extractor_output.json
    """
    doc_id: str
    version_id: str

    def prefix(self) -> str:
        return f"parsed/{self.doc_id}/{self.version_id}/"

    def items_jsonl(self) -> str:
        return self.prefix() + "items.jsonl"

    def assets_jsonl(self) -> str:
        return self.prefix() + "assets.jsonl"

    def raw_extractor_output(self) -> str:
        return self.prefix() + "raw/extractor_output.json"

    def asset_image(self, asset_id: str, ext: str = "png") -> str:
        return self.prefix() + f"assets/images/{asset_id}.{ext}"

    def asset_table(self, asset_id: str, ext: str = "json") -> str:
        return self.prefix() + f"assets/tables/{asset_id}.{ext}"


def gcs_uri(bucket: str, object_path: str) -> str:
    if object_path.startswith("gs://"):
        return object_path
    return f"gs://{bucket}/{object_path.lstrip('/')}"
