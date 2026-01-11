"""
Legacy dev script / test harness.

Updated to import from the new src package layout.
"""

from odonto_rag.rag.outline import generate_clinical_outline
from odonto_rag.rag.outline_refiner import refine_clinical_outline
from odonto_rag.deck.slide_writer import generate_slide_deck
from odonto_rag.ingest.gcloud_auth import gcloud_token
from odonto_rag.rag.providers.vertex import vertex_extract_text


def main():
    # Keep whatever logic you already had below.
    # If this file previously executed code at import-time,
    # move it into main() and call main() here.
    pass


if __name__ == "__main__":
    main()
