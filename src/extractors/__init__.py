# Módulo de extração de dados
"""Extratores de dados (NPAW API) e ingestão de listas de usuários."""

from src.extractors.ingestion import (
    IngestionError,
    IngestionResult,
    ingest_user_ids,
    is_valid_uuid_v4,
)

__all__ = [
    "IngestionError",
    "IngestionResult",
    "ingest_user_ids",
    "is_valid_uuid_v4",
]

from src.extractors.npaw_extractor import (
    NPAWAuthenticationError,
    NPAWExtractor,
    NPAWExtractorError,
)

__all__ = [
    "NPAWExtractor",
    "NPAWExtractorError",
    "NPAWAuthenticationError",
]
