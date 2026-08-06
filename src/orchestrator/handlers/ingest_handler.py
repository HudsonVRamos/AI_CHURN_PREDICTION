"""Lambda handler para o estágio de Ingestão do pipeline.

Responsável por validar e deduplicar listas de User IDs fornecidas
via CSV, JSON ou array direto.

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.common.logging import get_logger, set_execution_id
from src.extractors.ingestion import ingest_user_ids, IngestionError


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para o estágio de ingestão de User IDs.

    Valida e deduplica IDs de usuários fornecidos no evento.
    Persiste os IDs válidos no output do evento para o próximo estágio.

    Args:
        event: Dicionário com:
            - execution_id (opcional): UUID da execução do pipeline.
            - source: Conteúdo CSV/JSON ou lista de IDs.
            - source_format (opcional): "csv", "json" ou None.
        context: Contexto Lambda (não utilizado diretamente).

    Returns:
        Dicionário com event original enriquecido com:
            - valid_user_ids: Lista de IDs válidos e únicos.
            - invalid_user_ids: Lista de IDs rejeitados.
            - duplicates_removed: Contagem de duplicatas removidas.
            - stage_completed: "ingestion"

    Raises:
        IngestionError: Se nenhum ID válido for encontrado.
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("extraction")
    logger.log_stage_start()
    start_time = time.time()

    try:
        source = event.get("source", [])
        source_format = event.get("source_format")

        result = ingest_user_ids(
            source=source,
            source_format=source_format,
        )

        # Persistir resultados no output do evento (R17.4)
        output = {
            **event,
            "execution_id": execution_id,
            "valid_user_ids": result.valid_ids,
            "invalid_user_ids": result.invalid_ids,
            "duplicates_removed": result.duplicates_removed,
            "user_dates": result.user_dates,
            "users_count": len(result.valid_ids),
            "stage_completed": "ingestion",
        }

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except IngestionError as e:
        logger.error(
            f"Falha no estágio ingestion: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise

    except Exception as e:
        logger.critical(
            f"Erro sistêmico no estágio ingestion: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
