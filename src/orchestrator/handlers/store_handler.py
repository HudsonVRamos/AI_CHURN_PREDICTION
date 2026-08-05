"""Lambda handler para o estágio de armazenamento no Feature Store.

Responsável por persistir Feature Vectors no DynamoDB (Feature Store)
com versionamento imutável, ANTES da inferência (R17.4, R9.1).

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.common.logging import get_logger, set_execution_id
from src.common.models import FeatureVector
from src.store.feature_store import FeatureStore


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para o estágio de armazenamento de features.

    Persiste Feature Vectors no DynamoDB Feature Store antes da
    inferência (data integrity - R9.1, R17.4).

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução do pipeline.
            - feature_vectors: Lista de Feature Vectors serializados.
        context: Contexto Lambda.

    Returns:
        Dicionário com event enriquecido com:
            - stored_feature_versions: Dict user_id -> version atribuída.
            - users_stored: Contagem de features armazenadas.
            - users_store_failed: Contagem de falhas ao armazenar.
            - stage_completed: "store-features"
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("feature-engineering")
    logger.log_stage_start()
    start_time = time.time()

    try:
        feature_vectors_data = event.get("feature_vectors", [])
        store = FeatureStore()

        stored_versions: dict[str, int] = {}
        users_stored = 0
        users_store_failed = 0

        for fv_data in feature_vectors_data:
            user_id = fv_data.get("user_id", "unknown")
            try:
                fv = FeatureVector(**fv_data)
                version = store.store(fv)
                stored_versions[user_id] = version
                users_stored += 1
            except Exception as e:
                logger.error(
                    f"Falha ao armazenar features para user_id={user_id}: {e}",
                    extra={
                        "user_id": user_id,
                        "error_type": type(e).__name__,
                    },
                )
                users_store_failed += 1

        # Atualizar feature_vectors com versões corretas
        updated_feature_vectors = []
        for fv_data in feature_vectors_data:
            uid = fv_data.get("user_id")
            if uid in stored_versions:
                fv_data_copy = {**fv_data, "version": stored_versions[uid]}
                updated_feature_vectors.append(fv_data_copy)

        output = {
            **event,
            "execution_id": execution_id,
            "feature_vectors": updated_feature_vectors,
            "stored_feature_versions": stored_versions,
            "users_stored": users_stored,
            "users_store_failed": users_store_failed,
            "stage_completed": "store-features",
        }

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except Exception as e:
        logger.error(
            f"Falha no estágio store-features: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
