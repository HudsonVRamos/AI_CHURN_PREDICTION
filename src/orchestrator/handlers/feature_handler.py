"""Lambda handler para o estágio de Feature Engineering do pipeline.

Responsável por transformar sessões brutas extraídas da NPAW em
Feature Vectors comportamentais por usuário.

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.common.logging import get_logger, set_execution_id
from src.features.feature_engineer import FeatureEngineer


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para o estágio de feature engineering.

    Transforma sessões brutas em Feature Vectors. Persiste os
    resultados no output do evento antes do próximo estágio (R17.4).

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução do pipeline.
            - extracted_sessions: Dict mapeando user_id -> sessões.
        context: Contexto Lambda.

    Returns:
        Dicionário com event enriquecido com:
            - feature_vectors: Lista de Feature Vectors serializados.
            - users_with_features: Contagem de usuários com features.
            - users_insufficient_data: Contagem com dados insuficientes.
            - stage_completed: "feature-engineering"
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("feature-engineering")
    logger.log_stage_start()
    start_time = time.time()

    try:
        extracted_sessions = event.get("extracted_sessions", {})
        engineer = FeatureEngineer()

        feature_vectors = []
        users_insufficient_data = 0

        for user_id, sessions in extracted_sessions.items():
            try:
                fv = engineer.compute(user_id=user_id, sessions=sessions)
                if fv is not None:
                    feature_vectors.append(fv.model_dump())
                else:
                    users_insufficient_data += 1
            except Exception as e:
                logger.error(
                    f"Falha ao computar features para user_id={user_id}: {e}",
                    extra={
                        "user_id": user_id,
                        "error_type": type(e).__name__,
                    },
                )
                users_insufficient_data += 1

        # Persistir resultados no output (R17.4)
        output = {
            **event,
            "execution_id": execution_id,
            "feature_vectors": feature_vectors,
            "users_with_features": len(feature_vectors),
            "users_insufficient_data": users_insufficient_data,
            "stage_completed": "feature-engineering",
        }

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except Exception as e:
        logger.error(
            f"Falha no estágio feature-engineering: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
