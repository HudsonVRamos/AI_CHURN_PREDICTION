"""Lambda handler para o estágio de Feature Engineering do pipeline.

Responsável por transformar sessões brutas extraídas da NPAW em
Feature Vectors comportamentais por usuário.

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import boto3

from src.common.logging import get_logger, set_execution_id
from src.features.feature_engineer import FeatureEngineer


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para o estágio de feature engineering.

    Transforma sessões brutas em Feature Vectors. Lê dados do S3
    (persistidos pelo estágio de extração) e persiste os
    resultados no output do evento antes do próximo estágio (R17.4).

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução do pipeline.
            - extracted_data_s3_prefix: Prefixo S3 dos dados extraídos.
            - users_with_data: Lista de user_ids com dados no S3.
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
        # Ler sessões do S3 (dados persistidos pelo estágio de extração)
        s3_prefix = event.get("extracted_data_s3_prefix", "")
        users_with_data = event.get("users_with_data", [])
        bucket = os.environ.get("BUCKET_NAME", "sky-brazil-churn-prediction")

        s3_client = boto3.client("s3")
        extracted_sessions: dict[str, list] = {}

        if users_with_data and s3_prefix:
            # Extrair o prefixo relativo (sem s3://bucket/)
            prefix = s3_prefix.replace(f"s3://{bucket}/", "")

            for user_id in users_with_data:
                s3_key = f"{prefix}/{user_id}.json"
                try:
                    response = s3_client.get_object(Bucket=bucket, Key=s3_key)
                    sessions = json.loads(response["Body"].read().decode("utf-8"))
                    extracted_sessions[user_id] = sessions
                except Exception as e:
                    logger.warning(
                        f"Não foi possível ler dados do S3 para {user_id}: {e}",
                        extra={"user_id": user_id, "s3_key": s3_key},
                    )

        logger.info(
            f"Dados carregados do S3: {len(extracted_sessions)} usuários com sessões.",
            extra={"users_loaded": len(extracted_sessions)},
        )

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

        # Persistir feature vectors no S3 para não exceder limite do Step Functions
        features_s3_key = f"features/{execution_id}/feature_vectors.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=features_s3_key,
            Body=json.dumps(feature_vectors, ensure_ascii=False),
            ContentType="application/json",
        )

        # Output leve para o próximo estágio (sem dados pesados)
        output = {
            **event,
            "execution_id": execution_id,
            "feature_vectors_s3_key": f"s3://{bucket}/{features_s3_key}",
            "users_with_features": len(feature_vectors),
            "users_insufficient_data": users_insufficient_data,
            "stage_completed": "feature-engineering",
        }

        # Remover campos pesados do output para evitar DataLimitExceeded
        output.pop("users_with_data", None)
        output.pop("valid_user_ids", None)
        output.pop("invalid_user_ids", None)
        output.pop("user_dates", None)

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except Exception as e:
        logger.error(
            f"Falha no estágio feature-engineering: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
