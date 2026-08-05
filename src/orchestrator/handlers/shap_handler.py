"""Lambda handler para o estágio de Explicabilidade (SHAP) do pipeline.

Responsável por calcular SHAP values para cada predição, identificando
as features com maior impacto na classificação de churn.

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import boto3

from src.common.logging import get_logger, set_execution_id
from src.common.models import FeatureVector, PredictionResult
from src.explainability.shap_explainer import SHAPExplainer


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para o estágio de explicabilidade SHAP.

    Calcula SHAP values para cada predição do batch. Persiste
    resultados em S3 antes do próximo estágio (R17.4).

    Se o SHAP falhar para um usuário individual, a predição
    permanece válida e o explainability é None (R11.6).

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução do pipeline.
            - predictions: Lista de PredictionResult serializados.
            - feature_vectors: Lista de Feature Vectors serializados.
            - model: Modelo treinado (injetado).
            - training_data: DataFrame de treino (injetado).
            - bucket: Nome do bucket S3.
        context: Contexto Lambda.

    Returns:
        Dicionário com event enriquecido com:
            - explainability_results: Lista de ExplainabilityResult
              serializados (None para falhas individuais).
            - shap_success_count: Contagem de cálculos bem-sucedidos.
            - shap_failure_count: Contagem de falhas.
            - stage_completed: "explainability"
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("explainability")
    logger.log_stage_start()
    start_time = time.time()

    try:
        feature_vectors_data = event.get("feature_vectors", [])
        model = event.get("model")
        training_data = event.get("training_data")
        bucket = event.get("bucket", "sky-brazil-churn-prediction")

        # Reconstruir FeatureVectors
        feature_vectors = [
            FeatureVector(**fv_data) for fv_data in feature_vectors_data
        ]

        # Inicializar SHAP Explainer
        explainer = SHAPExplainer(
            model=model,
            training_data=training_data,
        )

        # Calcular SHAP em batch
        results = explainer.explain_batch(feature_vectors)

        # Serializar resultados (None para falhas individuais)
        explainability_serialized = []
        shap_success_count = 0
        shap_failure_count = 0

        for result in results:
            if result is not None:
                explainability_serialized.append(result.model_dump())
                shap_success_count += 1
            else:
                explainability_serialized.append(None)
                shap_failure_count += 1

        # Persistir resultados em S3 ANTES do próximo estágio (R17.4)
        s3_client = boto3.client("s3")
        s3_key = f"predictions/{execution_id}/shap_results.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(
                explainability_serialized, ensure_ascii=False, default=str
            ),
            ContentType="application/json",
        )

        output = {
            **event,
            "execution_id": execution_id,
            "explainability_results": explainability_serialized,
            "shap_results_s3": f"s3://{bucket}/{s3_key}",
            "shap_success_count": shap_success_count,
            "shap_failure_count": shap_failure_count,
            "stage_completed": "explainability",
        }

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except Exception as e:
        logger.error(
            f"Falha no estágio explainability: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
