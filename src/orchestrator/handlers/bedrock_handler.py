"""Lambda handler para o estágio de Explicação via AWS Bedrock.

Responsável por gerar explicações em linguagem natural (PT-BR)
sobre as predições de churn, com degradação graciosa se Bedrock
estiver indisponível (R12.6).

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import boto3

from src.common.logging import get_logger, set_execution_id
from src.common.models import (
    ExplainabilityResult,
    FeatureContribution,
    FeatureVector,
    PredictionResult,
)
from src.explanations.bedrock_explainer import BedrockExplainer


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para o estágio de explicação Bedrock.

    Gera explicações em linguagem natural via Claude 3 Haiku.
    Se Bedrock falhar, a predição permanece válida com
    explanation = None (R12.6).

    Persiste explicações em S3 antes do próximo estágio (R17.4).

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução do pipeline.
            - predictions: Lista de PredictionResult serializados.
            - explainability_results: Lista de ExplainabilityResult.
            - feature_vectors: Lista de Feature Vectors serializados.
            - population_stats: Estatísticas populacionais por feature.
            - bucket: Nome do bucket S3.
        context: Contexto Lambda.

    Returns:
        Dicionário com event enriquecido com:
            - explanations: Dict user_id -> texto da explicação (ou None).
            - explanation_success_count: Contagem de explicações geradas.
            - explanation_failure_count: Contagem de falhas.
            - stage_completed: "bedrock-explanation"
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("bedrock-explanation")
    logger.log_stage_start()
    start_time = time.time()

    try:
        predictions_data = event.get("predictions", [])
        explainability_data = event.get("explainability_results", [])
        feature_vectors_data = event.get("feature_vectors", [])
        population_stats = event.get("population_stats", {})
        bucket = event.get("bucket", "sky-brazil-churn-prediction")

        # Inicializar Bedrock Explainer
        bedrock_explainer = BedrockExplainer()

        # Mapear feature vectors por user_id para acesso rápido
        fv_map: dict[str, dict] = {}
        for fv_data in feature_vectors_data:
            fv_map[fv_data.get("user_id", "")] = fv_data

        explanations: dict[str, str | None] = {}
        explanation_success_count = 0
        explanation_failure_count = 0

        for i, pred_data in enumerate(predictions_data):
            user_id = pred_data.get("user_id", "")
            churn_prob = pred_data.get("churn_probability", 0.0)
            confidence = pred_data.get("confidence", 0.0)
            risk_tier = pred_data.get("risk_tier", "Low")

            # Apenas Medium e High risk recebem explicação (R12.3)
            if risk_tier == "Low":
                explanations[user_id] = None
                continue

            # Obter explicabilidade para este usuário
            expl_data = (
                explainability_data[i]
                if i < len(explainability_data)
                else None
            )

            if expl_data is None:
                explanations[user_id] = None
                explanation_failure_count += 1
                continue

            # Reconstruir top_features como FeatureContribution
            top_features = [
                FeatureContribution(**fc)
                for fc in expl_data.get("top_features", [])
            ]

            # Valores das features do assinante
            user_fv = fv_map.get(user_id, {})

            explanation = bedrock_explainer.generate_explanation(
                user_id=user_id,
                churn_probability=churn_prob,
                confidence=confidence,
                top_features=top_features,
                user_feature_values=user_fv,
                population_stats=population_stats,
            )

            explanations[user_id] = explanation
            if explanation is not None:
                explanation_success_count += 1
            else:
                explanation_failure_count += 1

        # Persistir explicações em S3 ANTES do próximo estágio (R17.4)
        s3_client = boto3.client("s3")
        s3_key = f"predictions/{execution_id}/explanations.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(explanations, ensure_ascii=False),
            ContentType="application/json",
        )

        output = {
            **event,
            "execution_id": execution_id,
            "explanations": explanations,
            "explanations_s3": f"s3://{bucket}/{s3_key}",
            "explanation_success_count": explanation_success_count,
            "explanation_failure_count": explanation_failure_count,
            "stage_completed": "bedrock-explanation",
        }

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except Exception as e:
        logger.error(
            f"Falha no estágio bedrock-explanation: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
