"""Lambda handler para o estágio de Batch Prediction.

Carrega modelo treinado do S3 (pickle) e executa inferência local.
Armazena resultados no S3 e DynamoDB.

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import io
import json
import os
import pickle
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
import numpy as np

from src.common.logging import get_logger, set_execution_id
from src.common.models import FeatureVector, PredictionResult


# Features numéricas usadas pelo modelo (na ordem correta)
MODEL_FEATURES = [
    "total_sessions",
    "total_viewing_hours",
    "avg_session_duration_min",
    "sessions_per_week",
    "distinct_channels",
    "avg_happiness_score",
    "avg_buffer_ratio",
    "error_rate",
    "avg_bitrate",
    "pct_episode",
    "pct_sport",
    "pct_live",
    "pct_show",
    "distinct_devices",
    "avg_pause_count",
    "avg_seek_count",
    "viewing_time_trend",
    "error_rate_trend",
    "session_frequency_trend",
]

# Thresholds de risco
RISK_THRESHOLDS = {"low_max": 0.30, "medium_max": 0.60}


def _classify_risk(probability: float) -> str:
    """Classifica probabilidade em tier de risco."""
    if probability <= RISK_THRESHOLDS["low_max"]:
        return "Low"
    elif probability <= RISK_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"


def _load_model(bucket: str, model_key: str) -> Any:
    """Carrega modelo pickle do S3."""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=model_key)
    model_bytes = response["Body"].read()
    return pickle.loads(model_bytes)


def _feature_vector_to_array(fv: FeatureVector) -> np.ndarray:
    """Converte FeatureVector em array numérico na ordem esperada pelo modelo."""
    values = []
    for feat in MODEL_FEATURES:
        val = getattr(fv, feat, None)
        values.append(float(val) if val is not None else 0.0)
    return np.array(values)


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para inferência de churn com modelo local.

    Carrega modelo do S3, calcula probabilidade de churn para cada
    usuário e persiste resultados.

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução do pipeline.
            - feature_vectors_s3_key: URI S3 dos feature vectors.
            - bucket: Nome do bucket S3.
        context: Contexto Lambda.

    Returns:
        Dicionário com:
            - predictions_s3_key: URI S3 dos resultados.
            - predictions_count: Contagem de predições.
            - model_version_used: Versão do modelo.
            - stage_completed: "ml-inference"
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("ml-inference")
    logger.log_stage_start()
    start_time = time.time()

    try:
        bucket = event.get("bucket", os.environ.get("BUCKET_NAME", "sky-brazil-churn-prediction"))
        model_s3_key = os.environ.get("MODEL_S3_KEY", "models/approved/churn_model.pkl")

        # 1. Carregar feature vectors do S3
        feature_vectors_data = event.get("feature_vectors", [])
        feature_vectors_s3_key = event.get("feature_vectors_s3_key", "")

        s3_client = boto3.client("s3")

        if not feature_vectors_data and feature_vectors_s3_key:
            logger.info(f"Carregando feature vectors do S3: {feature_vectors_s3_key}")
            s3_key = feature_vectors_s3_key.replace(f"s3://{bucket}/", "")
            response = s3_client.get_object(Bucket=bucket, Key=s3_key)
            feature_vectors_data = json.loads(response["Body"].read().decode("utf-8"))
            logger.info(f"Feature vectors carregados: {len(feature_vectors_data)} usuários")

        if not feature_vectors_data:
            logger.warning("Nenhum feature vector para processar.")
            output = {
                **event,
                "execution_id": execution_id,
                "predictions_s3_key": "",
                "predictions_count": 0,
                "model_version_used": model_s3_key,
                "stage_completed": "ml-inference",
            }
            output.pop("feature_vectors", None)
            output.pop("feature_vectors_s3_key", None)
            return output

        # 2. Reconstruir FeatureVectors
        feature_vectors = [FeatureVector(**fv_data) for fv_data in feature_vectors_data]

        # 3. Carregar modelo do S3
        logger.info(f"Carregando modelo: s3://{bucket}/{model_s3_key}")
        model = _load_model(bucket, model_s3_key)
        logger.info("Modelo carregado com sucesso.")

        # 4. Preparar matriz de features
        X = np.array([_feature_vector_to_array(fv) for fv in feature_vectors])

        # 5. Executar inferência
        probabilities = model.predict_proba(X)[:, 1]  # P(churn)
        logger.info(f"Inferência concluída: {len(probabilities)} predições")

        # 6. Montar PredictionResults
        now_iso = datetime.now(timezone.utc).isoformat()
        predictions: list[PredictionResult] = []

        for fv, prob in zip(feature_vectors, probabilities):
            pred = PredictionResult(
                user_id=fv.user_id,
                churn_probability=round(float(prob), 4),
                confidence=round(1.0 - abs(float(prob) - 0.5) * 2, 4),  # Maior confiança longe de 0.5
                risk_tier=_classify_risk(float(prob)),
                model_version=model_s3_key,
                feature_version=fv.version,
                timestamp=now_iso,
            )
            predictions.append(pred)

        # Log de distribuição de risco
        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        for p in predictions:
            risk_counts[p.risk_tier] += 1
        logger.info(
            f"Distribuição de risco: High={risk_counts['High']}, "
            f"Medium={risk_counts['Medium']}, Low={risk_counts['Low']}"
        )

        # 7. Salvar predições no S3
        predictions_serialized = [pred.model_dump() for pred in predictions]
        predictions_s3_key = f"predictions/{execution_id}/results.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=predictions_s3_key,
            Body=json.dumps(predictions_serialized, ensure_ascii=False, default=str),
            ContentType="application/json",
        )
        logger.info(f"Predições salvas: s3://{bucket}/{predictions_s3_key}")

        # 8. Salvar no DynamoDB
        try:
            dynamodb = boto3.resource("dynamodb")
            table = dynamodb.Table("churn_predictions")
            with table.batch_writer() as batch:
                for pred in predictions:
                    batch.put_item(Item={
                        "user_id": pred.user_id,
                        "execution_id": execution_id,
                        "churn_probability": str(pred.churn_probability),
                        "confidence": str(pred.confidence),
                        "risk_tier": pred.risk_tier,
                        "model_version": pred.model_version,
                        "timestamp": pred.timestamp,
                    })
            logger.info(f"Predições salvas no DynamoDB: {len(predictions)} registros")
        except Exception as e:
            logger.warning(f"Falha ao salvar no DynamoDB (não-bloqueante): {e}")

        # 9. Output leve
        output = {
            **event,
            "execution_id": execution_id,
            "predictions_s3_key": f"s3://{bucket}/{predictions_s3_key}",
            "predictions_count": len(predictions),
            "model_version_used": model_s3_key,
            "risk_distribution": risk_counts,
            "stage_completed": "ml-inference",
        }
        output.pop("feature_vectors", None)
        output.pop("feature_vectors_s3_key", None)

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except Exception as e:
        logger.error(
            f"Falha no estágio ml-inference: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
