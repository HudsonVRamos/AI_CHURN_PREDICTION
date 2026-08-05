"""Lambda handler para o estágio de Batch Prediction (SageMaker).

Responsável por executar inferência em lote via SageMaker Batch Transform
e armazenar resultados no DynamoDB ANTES do estágio de explicabilidade (R17.4).

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.common.logging import get_logger, set_execution_id
from src.common.models import FeatureVector
from src.ml.batch_inference import BatchInferenceProcessor


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para o estágio de inferência em lote.

    Executa predição de churn via SageMaker Batch Transform.
    Resultados são persistidos no DynamoDB antes de retornar (R17.4).

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução do pipeline.
            - feature_vectors: Lista de Feature Vectors serializados.
            - model_version (opcional): ARN do modelo a usar.
            - bucket: Nome do bucket S3.
            - sagemaker_pipeline: Instância do pipeline (injetada).
        context: Contexto Lambda.

    Returns:
        Dicionário com event enriquecido com:
            - predictions: Lista de PredictionResult serializados.
            - predictions_count: Contagem de predições geradas.
            - model_version_used: Versão do modelo utilizada.
            - stage_completed: "ml-inference"
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("ml-inference")
    logger.log_stage_start()
    start_time = time.time()

    try:
        feature_vectors_data = event.get("feature_vectors", [])
        model_version = event.get("model_version")
        bucket = event.get("bucket", "sky-brazil-churn-prediction")
        sagemaker_pipeline = event.get("sagemaker_pipeline")

        # Reconstruir FeatureVectors a partir dos dados serializados
        feature_vectors = [
            FeatureVector(**fv_data) for fv_data in feature_vectors_data
        ]

        # Inicializar processador de inferência
        processor = BatchInferenceProcessor(
            sagemaker_pipeline=sagemaker_pipeline,
            bucket=bucket,
        )

        # Executar inferência em lote
        predictions = processor.process(
            feature_vectors=feature_vectors,
            execution_id=execution_id,
            model_version=model_version,
        )

        # Serializar resultados para o próximo estágio
        predictions_serialized = [
            pred.model_dump() for pred in predictions
        ]

        output = {
            **event,
            "execution_id": execution_id,
            "predictions": predictions_serialized,
            "predictions_count": len(predictions),
            "model_version_used": (
                predictions[0].model_version if predictions else model_version
            ),
            "stage_completed": "ml-inference",
        }

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except Exception as e:
        logger.error(
            f"Falha no estágio ml-inference: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
