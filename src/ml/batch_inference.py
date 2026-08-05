"""Módulo de inferência em lote via SageMaker Batch Transform.

Orquestra o fluxo completo de batch inference:
1. Prepara input CSV com feature vectors para Batch Transform
2. Invoca predict_batch no SageMaker
3. Parseia output do Batch Transform em PredictionResult
4. Armazena resultados no DynamoDB (tabela churn_predictions)
5. Garante inferência determinística via seed=42 nos hyperparameters do modelo

Requirements: 10.4, 10.5, 10.6, 10.9
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3

from src.common.logging import get_logger
from src.common.models import FeatureVector, PredictionResult

logger = get_logger("ml-inference")

# Campos numéricos do FeatureVector usados como input para o modelo
# (mesma ordem usada no treinamento — sem user_id, version, timestamps, observation_*)
FEATURE_COLUMNS = [
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

# Thresholds de risco padrão (configuráveis)
DEFAULT_RISK_THRESHOLDS = {
    "low_max": 0.30,
    "medium_max": 0.60,
}


class BatchInferenceProcessor:
    """Processador de inferência em lote via SageMaker Batch Transform.

    Orquestra o fluxo completo:
    - Preparação do CSV de input (sem labels, apenas features)
    - Chamada ao predict_batch do SageMaker
    - Parsing dos resultados em PredictionResult
    - Armazenamento no DynamoDB (tabela churn_predictions)

    A inferência é determinística: seed=42 nos hyperparameters do modelo
    garante que os mesmos inputs produzem os mesmos outputs.

    Attributes:
        SEED: Seed fixa para inferência determinística.
    """

    SEED = 42

    def __init__(
        self,
        sagemaker_pipeline: Any,
        bucket: str,
        dynamodb_table_name: str = "churn_predictions",
        risk_thresholds: dict[str, float] | None = None,
        s3_client: Any | None = None,
        dynamodb_resource: Any | None = None,
    ) -> None:
        """Inicializa o processador de batch inference.

        Args:
            sagemaker_pipeline: Instância de SageMakerMLPipeline para executar
                o Batch Transform.
            bucket: Bucket S3 para input/output do Batch Transform.
            dynamodb_table_name: Nome da tabela DynamoDB para armazenar predições.
            risk_thresholds: Thresholds customizados de risco
                (chaves: low_max, medium_max).
            s3_client: Cliente boto3 S3 (para testes/injeção).
            dynamodb_resource: Resource boto3 DynamoDB (para testes/injeção).
        """
        self._pipeline = sagemaker_pipeline
        self._bucket = bucket
        self._table_name = dynamodb_table_name
        self._risk_thresholds = risk_thresholds or DEFAULT_RISK_THRESHOLDS

        self._s3_client = s3_client or boto3.client("s3")
        self._dynamodb = dynamodb_resource or boto3.resource("dynamodb")
        self._table = self._dynamodb.Table(self._table_name)

        logger.info(
            f"BatchInferenceProcessor inicializado: "
            f"bucket={bucket}, table={dynamodb_table_name}"
        )

    def process(
        self,
        feature_vectors: list[FeatureVector],
        execution_id: str,
        model_version: str | None = None,
    ) -> list[PredictionResult]:
        """Executa o fluxo completo de batch inference.

        1. Prepara input CSV no S3
        2. Executa Batch Transform via SageMaker
        3. Parseia output em PredictionResult
        4. Armazena no DynamoDB

        A inferência é determinística: para o mesmo conjunto de features
        e modelo, os resultados são idênticos (seed=42).

        Args:
            feature_vectors: Lista de FeatureVectors para scoring.
            execution_id: ID único da execução do pipeline.
            model_version: ARN do modelo (None = usar modelo aprovado).

        Returns:
            Lista de PredictionResult com churn_probability, confidence,
            timestamp e model_version.

        Raises:
            ValueError: Se feature_vectors estiver vazia.
            RuntimeError: Se o Batch Transform falhar.
        """
        if not feature_vectors:
            raise ValueError("feature_vectors não pode estar vazia")

        logger.info(
            f"Iniciando batch inference: "
            f"execution_id={execution_id}, "
            f"num_users={len(feature_vectors)}, "
            f"model_version={model_version or 'approved'}"
        )

        # 1. Determinar versão do modelo
        if model_version is None:
            active_model = self._pipeline.get_active_model()
            resolved_model_version = active_model.model_package_arn
        else:
            resolved_model_version = model_version

        # 2. Preparar input CSV no S3
        input_s3_path = self._prepare_input(feature_vectors, execution_id)

        # 3. Executar Batch Transform
        output_s3_path = self._pipeline.predict_batch(
            feature_vectors_s3=input_s3_path,
            model_version=resolved_model_version,
        )

        # 4. Parsear resultados
        predictions = self._parse_predictions(
            output_s3=output_s3_path,
            feature_vectors=feature_vectors,
            model_version=resolved_model_version,
            execution_id=execution_id,
        )

        # 5. Armazenar no DynamoDB
        self._store_predictions(predictions, execution_id)

        logger.info(
            f"Batch inference concluído: "
            f"execution_id={execution_id}, "
            f"predictions={len(predictions)}"
        )

        return predictions

    def _prepare_input(
        self,
        feature_vectors: list[FeatureVector],
        execution_id: str,
    ) -> str:
        """Prepara CSV de input para o Batch Transform e faz upload para S3.

        O CSV contém apenas features numéricas (sem header, sem labels),
        na mesma ordem utilizada no treinamento.

        Args:
            feature_vectors: Lista de FeatureVectors para converter.
            execution_id: ID da execução para organizar no S3.

        Returns:
            S3 URI do arquivo CSV preparado.
        """
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)

        for fv in feature_vectors:
            row = []
            for col in FEATURE_COLUMNS:
                value = getattr(fv, col)
                # Trends podem ser None — usar 0.0 como default
                row.append(value if value is not None else 0.0)
            writer.writerow(row)

        s3_key = f"inference/{execution_id}/input/features.csv"
        self._s3_client.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=csv_buffer.getvalue().encode("utf-8"),
            ContentType="text/csv",
        )

        s3_uri = f"s3://{self._bucket}/{s3_key}"
        logger.info(f"Input preparado: {s3_uri} ({len(feature_vectors)} registros)")
        return s3_uri

    def _parse_predictions(
        self,
        output_s3: str,
        feature_vectors: list[FeatureVector],
        model_version: str,
        execution_id: str,
    ) -> list[PredictionResult]:
        """Parseia output do Batch Transform em PredictionResult.

        O Batch Transform gera um arquivo .out com uma probabilidade
        por linha (na mesma ordem do input).

        Args:
            output_s3: S3 URI do diretório de output do Batch Transform.
            feature_vectors: Feature vectors originais (para mapear user_id).
            model_version: Versão do modelo usada.
            execution_id: ID da execução.

        Returns:
            Lista de PredictionResult com todos os campos preenchidos.
        """
        # O Batch Transform gera output com sufixo .out
        output_file_s3 = f"{output_s3}/features.csv.out"

        # Parse do S3 path
        bucket, key = self._parse_s3_path(output_file_s3)
        response = self._s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")

        # Cada linha é uma probabilidade
        probabilities = []
        for line in content.strip().split("\n"):
            if line.strip():
                probabilities.append(float(line.strip()))

        # Timestamp da inferência (único para todo o batch — determinístico)
        inference_timestamp = datetime.now(timezone.utc).isoformat()

        predictions: list[PredictionResult] = []
        for i, fv in enumerate(feature_vectors):
            if i >= len(probabilities):
                logger.warning(
                    f"Sem predição para user {fv.user_id} "
                    f"(index {i} > {len(probabilities)} resultados)"
                )
                break

            probability = probabilities[i]
            confidence = self._compute_confidence(probability)
            risk_tier = self._determine_risk_tier(probability)

            prediction = PredictionResult(
                user_id=fv.user_id,
                churn_probability=round(probability, 6),
                confidence=round(confidence, 6),
                risk_tier=risk_tier,
                model_version=model_version,
                feature_version=fv.version,
                timestamp=inference_timestamp,
            )
            predictions.append(prediction)

        logger.info(
            f"Predictions parseadas: {len(predictions)} de "
            f"{len(feature_vectors)} feature vectors"
        )
        return predictions

    def _store_predictions(
        self,
        predictions: list[PredictionResult],
        execution_id: str,
    ) -> None:
        """Armazena predições no DynamoDB (tabela churn_predictions).

        Schema DynamoDB:
        - PK: execution_id (String)
        - SK: user_id (String)
        - Attributes: churn_probability, confidence, risk_tier,
          model_version, feature_version, timestamp

        Args:
            predictions: Lista de PredictionResult para armazenar.
            execution_id: ID da execução (partition key).
        """
        logger.info(
            f"Armazenando {len(predictions)} predições no DynamoDB "
            f"(table={self._table_name}, execution_id={execution_id})"
        )

        with self._table.batch_writer() as batch:
            for pred in predictions:
                item = {
                    "execution_id": execution_id,
                    "user_id": pred.user_id,
                    "churn_probability": str(pred.churn_probability),
                    "confidence": str(pred.confidence),
                    "risk_tier": pred.risk_tier,
                    "model_version": pred.model_version,
                    "feature_version": pred.feature_version,
                    "timestamp": pred.timestamp,
                }
                batch.put_item(Item=item)

        logger.info(
            f"Predições armazenadas com sucesso: "
            f"execution_id={execution_id}, total={len(predictions)}"
        )

    @staticmethod
    def _compute_confidence(probability: float) -> float:
        """Calcula o grau de confiança a partir da probabilidade de churn.

        Fórmula: confidence = |probability - 0.5| * 2
        - probability = 0.0 ou 1.0 → confidence = 1.0 (máxima certeza)
        - probability = 0.5 → confidence = 0.0 (máxima incerteza)

        Args:
            probability: Probabilidade de churn (0.0 a 1.0).

        Returns:
            Grau de confiança (0.0 a 1.0).
        """
        return abs(probability - 0.5) * 2.0

    def _determine_risk_tier(self, probability: float) -> str:
        """Determina o tier de risco com base na probabilidade.

        Thresholds (configuráveis):
        - Low: 0.0 a low_max (default 0.30)
        - Medium: low_max+0.01 a medium_max (default 0.60)
        - High: medium_max+0.01 a 1.0

        Args:
            probability: Probabilidade de churn (0.0 a 1.0).

        Returns:
            String "Low", "Medium" ou "High".
        """
        low_max = self._risk_thresholds["low_max"]
        medium_max = self._risk_thresholds["medium_max"]

        if probability <= low_max:
            return "Low"
        elif probability <= medium_max:
            return "Medium"
        else:
            return "High"

    @staticmethod
    def _parse_s3_path(s3_path: str) -> tuple[str, str]:
        """Parseia um S3 URI em bucket e key.

        Args:
            s3_path: Path no formato s3://bucket/key.

        Returns:
            Tupla (bucket, key).

        Raises:
            ValueError: Se o path não está no formato esperado.
        """
        if not s3_path.startswith("s3://"):
            raise ValueError(
                f"S3 path deve iniciar com s3://, recebido: {s3_path}"
            )
        parts = s3_path[5:].split("/", 1)
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"S3 path sem key: {s3_path}")
        return parts[0], parts[1]
