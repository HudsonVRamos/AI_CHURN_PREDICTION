"""Módulo de monitoramento e detecção de data drift.

Publica métricas custom no AWS CloudWatch (namespace: ChurnPrediction)
e detecta drift nas features comparando distribuição atual vs treino.

Métricas publicadas:
- InferenceTime: tempo de inferência em milissegundos
- PredictionCount: contagem de predições executadas
- PredictionFailures: contagem de falhas de predição
- FeatureDriftDetected: drift detectado em feature
- BedrockTimeout: timeout de chamada ao Bedrock
- ExtractionErrors: erros de extração de dados

Alertas:
- InferenceTimeAlert: inference time > threshold configurável
- FailureRateAlert: failure rate > threshold configurável

Audit Trail:
- Vincula cada predição a feature_version, model_version,
  explainability e explanation

Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 17.1, 17.2, 17.5
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DriftAlert:
    """Alerta de data drift para uma feature individual.

    Gerado quando a média atual de uma feature difere da média de treino
    por mais de threshold_std desvios-padrão.
    """

    feature_name: str
    current_mean: float
    training_mean: float
    training_std: float
    shift_in_std: float


@dataclass
class TrainingStats:
    """Estatísticas de treino para uma feature (média e desvio-padrão)."""

    mean: float
    std: float


# ---------------------------------------------------------------------------
# MetricsPublisher
# ---------------------------------------------------------------------------


class MetricsPublisher:
    """Publica métricas custom no AWS CloudWatch.

    Namespace: ChurnPrediction

    Aceita um cliente CloudWatch como parâmetro para facilitar testes.
    Se não fornecido, cria um cliente boto3 padrão.

    Métricas publicadas:
    - InferenceTime (Milliseconds): tempo de inferência por predição
    - PredictionCount (Count): total de predições executadas
    - PredictionFailures (Count): total de falhas de predição
    - FeatureDriftDetected (Count): drift detectado em feature
    - BedrockTimeout (Count): timeouts do Bedrock
    - ExtractionErrors (Count): erros na extração de dados
    """

    NAMESPACE = "ChurnPrediction"

    def __init__(self, cloudwatch_client: Any | None = None) -> None:
        """Inicializa o publisher de métricas.

        Args:
            cloudwatch_client: Cliente boto3 do CloudWatch. Se None, cria um novo.
        """
        if cloudwatch_client is None:
            import boto3
            self._client = boto3.client("cloudwatch")
        else:
            self._client = cloudwatch_client

    def _put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        dimensions: list[dict[str, str]] | None = None,
    ) -> None:
        """Publica uma métrica no CloudWatch.

        Args:
            metric_name: Nome da métrica.
            value: Valor numérico da métrica.
            unit: Unidade da métrica (Milliseconds, Count, etc).
            dimensions: Dimensões opcionais da métrica.
        """
        metric_data: dict[str, Any] = {
            "MetricName": metric_name,
            "Timestamp": datetime.now(timezone.utc),
            "Value": value,
            "Unit": unit,
        }

        if dimensions:
            metric_data["Dimensions"] = dimensions

        try:
            self._client.put_metric_data(
                Namespace=self.NAMESPACE,
                MetricData=[metric_data],
            )
            logger.debug(f"Métrica publicada: {metric_name}={value} {unit}")
        except Exception as e:
            logger.error(f"Erro ao publicar métrica {metric_name}: {e}")
            raise

    def publish_inference_time(self, time_ms: float) -> None:
        """Publica o tempo de inferência em milissegundos.

        Args:
            time_ms: Tempo de inferência em milissegundos (>= 0).
        """
        if time_ms < 0:
            raise ValueError(f"time_ms deve ser >= 0, recebido: {time_ms}")
        self._put_metric("InferenceTime", time_ms, "Milliseconds")

    def publish_prediction_count(self, count: int) -> None:
        """Publica a contagem de predições executadas.

        Args:
            count: Número de predições (>= 0).
        """
        if count < 0:
            raise ValueError(f"count deve ser >= 0, recebido: {count}")
        self._put_metric("PredictionCount", float(count), "Count")

    def publish_prediction_failures(self, count: int) -> None:
        """Publica a contagem de falhas de predição.

        Args:
            count: Número de falhas (>= 0).
        """
        if count < 0:
            raise ValueError(f"count deve ser >= 0, recebido: {count}")
        self._put_metric("PredictionFailures", float(count), "Count")

    def publish_feature_drift(self, feature_name: str, drift_value: float) -> None:
        """Publica detecção de drift em uma feature específica.

        Args:
            feature_name: Nome da feature com drift.
            drift_value: Magnitude do drift (em desvios-padrão).
        """
        if not feature_name:
            raise ValueError("feature_name não pode ser vazio")
        dimensions = [{"Name": "FeatureName", "Value": feature_name}]
        self._put_metric(
            "FeatureDriftDetected", drift_value, "Count", dimensions=dimensions
        )

    def publish_bedrock_timeout(self) -> None:
        """Publica ocorrência de timeout no Bedrock."""
        self._put_metric("BedrockTimeout", 1.0, "Count")

    def publish_extraction_errors(self, count: int) -> None:
        """Publica a contagem de erros de extração.

        Args:
            count: Número de erros de extração (>= 0).
        """
        if count < 0:
            raise ValueError(f"count deve ser >= 0, recebido: {count}")
        self._put_metric("ExtractionErrors", float(count), "Count")


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Detecta data drift comparando features atuais vs distribuição de treino.

    Compara a média atual de cada feature com a média de treino.
    Se o shift exceder threshold_std desvios-padrão, gera um DriftAlert.

    O threshold padrão é 2.0 desvios-padrão, conforme configurado em
    config/settings.yaml (monitoring.drift_threshold_std).

    Validates: Requirement 16.2
    """

    def __init__(
        self,
        training_stats: dict[str, dict[str, float]],
        threshold_std: float = 2.0,
    ) -> None:
        """Inicializa o detector de drift.

        Args:
            training_stats: Dicionário com estatísticas de treino por feature.
                Formato: {"feature_name": {"mean": float, "std": float}}
            threshold_std: Número de desvios-padrão para considerar drift.
                Deve ser > 0. Default: 2.0.

        Raises:
            ValueError: Se training_stats estiver vazio ou threshold_std <= 0.
        """
        if not training_stats:
            raise ValueError("training_stats não pode ser vazio")
        if threshold_std <= 0:
            raise ValueError(
                f"threshold_std deve ser > 0, recebido: {threshold_std}"
            )

        self._training_stats = training_stats
        self._threshold_std = threshold_std

    @property
    def training_stats(self) -> dict[str, dict[str, float]]:
        """Retorna as estatísticas de treino."""
        return self._training_stats

    @property
    def threshold_std(self) -> float:
        """Retorna o threshold em desvios-padrão."""
        return self._threshold_std

    def detect_drift(self, current_features: dict[str, float]) -> list[DriftAlert]:
        """Detecta drift nas features atuais comparando com a distribuição de treino.

        Para cada feature presente tanto em current_features quanto em training_stats,
        calcula o shift em desvios-padrão: |current_mean - training_mean| / training_std.
        Se o shift exceder threshold_std, gera um DriftAlert.

        Features com std=0 no treino são ignoradas (não é possível calcular drift).

        Args:
            current_features: Valores atuais das features.
                Formato: {"feature_name": current_value}

        Returns:
            Lista de DriftAlert para features com drift significativo.
            Lista vazia se nenhum drift detectado.
        """
        alerts: list[DriftAlert] = []

        for feature_name, current_value in current_features.items():
            if feature_name not in self._training_stats:
                logger.debug(
                    f"Feature '{feature_name}' não encontrada nas stats de treino, ignorando."
                )
                continue

            stats = self._training_stats[feature_name]
            training_mean = stats["mean"]
            training_std = stats["std"]

            # Se std == 0, não é possível calcular drift significativo
            if training_std == 0:
                logger.debug(
                    f"Feature '{feature_name}' com std=0 no treino, ignorando."
                )
                continue

            shift_in_std = abs(current_value - training_mean) / training_std

            if shift_in_std > self._threshold_std:
                alert = DriftAlert(
                    feature_name=feature_name,
                    current_mean=current_value,
                    training_mean=training_mean,
                    training_std=training_std,
                    shift_in_std=shift_in_std,
                )
                alerts.append(alert)
                logger.warning(
                    f"Drift detectado em '{feature_name}': "
                    f"shift={shift_in_std:.2f} std (threshold={self._threshold_std})"
                )

        return alerts


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager:
    """Gerencia alertas baseados em thresholds configuráveis.

    Gera alertas quando:
    - Inference time excede threshold (default: 5000ms / 5s)
    - Failure rate excede threshold (default: 5%)

    Loga timing separado para cada estágio do pipeline
    (ML inference, Bedrock explanation, etc).

    Validates: Requirements 16.4, 16.5, 17.2
    """

    def __init__(
        self,
        metrics_publisher: MetricsPublisher,
        max_inference_time_ms: float = 5000.0,
        max_failure_rate_pct: float = 5.0,
    ) -> None:
        """Inicializa o gerenciador de alertas.

        Args:
            metrics_publisher: Publisher de métricas CloudWatch.
            max_inference_time_ms: Threshold máximo de tempo de
                inferência em milissegundos. Default: 5000 (5s).
            max_failure_rate_pct: Threshold máximo de taxa de falha
                em percentual. Default: 5.0%.

        Raises:
            ValueError: Se thresholds forem <= 0.
        """
        if max_inference_time_ms <= 0:
            raise ValueError(
                "max_inference_time_ms deve ser > 0, "
                f"recebido: {max_inference_time_ms}"
            )
        if max_failure_rate_pct <= 0:
            raise ValueError(
                "max_failure_rate_pct deve ser > 0, "
                f"recebido: {max_failure_rate_pct}"
            )

        self._publisher = metrics_publisher
        self._max_inference_time_ms = max_inference_time_ms
        self._max_failure_rate_pct = max_failure_rate_pct

    @property
    def max_inference_time_ms(self) -> float:
        """Retorna o threshold de inference time em ms."""
        return self._max_inference_time_ms

    @property
    def max_failure_rate_pct(self) -> float:
        """Retorna o threshold de failure rate em %."""
        return self._max_failure_rate_pct

    def check_inference_time(self, time_ms: float) -> bool:
        """Verifica se inference time excede o threshold.

        Se exceder, publica métrica de alerta e loga WARNING.

        Args:
            time_ms: Tempo de inferência em milissegundos.

        Returns:
            True se alerta foi gerado (excedeu threshold).

        Raises:
            ValueError: Se time_ms < 0.
        """
        if time_ms < 0:
            raise ValueError(
                f"time_ms deve ser >= 0, recebido: {time_ms}"
            )

        exceeded = time_ms > self._max_inference_time_ms

        if exceeded:
            logger.warning(
                "ALERTA: Inference time %.1fms excede "
                "threshold de %.1fms",
                time_ms,
                self._max_inference_time_ms,
            )
            self._publisher._put_metric(
                "InferenceTimeAlert", time_ms, "Milliseconds"
            )

        return exceeded

    def check_failure_rate(
        self, total: int, failures: int
    ) -> bool:
        """Verifica se a taxa de falha excede o threshold.

        Se exceder, publica métrica de alerta e loga WARNING.

        Args:
            total: Total de predições no batch.
            failures: Total de falhas no batch.

        Returns:
            True se alerta foi gerado (rate > threshold).

        Raises:
            ValueError: Se total <= 0 ou failures < 0 ou
                failures > total.
        """
        if total <= 0:
            raise ValueError(
                f"total deve ser > 0, recebido: {total}"
            )
        if failures < 0:
            raise ValueError(
                f"failures deve ser >= 0, recebido: {failures}"
            )
        if failures > total:
            raise ValueError(
                "failures não pode ser > total: "
                f"{failures} > {total}"
            )

        rate_pct = (failures / total) * 100.0
        exceeded = rate_pct > self._max_failure_rate_pct

        if exceeded:
            logger.warning(
                "ALERTA: Failure rate %.2f%% excede "
                "threshold de %.2f%% (%d/%d)",
                rate_pct,
                self._max_failure_rate_pct,
                failures,
                total,
            )
            self._publisher._put_metric(
                "FailureRateAlert", rate_pct, "Percent"
            )

        return exceeded

    def log_timing(self, stage: str, duration_ms: float) -> None:
        """Loga timing separado para um estágio específico.

        Publica métrica com dimensão de estágio e loga INFO.

        Estágios esperados:
        - ml-inference: tempo de inferência do modelo ML
        - bedrock-explanation: tempo de geração de explicação

        Args:
            stage: Nome do estágio (ex: 'ml-inference').
            duration_ms: Duração do estágio em milissegundos.

        Raises:
            ValueError: Se stage vazio ou duration_ms < 0.
        """
        if not stage:
            raise ValueError("stage não pode ser vazio")
        if duration_ms < 0:
            raise ValueError(
                f"duration_ms deve ser >= 0, recebido: {duration_ms}"
            )

        logger.info(
            "Timing [%s]: %.1fms",
            stage,
            duration_ms,
        )

        dimensions = [{"Name": "Stage", "Value": stage}]
        self._publisher._put_metric(
            "StageTiming",
            duration_ms,
            "Milliseconds",
            dimensions=dimensions,
        )


# ---------------------------------------------------------------------------
# AuditTrail
# ---------------------------------------------------------------------------


@dataclass
class AuditRecord:
    """Registro de auditoria de uma predição individual.

    Vincula a predição a todas as suas dependências para
    rastreabilidade completa.

    Validates: Requirement 17.5
    """

    audit_id: str
    execution_id: str
    user_id: str
    feature_version: int
    model_version: str
    explainability_computed: bool
    explanation_generated: bool
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditTrail:
    """Registro de auditoria vinculando predição a dependências.

    Cada predição é vinculada a:
    - feature_version: versão do Feature_Vector usado
    - model_version: versão do modelo (SageMaker Model Package ARN)
    - explainability_computed: se SHAP foi calculado
    - explanation_generated: se Bedrock gerou explicação

    Validates: Requirements 17.1, 17.5
    """

    def __init__(self) -> None:
        """Inicializa o audit trail com lista vazia de records."""
        self._records: list[AuditRecord] = []

    @property
    def records(self) -> list[AuditRecord]:
        """Retorna todos os records de auditoria."""
        return list(self._records)

    def record_prediction_audit(
        self,
        execution_id: str,
        user_id: str,
        feature_version: int,
        model_version: str,
        explainability_computed: bool,
        explanation_generated: bool,
    ) -> dict[str, Any]:
        """Registra trail de auditoria e retorna o record.

        Args:
            execution_id: UUID da execução do pipeline.
            user_id: UUID do assinante.
            feature_version: Versão do Feature_Vector.
            model_version: Versão/ARN do modelo.
            explainability_computed: Se SHAP foi calculado.
            explanation_generated: Se Bedrock gerou explicação.

        Returns:
            Dicionário com os dados do audit record criado.

        Raises:
            ValueError: Se execution_id ou user_id vazios,
                ou feature_version < 0.
        """
        if not execution_id:
            raise ValueError("execution_id não pode ser vazio")
        if not user_id:
            raise ValueError("user_id não pode ser vazio")
        if feature_version < 0:
            raise ValueError(
                "feature_version deve ser >= 0, "
                f"recebido: {feature_version}"
            )
        if not model_version:
            raise ValueError("model_version não pode ser vazio")

        audit_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        record = AuditRecord(
            audit_id=audit_id,
            execution_id=execution_id,
            user_id=user_id,
            feature_version=feature_version,
            model_version=model_version,
            explainability_computed=explainability_computed,
            explanation_generated=explanation_generated,
            timestamp=timestamp,
        )

        self._records.append(record)

        logger.info(
            "Audit trail registrado: execution=%s user=%s "
            "feature_v=%d model_v=%s shap=%s bedrock=%s",
            execution_id,
            user_id,
            feature_version,
            model_version,
            explainability_computed,
            explanation_generated,
        )

        return {
            "audit_id": record.audit_id,
            "execution_id": record.execution_id,
            "user_id": record.user_id,
            "feature_version": record.feature_version,
            "model_version": record.model_version,
            "explainability_computed": record.explainability_computed,
            "explanation_generated": record.explanation_generated,
            "timestamp": record.timestamp,
        }

    def get_records_by_execution(
        self, execution_id: str
    ) -> list[AuditRecord]:
        """Retorna records filtrados por execution_id.

        Args:
            execution_id: UUID da execução do pipeline.

        Returns:
            Lista de AuditRecords da execução especificada.
        """
        return [
            r for r in self._records
            if r.execution_id == execution_id
        ]

    def get_records_by_user(
        self, user_id: str
    ) -> list[AuditRecord]:
        """Retorna records filtrados por user_id.

        Args:
            user_id: UUID do assinante.

        Returns:
            Lista de AuditRecords do usuário especificado.
        """
        return [
            r for r in self._records
            if r.user_id == user_id
        ]
