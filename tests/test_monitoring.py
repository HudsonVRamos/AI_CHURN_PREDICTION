"""Testes para o módulo de monitoramento (métricas CloudWatch e detecção de drift).

Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5, 17.1, 17.2, 17.5
"""

from __future__ import annotations

from unittest.mock import MagicMock
from datetime import datetime

import pytest

from src.common.monitoring import (
    AlertManager,
    AuditRecord,
    AuditTrail,
    DriftAlert,
    DriftDetector,
    MetricsPublisher,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cloudwatch_client() -> MagicMock:
    """Cliente CloudWatch mockado para testes."""
    return MagicMock()


@pytest.fixture
def publisher(mock_cloudwatch_client: MagicMock) -> MetricsPublisher:
    """MetricsPublisher com cliente mockado."""
    return MetricsPublisher(cloudwatch_client=mock_cloudwatch_client)


@pytest.fixture
def training_stats() -> dict[str, dict[str, float]]:
    """Estatísticas de treino de exemplo."""
    return {
        "total_sessions": {"mean": 50.0, "std": 10.0},
        "avg_happiness_score": {"mean": 7.5, "std": 1.5},
        "error_rate": {"mean": 0.05, "std": 0.02},
        "sessions_per_week": {"mean": 5.0, "std": 2.0},
        "total_viewing_hours": {"mean": 30.0, "std": 8.0},
    }


@pytest.fixture
def drift_detector(training_stats: dict) -> DriftDetector:
    """DriftDetector com stats de treino padrão e threshold=2.0."""
    return DriftDetector(training_stats=training_stats, threshold_std=2.0)


# ---------------------------------------------------------------------------
# Testes MetricsPublisher
# ---------------------------------------------------------------------------


class TestMetricsPublisher:
    """Testes para publicação de métricas no CloudWatch."""

    def test_namespace_is_churn_prediction(self) -> None:
        """Namespace deve ser 'ChurnPrediction'."""
        assert MetricsPublisher.NAMESPACE == "ChurnPrediction"

    def test_publish_inference_time(
        self, publisher: MetricsPublisher, mock_cloudwatch_client: MagicMock
    ) -> None:
        """Publica InferenceTime com valor e unidade corretos."""
        publisher.publish_inference_time(150.5)

        mock_cloudwatch_client.put_metric_data.assert_called_once()
        call_kwargs = mock_cloudwatch_client.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == "ChurnPrediction"
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "InferenceTime"
        assert metric["Value"] == 150.5
        assert metric["Unit"] == "Milliseconds"
        assert isinstance(metric["Timestamp"], datetime)

    def test_publish_inference_time_rejects_negative(
        self, publisher: MetricsPublisher
    ) -> None:
        """Rejeita tempo de inferência negativo."""
        with pytest.raises(ValueError, match="time_ms deve ser >= 0"):
            publisher.publish_inference_time(-1.0)

    def test_publish_prediction_count(
        self, publisher: MetricsPublisher, mock_cloudwatch_client: MagicMock
    ) -> None:
        """Publica PredictionCount como Count."""
        publisher.publish_prediction_count(42)

        call_kwargs = mock_cloudwatch_client.put_metric_data.call_args[1]
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "PredictionCount"
        assert metric["Value"] == 42.0
        assert metric["Unit"] == "Count"

    def test_publish_prediction_count_rejects_negative(
        self, publisher: MetricsPublisher
    ) -> None:
        """Rejeita contagem negativa de predições."""
        with pytest.raises(ValueError, match="count deve ser >= 0"):
            publisher.publish_prediction_count(-5)

    def test_publish_prediction_failures(
        self, publisher: MetricsPublisher, mock_cloudwatch_client: MagicMock
    ) -> None:
        """Publica PredictionFailures como Count."""
        publisher.publish_prediction_failures(3)

        call_kwargs = mock_cloudwatch_client.put_metric_data.call_args[1]
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "PredictionFailures"
        assert metric["Value"] == 3.0
        assert metric["Unit"] == "Count"

    def test_publish_prediction_failures_rejects_negative(
        self, publisher: MetricsPublisher
    ) -> None:
        """Rejeita contagem negativa de falhas."""
        with pytest.raises(ValueError, match="count deve ser >= 0"):
            publisher.publish_prediction_failures(-1)

    def test_publish_feature_drift(
        self, publisher: MetricsPublisher, mock_cloudwatch_client: MagicMock
    ) -> None:
        """Publica FeatureDriftDetected com dimensão FeatureName."""
        publisher.publish_feature_drift("total_sessions", 3.5)

        call_kwargs = mock_cloudwatch_client.put_metric_data.call_args[1]
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "FeatureDriftDetected"
        assert metric["Value"] == 3.5
        assert metric["Unit"] == "Count"
        assert metric["Dimensions"] == [
            {"Name": "FeatureName", "Value": "total_sessions"}
        ]

    def test_publish_feature_drift_rejects_empty_name(
        self, publisher: MetricsPublisher
    ) -> None:
        """Rejeita feature_name vazio."""
        with pytest.raises(ValueError, match="feature_name não pode ser vazio"):
            publisher.publish_feature_drift("", 2.5)

    def test_publish_bedrock_timeout(
        self, publisher: MetricsPublisher, mock_cloudwatch_client: MagicMock
    ) -> None:
        """Publica BedrockTimeout com valor 1."""
        publisher.publish_bedrock_timeout()

        call_kwargs = mock_cloudwatch_client.put_metric_data.call_args[1]
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "BedrockTimeout"
        assert metric["Value"] == 1.0
        assert metric["Unit"] == "Count"

    def test_publish_extraction_errors(
        self, publisher: MetricsPublisher, mock_cloudwatch_client: MagicMock
    ) -> None:
        """Publica ExtractionErrors como Count."""
        publisher.publish_extraction_errors(7)

        call_kwargs = mock_cloudwatch_client.put_metric_data.call_args[1]
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "ExtractionErrors"
        assert metric["Value"] == 7.0
        assert metric["Unit"] == "Count"

    def test_publish_extraction_errors_rejects_negative(
        self, publisher: MetricsPublisher
    ) -> None:
        """Rejeita contagem negativa de erros de extração."""
        with pytest.raises(ValueError, match="count deve ser >= 0"):
            publisher.publish_extraction_errors(-1)

    def test_publish_inference_time_zero(
        self, publisher: MetricsPublisher, mock_cloudwatch_client: MagicMock
    ) -> None:
        """Aceita tempo de inferência zero."""
        publisher.publish_inference_time(0.0)

        call_kwargs = mock_cloudwatch_client.put_metric_data.call_args[1]
        metric = call_kwargs["MetricData"][0]
        assert metric["Value"] == 0.0

    def test_cloudwatch_error_propagates(
        self, publisher: MetricsPublisher, mock_cloudwatch_client: MagicMock
    ) -> None:
        """Erros do CloudWatch são propagados."""
        mock_cloudwatch_client.put_metric_data.side_effect = Exception("AWS Error")
        with pytest.raises(Exception, match="AWS Error"):
            publisher.publish_prediction_count(1)


# ---------------------------------------------------------------------------
# Testes DriftDetector
# ---------------------------------------------------------------------------


class TestDriftDetector:
    """Testes para detecção de data drift."""

    def test_no_drift_within_threshold(self, drift_detector: DriftDetector) -> None:
        """Sem drift quando valores estão dentro do threshold."""
        current = {
            "total_sessions": 55.0,  # shift = 0.5 std (< 2.0)
            "avg_happiness_score": 7.0,  # shift = 0.33 std
            "error_rate": 0.06,  # shift = 0.5 std
        }
        alerts = drift_detector.detect_drift(current)
        assert alerts == []

    def test_drift_detected_above_threshold(
        self, drift_detector: DriftDetector
    ) -> None:
        """Drift detectado quando shift > threshold."""
        current = {
            "total_sessions": 80.0,  # shift = 3.0 std (> 2.0) -> DRIFT
            "avg_happiness_score": 7.5,  # shift = 0.0 std -> OK
        }
        alerts = drift_detector.detect_drift(current)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.feature_name == "total_sessions"
        assert alert.current_mean == 80.0
        assert alert.training_mean == 50.0
        assert alert.training_std == 10.0
        assert alert.shift_in_std == 3.0

    def test_multiple_drifts_detected(self, drift_detector: DriftDetector) -> None:
        """Múltiplas features com drift são todas reportadas."""
        current = {
            "total_sessions": 80.0,  # shift = 3.0 std -> DRIFT
            "avg_happiness_score": 3.0,  # shift = 3.0 std -> DRIFT
            "error_rate": 0.06,  # shift = 0.5 std -> OK
        }
        alerts = drift_detector.detect_drift(current)

        assert len(alerts) == 2
        feature_names = {a.feature_name for a in alerts}
        assert feature_names == {"total_sessions", "avg_happiness_score"}

    def test_unknown_feature_ignored(self, drift_detector: DriftDetector) -> None:
        """Features não presentes nas stats de treino são ignoradas."""
        current = {
            "unknown_feature": 999.0,
            "total_sessions": 50.0,  # sem drift
        }
        alerts = drift_detector.detect_drift(current)
        assert alerts == []

    def test_zero_std_feature_ignored(self) -> None:
        """Features com std=0 no treino são ignoradas (divisão por zero)."""
        stats = {
            "constant_feature": {"mean": 5.0, "std": 0.0},
            "normal_feature": {"mean": 10.0, "std": 2.0},
        }
        detector = DriftDetector(training_stats=stats)

        current = {
            "constant_feature": 100.0,  # Ignorado (std=0)
            "normal_feature": 10.5,  # shift = 0.25 std -> OK
        }
        alerts = detector.detect_drift(current)
        assert alerts == []

    def test_negative_drift_detected(self, drift_detector: DriftDetector) -> None:
        """Drift detectado tanto para valores acima quanto abaixo da média."""
        current = {
            "total_sessions": 20.0,  # shift = |-30| / 10 = 3.0 std -> DRIFT
        }
        alerts = drift_detector.detect_drift(current)

        assert len(alerts) == 1
        assert alerts[0].shift_in_std == 3.0

    def test_exact_threshold_no_drift(self, drift_detector: DriftDetector) -> None:
        """Shift exatamente no threshold (==) NÃO gera drift (> é necessário)."""
        # threshold = 2.0, std = 10.0, shift exato = 2.0
        current = {
            "total_sessions": 70.0,  # shift = |70 - 50| / 10 = 2.0 (== threshold)
        }
        alerts = drift_detector.detect_drift(current)
        assert alerts == []

    def test_custom_threshold(self) -> None:
        """Threshold customizado altera a sensibilidade da detecção."""
        stats = {"feature_a": {"mean": 100.0, "std": 10.0}}
        detector = DriftDetector(training_stats=stats, threshold_std=1.0)

        current = {"feature_a": 115.0}  # shift = 1.5 std (> 1.0)
        alerts = detector.detect_drift(current)
        assert len(alerts) == 1
        assert alerts[0].shift_in_std == 1.5

    def test_empty_current_features(self, drift_detector: DriftDetector) -> None:
        """Dicionário vazio de features atuais retorna lista vazia."""
        alerts = drift_detector.detect_drift({})
        assert alerts == []

    def test_init_rejects_empty_stats(self) -> None:
        """Rejeita training_stats vazio."""
        with pytest.raises(ValueError, match="training_stats não pode ser vazio"):
            DriftDetector(training_stats={})

    def test_init_rejects_negative_threshold(self) -> None:
        """Rejeita threshold <= 0."""
        with pytest.raises(ValueError, match="threshold_std deve ser > 0"):
            DriftDetector(
                training_stats={"f": {"mean": 1.0, "std": 1.0}},
                threshold_std=0.0,
            )

    def test_init_rejects_zero_threshold(self) -> None:
        """Rejeita threshold == 0."""
        with pytest.raises(ValueError, match="threshold_std deve ser > 0"):
            DriftDetector(
                training_stats={"f": {"mean": 1.0, "std": 1.0}},
                threshold_std=0.0,
            )

    def test_properties(self, training_stats: dict) -> None:
        """Verifica que properties retornam valores configurados."""
        detector = DriftDetector(training_stats=training_stats, threshold_std=3.0)
        assert detector.training_stats == training_stats
        assert detector.threshold_std == 3.0


# ---------------------------------------------------------------------------
# Testes DriftAlert dataclass
# ---------------------------------------------------------------------------


class TestDriftAlert:
    """Testes para o dataclass DriftAlert."""

    def test_creation(self) -> None:
        """Criação com todos os campos."""
        alert = DriftAlert(
            feature_name="total_sessions",
            current_mean=80.0,
            training_mean=50.0,
            training_std=10.0,
            shift_in_std=3.0,
        )
        assert alert.feature_name == "total_sessions"
        assert alert.current_mean == 80.0
        assert alert.training_mean == 50.0
        assert alert.training_std == 10.0
        assert alert.shift_in_std == 3.0

    def test_equality(self) -> None:
        """Dois DriftAlerts com mesmos valores são iguais."""
        alert1 = DriftAlert("f", 1.0, 2.0, 3.0, 4.0)
        alert2 = DriftAlert("f", 1.0, 2.0, 3.0, 4.0)
        assert alert1 == alert2



# ---------------------------------------------------------------------------
# Fixtures AlertManager e AuditTrail
# ---------------------------------------------------------------------------


@pytest.fixture
def alert_manager(publisher: MetricsPublisher) -> AlertManager:
    """AlertManager com thresholds padrão (5000ms, 5%)."""
    return AlertManager(metrics_publisher=publisher)


@pytest.fixture
def alert_manager_custom(publisher: MetricsPublisher) -> AlertManager:
    """AlertManager com thresholds customizados."""
    return AlertManager(
        metrics_publisher=publisher,
        max_inference_time_ms=2000.0,
        max_failure_rate_pct=10.0,
    )


@pytest.fixture
def audit_trail() -> AuditTrail:
    """AuditTrail vazio para testes."""
    return AuditTrail()


# ---------------------------------------------------------------------------
# Testes AlertManager
# ---------------------------------------------------------------------------


class TestAlertManager:
    """Testes para alertas baseados em thresholds.

    Validates: Requirements 16.4, 16.5, 17.2
    """

    def test_init_default_thresholds(
        self, alert_manager: AlertManager
    ) -> None:
        """Thresholds padrão: 5000ms e 5%."""
        assert alert_manager.max_inference_time_ms == 5000.0
        assert alert_manager.max_failure_rate_pct == 5.0

    def test_init_custom_thresholds(
        self, alert_manager_custom: AlertManager
    ) -> None:
        """Thresholds customizados são respeitados."""
        assert alert_manager_custom.max_inference_time_ms == 2000.0
        assert alert_manager_custom.max_failure_rate_pct == 10.0

    def test_init_rejects_zero_inference_time(
        self, publisher: MetricsPublisher
    ) -> None:
        """Rejeita max_inference_time_ms <= 0."""
        with pytest.raises(
            ValueError, match="max_inference_time_ms deve ser > 0"
        ):
            AlertManager(
                metrics_publisher=publisher,
                max_inference_time_ms=0,
            )

    def test_init_rejects_negative_failure_rate(
        self, publisher: MetricsPublisher
    ) -> None:
        """Rejeita max_failure_rate_pct <= 0."""
        with pytest.raises(
            ValueError, match="max_failure_rate_pct deve ser > 0"
        ):
            AlertManager(
                metrics_publisher=publisher,
                max_failure_rate_pct=-1.0,
            )

    # --- check_inference_time ---

    def test_inference_time_no_alert_below_threshold(
        self, alert_manager: AlertManager
    ) -> None:
        """Sem alerta quando tempo está abaixo do threshold."""
        result = alert_manager.check_inference_time(3000.0)
        assert result is False

    def test_inference_time_no_alert_at_threshold(
        self, alert_manager: AlertManager
    ) -> None:
        """Sem alerta quando tempo é exatamente o threshold."""
        result = alert_manager.check_inference_time(5000.0)
        assert result is False

    def test_inference_time_alert_above_threshold(
        self,
        alert_manager: AlertManager,
        mock_cloudwatch_client: MagicMock,
    ) -> None:
        """Alerta gerado quando tempo excede threshold."""
        result = alert_manager.check_inference_time(6000.0)
        assert result is True

        # Verifica publicação da métrica de alerta
        call_kwargs = (
            mock_cloudwatch_client.put_metric_data.call_args[1]
        )
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "InferenceTimeAlert"
        assert metric["Value"] == 6000.0
        assert metric["Unit"] == "Milliseconds"

    def test_inference_time_alert_custom_threshold(
        self,
        alert_manager_custom: AlertManager,
        mock_cloudwatch_client: MagicMock,
    ) -> None:
        """Alerta com threshold customizado (2000ms)."""
        # Abaixo do threshold custom
        assert alert_manager_custom.check_inference_time(1500.0) is False
        # Acima do threshold custom
        assert alert_manager_custom.check_inference_time(2500.0) is True

    def test_inference_time_rejects_negative(
        self, alert_manager: AlertManager
    ) -> None:
        """Rejeita time_ms negativo."""
        with pytest.raises(ValueError, match="time_ms deve ser >= 0"):
            alert_manager.check_inference_time(-1.0)

    # --- check_failure_rate ---

    def test_failure_rate_no_alert_below_threshold(
        self, alert_manager: AlertManager
    ) -> None:
        """Sem alerta quando taxa de falha está abaixo de 5%."""
        result = alert_manager.check_failure_rate(100, 3)
        assert result is False

    def test_failure_rate_no_alert_at_threshold(
        self, alert_manager: AlertManager
    ) -> None:
        """Sem alerta quando taxa é exatamente 5%."""
        result = alert_manager.check_failure_rate(100, 5)
        assert result is False

    def test_failure_rate_alert_above_threshold(
        self,
        alert_manager: AlertManager,
        mock_cloudwatch_client: MagicMock,
    ) -> None:
        """Alerta gerado quando taxa excede 5%."""
        result = alert_manager.check_failure_rate(100, 6)
        assert result is True

        # Verifica publicação da métrica de alerta
        call_kwargs = (
            mock_cloudwatch_client.put_metric_data.call_args[1]
        )
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "FailureRateAlert"
        assert metric["Value"] == 6.0
        assert metric["Unit"] == "Percent"

    def test_failure_rate_custom_threshold(
        self,
        alert_manager_custom: AlertManager,
    ) -> None:
        """Threshold customizado (10%) altera sensibilidade."""
        # 8% abaixo do threshold de 10%
        assert (
            alert_manager_custom.check_failure_rate(100, 8) is False
        )
        # 11% acima do threshold de 10%
        assert (
            alert_manager_custom.check_failure_rate(100, 11) is True
        )

    def test_failure_rate_rejects_zero_total(
        self, alert_manager: AlertManager
    ) -> None:
        """Rejeita total <= 0."""
        with pytest.raises(ValueError, match="total deve ser > 0"):
            alert_manager.check_failure_rate(0, 0)

    def test_failure_rate_rejects_negative_failures(
        self, alert_manager: AlertManager
    ) -> None:
        """Rejeita failures < 0."""
        with pytest.raises(
            ValueError, match="failures deve ser >= 0"
        ):
            alert_manager.check_failure_rate(10, -1)

    def test_failure_rate_rejects_failures_exceeding_total(
        self, alert_manager: AlertManager
    ) -> None:
        """Rejeita failures > total."""
        with pytest.raises(
            ValueError, match="failures não pode ser > total"
        ):
            alert_manager.check_failure_rate(10, 15)

    # --- log_timing ---

    def test_log_timing_publishes_metric(
        self,
        alert_manager: AlertManager,
        mock_cloudwatch_client: MagicMock,
    ) -> None:
        """log_timing publica métrica StageTiming com dimensão."""
        alert_manager.log_timing("ml-inference", 250.0)

        call_kwargs = (
            mock_cloudwatch_client.put_metric_data.call_args[1]
        )
        metric = call_kwargs["MetricData"][0]
        assert metric["MetricName"] == "StageTiming"
        assert metric["Value"] == 250.0
        assert metric["Unit"] == "Milliseconds"
        assert metric["Dimensions"] == [
            {"Name": "Stage", "Value": "ml-inference"}
        ]

    def test_log_timing_bedrock_stage(
        self,
        alert_manager: AlertManager,
        mock_cloudwatch_client: MagicMock,
    ) -> None:
        """log_timing funciona para estágio bedrock-explanation."""
        alert_manager.log_timing("bedrock-explanation", 1500.0)

        call_kwargs = (
            mock_cloudwatch_client.put_metric_data.call_args[1]
        )
        metric = call_kwargs["MetricData"][0]
        assert metric["Dimensions"] == [
            {"Name": "Stage", "Value": "bedrock-explanation"}
        ]

    def test_log_timing_rejects_empty_stage(
        self, alert_manager: AlertManager
    ) -> None:
        """Rejeita stage vazio."""
        with pytest.raises(ValueError, match="stage não pode ser vazio"):
            alert_manager.log_timing("", 100.0)

    def test_log_timing_rejects_negative_duration(
        self, alert_manager: AlertManager
    ) -> None:
        """Rejeita duration_ms negativo."""
        with pytest.raises(
            ValueError, match="duration_ms deve ser >= 0"
        ):
            alert_manager.log_timing("ml-inference", -1.0)

    def test_log_timing_zero_duration(
        self,
        alert_manager: AlertManager,
        mock_cloudwatch_client: MagicMock,
    ) -> None:
        """Aceita duration_ms = 0."""
        alert_manager.log_timing("ml-inference", 0.0)

        call_kwargs = (
            mock_cloudwatch_client.put_metric_data.call_args[1]
        )
        metric = call_kwargs["MetricData"][0]
        assert metric["Value"] == 0.0


# ---------------------------------------------------------------------------
# Testes AuditTrail
# ---------------------------------------------------------------------------


class TestAuditTrail:
    """Testes para registro de auditoria de predições.

    Validates: Requirements 17.1, 17.5
    """

    def test_record_prediction_audit_returns_dict(
        self, audit_trail: AuditTrail
    ) -> None:
        """record_prediction_audit retorna dicionário completo."""
        result = audit_trail.record_prediction_audit(
            execution_id="exec-001",
            user_id="user-abc",
            feature_version=3,
            model_version="model-v1.2",
            explainability_computed=True,
            explanation_generated=True,
        )

        assert result["execution_id"] == "exec-001"
        assert result["user_id"] == "user-abc"
        assert result["feature_version"] == 3
        assert result["model_version"] == "model-v1.2"
        assert result["explainability_computed"] is True
        assert result["explanation_generated"] is True
        assert "audit_id" in result
        assert "timestamp" in result

    def test_record_stores_in_internal_list(
        self, audit_trail: AuditTrail
    ) -> None:
        """Cada registro é armazenado internamente."""
        audit_trail.record_prediction_audit(
            execution_id="exec-001",
            user_id="user-1",
            feature_version=1,
            model_version="v1",
            explainability_computed=True,
            explanation_generated=False,
        )
        audit_trail.record_prediction_audit(
            execution_id="exec-001",
            user_id="user-2",
            feature_version=2,
            model_version="v1",
            explainability_computed=False,
            explanation_generated=False,
        )

        assert len(audit_trail.records) == 2

    def test_record_with_no_explainability(
        self, audit_trail: AuditTrail
    ) -> None:
        """Aceita predição sem explainability/explanation."""
        result = audit_trail.record_prediction_audit(
            execution_id="exec-002",
            user_id="user-xyz",
            feature_version=1,
            model_version="model-v2.0",
            explainability_computed=False,
            explanation_generated=False,
        )

        assert result["explainability_computed"] is False
        assert result["explanation_generated"] is False

    def test_record_timestamp_iso_format(
        self, audit_trail: AuditTrail
    ) -> None:
        """Timestamp é gerado em formato ISO 8601."""
        result = audit_trail.record_prediction_audit(
            execution_id="exec-003",
            user_id="user-ts",
            feature_version=1,
            model_version="v1",
            explainability_computed=True,
            explanation_generated=True,
        )

        # Deve ser parseable como ISO datetime
        ts = result["timestamp"]
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_record_unique_audit_ids(
        self, audit_trail: AuditTrail
    ) -> None:
        """Cada record recebe um audit_id único."""
        r1 = audit_trail.record_prediction_audit(
            execution_id="exec-004",
            user_id="user-1",
            feature_version=1,
            model_version="v1",
            explainability_computed=True,
            explanation_generated=True,
        )
        r2 = audit_trail.record_prediction_audit(
            execution_id="exec-004",
            user_id="user-2",
            feature_version=1,
            model_version="v1",
            explainability_computed=True,
            explanation_generated=True,
        )

        assert r1["audit_id"] != r2["audit_id"]

    def test_rejects_empty_execution_id(
        self, audit_trail: AuditTrail
    ) -> None:
        """Rejeita execution_id vazio."""
        with pytest.raises(
            ValueError, match="execution_id não pode ser vazio"
        ):
            audit_trail.record_prediction_audit(
                execution_id="",
                user_id="user-1",
                feature_version=1,
                model_version="v1",
                explainability_computed=True,
                explanation_generated=True,
            )

    def test_rejects_empty_user_id(
        self, audit_trail: AuditTrail
    ) -> None:
        """Rejeita user_id vazio."""
        with pytest.raises(
            ValueError, match="user_id não pode ser vazio"
        ):
            audit_trail.record_prediction_audit(
                execution_id="exec-005",
                user_id="",
                feature_version=1,
                model_version="v1",
                explainability_computed=True,
                explanation_generated=True,
            )

    def test_rejects_negative_feature_version(
        self, audit_trail: AuditTrail
    ) -> None:
        """Rejeita feature_version < 0."""
        with pytest.raises(
            ValueError, match="feature_version deve ser >= 0"
        ):
            audit_trail.record_prediction_audit(
                execution_id="exec-006",
                user_id="user-1",
                feature_version=-1,
                model_version="v1",
                explainability_computed=True,
                explanation_generated=True,
            )

    def test_rejects_empty_model_version(
        self, audit_trail: AuditTrail
    ) -> None:
        """Rejeita model_version vazio."""
        with pytest.raises(
            ValueError, match="model_version não pode ser vazio"
        ):
            audit_trail.record_prediction_audit(
                execution_id="exec-007",
                user_id="user-1",
                feature_version=1,
                model_version="",
                explainability_computed=True,
                explanation_generated=True,
            )

    def test_get_records_by_execution(
        self, audit_trail: AuditTrail
    ) -> None:
        """Filtra records por execution_id."""
        audit_trail.record_prediction_audit(
            execution_id="exec-A",
            user_id="user-1",
            feature_version=1,
            model_version="v1",
            explainability_computed=True,
            explanation_generated=True,
        )
        audit_trail.record_prediction_audit(
            execution_id="exec-B",
            user_id="user-2",
            feature_version=2,
            model_version="v2",
            explainability_computed=False,
            explanation_generated=False,
        )
        audit_trail.record_prediction_audit(
            execution_id="exec-A",
            user_id="user-3",
            feature_version=1,
            model_version="v1",
            explainability_computed=True,
            explanation_generated=False,
        )

        records_a = audit_trail.get_records_by_execution("exec-A")
        assert len(records_a) == 2
        assert all(r.execution_id == "exec-A" for r in records_a)

        records_b = audit_trail.get_records_by_execution("exec-B")
        assert len(records_b) == 1

    def test_get_records_by_user(
        self, audit_trail: AuditTrail
    ) -> None:
        """Filtra records por user_id."""
        audit_trail.record_prediction_audit(
            execution_id="exec-1",
            user_id="user-X",
            feature_version=1,
            model_version="v1",
            explainability_computed=True,
            explanation_generated=True,
        )
        audit_trail.record_prediction_audit(
            execution_id="exec-2",
            user_id="user-X",
            feature_version=2,
            model_version="v2",
            explainability_computed=True,
            explanation_generated=True,
        )
        audit_trail.record_prediction_audit(
            execution_id="exec-1",
            user_id="user-Y",
            feature_version=1,
            model_version="v1",
            explainability_computed=False,
            explanation_generated=False,
        )

        records_x = audit_trail.get_records_by_user("user-X")
        assert len(records_x) == 2
        assert all(r.user_id == "user-X" for r in records_x)

    def test_records_returns_copy(
        self, audit_trail: AuditTrail
    ) -> None:
        """Property records retorna cópia (imutabilidade)."""
        audit_trail.record_prediction_audit(
            execution_id="exec-1",
            user_id="user-1",
            feature_version=1,
            model_version="v1",
            explainability_computed=True,
            explanation_generated=True,
        )

        records = audit_trail.records
        records.clear()  # Limpa a cópia

        # Original não é afetado
        assert len(audit_trail.records) == 1


# ---------------------------------------------------------------------------
# Testes AuditRecord dataclass
# ---------------------------------------------------------------------------


class TestAuditRecord:
    """Testes para o dataclass AuditRecord."""

    def test_creation(self) -> None:
        """Criação com todos os campos."""
        record = AuditRecord(
            audit_id="id-1",
            execution_id="exec-1",
            user_id="user-1",
            feature_version=3,
            model_version="v2.1",
            explainability_computed=True,
            explanation_generated=False,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        assert record.audit_id == "id-1"
        assert record.execution_id == "exec-1"
        assert record.user_id == "user-1"
        assert record.feature_version == 3
        assert record.model_version == "v2.1"
        assert record.explainability_computed is True
        assert record.explanation_generated is False
        assert record.timestamp == "2024-01-01T00:00:00+00:00"
        assert record.metadata == {}

    def test_metadata_field(self) -> None:
        """Metadata opcional funciona corretamente."""
        record = AuditRecord(
            audit_id="id-2",
            execution_id="exec-2",
            user_id="user-2",
            feature_version=1,
            model_version="v1.0",
            explainability_computed=False,
            explanation_generated=False,
            timestamp="2024-06-15T12:00:00+00:00",
            metadata={"bedrock_model_id": "anthropic.claude-3-haiku"},
        )
        assert record.metadata == {
            "bedrock_model_id": "anthropic.claude-3-haiku"
        }
