"""Testes para o ReportGenerator.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 13.1, 13.2, 13.3, 13.4, 13.5
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from src.common.models import (
    ExplainabilityResult,
    FeatureContribution,
    PredictionResult,
)
from src.reports.report_generator import ReportGenerator


# --- Fixtures ---


@pytest.fixture
def report_generator() -> ReportGenerator:
    """ReportGenerator sem S3 para testes locais."""
    return ReportGenerator(
        s3_client=None,
        bucket="test-bucket",
        model_version="v1.2.0",
        feature_version="42",
        analysis_period_start="2024-01-01T00:00:00Z",
        analysis_period_end="2024-06-30T23:59:59Z",
    )


@pytest.fixture
def sample_prediction_high() -> PredictionResult:
    """Predição de alto risco."""
    return PredictionResult(
        user_id="user-001",
        churn_probability=0.85,
        confidence=0.92,
        risk_tier="High",
        model_version="v1.2.0",
        feature_version=42,
        timestamp="2024-07-01T10:00:00Z",
    )


@pytest.fixture
def sample_prediction_medium() -> PredictionResult:
    """Predição de risco médio."""
    return PredictionResult(
        user_id="user-002",
        churn_probability=0.45,
        confidence=0.80,
        risk_tier="Medium",
        model_version="v1.2.0",
        feature_version=42,
        timestamp="2024-07-01T10:00:00Z",
    )


@pytest.fixture
def sample_prediction_low() -> PredictionResult:
    """Predição de baixo risco."""
    return PredictionResult(
        user_id="user-003",
        churn_probability=0.15,
        confidence=0.88,
        risk_tier="Low",
        model_version="v1.2.0",
        feature_version=42,
        timestamp="2024-07-01T10:00:00Z",
    )


@pytest.fixture
def sample_explainability() -> ExplainabilityResult:
    """Resultado de explicabilidade para user-001."""
    return ExplainabilityResult(
        user_id="user-001",
        top_features=[
            FeatureContribution(
                feature_name="sessions_per_week",
                contribution_weight=-0.35,
                normalized_impact=-0.8,
            ),
            FeatureContribution(
                feature_name="error_rate",
                contribution_weight=0.25,
                normalized_impact=0.6,
            ),
            FeatureContribution(
                feature_name="viewing_time_trend",
                contribution_weight=-0.20,
                normalized_impact=-0.5,
            ),
        ],
        base_value=0.3,
        prediction_value=0.85,
    )


@pytest.fixture
def sample_explainability_medium() -> ExplainabilityResult:
    """Resultado de explicabilidade para user-002."""
    return ExplainabilityResult(
        user_id="user-002",
        top_features=[
            FeatureContribution(
                feature_name="avg_happiness_score",
                contribution_weight=-0.15,
                normalized_impact=-0.4,
            ),
            FeatureContribution(
                feature_name="sessions_per_week",
                contribution_weight=-0.10,
                normalized_impact=-0.3,
            ),
        ],
        base_value=0.3,
        prediction_value=0.45,
    )


@pytest.fixture
def sample_explainability_low() -> ExplainabilityResult:
    """Resultado de explicabilidade para user-003."""
    return ExplainabilityResult(
        user_id="user-003",
        top_features=[
            FeatureContribution(
                feature_name="total_sessions",
                contribution_weight=-0.05,
                normalized_impact=-0.1,
            ),
        ],
        base_value=0.3,
        prediction_value=0.15,
    )


# --- Testes de Relatório Individual (R13.1, R13.4, R13.5) ---


class TestIndividualReport:
    """Testes para generate_individual_report."""

    def test_individual_report_fields(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_explainability: ExplainabilityResult,
    ) -> None:
        """R13.1: Relatório contém todos os campos obrigatórios."""
        report = report_generator.generate_individual_report(
            user_id="user-001",
            prediction=sample_prediction_high,
            explainability=sample_explainability,
            explanation="O assinante apresenta queda de engajamento.",
        )

        assert report["report_type"] == "individual"
        assert report["user_id"] == "user-001"
        assert report["churn_probability"] == 0.85
        assert report["confidence"] == 0.92
        assert report["risk_tier"] == "High"
        assert len(report["top_features"]) == 3
        assert report["explanation"] == (
            "O assinante apresenta queda de engajamento."
        )
        assert report["explanation_status"] == "available"

    def test_individual_report_metadata(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_explainability: ExplainabilityResult,
    ) -> None:
        """R13.4: Metadados incluem model_version, feature_version, período, timestamp."""
        report = report_generator.generate_individual_report(
            user_id="user-001",
            prediction=sample_prediction_high,
            explainability=sample_explainability,
            explanation="Teste.",
        )

        metadata = report["metadata"]
        assert metadata["model_version"] == "v1.2.0"
        assert metadata["feature_version"] == "42"
        assert metadata["analysis_period_start"] == "2024-01-01T00:00:00Z"
        assert metadata["analysis_period_end"] == "2024-06-30T23:59:59Z"
        # Timestamp ISO 8601 com timezone
        assert "T" in metadata["timestamp"]
        assert "+" in metadata["timestamp"] or "Z" in metadata["timestamp"]

    def test_individual_report_explanation_unavailable(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_explainability: ExplainabilityResult,
    ) -> None:
        """R13.5: Se explicação indisponível, status='unavailable'."""
        report = report_generator.generate_individual_report(
            user_id="user-001",
            prediction=sample_prediction_high,
            explainability=sample_explainability,
            explanation=None,
        )

        assert report["explanation"] is None
        assert report["explanation_status"] == "unavailable"

    def test_individual_report_no_explainability(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
    ) -> None:
        """R13.5: Se explainability indisponível, top_features vazio."""
        report = report_generator.generate_individual_report(
            user_id="user-001",
            prediction=sample_prediction_high,
            explainability=None,
            explanation=None,
        )

        assert report["top_features"] == []
        assert report["explanation_status"] == "unavailable"

    def test_individual_report_top_features_structure(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_explainability: ExplainabilityResult,
    ) -> None:
        """R13.1: Top features contém nome, peso e impacto normalizado."""
        report = report_generator.generate_individual_report(
            user_id="user-001",
            prediction=sample_prediction_high,
            explainability=sample_explainability,
            explanation="Teste",
        )

        feat = report["top_features"][0]
        assert "feature_name" in feat
        assert "contribution_weight" in feat
        assert "normalized_impact" in feat
        assert feat["feature_name"] == "sessions_per_week"
        assert feat["contribution_weight"] == -0.35
        assert feat["normalized_impact"] == -0.8


# --- Testes de Relatório Executivo (R6.1, R6.2, R6.5, R6.6, R13.2) ---


class TestExecutiveReport:
    """Testes para generate_executive_report."""

    def test_executive_report_summary(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_prediction_medium: PredictionResult,
        sample_prediction_low: PredictionResult,
        sample_explainability: ExplainabilityResult,
        sample_explainability_medium: ExplainabilityResult,
        sample_explainability_low: ExplainabilityResult,
    ) -> None:
        """R6.1, R13.2: Resumo com total, distribuição e média."""
        predictions = [
            sample_prediction_high,
            sample_prediction_medium,
            sample_prediction_low,
        ]
        explainabilities = [
            sample_explainability,
            sample_explainability_medium,
            sample_explainability_low,
        ]

        report = report_generator.generate_executive_report(
            predictions=predictions,
            explainabilities=explainabilities,
        )

        assert report["report_type"] == "executive"
        assert report["total_analyzed"] == 3
        assert report["distribution"]["High"]["count"] == 1
        assert report["distribution"]["Medium"]["count"] == 1
        assert report["distribution"]["Low"]["count"] == 1

        # Percentuais
        assert report["distribution"]["High"]["percentage"] == pytest.approx(
            33.33, abs=0.01
        )

        # Média
        expected_avg = (0.85 + 0.45 + 0.15) / 3
        assert report["average_churn_probability"] == pytest.approx(
            expected_avg, abs=0.001
        )

    def test_executive_report_top_factors(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_prediction_medium: PredictionResult,
        sample_explainability: ExplainabilityResult,
        sample_explainability_medium: ExplainabilityResult,
    ) -> None:
        """R6.1, R13.2: Top fatores da população são agregados."""
        predictions = [sample_prediction_high, sample_prediction_medium]
        explainabilities = [
            sample_explainability,
            sample_explainability_medium,
        ]

        report = report_generator.generate_executive_report(
            predictions=predictions,
            explainabilities=explainabilities,
        )

        factors = report["top_population_factors"]
        assert len(factors) > 0
        # sessions_per_week aparece em ambos os explainabilities
        factor_names = [f["feature_name"] for f in factors]
        assert "sessions_per_week" in factor_names

        # O mais frequente deve estar primeiro
        assert factors[0]["occurrence_count"] >= factors[-1]["occurrence_count"]

    def test_executive_report_high_risk_list(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_prediction_low: PredictionResult,
        sample_explainability: ExplainabilityResult,
        sample_explainability_low: ExplainabilityResult,
    ) -> None:
        """R6.2: Lista de alto risco ordenada por probabilidade desc."""
        # Criar segundo high risk
        second_high = PredictionResult(
            user_id="user-004",
            churn_probability=0.92,
            confidence=0.95,
            risk_tier="High",
            model_version="v1.2.0",
            feature_version=42,
            timestamp="2024-07-01T10:00:00Z",
        )
        second_expl = ExplainabilityResult(
            user_id="user-004",
            top_features=[
                FeatureContribution(
                    feature_name="error_rate",
                    contribution_weight=0.40,
                    normalized_impact=0.9,
                ),
            ],
            base_value=0.3,
            prediction_value=0.92,
        )

        predictions = [
            sample_prediction_high,
            second_high,
            sample_prediction_low,
        ]
        explainabilities = [
            sample_explainability,
            second_expl,
            sample_explainability_low,
        ]

        report = report_generator.generate_executive_report(
            predictions=predictions,
            explainabilities=explainabilities,
        )

        high_risk = report["high_risk_users"]
        assert len(high_risk) == 2
        # Ordenado por churn_probability desc
        assert high_risk[0]["user_id"] == "user-004"
        assert high_risk[0]["churn_probability"] == 0.92
        assert high_risk[1]["user_id"] == "user-001"
        assert high_risk[1]["churn_probability"] == 0.85

    def test_executive_report_no_high_risk(
        self,
        report_generator: ReportGenerator,
        sample_prediction_low: PredictionResult,
        sample_explainability_low: ExplainabilityResult,
    ) -> None:
        """R6.6: Se nenhum alto risco, lista vazia e contagem zero."""
        report = report_generator.generate_executive_report(
            predictions=[sample_prediction_low],
            explainabilities=[sample_explainability_low],
        )

        assert report["distribution"]["High"]["count"] == 0
        assert report["high_risk_users"] == []

    def test_executive_report_metadata(
        self,
        report_generator: ReportGenerator,
        sample_prediction_low: PredictionResult,
        sample_explainability_low: ExplainabilityResult,
    ) -> None:
        """R6.5, R13.4: Metadados presentes no relatório executivo."""
        report = report_generator.generate_executive_report(
            predictions=[sample_prediction_low],
            explainabilities=[sample_explainability_low],
        )

        metadata = report["metadata"]
        assert metadata["model_version"] == "v1.2.0"
        assert metadata["feature_version"] == "42"
        assert "T" in metadata["timestamp"]

    def test_executive_report_empty_predictions(
        self,
        report_generator: ReportGenerator,
    ) -> None:
        """Edge case: lista vazia de predições."""
        report = report_generator.generate_executive_report(
            predictions=[],
            explainabilities=[],
        )

        assert report["total_analyzed"] == 0
        assert report["average_churn_probability"] == 0.0
        assert report["high_risk_users"] == []


# --- Testes de Exportação JSON (R6.3, R13.3) ---


class TestExportJson:
    """Testes para export_json."""

    def test_export_json_creates_valid_file(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_explainability: ExplainabilityResult,
    ) -> None:
        """R6.3: Exporta JSON válido para consumo programático."""
        report = report_generator.generate_individual_report(
            user_id="user-001",
            prediction=sample_prediction_high,
            explainability=sample_explainability,
            explanation="Teste",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            output_path = f.name

        try:
            report_generator.export_json(report, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded["user_id"] == "user-001"
            assert loaded["churn_probability"] == 0.85
        finally:
            os.unlink(output_path)

    def test_export_json_utf8_encoding(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_explainability: ExplainabilityResult,
    ) -> None:
        """R13.3: JSON preserva caracteres UTF-8 (português)."""
        report = report_generator.generate_individual_report(
            user_id="user-001",
            prediction=sample_prediction_high,
            explainability=sample_explainability,
            explanation="Assinante com risco elevado de cancelação.",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            output_path = f.name

        try:
            report_generator.export_json(report, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            # ensure_ascii=False preserva caracteres especiais
            assert "cancelação" in content
        finally:
            os.unlink(output_path)


# --- Testes de Exportação Markdown (R6.4, R13.3) ---


class TestExportMarkdown:
    """Testes para export_markdown."""

    def test_export_individual_markdown(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_explainability: ExplainabilityResult,
    ) -> None:
        """R6.4: Markdown individual legível com formatação correta."""
        report = report_generator.generate_individual_report(
            user_id="user-001",
            prediction=sample_prediction_high,
            explainability=sample_explainability,
            explanation="Queda significativa no engajamento.",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            output_path = f.name

        try:
            report_generator.export_markdown(report, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "# Relatório Individual de Churn" in content
            assert "user-001" in content
            assert "sessions_per_week" in content
            assert "Queda significativa no engajamento." in content
            assert "v1.2.0" in content
        finally:
            os.unlink(output_path)

    def test_export_executive_markdown(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_prediction_low: PredictionResult,
        sample_explainability: ExplainabilityResult,
        sample_explainability_low: ExplainabilityResult,
    ) -> None:
        """R6.4: Markdown executivo com tabelas e seções."""
        predictions = [sample_prediction_high, sample_prediction_low]
        explainabilities = [
            sample_explainability,
            sample_explainability_low,
        ]

        report = report_generator.generate_executive_report(
            predictions=predictions,
            explainabilities=explainabilities,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            output_path = f.name

        try:
            report_generator.export_markdown(report, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "# Relatório Executivo de Churn" in content
            assert "Distribuição por Nível de Risco" in content
            assert "Alto Risco" in content
            assert "user-001" in content
            assert "v1.2.0" in content
        finally:
            os.unlink(output_path)

    def test_export_markdown_explanation_unavailable(
        self,
        report_generator: ReportGenerator,
        sample_prediction_high: PredictionResult,
        sample_explainability: ExplainabilityResult,
    ) -> None:
        """R13.5: Markdown mostra mensagem quando explicação indisponível."""
        report = report_generator.generate_individual_report(
            user_id="user-001",
            prediction=sample_prediction_high,
            explainability=sample_explainability,
            explanation=None,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            output_path = f.name

        try:
            report_generator.export_markdown(report, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "indisponível" in content
        finally:
            os.unlink(output_path)


# --- Testes de Upload S3 ---


class TestS3Upload:
    """Testes para upload_to_s3."""

    def test_upload_without_s3_client(
        self,
        report_generator: ReportGenerator,
    ) -> None:
        """Upload retorna None quando S3 não configurado."""
        result = report_generator.upload_to_s3(
            local_path="fake.json",
            execution_id="exec-001",
            filename="report.json",
        )
        assert result is None

    def test_upload_with_s3_client(self) -> None:
        """Upload envia arquivo para S3 na estrutura correta."""
        mock_s3 = MagicMock()
        gen = ReportGenerator(
            s3_client=mock_s3,
            bucket="my-bucket",
            model_version="v1.0",
            feature_version="1",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"test": true}')
            local_path = f.name

        try:
            result = gen.upload_to_s3(
                local_path=local_path,
                execution_id="exec-123",
                filename="executive_report.json",
            )

            assert result == (
                "s3://my-bucket/reports/exec-123/executive_report.json"
            )
            mock_s3.put_object.assert_called_once()
            call_kwargs = mock_s3.put_object.call_args[1]
            assert call_kwargs["Bucket"] == "my-bucket"
            assert call_kwargs["Key"] == (
                "reports/exec-123/executive_report.json"
            )
        finally:
            os.unlink(local_path)

    def test_upload_s3_error_raises(self) -> None:
        """Upload propaga exceção quando S3 falha."""
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = Exception("S3 error")
        gen = ReportGenerator(
            s3_client=mock_s3,
            bucket="my-bucket",
            model_version="v1.0",
            feature_version="1",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"test": true}')
            local_path = f.name

        try:
            with pytest.raises(Exception, match="S3 error"):
                gen.upload_to_s3(
                    local_path=local_path,
                    execution_id="exec-123",
                    filename="report.json",
                )
        finally:
            os.unlink(local_path)
