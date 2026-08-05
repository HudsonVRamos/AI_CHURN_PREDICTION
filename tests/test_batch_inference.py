"""Testes unitários para o módulo src.ml.batch_inference.

Valida:
- Preparação de input CSV para Batch Transform
- Parsing de output do Batch Transform em PredictionResult
- Cálculo de confidence (|prob - 0.5| * 2)
- Determinação de risk tier (Low/Medium/High)
- Armazenamento no DynamoDB
- Fluxo completo do process()
- Inferência determinística (seed=42)

Requirements: 10.4, 10.5, 10.6, 10.9
"""

from __future__ import annotations

import io
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.common.models import FeatureVector, PredictionResult
from src.ml.batch_inference import (
    BatchInferenceProcessor,
    DEFAULT_RISK_THRESHOLDS,
    FEATURE_COLUMNS,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def sample_feature_vector() -> FeatureVector:
    """FeatureVector de exemplo para testes."""
    return FeatureVector(
        user_id="user-001",
        version=1,
        generated_at="2024-01-15T10:00:00+00:00",
        observation_start="2023-07-15T00:00:00+00:00",
        observation_end="2024-01-15T00:00:00+00:00",
        total_sessions=120,
        total_viewing_hours=85.5,
        avg_session_duration_min=42.5,
        sessions_per_week=4.8,
        distinct_channels=12,
        avg_happiness_score=7.2,
        avg_buffer_ratio=0.02,
        error_rate=0.05,
        avg_bitrate=5000000.0,
        pct_episode=40.0,
        pct_sport=25.0,
        pct_live=20.0,
        pct_show=15.0,
        distinct_devices=3,
        avg_pause_count=2.1,
        avg_seek_count=1.5,
        viewing_time_trend=0.3,
        error_rate_trend=-0.01,
        session_frequency_trend=0.2,
    )


@pytest.fixture
def sample_feature_vectors() -> list[FeatureVector]:
    """Lista de FeatureVectors para testes de batch."""
    base_kwargs = {
        "version": 1,
        "generated_at": "2024-01-15T10:00:00+00:00",
        "observation_start": "2023-07-15T00:00:00+00:00",
        "observation_end": "2024-01-15T00:00:00+00:00",
        "total_sessions": 100,
        "total_viewing_hours": 50.0,
        "avg_session_duration_min": 30.0,
        "sessions_per_week": 4.0,
        "distinct_channels": 8,
        "avg_happiness_score": 6.5,
        "avg_buffer_ratio": 0.03,
        "error_rate": 0.08,
        "avg_bitrate": 4000000.0,
        "pct_episode": 50.0,
        "pct_sport": 20.0,
        "pct_live": 15.0,
        "pct_show": 15.0,
        "distinct_devices": 2,
        "avg_pause_count": 1.5,
        "avg_seek_count": 1.0,
        "viewing_time_trend": 0.1,
        "error_rate_trend": 0.0,
        "session_frequency_trend": -0.1,
    }
    return [
        FeatureVector(user_id=f"user-{i:03d}", **base_kwargs)
        for i in range(1, 6)
    ]


@pytest.fixture
def mock_sagemaker_pipeline():
    """Mock do SageMakerMLPipeline."""
    pipeline = MagicMock()
    pipeline.predict_batch.return_value = (
        "s3://test-bucket/predictions/batch-123"
    )
    pipeline.get_active_model.return_value = MagicMock(
        model_package_arn="arn:aws:sagemaker:us-east-1:123:pkg/v1"
    )
    return pipeline


@pytest.fixture
def mock_s3_client():
    """Mock do cliente S3."""
    client = MagicMock()
    # Default: retorna 5 probabilidades
    probs = "0.75\n0.20\n0.45\n0.90\n0.10\n"
    mock_body = MagicMock()
    mock_body.read.return_value = probs.encode("utf-8")
    client.get_object.return_value = {"Body": mock_body}
    return client


@pytest.fixture
def mock_dynamodb_resource():
    """Mock do resource DynamoDB."""
    resource = MagicMock()
    table = MagicMock()
    resource.Table.return_value = table
    # batch_writer como context manager
    batch_writer = MagicMock()
    table.batch_writer.return_value.__enter__ = MagicMock(
        return_value=batch_writer
    )
    table.batch_writer.return_value.__exit__ = MagicMock(
        return_value=False
    )
    return resource


@pytest.fixture
def processor(mock_sagemaker_pipeline, mock_s3_client, mock_dynamodb_resource):
    """Instância do BatchInferenceProcessor com mocks."""
    return BatchInferenceProcessor(
        sagemaker_pipeline=mock_sagemaker_pipeline,
        bucket="test-bucket",
        dynamodb_table_name="churn_predictions",
        s3_client=mock_s3_client,
        dynamodb_resource=mock_dynamodb_resource,
    )


# ---------------------------------------------------------------
# Testes de _compute_confidence
# ---------------------------------------------------------------


class TestComputeConfidence:
    """Testes do cálculo de confiança: |prob - 0.5| * 2."""

    def test_probabilidade_zero(self):
        """prob=0.0 → confidence=1.0 (certeza total de não-churn)."""
        assert BatchInferenceProcessor._compute_confidence(0.0) == 1.0

    def test_probabilidade_um(self):
        """prob=1.0 → confidence=1.0 (certeza total de churn)."""
        assert BatchInferenceProcessor._compute_confidence(1.0) == 1.0

    def test_probabilidade_meio(self):
        """prob=0.5 → confidence=0.0 (máxima incerteza)."""
        assert BatchInferenceProcessor._compute_confidence(0.5) == 0.0

    def test_probabilidade_alta(self):
        """prob=0.8 → confidence=0.6."""
        result = BatchInferenceProcessor._compute_confidence(0.8)
        assert abs(result - 0.6) < 1e-10

    def test_probabilidade_baixa(self):
        """prob=0.2 → confidence=0.6."""
        result = BatchInferenceProcessor._compute_confidence(0.2)
        assert abs(result - 0.6) < 1e-10

    def test_simetria(self):
        """Confidence é simétrica: f(0.3) == f(0.7)."""
        c1 = BatchInferenceProcessor._compute_confidence(0.3)
        c2 = BatchInferenceProcessor._compute_confidence(0.7)
        assert abs(c1 - c2) < 1e-10

    def test_resultado_entre_zero_e_um(self):
        """Confidence deve estar entre 0.0 e 1.0 para qualquer input válido."""
        for prob in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            c = BatchInferenceProcessor._compute_confidence(prob)
            assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------
# Testes de _determine_risk_tier
# ---------------------------------------------------------------


class TestDetermineRiskTier:
    """Testes da classificação de risco."""

    def test_low_risk_zero(self, processor):
        """prob=0.0 → Low."""
        assert processor._determine_risk_tier(0.0) == "Low"

    def test_low_risk_limite(self, processor):
        """prob=0.30 → Low (limite superior)."""
        assert processor._determine_risk_tier(0.30) == "Low"

    def test_medium_risk_inicio(self, processor):
        """prob=0.31 → Medium."""
        assert processor._determine_risk_tier(0.31) == "Medium"

    def test_medium_risk_limite(self, processor):
        """prob=0.60 → Medium (limite superior)."""
        assert processor._determine_risk_tier(0.60) == "Medium"

    def test_high_risk_inicio(self, processor):
        """prob=0.61 → High."""
        assert processor._determine_risk_tier(0.61) == "High"

    def test_high_risk_maximo(self, processor):
        """prob=1.0 → High."""
        assert processor._determine_risk_tier(1.0) == "High"

    def test_thresholds_customizados(
        self, mock_sagemaker_pipeline, mock_s3_client, mock_dynamodb_resource
    ):
        """Thresholds customizados devem ser respeitados."""
        custom = {"low_max": 0.20, "medium_max": 0.50}
        proc = BatchInferenceProcessor(
            sagemaker_pipeline=mock_sagemaker_pipeline,
            bucket="test-bucket",
            risk_thresholds=custom,
            s3_client=mock_s3_client,
            dynamodb_resource=mock_dynamodb_resource,
        )
        assert proc._determine_risk_tier(0.20) == "Low"
        assert proc._determine_risk_tier(0.21) == "Medium"
        assert proc._determine_risk_tier(0.50) == "Medium"
        assert proc._determine_risk_tier(0.51) == "High"


# ---------------------------------------------------------------
# Testes de _prepare_input
# ---------------------------------------------------------------


class TestPrepareInput:
    """Testes da preparação do CSV de input."""

    def test_upload_csv_para_s3(
        self, processor, mock_s3_client, sample_feature_vector
    ):
        """Deve fazer upload do CSV para S3 no path correto."""
        result = processor._prepare_input(
            [sample_feature_vector], "exec-001"
        )

        assert result == "s3://test-bucket/inference/exec-001/input/features.csv"
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "inference/exec-001/input/features.csv"
        assert call_kwargs["ContentType"] == "text/csv"

    def test_csv_sem_header(
        self, processor, mock_s3_client, sample_feature_vector
    ):
        """CSV deve ser sem header (apenas valores numéricos)."""
        processor._prepare_input([sample_feature_vector], "exec-001")

        call_kwargs = mock_s3_client.put_object.call_args[1]
        body = call_kwargs["Body"].decode("utf-8")
        lines = body.strip().split("\n")
        # Deve ter exatamente 1 linha (1 feature vector)
        assert len(lines) == 1
        # Não deve ter header (primeiro valor deve ser numérico)
        first_val = lines[0].split(",")[0]
        float(first_val)  # Não deve levantar exceção

    def test_csv_colunas_na_ordem_correta(
        self, processor, mock_s3_client, sample_feature_vector
    ):
        """CSV deve ter colunas na ordem definida em FEATURE_COLUMNS."""
        processor._prepare_input([sample_feature_vector], "exec-001")

        call_kwargs = mock_s3_client.put_object.call_args[1]
        body = call_kwargs["Body"].decode("utf-8")
        values = body.strip().split(",")

        # Verificar que o número de colunas é correto
        assert len(values) == len(FEATURE_COLUMNS)

        # Verificar valor da primeira coluna (total_sessions=120)
        assert float(values[0]) == 120.0

    def test_trends_none_substituidos_por_zero(
        self, processor, mock_s3_client
    ):
        """Trends com valor None devem ser substituídos por 0.0 no CSV."""
        fv = FeatureVector(
            user_id="user-null-trends",
            version=1,
            generated_at="2024-01-15T10:00:00+00:00",
            observation_start="2023-12-15T00:00:00+00:00",
            observation_end="2024-01-15T00:00:00+00:00",
            total_sessions=20,
            total_viewing_hours=10.0,
            avg_session_duration_min=30.0,
            sessions_per_week=5.0,
            distinct_channels=3,
            avg_happiness_score=7.0,
            avg_buffer_ratio=0.01,
            error_rate=0.02,
            avg_bitrate=3000000.0,
            pct_episode=100.0,
            pct_sport=0.0,
            pct_live=0.0,
            pct_show=0.0,
            distinct_devices=1,
            avg_pause_count=1.0,
            avg_seek_count=0.5,
            viewing_time_trend=None,
            error_rate_trend=None,
            session_frequency_trend=None,
        )
        processor._prepare_input([fv], "exec-002")

        call_kwargs = mock_s3_client.put_object.call_args[1]
        body = call_kwargs["Body"].decode("utf-8")
        values = body.strip().split(",")

        # Últimas 3 colunas são trends — devem ser 0.0
        assert float(values[-1]) == 0.0
        assert float(values[-2]) == 0.0
        assert float(values[-3]) == 0.0

    def test_multiplos_feature_vectors(
        self, processor, mock_s3_client, sample_feature_vectors
    ):
        """Múltiplos feature vectors devem gerar múltiplas linhas."""
        processor._prepare_input(sample_feature_vectors, "exec-003")

        call_kwargs = mock_s3_client.put_object.call_args[1]
        body = call_kwargs["Body"].decode("utf-8")
        lines = body.strip().split("\n")
        assert len(lines) == 5


# ---------------------------------------------------------------
# Testes de _parse_predictions
# ---------------------------------------------------------------


class TestParsePredictions:
    """Testes do parsing de output do Batch Transform."""

    def test_parse_cria_prediction_results(
        self, processor, mock_s3_client, sample_feature_vectors
    ):
        """Deve criar PredictionResult para cada feature vector."""
        predictions = processor._parse_predictions(
            output_s3="s3://test-bucket/predictions/batch-123",
            feature_vectors=sample_feature_vectors,
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
            execution_id="exec-001",
        )

        assert len(predictions) == 5
        for pred in predictions:
            assert isinstance(pred, PredictionResult)

    def test_parse_mapeia_user_ids_corretamente(
        self, processor, mock_s3_client, sample_feature_vectors
    ):
        """User IDs devem corresponder aos feature vectors."""
        predictions = processor._parse_predictions(
            output_s3="s3://test-bucket/predictions/batch-123",
            feature_vectors=sample_feature_vectors,
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
            execution_id="exec-001",
        )

        assert predictions[0].user_id == "user-001"
        assert predictions[4].user_id == "user-005"

    def test_parse_calcula_confidence_corretamente(
        self, processor, mock_s3_client, sample_feature_vectors
    ):
        """Confidence deve ser |prob - 0.5| * 2."""
        # Probs: 0.75, 0.20, 0.45, 0.90, 0.10
        predictions = processor._parse_predictions(
            output_s3="s3://test-bucket/predictions/batch-123",
            feature_vectors=sample_feature_vectors,
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
            execution_id="exec-001",
        )

        # prob=0.75 → confidence = |0.75-0.5|*2 = 0.5
        assert abs(predictions[0].confidence - 0.5) < 1e-4
        # prob=0.20 → confidence = |0.20-0.5|*2 = 0.6
        assert abs(predictions[1].confidence - 0.6) < 1e-4
        # prob=0.45 → confidence = |0.45-0.5|*2 = 0.1
        assert abs(predictions[2].confidence - 0.1) < 1e-4

    def test_parse_determina_risk_tier_corretamente(
        self, processor, mock_s3_client, sample_feature_vectors
    ):
        """Risk tier deve seguir thresholds."""
        # Probs: 0.75, 0.20, 0.45, 0.90, 0.10
        predictions = processor._parse_predictions(
            output_s3="s3://test-bucket/predictions/batch-123",
            feature_vectors=sample_feature_vectors,
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
            execution_id="exec-001",
        )

        assert predictions[0].risk_tier == "High"    # 0.75
        assert predictions[1].risk_tier == "Low"     # 0.20
        assert predictions[2].risk_tier == "Medium"  # 0.45
        assert predictions[3].risk_tier == "High"    # 0.90
        assert predictions[4].risk_tier == "Low"     # 0.10

    def test_parse_inclui_model_version(
        self, processor, mock_s3_client, sample_feature_vectors
    ):
        """Cada predição deve incluir a versão do modelo."""
        model_arn = "arn:aws:sagemaker:us-east-1:123:pkg/v2"
        predictions = processor._parse_predictions(
            output_s3="s3://test-bucket/predictions/batch-123",
            feature_vectors=sample_feature_vectors,
            model_version=model_arn,
            execution_id="exec-001",
        )

        for pred in predictions:
            assert pred.model_version == model_arn

    def test_parse_inclui_timestamp_iso8601(
        self, processor, mock_s3_client, sample_feature_vectors
    ):
        """Timestamp deve estar em formato ISO 8601."""
        predictions = processor._parse_predictions(
            output_s3="s3://test-bucket/predictions/batch-123",
            feature_vectors=sample_feature_vectors,
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
            execution_id="exec-001",
        )

        for pred in predictions:
            # ISO 8601 deve conter 'T' e timezone info
            assert "T" in pred.timestamp
            assert "+" in pred.timestamp or "Z" in pred.timestamp

    def test_parse_inclui_feature_version(
        self, processor, mock_s3_client, sample_feature_vectors
    ):
        """Cada predição deve registrar a versão do feature vector usado."""
        predictions = processor._parse_predictions(
            output_s3="s3://test-bucket/predictions/batch-123",
            feature_vectors=sample_feature_vectors,
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
            execution_id="exec-001",
        )

        for pred in predictions:
            assert pred.feature_version == 1


# ---------------------------------------------------------------
# Testes de _store_predictions
# ---------------------------------------------------------------


class TestStorePredictions:
    """Testes do armazenamento no DynamoDB."""

    def test_store_chama_batch_writer(
        self, processor, mock_dynamodb_resource
    ):
        """Deve usar batch_writer para inserção eficiente."""
        predictions = [
            PredictionResult(
                user_id="user-001",
                churn_probability=0.75,
                confidence=0.5,
                risk_tier="High",
                model_version="arn:model/v1",
                feature_version=1,
                timestamp="2024-01-15T10:00:00+00:00",
            ),
        ]

        processor._store_predictions(predictions, "exec-001")

        table = mock_dynamodb_resource.Table.return_value
        table.batch_writer.assert_called_once()

    def test_store_item_com_campos_corretos(
        self, processor, mock_dynamodb_resource
    ):
        """Item no DynamoDB deve ter execution_id como PK e user_id como SK."""
        predictions = [
            PredictionResult(
                user_id="user-001",
                churn_probability=0.75,
                confidence=0.5,
                risk_tier="High",
                model_version="arn:model/v1",
                feature_version=1,
                timestamp="2024-01-15T10:00:00+00:00",
            ),
        ]

        processor._store_predictions(predictions, "exec-001")

        table = mock_dynamodb_resource.Table.return_value
        batch_ctx = table.batch_writer.return_value.__enter__.return_value
        batch_ctx.put_item.assert_called_once()

        item = batch_ctx.put_item.call_args[1]["Item"]
        assert item["execution_id"] == "exec-001"
        assert item["user_id"] == "user-001"
        assert item["churn_probability"] == "0.75"
        assert item["confidence"] == "0.5"
        assert item["risk_tier"] == "High"
        assert item["model_version"] == "arn:model/v1"
        assert item["feature_version"] == 1
        assert item["timestamp"] == "2024-01-15T10:00:00+00:00"

    def test_store_multiplas_predictions(
        self, processor, mock_dynamodb_resource
    ):
        """Deve armazenar todas as predições no batch."""
        predictions = [
            PredictionResult(
                user_id=f"user-{i:03d}",
                churn_probability=0.5,
                confidence=0.0,
                risk_tier="Medium",
                model_version="arn:model/v1",
                feature_version=1,
                timestamp="2024-01-15T10:00:00+00:00",
            )
            for i in range(1, 4)
        ]

        processor._store_predictions(predictions, "exec-001")

        table = mock_dynamodb_resource.Table.return_value
        batch_ctx = table.batch_writer.return_value.__enter__.return_value
        assert batch_ctx.put_item.call_count == 3


# ---------------------------------------------------------------
# Testes do fluxo completo (process)
# ---------------------------------------------------------------


class TestProcess:
    """Testes do fluxo completo de batch inference."""

    def test_process_retorna_predictions(
        self, processor, sample_feature_vectors
    ):
        """process() deve retornar lista de PredictionResult."""
        result = processor.process(
            feature_vectors=sample_feature_vectors,
            execution_id="exec-001",
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
        )

        assert len(result) == 5
        for pred in result:
            assert isinstance(pred, PredictionResult)
            assert 0.0 <= pred.churn_probability <= 1.0
            assert 0.0 <= pred.confidence <= 1.0
            assert pred.risk_tier in ("Low", "Medium", "High")

    def test_process_sem_model_version_usa_approved(
        self, processor, mock_sagemaker_pipeline, sample_feature_vectors
    ):
        """Sem model_version explícito, deve usar get_active_model()."""
        processor.process(
            feature_vectors=sample_feature_vectors,
            execution_id="exec-002",
        )

        mock_sagemaker_pipeline.get_active_model.assert_called_once()

    def test_process_com_model_version_nao_chama_get_active(
        self, processor, mock_sagemaker_pipeline, sample_feature_vectors
    ):
        """Com model_version explícito, NÃO deve chamar get_active_model."""
        processor.process(
            feature_vectors=sample_feature_vectors,
            execution_id="exec-003",
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
        )

        mock_sagemaker_pipeline.get_active_model.assert_not_called()

    def test_process_chama_predict_batch(
        self, processor, mock_sagemaker_pipeline, sample_feature_vectors
    ):
        """Deve invocar predict_batch no pipeline SageMaker."""
        processor.process(
            feature_vectors=sample_feature_vectors,
            execution_id="exec-004",
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
        )

        mock_sagemaker_pipeline.predict_batch.assert_called_once()
        call_kwargs = mock_sagemaker_pipeline.predict_batch.call_args[1]
        assert "feature_vectors_s3" in call_kwargs
        assert call_kwargs["model_version"] == (
            "arn:aws:sagemaker:us-east-1:123:pkg/v1"
        )

    def test_process_armazena_no_dynamodb(
        self, processor, mock_dynamodb_resource, sample_feature_vectors
    ):
        """Deve armazenar resultados no DynamoDB."""
        processor.process(
            feature_vectors=sample_feature_vectors,
            execution_id="exec-005",
            model_version="arn:aws:sagemaker:us-east-1:123:pkg/v1",
        )

        table = mock_dynamodb_resource.Table.return_value
        table.batch_writer.assert_called_once()

    def test_process_lista_vazia_levanta_erro(self, processor):
        """feature_vectors vazia deve levantar ValueError."""
        with pytest.raises(ValueError, match="não pode estar vazia"):
            processor.process(
                feature_vectors=[],
                execution_id="exec-006",
            )


# ---------------------------------------------------------------
# Testes de inferência determinística (seed=42)
# ---------------------------------------------------------------


class TestDeterministicInference:
    """Verifica que a inferência é determinística (R10.5).

    O seed=42 nos hyperparameters do modelo garante que os mesmos inputs
    produzem os mesmos outputs. Aqui verificamos que o processador
    não introduz aleatoriedade no fluxo.
    """

    def test_seed_fixo_definido(self):
        """Classe deve ter SEED=42 como constante."""
        assert BatchInferenceProcessor.SEED == 42

    def test_mesmos_inputs_mesmos_outputs(
        self, processor, mock_s3_client, sample_feature_vectors
    ):
        """Dois processamentos com mesmos inputs devem gerar mesma saída."""
        # Executar duas vezes com o mesmo input
        result1 = processor._parse_predictions(
            output_s3="s3://test-bucket/predictions/batch-123",
            feature_vectors=sample_feature_vectors,
            model_version="arn:model/v1",
            execution_id="exec-001",
        )

        # Resetar mock para simular segunda chamada idêntica
        probs = "0.75\n0.20\n0.45\n0.90\n0.10\n"
        mock_body = MagicMock()
        mock_body.read.return_value = probs.encode("utf-8")
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        result2 = processor._parse_predictions(
            output_s3="s3://test-bucket/predictions/batch-123",
            feature_vectors=sample_feature_vectors,
            model_version="arn:model/v1",
            execution_id="exec-001",
        )

        # Comparar probabilidades e confidences (timestamps podem diferir)
        for p1, p2 in zip(result1, result2):
            assert p1.churn_probability == p2.churn_probability
            assert p1.confidence == p2.confidence
            assert p1.risk_tier == p2.risk_tier
            assert p1.user_id == p2.user_id
            assert p1.model_version == p2.model_version

    def test_nao_gera_explicacoes_texto(
        self, processor, sample_feature_vectors
    ):
        """ML Pipeline NÃO deve gerar explicações em linguagem natural (R10.6).

        PredictionResult contém apenas campos numéricos e metadados.
        """
        result = processor.process(
            feature_vectors=sample_feature_vectors,
            execution_id="exec-007",
            model_version="arn:model/v1",
        )

        for pred in result:
            # PredictionResult não possui campo de explicação textual
            assert not hasattr(pred, "explanation")
            assert not hasattr(pred, "natural_language_explanation")


# ---------------------------------------------------------------
# Testes de _parse_s3_path
# ---------------------------------------------------------------


class TestParseS3Path:
    """Testes do parsing de S3 paths."""

    def test_path_valido(self):
        """Path S3 válido deve retornar bucket e key."""
        bucket, key = BatchInferenceProcessor._parse_s3_path(
            "s3://my-bucket/path/to/file.csv"
        )
        assert bucket == "my-bucket"
        assert key == "path/to/file.csv"

    def test_path_sem_prefixo(self):
        """Path sem s3:// deve levantar ValueError."""
        with pytest.raises(ValueError, match="deve iniciar com s3://"):
            BatchInferenceProcessor._parse_s3_path("/local/path")

    def test_path_sem_key(self):
        """Path sem key deve levantar ValueError."""
        with pytest.raises(ValueError, match="sem key"):
            BatchInferenceProcessor._parse_s3_path("s3://bucket-only")
