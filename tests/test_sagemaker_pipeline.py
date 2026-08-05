"""Testes unitários para o módulo src.ml.sagemaker_pipeline.

Valida:
- Inicialização e validação de algoritmos
- Split estratificado com proporções corretas
- Fluxo de train com mocks dos serviços AWS
- Fluxo de predict_batch
- get_active_model
- Parsing de S3 paths
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.sagemaker_pipeline import (
    DEFAULT_HYPERPARAMETERS,
    SageMakerMLPipeline,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def mock_sagemaker_client():
    """Mock do cliente boto3 SageMaker."""
    return MagicMock()


@pytest.fixture
def mock_s3_client():
    """Mock do cliente boto3 S3."""
    return MagicMock()


@pytest.fixture
def pipeline(mock_sagemaker_client, mock_s3_client):
    """Instância do pipeline com clientes mockados."""
    return SageMakerMLPipeline(
        region="us-east-1",
        model_package_group="churn-prediction-models",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        bucket="sky-brazil-churn-prediction",
        algorithm="xgboost",
        sagemaker_client=mock_sagemaker_client,
        s3_client=mock_s3_client,
    )


@pytest.fixture
def sample_training_df():
    """DataFrame de treinamento com features e labels balanceados."""
    np.random.seed(42)
    n_samples = 200
    n_churned = 60  # ~30% churn
    n_active = n_samples - n_churned

    data = {
        "label": [1] * n_churned + [0] * n_active,
        "total_sessions": np.random.randint(5, 500, n_samples).tolist(),
        "total_viewing_hours": np.random.uniform(1, 100, n_samples).tolist(),
        "avg_session_duration_min": np.random.uniform(5, 120, n_samples).tolist(),
        "sessions_per_week": np.random.uniform(0.5, 20, n_samples).tolist(),
        "distinct_channels": np.random.randint(1, 50, n_samples).tolist(),
        "avg_happiness_score": np.random.uniform(3, 9, n_samples).tolist(),
        "avg_buffer_ratio": np.random.uniform(0, 0.1, n_samples).tolist(),
        "error_rate": np.random.uniform(0, 0.3, n_samples).tolist(),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------
# Testes de inicialização
# ---------------------------------------------------------------


class TestInit:
    """Testes de inicialização do SageMakerMLPipeline."""

    def test_init_algoritmo_valido(self, mock_sagemaker_client, mock_s3_client):
        """Inicialização com algoritmo válido deve funcionar."""
        pipeline = SageMakerMLPipeline(
            region="us-east-1",
            model_package_group="test-group",
            role_arn="arn:aws:iam::123:role/test",
            bucket="test-bucket",
            algorithm="xgboost",
            sagemaker_client=mock_sagemaker_client,
            s3_client=mock_s3_client,
        )
        assert pipeline.algorithm == "xgboost"
        assert pipeline.region == "us-east-1"

    def test_init_todos_algoritmos_suportados(
        self, mock_sagemaker_client, mock_s3_client
    ):
        """Deve aceitar todos os algoritmos suportados."""
        for algo in SageMakerMLPipeline.SUPPORTED_ALGORITHMS:
            p = SageMakerMLPipeline(
                region="us-east-1",
                model_package_group="test-group",
                role_arn="arn:aws:iam::123:role/test",
                bucket="test-bucket",
                algorithm=algo,
                sagemaker_client=mock_sagemaker_client,
                s3_client=mock_s3_client,
            )
            assert p.algorithm == algo

    def test_init_algoritmo_invalido(
        self, mock_sagemaker_client, mock_s3_client
    ):
        """Algoritmo inválido deve levantar ValueError."""
        with pytest.raises(ValueError, match="não suportado"):
            SageMakerMLPipeline(
                region="us-east-1",
                model_package_group="test-group",
                role_arn="arn:aws:iam::123:role/test",
                bucket="test-bucket",
                algorithm="random_forest",
                sagemaker_client=mock_sagemaker_client,
                s3_client=mock_s3_client,
            )

    def test_init_case_insensitive(
        self, mock_sagemaker_client, mock_s3_client
    ):
        """Algoritmo deve ser case-insensitive."""
        p = SageMakerMLPipeline(
            region="us-east-1",
            model_package_group="test-group",
            role_arn="arn:aws:iam::123:role/test",
            bucket="test-bucket",
            algorithm="XGBoost",
            sagemaker_client=mock_sagemaker_client,
            s3_client=mock_s3_client,
        )
        assert p.algorithm == "xgboost"


# ---------------------------------------------------------------
# Testes do split estratificado
# ---------------------------------------------------------------


class TestStratifiedSplit:
    """Testes do split estratificado."""

    def test_split_proporcoes_corretas(self, pipeline, sample_training_df):
        """Split deve gerar proporções 70/15/15 (com tolerância)."""
        train_df, val_df, test_df = pipeline._stratified_split(
            sample_training_df
        )
        total = len(sample_training_df)

        # Verificar proporções (tolerância de ±2%)
        assert abs(len(train_df) / total - 0.70) < 0.02
        assert abs(len(val_df) / total - 0.15) < 0.02
        assert abs(len(test_df) / total - 0.15) < 0.02

    def test_split_todos_registros_preservados(
        self, pipeline, sample_training_df
    ):
        """Split não deve perder registros."""
        train_df, val_df, test_df = pipeline._stratified_split(
            sample_training_df
        )
        assert len(train_df) + len(val_df) + len(test_df) == len(
            sample_training_df
        )

    def test_split_estratificado_mantem_proporcao_classes(
        self, pipeline, sample_training_df
    ):
        """Cada split deve manter proporção de classes similar ao original."""
        original_ratio = sample_training_df["label"].mean()
        train_df, val_df, test_df = pipeline._stratified_split(
            sample_training_df
        )

        # Tolerância de ±5% na proporção de classes
        for split_df in [train_df, val_df, test_df]:
            split_ratio = split_df["label"].mean()
            assert abs(split_ratio - original_ratio) < 0.05

    def test_split_sem_sobreposicao(self, pipeline, sample_training_df):
        """Splits não devem ter registros em comum."""
        train_df, val_df, test_df = pipeline._stratified_split(
            sample_training_df
        )

        train_idx = set(train_df.index)
        val_idx = set(val_df.index)
        test_idx = set(test_df.index)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0


# ---------------------------------------------------------------
# Testes de parsing S3
# ---------------------------------------------------------------


class TestParseS3Path:
    """Testes do parsing de paths S3."""

    def test_parse_path_valido(self):
        """Path S3 válido deve retornar bucket e key."""
        bucket, key = SageMakerMLPipeline._parse_s3_path(
            "s3://my-bucket/path/to/file.csv"
        )
        assert bucket == "my-bucket"
        assert key == "path/to/file.csv"

    def test_parse_path_sem_prefixo_s3(self):
        """Path sem prefixo s3:// deve levantar ValueError."""
        with pytest.raises(ValueError, match="deve iniciar com s3://"):
            SageMakerMLPipeline._parse_s3_path("/local/path/file.csv")

    def test_parse_path_sem_key(self):
        """Path S3 sem key deve levantar ValueError."""
        with pytest.raises(ValueError, match="sem key"):
            SageMakerMLPipeline._parse_s3_path("s3://bucket-only")

    def test_parse_path_com_key_complexa(self):
        """Path S3 com múltiplos níveis no key."""
        bucket, key = SageMakerMLPipeline._parse_s3_path(
            "s3://bucket/a/b/c/d/file.tar.gz"
        )
        assert bucket == "bucket"
        assert key == "a/b/c/d/file.tar.gz"


# ---------------------------------------------------------------
# Testes de hyperparameters
# ---------------------------------------------------------------


class TestHyperparameters:
    """Testes de mescla de hyperparameters."""

    def test_defaults_xgboost(self, pipeline):
        """XGBoost deve usar defaults corretos."""
        params = pipeline._get_hyperparameters("xgboost", None)
        assert params["objective"] == "binary:logistic"
        assert params["eval_metric"] == "auc"
        assert params["seed"] == "42"

    def test_defaults_lightgbm(self, pipeline):
        """LightGBM deve usar defaults corretos."""
        params = pipeline._get_hyperparameters("lightgbm", None)
        assert params["objective"] == "binary"
        assert params["metric"] == "auc"

    def test_defaults_catboost(self, pipeline):
        """CatBoost deve usar defaults corretos."""
        params = pipeline._get_hyperparameters("catboost", None)
        assert params["loss_function"] == "Logloss"
        assert params["eval_metric"] == "AUC"

    def test_override_hyperparameters(self, pipeline):
        """Overrides devem sobrescrever defaults."""
        overrides = {"max_depth": 10, "eta": 0.1}
        params = pipeline._get_hyperparameters("xgboost", overrides)
        assert params["max_depth"] == "10"
        assert params["eta"] == "0.1"
        # Outros defaults devem permanecer
        assert params["objective"] == "binary:logistic"

    def test_override_valores_convertidos_para_string(self, pipeline):
        """Todos os valores devem ser convertidos para string."""
        overrides = {"num_round": 200, "eta": 0.05}
        params = pipeline._get_hyperparameters("xgboost", overrides)
        for value in params.values():
            assert isinstance(value, str)


# ---------------------------------------------------------------
# Testes do train (fluxo completo com mocks)
# ---------------------------------------------------------------


class TestTrain:
    """Testes do método train."""

    def test_train_algoritmo_invalido_no_metodo(self, pipeline):
        """Passar algoritmo inválido em train() deve levantar ValueError."""
        with pytest.raises(ValueError, match="não suportado"):
            pipeline.train(
                training_data_s3="s3://bucket/data.csv",
                algorithm="invalid_algo",
            )

    def test_train_fluxo_completo(
        self, pipeline, mock_sagemaker_client, mock_s3_client, sample_training_df
    ):
        """Train completo deve chamar todos os serviços AWS."""
        # Mock do download S3
        csv_buffer = io.StringIO()
        sample_training_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        mock_body = MagicMock()
        mock_body.read.return_value = csv_bytes
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        # Mock do training job
        mock_sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "ModelArtifacts": {
                "S3ModelArtifacts": "s3://bucket/output/model.tar.gz"
            },
        }

        # Mock do batch transform para avaliação
        mock_sagemaker_client.describe_transform_job.return_value = {
            "TransformJobStatus": "Completed",
        }

        # Mock das predições do test set
        n_test = int(len(sample_training_df) * 0.15)
        predictions = "\n".join(
            [str(np.random.uniform(0, 1)) for _ in range(n_test)]
        )
        mock_pred_body = MagicMock()
        mock_pred_body.read.return_value = predictions.encode("utf-8")

        # Configurar get_object para retornar dados diferentes por chamada
        mock_s3_client.get_object.side_effect = [
            {"Body": mock_body},        # Download training data
            {"Body": mock_pred_body},   # Download predictions
        ]

        # Mock do registro no model registry
        mock_sagemaker_client.create_model_package.return_value = {
            "ModelPackageArn": "arn:aws:sagemaker:us-east-1:123:model-package/v1"
        }

        # Mock do STS para get_training_image
        with patch("boto3.client") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_boto3.return_value = mock_sts

            result = pipeline.train(
                training_data_s3="s3://bucket/features/training_data.csv"
            )

        # Verificações
        assert result.algorithm == "xgboost"
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "f1" in result.metrics
        assert "roc_auc" in result.metrics
        assert 0.0 <= result.metrics["precision"] <= 1.0
        assert 0.0 <= result.metrics["recall"] <= 1.0
        assert 0.0 <= result.metrics["f1"] <= 1.0
        assert 0.0 <= result.metrics["roc_auc"] <= 1.0

        # Deve ter chamado create_training_job
        mock_sagemaker_client.create_training_job.assert_called_once()

        # Deve ter registrado o modelo
        mock_sagemaker_client.create_model_package.assert_called_once()


# ---------------------------------------------------------------
# Testes do predict_batch
# ---------------------------------------------------------------


class TestPredictBatch:
    """Testes do método predict_batch."""

    def test_predict_batch_com_model_version(
        self, pipeline, mock_sagemaker_client
    ):
        """predict_batch com model_version deve usar o ARN fornecido."""
        mock_sagemaker_client.describe_transform_job.return_value = {
            "TransformJobStatus": "Completed",
        }

        with patch("boto3.client") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123"}
            mock_boto3.return_value = mock_sts

            result = pipeline.predict_batch(
                feature_vectors_s3="s3://bucket/features/data.csv",
                model_version="arn:aws:sagemaker:us-east-1:123:model/v1",
            )

        assert result.startswith("s3://")
        mock_sagemaker_client.create_transform_job.assert_called_once()

    def test_predict_batch_sem_model_version_usa_approved(
        self, pipeline, mock_sagemaker_client
    ):
        """predict_batch sem model_version deve buscar o modelo aprovado."""
        # Mock para get_active_model
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": "arn:aws:sagemaker:us-east-1:123:pkg/v2"}
            ]
        }
        mock_sagemaker_client.describe_model_package.return_value = {
            "CustomerMetadataProperties": {
                "algorithm": "xgboost",
                "training_date": "2024-01-01T00:00:00",
                "dataset_version": "abc123",
                "precision": "0.85",
                "recall": "0.80",
                "f1": "0.82",
                "roc_auc": "0.90",
            },
        }
        mock_sagemaker_client.describe_transform_job.return_value = {
            "TransformJobStatus": "Completed",
        }

        with patch("boto3.client") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123"}
            mock_boto3.return_value = mock_sts

            result = pipeline.predict_batch(
                feature_vectors_s3="s3://bucket/features/data.csv",
            )

        assert result.startswith("s3://")
        mock_sagemaker_client.list_model_packages.assert_called_once()


# ---------------------------------------------------------------
# Testes do get_active_model
# ---------------------------------------------------------------


class TestGetActiveModel:
    """Testes do método get_active_model."""

    def test_get_active_model_sucesso(
        self, pipeline, mock_sagemaker_client
    ):
        """Deve retornar ModelVersion do modelo aprovado."""
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {
                    "ModelPackageArn": "arn:aws:sagemaker:us-east-1:123:pkg/v1",
                    "CreationTime": "2024-01-15T10:00:00",
                }
            ]
        }
        mock_sagemaker_client.describe_model_package.return_value = {
            "CustomerMetadataProperties": {
                "algorithm": "lightgbm",
                "training_date": "2024-01-15T10:00:00",
                "dataset_version": "ds-v3",
                "precision": "0.88",
                "recall": "0.82",
                "f1": "0.85",
                "roc_auc": "0.92",
            },
        }

        result = pipeline.get_active_model()

        assert result.algorithm == "lightgbm"
        assert result.metrics["precision"] == 0.88
        assert result.metrics["roc_auc"] == 0.92
        assert result.model_package_arn == (
            "arn:aws:sagemaker:us-east-1:123:pkg/v1"
        )

    def test_get_active_model_nenhum_aprovado(
        self, pipeline, mock_sagemaker_client
    ):
        """Deve levantar RuntimeError se nenhum modelo aprovado."""
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": []
        }

        with pytest.raises(RuntimeError, match="Nenhum modelo aprovado"):
            pipeline.get_active_model()


# ---------------------------------------------------------------
# Testes de upload de splits
# ---------------------------------------------------------------


class TestUploadSplit:
    """Testes do upload de splits para S3."""

    def test_upload_formato_csv_sem_header(
        self, pipeline, mock_s3_client, sample_training_df
    ):
        """Upload deve gerar CSV sem header com label na primeira coluna."""
        pipeline._upload_split(sample_training_df, "models/v1", "train")

        # Verificar que put_object foi chamado
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]

        assert call_kwargs["Bucket"] == "sky-brazil-churn-prediction"
        assert call_kwargs["Key"] == "models/v1/train/data.csv"
        assert call_kwargs["ContentType"] == "text/csv"

        # Verificar que não tem header (primeira linha é dado)
        body = call_kwargs["Body"].decode("utf-8")
        first_line = body.split("\n")[0]
        # Primeira coluna deve ser 0 ou 1 (label numérico)
        first_value = first_line.split(",")[0]
        assert first_value in ("0", "1", "0.0", "1.0")
