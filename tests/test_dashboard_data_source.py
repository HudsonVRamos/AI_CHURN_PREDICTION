"""Testes unitários para o DashboardDataSource.

Usa moto para simular DynamoDB local.
Valida: Requirements 14.1, 14.3, 14.4, 14.6
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import boto3
import pytest
from moto import mock_aws

from src.dashboard.data_source import DashboardDataSource


def _create_executions_table(dynamodb_resource, table_name: str):
    """Cria tabela DynamoDB simulada para churn_executions."""
    dynamodb_resource.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "execution_id", "KeyType": "HASH"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "execution_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


def _create_predictions_table(dynamodb_resource, table_name: str):
    """Cria tabela DynamoDB simulada para churn_predictions."""
    dynamodb_resource.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "execution_id", "KeyType": "HASH"},
            {"AttributeName": "user_id", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "execution_id", "AttributeType": "S"},
            {"AttributeName": "user_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


def _create_feature_store_table(dynamodb_resource, table_name: str):
    """Cria tabela DynamoDB simulada para churn_feature_store."""
    dynamodb_resource.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "user_id", "KeyType": "HASH"},
            {"AttributeName": "version", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "version", "AttributeType": "N"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


def _insert_execution(table, execution: dict[str, Any]):
    """Insere uma execução na tabela."""
    table.put_item(Item=execution)


def _insert_prediction(table, prediction: dict[str, Any]):
    """Insere uma predição na tabela."""
    table.put_item(Item=prediction)


def _insert_feature(table, feature: dict[str, Any]):
    """Insere um feature vector na tabela."""
    table.put_item(Item=feature)


@pytest.fixture
def dashboard_setup():
    """Fixture que cria DynamoDB mock com as 3 tabelas do dashboard."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")

        exec_table = "test_executions"
        pred_table = "test_predictions"
        feat_table = "test_features"

        _create_executions_table(dynamodb, exec_table)
        _create_predictions_table(dynamodb, pred_table)
        _create_feature_store_table(dynamodb, feat_table)

        ds = DashboardDataSource(
            dynamodb_resource=dynamodb,
            s3_client=None,
            executions_table=exec_table,
            predictions_table=pred_table,
            feature_store_table=feat_table,
        )

        yield {
            "data_source": ds,
            "dynamodb": dynamodb,
            "exec_table": dynamodb.Table(exec_table),
            "pred_table": dynamodb.Table(pred_table),
            "feat_table": dynamodb.Table(feat_table),
        }


class TestGetLatestExecution:
    """Testes para o método get_latest_execution()."""

    def test_retorna_none_sem_execucoes(self, dashboard_setup):
        """R14.1: Retorna None quando não há execuções."""
        ds = dashboard_setup["data_source"]

        result = ds.get_latest_execution()

        assert result is None

    def test_retorna_execucao_mais_recente(self, dashboard_setup):
        """R14.1: Retorna a execução com start_time mais recente."""
        ds = dashboard_setup["data_source"]
        table = dashboard_setup["exec_table"]

        _insert_execution(table, {
            "execution_id": "exec-001",
            "start_time": "2024-01-01T10:00:00Z",
            "end_time": "2024-01-01T11:00:00Z",
            "mode": "predict",
            "model_version": "v1.0",
            "users_processed": 100,
            "users_failed": 2,
            "status": "completed",
            "output_s3_path": "s3://bucket/reports/exec-001/",
        })
        _insert_execution(table, {
            "execution_id": "exec-002",
            "start_time": "2024-02-01T10:00:00Z",
            "end_time": "2024-02-01T11:30:00Z",
            "mode": "predict",
            "model_version": "v1.1",
            "users_processed": 150,
            "users_failed": 0,
            "status": "completed",
            "output_s3_path": "s3://bucket/reports/exec-002/",
        })

        result = ds.get_latest_execution()

        assert result is not None
        assert result.execution_id == "exec-002"
        assert result.model_version == "v1.1"
        assert result.users_processed == 150
        assert result.status == "completed"

    def test_retorna_execution_summary_valido(self, dashboard_setup):
        """R14.1: ExecutionSummary contém todos os campos obrigatórios."""
        ds = dashboard_setup["data_source"]
        table = dashboard_setup["exec_table"]

        _insert_execution(table, {
            "execution_id": "exec-100",
            "start_time": "2024-03-15T08:00:00Z",
            "end_time": "2024-03-15T09:30:00Z",
            "mode": "train",
            "model_version": "v2.0",
            "users_processed": 500,
            "users_failed": 10,
            "status": "completed",
            "output_s3_path": "s3://bucket/models/v2.0/",
        })

        result = ds.get_latest_execution()

        assert result.execution_id == "exec-100"
        assert result.start_time == "2024-03-15T08:00:00Z"
        assert result.end_time == "2024-03-15T09:30:00Z"
        assert result.mode == "train"
        assert result.model_version == "v2.0"
        assert result.users_processed == 500
        assert result.users_failed == 10
        assert result.output_s3_path == "s3://bucket/models/v2.0/"


class TestGetPredictions:
    """Testes para o método get_predictions()."""

    def test_retorna_lista_vazia_sem_predicoes(self, dashboard_setup):
        """R14.3: Lista vazia quando não há predições."""
        ds = dashboard_setup["data_source"]

        result = ds.get_predictions()

        assert result == []

    def test_retorna_todas_predicoes_sem_filtro(self, dashboard_setup):
        """R14.1: Retorna todas as predições sem filtro."""
        ds = dashboard_setup["data_source"]
        table = dashboard_setup["pred_table"]

        _insert_prediction(table, {
            "execution_id": "exec-001",
            "user_id": "user-A",
            "churn_probability": Decimal("0.85"),
            "confidence": Decimal("0.92"),
            "risk_tier": "High",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })
        _insert_prediction(table, {
            "execution_id": "exec-001",
            "user_id": "user-B",
            "churn_probability": Decimal("0.20"),
            "confidence": Decimal("0.88"),
            "risk_tier": "Low",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })

        result = ds.get_predictions()

        assert len(result) == 2

    def test_filtra_por_risk_level(self, dashboard_setup):
        """R14.3: Filtro por risk_level retorna apenas o nível solicitado."""
        ds = dashboard_setup["data_source"]
        table = dashboard_setup["pred_table"]

        _insert_prediction(table, {
            "execution_id": "exec-001",
            "user_id": "user-high",
            "churn_probability": Decimal("0.85"),
            "confidence": Decimal("0.90"),
            "risk_tier": "High",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })
        _insert_prediction(table, {
            "execution_id": "exec-001",
            "user_id": "user-low",
            "churn_probability": Decimal("0.15"),
            "confidence": Decimal("0.85"),
            "risk_tier": "Low",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })
        _insert_prediction(table, {
            "execution_id": "exec-001",
            "user_id": "user-med",
            "churn_probability": Decimal("0.45"),
            "confidence": Decimal("0.80"),
            "risk_tier": "Medium",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })

        result = ds.get_predictions(risk_level="High")

        assert len(result) == 1
        assert result[0]["risk_tier"] == "High"
        assert result[0]["user_id"] == "user-high"

    def test_filtro_all_retorna_todos(self, dashboard_setup):
        """R14.3: risk_level='All' retorna todas as predições."""
        ds = dashboard_setup["data_source"]
        table = dashboard_setup["pred_table"]

        _insert_prediction(table, {
            "execution_id": "exec-001",
            "user_id": "user-1",
            "churn_probability": Decimal("0.85"),
            "confidence": Decimal("0.90"),
            "risk_tier": "High",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })
        _insert_prediction(table, {
            "execution_id": "exec-001",
            "user_id": "user-2",
            "churn_probability": Decimal("0.20"),
            "confidence": Decimal("0.88"),
            "risk_tier": "Low",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })

        result = ds.get_predictions(risk_level="All")

        assert len(result) == 2

    def test_filtra_por_periodo(self, dashboard_setup):
        """R14.3: Filtro por período (date range) funciona."""
        ds = dashboard_setup["data_source"]
        table = dashboard_setup["pred_table"]

        _insert_prediction(table, {
            "execution_id": "exec-jan",
            "user_id": "user-A",
            "churn_probability": Decimal("0.70"),
            "confidence": Decimal("0.85"),
            "risk_tier": "High",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-10T10:00:00Z",
        })
        _insert_prediction(table, {
            "execution_id": "exec-mar",
            "user_id": "user-A",
            "churn_probability": Decimal("0.65"),
            "confidence": Decimal("0.88"),
            "risk_tier": "High",
            "model_version": "v1.1",
            "feature_version": 2,
            "timestamp": "2024-03-10T10:00:00Z",
        })

        result = ds.get_predictions(
            period_start="2024-02-01T00:00:00Z",
            period_end="2024-04-01T00:00:00Z",
        )

        assert len(result) == 1
        assert result[0]["timestamp"] == "2024-03-10T10:00:00Z"

    def test_filtra_por_risk_level_e_periodo_combinados(self, dashboard_setup):
        """R14.3: Filtros combinados (risk_level + período)."""
        ds = dashboard_setup["data_source"]
        table = dashboard_setup["pred_table"]

        _insert_prediction(table, {
            "execution_id": "exec-001",
            "user_id": "user-A",
            "churn_probability": Decimal("0.80"),
            "confidence": Decimal("0.90"),
            "risk_tier": "High",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })
        _insert_prediction(table, {
            "execution_id": "exec-002",
            "user_id": "user-B",
            "churn_probability": Decimal("0.75"),
            "confidence": Decimal("0.87"),
            "risk_tier": "High",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-03-15T10:00:00Z",
        })
        _insert_prediction(table, {
            "execution_id": "exec-002",
            "user_id": "user-C",
            "churn_probability": Decimal("0.25"),
            "confidence": Decimal("0.88"),
            "risk_tier": "Low",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-03-15T10:00:00Z",
        })

        result = ds.get_predictions(
            risk_level="High",
            period_start="2024-02-01T00:00:00Z",
        )

        assert len(result) == 1
        assert result[0]["user_id"] == "user-B"
        assert result[0]["risk_tier"] == "High"


class TestGetSubscriberDetail:
    """Testes para o método get_subscriber_detail()."""

    def test_retorna_none_se_user_nao_existe(self, dashboard_setup):
        """R14.4: Retorna None se não encontrar predição para o user."""
        ds = dashboard_setup["data_source"]

        result = ds.get_subscriber_detail("user-inexistente")

        assert result is None

    def test_retorna_detalhes_completos(self, dashboard_setup):
        """R14.4: Retorna churn score, confiança, features e explicação."""
        ds = dashboard_setup["data_source"]
        pred_table = dashboard_setup["pred_table"]
        feat_table = dashboard_setup["feat_table"]

        _insert_prediction(pred_table, {
            "execution_id": "exec-001",
            "user_id": "user-detail",
            "churn_probability": Decimal("0.72"),
            "confidence": Decimal("0.91"),
            "risk_tier": "High",
            "model_version": "v1.0",
            "feature_version": 3,
            "timestamp": "2024-03-01T10:00:00Z",
            "shap_results": {
                "total_sessions": Decimal("-0.15"),
                "avg_happiness_score": Decimal("0.25"),
            },
            "bedrock_explanation": "O assinante apresenta queda no engajamento.",
            "explanation_status": "available",
        })

        _insert_feature(feat_table, {
            "user_id": "user-detail",
            "version": 3,
            "generated_at": "2024-02-28T10:00:00Z",
            "observation_start": "2023-08-28T00:00:00Z",
            "observation_end": "2024-02-28T00:00:00Z",
            "features": {
                "total_sessions": 45,
                "total_viewing_hours": Decimal("22.5"),
                "avg_happiness_score": Decimal("5.8"),
                "sessions_per_week": Decimal("2.1"),
            },
        })

        result = ds.get_subscriber_detail("user-detail")

        assert result is not None
        assert result["user_id"] == "user-detail"
        assert result["churn_probability"] == pytest.approx(0.72)
        assert result["confidence"] == pytest.approx(0.91)
        assert result["risk_tier"] == "High"
        assert result["model_version"] == "v1.0"
        assert result["bedrock_explanation"] == (
            "O assinante apresenta queda no engajamento."
        )
        assert result["explanation_status"] == "available"
        assert result["features"] is not None
        assert result["features"]["total_sessions"] == 45
        assert result["features"]["total_viewing_hours"] == pytest.approx(22.5)

    def test_retorna_detalhes_sem_explicacao(self, dashboard_setup):
        """R14.4: Funciona quando bedrock_explanation é None."""
        ds = dashboard_setup["data_source"]
        pred_table = dashboard_setup["pred_table"]

        _insert_prediction(pred_table, {
            "execution_id": "exec-001",
            "user_id": "user-no-expl",
            "churn_probability": Decimal("0.45"),
            "confidence": Decimal("0.80"),
            "risk_tier": "Medium",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-03-01T10:00:00Z",
            "explanation_status": "unavailable",
        })

        result = ds.get_subscriber_detail("user-no-expl")

        assert result is not None
        assert result["bedrock_explanation"] is None
        assert result["explanation_status"] == "unavailable"

    def test_retorna_predicao_mais_recente(self, dashboard_setup):
        """R14.4: Se houver múltiplas predições, retorna a mais recente."""
        ds = dashboard_setup["data_source"]
        pred_table = dashboard_setup["pred_table"]

        _insert_prediction(pred_table, {
            "execution_id": "exec-old",
            "user_id": "user-multi",
            "churn_probability": Decimal("0.60"),
            "confidence": Decimal("0.85"),
            "risk_tier": "Medium",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-01T10:00:00Z",
        })
        _insert_prediction(pred_table, {
            "execution_id": "exec-new",
            "user_id": "user-multi",
            "churn_probability": Decimal("0.80"),
            "confidence": Decimal("0.92"),
            "risk_tier": "High",
            "model_version": "v1.1",
            "feature_version": 2,
            "timestamp": "2024-03-01T10:00:00Z",
        })

        result = ds.get_subscriber_detail("user-multi")

        assert result is not None
        assert result["churn_probability"] == pytest.approx(0.80)
        assert result["risk_tier"] == "High"
        assert result["model_version"] == "v1.1"


class TestGetHistory:
    """Testes para o método get_history()."""

    def test_retorna_lista_vazia_se_user_nao_existe(self, dashboard_setup):
        """Retorna lista vazia para user sem histórico."""
        ds = dashboard_setup["data_source"]

        result = ds.get_history("user-fantasma")

        assert result == []

    def test_retorna_historico_ordenado_por_timestamp(self, dashboard_setup):
        """R14.4: Histórico ordenado cronologicamente."""
        ds = dashboard_setup["data_source"]
        pred_table = dashboard_setup["pred_table"]

        _insert_prediction(pred_table, {
            "execution_id": "exec-mar",
            "user_id": "user-hist",
            "churn_probability": Decimal("0.75"),
            "confidence": Decimal("0.90"),
            "risk_tier": "High",
            "model_version": "v1.1",
            "feature_version": 2,
            "timestamp": "2024-03-01T10:00:00Z",
        })
        _insert_prediction(pred_table, {
            "execution_id": "exec-jan",
            "user_id": "user-hist",
            "churn_probability": Decimal("0.50"),
            "confidence": Decimal("0.85"),
            "risk_tier": "Medium",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-01T10:00:00Z",
        })
        _insert_prediction(pred_table, {
            "execution_id": "exec-feb",
            "user_id": "user-hist",
            "churn_probability": Decimal("0.65"),
            "confidence": Decimal("0.88"),
            "risk_tier": "High",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-02-01T10:00:00Z",
        })

        result = ds.get_history("user-hist")

        assert len(result) == 3
        assert result[0]["timestamp"] == "2024-01-01T10:00:00Z"
        assert result[1]["timestamp"] == "2024-02-01T10:00:00Z"
        assert result[2]["timestamp"] == "2024-03-01T10:00:00Z"
        # Evolução de risco
        assert result[0]["risk_tier"] == "Medium"
        assert result[2]["risk_tier"] == "High"

    def test_retorna_apenas_predicoes_do_user_especificado(
        self, dashboard_setup
    ):
        """Não mistura dados de outros users."""
        ds = dashboard_setup["data_source"]
        pred_table = dashboard_setup["pred_table"]

        _insert_prediction(pred_table, {
            "execution_id": "exec-001",
            "user_id": "user-target",
            "churn_probability": Decimal("0.70"),
            "confidence": Decimal("0.85"),
            "risk_tier": "High",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })
        _insert_prediction(pred_table, {
            "execution_id": "exec-001",
            "user_id": "user-other",
            "churn_probability": Decimal("0.30"),
            "confidence": Decimal("0.90"),
            "risk_tier": "Low",
            "model_version": "v1.0",
            "feature_version": 1,
            "timestamp": "2024-01-15T10:00:00Z",
        })

        result = ds.get_history("user-target")

        assert len(result) == 1
        assert result[0]["user_id"] == "user-target"


class TestDeserializacao:
    """Testes de desserialização de Decimal para float."""

    def test_decimal_convertido_para_float_em_predicoes(self, dashboard_setup):
        """Valores Decimal do DynamoDB são convertidos para float."""
        ds = dashboard_setup["data_source"]
        pred_table = dashboard_setup["pred_table"]

        _insert_prediction(pred_table, {
            "execution_id": "exec-001",
            "user_id": "user-dec",
            "churn_probability": Decimal("0.123456"),
            "confidence": Decimal("0.987654"),
            "risk_tier": "Low",
            "model_version": "v1.0",
            "feature_version": 5,
            "timestamp": "2024-01-01T10:00:00Z",
        })

        result = ds.get_predictions()

        assert len(result) == 1
        assert isinstance(result[0]["churn_probability"], float)
        assert isinstance(result[0]["confidence"], float)
        assert result[0]["churn_probability"] == pytest.approx(0.123456)
        assert result[0]["confidence"] == pytest.approx(0.987654)
        assert result[0]["feature_version"] == 5
