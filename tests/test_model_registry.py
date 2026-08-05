"""Testes unitários para o módulo src.ml.model_registry.

Valida:
- Listagem de versões de modelos
- Obtenção do modelo ativo (aprovado)
- Aprovação e rejeição de modelos
- Rollback para versão anterior
- Bloqueio de deleção de modelos em produção
- Upload de artefatos para S3
- Verificação de uso em produção via DynamoDB

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ml.model_registry import (
    ModelInProductionError,
    ModelNotFoundError,
    ModelRegistry,
    NoApprovedModelError,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def mock_sagemaker_client():
    """Mock do cliente boto3 SageMaker."""
    client = MagicMock()
    # Configurar exceptions como atributo para validação
    client.exceptions = MagicMock()
    client.exceptions.ClientError = Exception
    return client


@pytest.fixture
def mock_s3_client():
    """Mock do cliente boto3 S3."""
    return MagicMock()


@pytest.fixture
def mock_dynamodb_resource():
    """Mock do resource boto3 DynamoDB."""
    resource = MagicMock()
    return resource


@pytest.fixture
def registry(
    mock_sagemaker_client, mock_s3_client, mock_dynamodb_resource
):
    """Instância do ModelRegistry com clientes mockados."""
    return ModelRegistry(
        model_package_group="churn-prediction-models",
        region="us-east-1",
        bucket="sky-brazil-churn-prediction",
        sagemaker_client=mock_sagemaker_client,
        s3_client=mock_s3_client,
        dynamodb_resource=mock_dynamodb_resource,
        predictions_table_name="churn_predictions",
    )


@pytest.fixture
def sample_model_metadata():
    """Metadados simulados de um model package."""
    return {
        "CustomerMetadataProperties": {
            "algorithm": "xgboost",
            "training_date": "2024-06-15T10:00:00+00:00",
            "dataset_version": "ds-v1",
            "precision": "0.88",
            "recall": "0.82",
            "f1": "0.85",
            "roc_auc": "0.92",
        },
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": "123.dkr.ecr.us-east-1.amazonaws.com/xgboost",
                    "ModelDataUrl": "s3://bucket/models/v1/model.tar.gz",
                }
            ],
        },
    }


# ---------------------------------------------------------------
# Testes de listagem de versões
# ---------------------------------------------------------------


class TestListVersions:
    """Testes do método list_versions."""

    def test_lista_versoes_com_sucesso(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Deve listar todas as versões do grupo."""
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {
                    "ModelPackageArn": (
                        "arn:aws:sagemaker:us-east-1:123:pkg/v2"
                    )
                },
                {
                    "ModelPackageArn": (
                        "arn:aws:sagemaker:us-east-1:123:pkg/v1"
                    )
                },
            ]
        }
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        versions = registry.list_versions()

        assert len(versions) == 2
        assert versions[0].algorithm == "xgboost"
        assert versions[0].metrics["precision"] == 0.88

    def test_lista_vazia_sem_modelos(
        self, registry, mock_sagemaker_client
    ):
        """Deve retornar lista vazia se nenhum modelo no grupo."""
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": []
        }

        versions = registry.list_versions()

        assert versions == []

    def test_lista_com_paginacao(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Deve paginar se houver NextToken."""
        mock_sagemaker_client.list_model_packages.side_effect = [
            {
                "ModelPackageSummaryList": [
                    {
                        "ModelPackageArn": (
                            "arn:aws:sagemaker:us-east-1:123:pkg/v1"
                        )
                    },
                ],
                "NextToken": "token123",
            },
            {
                "ModelPackageSummaryList": [
                    {
                        "ModelPackageArn": (
                            "arn:aws:sagemaker:us-east-1:123:pkg/v2"
                        )
                    },
                ],
            },
        ]
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        versions = registry.list_versions()

        assert len(versions) == 2
        assert mock_sagemaker_client.list_model_packages.call_count == 2


# ---------------------------------------------------------------
# Testes de get_active_model
# ---------------------------------------------------------------


class TestGetActiveModel:
    """Testes do método get_active_model."""

    def test_retorna_modelo_aprovado(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Deve retornar o modelo com status Approved."""
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {
                    "ModelPackageArn": (
                        "arn:aws:sagemaker:us-east-1:123:pkg/v2"
                    )
                }
            ]
        }
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        result = registry.get_active_model()

        assert result is not None
        assert result.algorithm == "xgboost"
        assert result.metrics["roc_auc"] == 0.92
        # Deve ter filtrado por Approved
        call_kwargs = (
            mock_sagemaker_client.list_model_packages.call_args[1]
        )
        assert call_kwargs["ModelApprovalStatus"] == "Approved"

    def test_nenhum_modelo_aprovado_levanta_erro(
        self, registry, mock_sagemaker_client
    ):
        """Deve levantar NoApprovedModelError se nenhum aprovado."""
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": []
        }

        with pytest.raises(NoApprovedModelError):
            registry.get_active_model()


# ---------------------------------------------------------------
# Testes de approve_model
# ---------------------------------------------------------------


class TestApproveModel:
    """Testes do método approve_model."""

    def test_aprova_modelo_existente(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Deve atualizar status para Approved."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        registry.approve_model(arn)

        mock_sagemaker_client.update_model_package.assert_called_once_with(
            ModelPackageArn=arn,
            ModelApprovalStatus="Approved",
        )

    def test_aprova_modelo_inexistente_levanta_erro(
        self, registry, mock_sagemaker_client
    ):
        """Deve levantar ModelNotFoundError se modelo não existe."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/inexistente"
        mock_sagemaker_client.describe_model_package.side_effect = (
            mock_sagemaker_client.exceptions.ClientError
        )

        with pytest.raises(ModelNotFoundError):
            registry.approve_model(arn)


# ---------------------------------------------------------------
# Testes de reject_model
# ---------------------------------------------------------------


class TestRejectModel:
    """Testes do método reject_model."""

    def test_rejeita_modelo_existente(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Deve atualizar status para Rejected."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        registry.reject_model(arn)

        mock_sagemaker_client.update_model_package.assert_called_once_with(
            ModelPackageArn=arn,
            ModelApprovalStatus="Rejected",
        )

    def test_rejeita_modelo_inexistente_levanta_erro(
        self, registry, mock_sagemaker_client
    ):
        """Deve levantar ModelNotFoundError se modelo não existe."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/inexistente"
        mock_sagemaker_client.describe_model_package.side_effect = (
            mock_sagemaker_client.exceptions.ClientError
        )

        with pytest.raises(ModelNotFoundError):
            registry.reject_model(arn)


# ---------------------------------------------------------------
# Testes de rollback
# ---------------------------------------------------------------


class TestRollback:
    """Testes do método rollback."""

    def test_rollback_aprova_alvo_rejeita_atual(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Rollback deve aprovar alvo e rejeitar modelo atual."""
        target_arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"
        current_arn = "arn:aws:sagemaker:us-east-1:123:pkg/v2"

        # Mock describe para validação de existência
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        # Mock list_model_packages para get_active_model
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": current_arn}
            ]
        }

        registry.rollback(target_arn)

        # Verificar chamadas de update
        calls = mock_sagemaker_client.update_model_package.call_args_list
        assert len(calls) == 2

        # Primeira chamada: aprovar alvo
        assert calls[0][1]["ModelPackageArn"] == target_arn
        assert calls[0][1]["ModelApprovalStatus"] == "Approved"

        # Segunda chamada: rejeitar atual
        assert calls[1][1]["ModelPackageArn"] == current_arn
        assert calls[1][1]["ModelApprovalStatus"] == "Rejected"

    def test_rollback_sem_modelo_ativo_apenas_aprova(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Rollback sem modelo ativo deve apenas aprovar o alvo."""
        target_arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"

        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        # Sem modelo aprovado
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": []
        }

        registry.rollback(target_arn)

        # Apenas uma chamada de update (aprovar alvo)
        calls = mock_sagemaker_client.update_model_package.call_args_list
        assert len(calls) == 1
        assert calls[0][1]["ModelPackageArn"] == target_arn
        assert calls[0][1]["ModelApprovalStatus"] == "Approved"

    def test_rollback_para_mesmo_modelo_ativo(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Rollback para o mesmo modelo ativo não deve rejeitar."""
        same_arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"

        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        # Modelo ativo é o mesmo que o alvo
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": same_arn}
            ]
        }

        registry.rollback(same_arn)

        # Apenas uma chamada (aprovar o alvo, não rejeita pois é o mesmo)
        calls = mock_sagemaker_client.update_model_package.call_args_list
        assert len(calls) == 1
        assert calls[0][1]["ModelApprovalStatus"] == "Approved"

    def test_rollback_modelo_inexistente_levanta_erro(
        self, registry, mock_sagemaker_client
    ):
        """Rollback para modelo inexistente deve levantar erro."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/inexistente"
        mock_sagemaker_client.describe_model_package.side_effect = (
            mock_sagemaker_client.exceptions.ClientError
        )

        with pytest.raises(ModelNotFoundError):
            registry.rollback(arn)


# ---------------------------------------------------------------
# Testes de delete_model
# ---------------------------------------------------------------


class TestDeleteModel:
    """Testes do método delete_model."""

    def test_deleta_modelo_nao_usado(
        self,
        registry,
        mock_sagemaker_client,
        mock_dynamodb_resource,
        sample_model_metadata,
    ):
        """Deve deletar modelo não usado em produção."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        # Mock DynamoDB: nenhuma predição usa este modelo
        mock_table = MagicMock()
        mock_table.scan.return_value = {"Count": 0}
        mock_dynamodb_resource.Table.return_value = mock_table

        registry.delete_model(arn)

        mock_sagemaker_client.delete_model_package.assert_called_once_with(
            ModelPackageName=arn
        )

    def test_bloqueia_delecao_modelo_em_producao(
        self,
        registry,
        mock_sagemaker_client,
        mock_dynamodb_resource,
        sample_model_metadata,
    ):
        """Deve bloquear deleção de modelo usado em predições."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        # Mock DynamoDB: modelo usado em predições
        mock_table = MagicMock()
        mock_table.scan.return_value = {"Count": 3}
        mock_dynamodb_resource.Table.return_value = mock_table

        with pytest.raises(ModelInProductionError) as exc_info:
            registry.delete_model(arn)

        assert arn in str(exc_info.value)
        # Não deve ter chamado delete
        mock_sagemaker_client.delete_model_package.assert_not_called()

    def test_deleta_modelo_inexistente_levanta_erro(
        self, registry, mock_sagemaker_client
    ):
        """Deve levantar ModelNotFoundError se modelo não existe."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/inexistente"
        mock_sagemaker_client.describe_model_package.side_effect = (
            mock_sagemaker_client.exceptions.ClientError
        )

        with pytest.raises(ModelNotFoundError):
            registry.delete_model(arn)


# ---------------------------------------------------------------
# Testes de is_model_in_production
# ---------------------------------------------------------------


class TestIsModelInProduction:
    """Testes do método is_model_in_production."""

    def test_modelo_em_producao_retorna_true(
        self, registry, mock_dynamodb_resource
    ):
        """Deve retornar True se modelo referenciado em predições."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"

        mock_table = MagicMock()
        mock_table.scan.return_value = {"Count": 5}
        mock_dynamodb_resource.Table.return_value = mock_table

        assert registry.is_model_in_production(arn) is True

    def test_modelo_nao_usado_retorna_false(
        self, registry, mock_dynamodb_resource
    ):
        """Deve retornar False se modelo não referenciado."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"

        mock_table = MagicMock()
        mock_table.scan.return_value = {"Count": 0}
        mock_dynamodb_resource.Table.return_value = mock_table

        assert registry.is_model_in_production(arn) is False

    def test_consulta_tabela_correta(
        self, registry, mock_dynamodb_resource
    ):
        """Deve consultar a tabela de predições configurada."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"

        mock_table = MagicMock()
        mock_table.scan.return_value = {"Count": 0}
        mock_dynamodb_resource.Table.return_value = mock_table

        registry.is_model_in_production(arn)

        mock_dynamodb_resource.Table.assert_called_with(
            "churn_predictions"
        )


# ---------------------------------------------------------------
# Testes de upload_model_artifacts
# ---------------------------------------------------------------


class TestUploadModelArtifacts:
    """Testes do método upload_model_artifacts."""

    def test_upload_todos_artefatos(self, registry, mock_s3_client):
        """Deve fazer upload de model.tar.gz, hyperparameters e metrics."""
        model_data = b"fake-model-binary-data"
        hyperparameters = {"max_depth": "6", "eta": "0.3"}
        metrics = {"precision": 0.88, "recall": 0.82, "f1": 0.85, "roc_auc": 0.92}

        result = registry.upload_model_artifacts(
            model_version_id="abc123",
            model_data=model_data,
            hyperparameters=hyperparameters,
            metrics=metrics,
        )

        # Deve retornar o S3 URI do modelo
        assert result == (
            "s3://sky-brazil-churn-prediction/models/abc123/model.tar.gz"
        )

        # Deve ter feito 3 uploads (model, hyperparams, metrics)
        assert mock_s3_client.put_object.call_count == 3

        # Verificar keys dos uploads
        calls = mock_s3_client.put_object.call_args_list
        keys = [c[1]["Key"] for c in calls]
        assert "models/abc123/model.tar.gz" in keys
        assert "models/abc123/hyperparameters.json" in keys
        assert "models/abc123/metrics.json" in keys

    def test_upload_conteudo_correto_model(
        self, registry, mock_s3_client
    ):
        """O model.tar.gz deve conter os bytes fornecidos."""
        model_data = b"\x1f\x8b\x08fake-gzip-data"

        registry.upload_model_artifacts(
            model_version_id="v1",
            model_data=model_data,
            hyperparameters={},
            metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5},
        )

        # Verificar o body do model.tar.gz
        model_call = None
        for call in mock_s3_client.put_object.call_args_list:
            if "model.tar.gz" in call[1]["Key"]:
                model_call = call
                break

        assert model_call is not None
        assert model_call[1]["Body"] == model_data
        assert model_call[1]["ContentType"] == "application/gzip"

    def test_upload_conteudo_correto_metrics(
        self, registry, mock_s3_client
    ):
        """O metrics.json deve conter JSON válido com as métricas."""
        import json

        metrics = {
            "precision": 0.88,
            "recall": 0.82,
            "f1": 0.85,
            "roc_auc": 0.92,
        }

        registry.upload_model_artifacts(
            model_version_id="v2",
            model_data=b"data",
            hyperparameters={"algo": "xgboost"},
            metrics=metrics,
        )

        # Verificar o body do metrics.json
        metrics_call = None
        for call in mock_s3_client.put_object.call_args_list:
            if "metrics.json" in call[1]["Key"]:
                metrics_call = call
                break

        assert metrics_call is not None
        body = metrics_call[1]["Body"].decode("utf-8")
        parsed = json.loads(body)
        assert parsed["precision"] == 0.88
        assert parsed["roc_auc"] == 0.92


# ---------------------------------------------------------------
# Testes de get_model_artifacts_s3_path
# ---------------------------------------------------------------


class TestGetModelArtifactsPath:
    """Testes do método get_model_artifacts_s3_path."""

    def test_retorna_path_do_artefato(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Deve retornar o ModelDataUrl do container."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        path = registry.get_model_artifacts_s3_path(arn)

        assert path == "s3://bucket/models/v1/model.tar.gz"

    def test_retorna_vazio_sem_containers(
        self, registry, mock_sagemaker_client
    ):
        """Deve retornar string vazia se sem InferenceSpecification."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"
        mock_sagemaker_client.describe_model_package.return_value = {
            "CustomerMetadataProperties": {
                "algorithm": "xgboost",
                "training_date": "2024-01-01",
                "dataset_version": "v1",
                "precision": "0.8",
                "recall": "0.8",
                "f1": "0.8",
                "roc_auc": "0.8",
            },
        }

        path = registry.get_model_artifacts_s3_path(arn)

        assert path == ""


# ---------------------------------------------------------------
# Testes de integração dos fluxos
# ---------------------------------------------------------------


class TestFluxoIntegrado:
    """Testes que validam fluxos completos de uso."""

    def test_ciclo_completo_treino_aprovacao_predicao(
        self, registry, mock_sagemaker_client, sample_model_metadata
    ):
        """Fluxo: upload artefatos → aprovar → buscar ativo."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"

        # Descrever modelo
        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )
        mock_sagemaker_client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": arn}
            ]
        }

        # Aprovar
        registry.approve_model(arn)

        # Buscar ativo
        active = registry.get_active_model()

        assert active is not None
        assert active.model_package_arn == arn

    def test_protecao_modelo_em_producao_completa(
        self,
        registry,
        mock_sagemaker_client,
        mock_dynamodb_resource,
        sample_model_metadata,
    ):
        """Modelo em produção: aprovar OK, deletar bloqueado."""
        arn = "arn:aws:sagemaker:us-east-1:123:pkg/v1"

        mock_sagemaker_client.describe_model_package.return_value = (
            sample_model_metadata
        )

        # Modelo usado em produção
        mock_table = MagicMock()
        mock_table.scan.return_value = {"Count": 10}
        mock_dynamodb_resource.Table.return_value = mock_table

        # Aprovar deve funcionar normalmente
        registry.approve_model(arn)
        assert mock_sagemaker_client.update_model_package.called

        # Deletar deve ser bloqueado
        with pytest.raises(ModelInProductionError):
            registry.delete_model(arn)
