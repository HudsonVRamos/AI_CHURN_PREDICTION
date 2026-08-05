"""Model Registry - Gerenciamento do ciclo de vida de modelos.

Integra com o Amazon SageMaker Model Registry para:
- Listar versões de modelos registrados
- Obter modelo ativo (aprovado) para predições
- Aprovar/rejeitar modelos via approval status
- Rollback para versão anterior
- Impedir deleção de modelos usados em produção

Artefatos armazenados em S3:
- model.tar.gz (artefato serializado do modelo)
- hyperparameters.json (parâmetros de treino)
- metrics.json (métricas de avaliação)

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import boto3

from src.common.logging import get_logger
from src.common.models import ModelVersion

logger = get_logger("ml-inference")


class ModelInProductionError(Exception):
    """Erro ao tentar deletar modelo usado em produção."""

    def __init__(self, model_package_arn: str) -> None:
        self.model_package_arn = model_package_arn
        super().__init__(
            f"Não é possível deletar modelo usado em produção: "
            f"{model_package_arn}"
        )


class ModelNotFoundError(Exception):
    """Erro quando modelo não é encontrado no registry."""

    def __init__(self, model_package_arn: str) -> None:
        self.model_package_arn = model_package_arn
        super().__init__(
            f"Modelo não encontrado no registry: {model_package_arn}"
        )


class NoApprovedModelError(Exception):
    """Erro quando nenhum modelo aprovado é encontrado."""

    def __init__(self, model_package_group: str) -> None:
        self.model_package_group = model_package_group
        super().__init__(
            f"Nenhum modelo aprovado no grupo: "
            f"{model_package_group}"
        )


class ModelRegistry:
    """Gerencia ciclo de vida de modelos no SageMaker Model Registry.

    Responsável por controlar versões, aprovação, rollback e proteção
    contra deleção de modelos usados em predições de produção.

    Attributes:
        model_package_group: Nome do grupo no Model Registry.
        region: Região AWS.
    """

    def __init__(
        self,
        model_package_group: str,
        region: str,
        bucket: str,
        sagemaker_client: Any | None = None,
        s3_client: Any | None = None,
        dynamodb_resource: Any | None = None,
        predictions_table_name: str = "churn_predictions",
    ) -> None:
        """Inicializa o Model Registry.

        Args:
            model_package_group: Nome do grupo no Model Registry.
            region: Região AWS para os serviços.
            bucket: Bucket S3 para artefatos de modelo.
            sagemaker_client: Cliente boto3 SageMaker (para testes).
            s3_client: Cliente boto3 S3 (para testes).
            dynamodb_resource: Resource boto3 DynamoDB (para testes).
            predictions_table_name: Nome da tabela de predições.
        """
        self.model_package_group = model_package_group
        self.region = region
        self.bucket = bucket
        self.predictions_table_name = predictions_table_name

        self._sagemaker_client = sagemaker_client or boto3.client(
            "sagemaker", region_name=region
        )
        self._s3_client = s3_client or boto3.client(
            "s3", region_name=region
        )
        self._dynamodb_resource = dynamodb_resource or boto3.resource(
            "dynamodb", region_name=region
        )

        logger.info(
            f"ModelRegistry inicializado: group={model_package_group}, "
            f"region={region}, bucket={bucket}"
        )

    def list_versions(self) -> list[ModelVersion]:
        """Lista todas as versões do modelo no grupo.

        Consulta o SageMaker Model Registry e retorna todas as versões
        ordenadas por data de criação (mais recente primeiro).

        Returns:
            Lista de ModelVersion com metadados de cada versão.
        """
        logger.info(
            f"Listando versões do grupo: {self.model_package_group}"
        )

        versions: list[ModelVersion] = []
        next_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "ModelPackageGroupName": self.model_package_group,
                "SortBy": "CreationTime",
                "SortOrder": "Descending",
                "MaxResults": 100,
            }
            if next_token:
                kwargs["NextToken"] = next_token

            response = self._sagemaker_client.list_model_packages(
                **kwargs
            )

            for package_summary in response.get(
                "ModelPackageSummaryList", []
            ):
                arn = package_summary["ModelPackageArn"]
                model_version = self._describe_model_version(arn)
                if model_version:
                    versions.append(model_version)

            next_token = response.get("NextToken")
            if not next_token:
                break

        logger.info(f"Encontradas {len(versions)} versões no grupo")
        return versions

    def get_active_model(self) -> ModelVersion | None:
        """Retorna o modelo aprovado (ativo para predições).

        Busca o modelo com status 'Approved' mais recente no grupo.

        Returns:
            ModelVersion do modelo ativo ou None se nenhum aprovado.

        Raises:
            NoApprovedModelError: Se nenhum modelo aprovado encontrado.
        """
        logger.info(
            f"Buscando modelo ativo no grupo: "
            f"{self.model_package_group}"
        )

        response = self._sagemaker_client.list_model_packages(
            ModelPackageGroupName=self.model_package_group,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )

        packages = response.get("ModelPackageSummaryList", [])
        if not packages:
            logger.warning(
                f"Nenhum modelo aprovado no grupo: "
                f"{self.model_package_group}"
            )
            raise NoApprovedModelError(self.model_package_group)

        arn = packages[0]["ModelPackageArn"]
        model_version = self._describe_model_version(arn)

        if model_version:
            logger.info(
                f"Modelo ativo encontrado: {model_version.model_package_arn}"
            )

        return model_version

    def approve_model(self, model_package_arn: str) -> None:
        """Aprova um modelo para uso em produção.

        Altera o approval status do modelo para 'Approved', tornando-o
        disponível para inferência via Batch Transform.

        Args:
            model_package_arn: ARN do Model Package a aprovar.

        Raises:
            ModelNotFoundError: Se o modelo não existe.
        """
        logger.info(f"Aprovando modelo: {model_package_arn}")

        self._validate_model_exists(model_package_arn)

        self._sagemaker_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Approved",
        )

        logger.info(f"Modelo aprovado com sucesso: {model_package_arn}")

    def reject_model(self, model_package_arn: str) -> None:
        """Rejeita um modelo (não pode ser usado para predições).

        Altera o approval status para 'Rejected'. Um modelo rejeitado
        não será retornado por get_active_model().

        Args:
            model_package_arn: ARN do Model Package a rejeitar.

        Raises:
            ModelNotFoundError: Se o modelo não existe.
        """
        logger.info(f"Rejeitando modelo: {model_package_arn}")

        self._validate_model_exists(model_package_arn)

        self._sagemaker_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Rejected",
        )

        logger.info(f"Modelo rejeitado: {model_package_arn}")

    def rollback(self, target_model_package_arn: str) -> None:
        """Rollback: aprova o modelo alvo e rejeita o atual.

        Realiza rollback atômico: primeiro aprova o modelo alvo,
        depois rejeita o modelo atualmente ativo. Se não houver
        modelo ativo, apenas aprova o alvo.

        Args:
            target_model_package_arn: ARN do modelo para restaurar.

        Raises:
            ModelNotFoundError: Se o modelo alvo não existe.
        """
        logger.info(
            f"Iniciando rollback para: {target_model_package_arn}"
        )

        self._validate_model_exists(target_model_package_arn)

        # Buscar modelo atualmente ativo (se existir)
        current_active_arn: str | None = None
        try:
            current_active = self.get_active_model()
            if current_active:
                current_active_arn = current_active.model_package_arn
        except NoApprovedModelError:
            current_active_arn = None

        # Aprovar modelo alvo
        self.approve_model(target_model_package_arn)

        # Rejeitar modelo anterior (se diferente do alvo)
        if (
            current_active_arn
            and current_active_arn != target_model_package_arn
        ):
            self.reject_model(current_active_arn)
            logger.info(
                f"Rollback completo: "
                f"{current_active_arn} → {target_model_package_arn}"
            )
        else:
            logger.info(
                f"Rollback completo: ativado {target_model_package_arn}"
            )

    def delete_model(self, model_package_arn: str) -> None:
        """Deleta um modelo do registry.

        Verifica se o modelo foi usado em predições de produção antes
        de permitir a deleção. Modelos usados em produção NÃO podem
        ser deletados (R15.6).

        Args:
            model_package_arn: ARN do Model Package a deletar.

        Raises:
            ModelInProductionError: Se modelo foi usado em produção.
            ModelNotFoundError: Se o modelo não existe.
        """
        logger.info(
            f"Solicitação de deleção: {model_package_arn}"
        )

        self._validate_model_exists(model_package_arn)

        # Verificar se modelo está em uso em produção
        if self.is_model_in_production(model_package_arn):
            logger.error(
                f"Deleção bloqueada - modelo em produção: "
                f"{model_package_arn}"
            )
            raise ModelInProductionError(model_package_arn)

        # Deletar modelo do registry
        self._sagemaker_client.delete_model_package(
            ModelPackageName=model_package_arn
        )

        logger.info(f"Modelo deletado com sucesso: {model_package_arn}")

    def is_model_in_production(
        self, model_package_arn: str
    ) -> bool:
        """Verifica se o modelo foi usado em predições de produção.

        Consulta a tabela de predições (DynamoDB) para verificar se
        existem registros que referenciam este modelo.

        Args:
            model_package_arn: ARN do Model Package a verificar.

        Returns:
            True se o modelo foi usado em produção, False caso contrário.
        """
        logger.debug(
            f"Verificando uso em produção: {model_package_arn}"
        )

        table = self._dynamodb_resource.Table(
            self.predictions_table_name
        )

        # Scan com filtro pelo model_version
        # (em produção, usar GSI para eficiência)
        response = table.scan(
            FilterExpression="model_version = :mv",
            ExpressionAttributeValues={
                ":mv": model_package_arn,
            },
            Limit=1,
            Select="COUNT",
        )

        count = response.get("Count", 0)
        in_production = count > 0

        logger.debug(
            f"Modelo {model_package_arn}: "
            f"{'em produção' if in_production else 'não utilizado'}"
        )
        return in_production

    def get_model_artifacts_s3_path(
        self, model_package_arn: str
    ) -> str:
        """Retorna o path S3 dos artefatos do modelo.

        Args:
            model_package_arn: ARN do Model Package.

        Returns:
            S3 URI do artefato do modelo (model.tar.gz).

        Raises:
            ModelNotFoundError: Se o modelo não existe.
        """
        detail = self._get_model_package_detail(model_package_arn)

        inference_spec = detail.get("InferenceSpecification", {})
        containers = inference_spec.get("Containers", [])

        if containers:
            return containers[0].get("ModelDataUrl", "")

        return ""

    def upload_model_artifacts(
        self,
        model_version_id: str,
        model_data: bytes,
        hyperparameters: dict[str, str],
        metrics: dict[str, float],
    ) -> str:
        """Armazena artefatos do modelo em S3.

        Salva model.tar.gz, hyperparameters.json e metrics.json na
        estrutura definida: s3://bucket/models/{version_id}/

        Args:
            model_version_id: Identificador da versão.
            model_data: Bytes do modelo serializado (tar.gz).
            hyperparameters: Parâmetros de treino.
            metrics: Métricas de avaliação.

        Returns:
            S3 URI do artefato model.tar.gz.
        """
        s3_prefix = f"models/{model_version_id}"

        # Upload model.tar.gz
        model_key = f"{s3_prefix}/model.tar.gz"
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=model_key,
            Body=model_data,
            ContentType="application/gzip",
        )

        # Upload hyperparameters.json
        hp_key = f"{s3_prefix}/hyperparameters.json"
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=hp_key,
            Body=json.dumps(
                hyperparameters, indent=2
            ).encode("utf-8"),
            ContentType="application/json",
        )

        # Upload metrics.json
        metrics_key = f"{s3_prefix}/metrics.json"
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=metrics_key,
            Body=json.dumps(metrics, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        model_s3_uri = f"s3://{self.bucket}/{model_key}"
        logger.info(
            f"Artefatos armazenados: {model_s3_uri} "
            f"(hyperparameters + metrics)"
        )
        return model_s3_uri

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _validate_model_exists(
        self, model_package_arn: str
    ) -> None:
        """Valida que um modelo existe no registry.

        Args:
            model_package_arn: ARN do Model Package.

        Raises:
            ModelNotFoundError: Se o modelo não existe.
        """
        try:
            self._sagemaker_client.describe_model_package(
                ModelPackageName=model_package_arn
            )
        except self._sagemaker_client.exceptions.ClientError:
            raise ModelNotFoundError(model_package_arn)
        except AttributeError:
            # Em testes com mock, exceptions pode não existir
            # Nesse caso, se describe_model_package não lançar exceção,
            # o modelo existe
            pass
        except Exception as e:
            error_code = ""
            if hasattr(e, "response"):
                error_code = e.response.get("Error", {}).get(
                    "Code", ""
                )
            if error_code in (
                "ValidationException",
                "ResourceNotFoundException",
            ):
                raise ModelNotFoundError(model_package_arn)
            raise

    def _get_model_package_detail(
        self, model_package_arn: str
    ) -> dict[str, Any]:
        """Obtém detalhes completos de um model package.

        Args:
            model_package_arn: ARN do Model Package.

        Returns:
            Dict com detalhes do model package.

        Raises:
            ModelNotFoundError: Se o modelo não existe.
        """
        self._validate_model_exists(model_package_arn)
        return self._sagemaker_client.describe_model_package(
            ModelPackageName=model_package_arn
        )

    def _describe_model_version(
        self, model_package_arn: str
    ) -> ModelVersion | None:
        """Descreve um model package e retorna como ModelVersion.

        Args:
            model_package_arn: ARN do Model Package.

        Returns:
            ModelVersion ou None se não for possível descrever.
        """
        try:
            detail = self._sagemaker_client.describe_model_package(
                ModelPackageName=model_package_arn
            )
        except Exception as e:
            logger.warning(
                f"Erro ao descrever modelo {model_package_arn}: {e}"
            )
            return None

        custom_metadata = detail.get(
            "CustomerMetadataProperties", {}
        )
        if not isinstance(custom_metadata, dict):
            custom_metadata = {}

        # Extrair métricas
        metrics = {
            "precision": float(
                custom_metadata.get("precision", 0.0)
            ),
            "recall": float(
                custom_metadata.get("recall", 0.0)
            ),
            "f1": float(custom_metadata.get("f1", 0.0)),
            "roc_auc": float(
                custom_metadata.get("roc_auc", 0.0)
            ),
        }

        # Extrair algoritmo
        algorithm = custom_metadata.get("algorithm", "xgboost")

        # Extrair training_date
        training_date = custom_metadata.get(
            "training_date",
            datetime.now(timezone.utc).isoformat(),
        )

        # Extrair dataset_version
        dataset_version = custom_metadata.get(
            "dataset_version", "unknown"
        )

        return ModelVersion(
            model_package_arn=model_package_arn,
            algorithm=algorithm,
            training_date=training_date,
            dataset_version=dataset_version,
            metrics=metrics,
        )
