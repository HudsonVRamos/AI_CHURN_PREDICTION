"""Data source do dashboard analítico.

Lê dados de execuções, predições e features do DynamoDB e S3
para alimentar o dashboard interativo.

Suporta filtros por período (date range), risk_level e user_id.

Requirements: 14.1, 14.3, 14.4, 14.6
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr, Key

from src.common.logging import get_logger
from src.common.models import ExecutionSummary

logger = get_logger("dashboard")


class DashboardDataSource:
    """Lê dados de predição do DynamoDB e S3 para exibição no dashboard.

    Conecta-se às tabelas:
    - churn_executions: resumo de execuções do pipeline
    - churn_predictions: resultados de predição por assinante
    - churn_feature_store: features versionadas dos assinantes

    Args:
        dynamodb_resource: Recurso boto3 DynamoDB (injetável para testes).
        s3_client: Cliente boto3 S3 (injetável para testes).
        executions_table: Nome da tabela de execuções.
        predictions_table: Nome da tabela de predições.
        feature_store_table: Nome da tabela de feature store.
    """

    DEFAULT_EXECUTIONS_TABLE = "churn_executions"
    DEFAULT_PREDICTIONS_TABLE = "churn_predictions"
    DEFAULT_FEATURE_STORE_TABLE = "churn_feature_store"

    def __init__(
        self,
        dynamodb_resource: Any = None,
        s3_client: Any = None,
        executions_table: str | None = None,
        predictions_table: str | None = None,
        feature_store_table: str | None = None,
    ) -> None:
        self._dynamodb = dynamodb_resource or boto3.resource("dynamodb")
        self._s3 = s3_client or boto3.client("s3")
        self._executions_table = self._dynamodb.Table(
            executions_table or self.DEFAULT_EXECUTIONS_TABLE
        )
        self._predictions_table = self._dynamodb.Table(
            predictions_table or self.DEFAULT_PREDICTIONS_TABLE
        )
        self._feature_store_table = self._dynamodb.Table(
            feature_store_table or self.DEFAULT_FEATURE_STORE_TABLE
        )

    def get_latest_execution(self) -> ExecutionSummary | None:
        """Retorna o resumo da execução mais recente do pipeline.

        Faz scan na tabela churn_executions e retorna a execução com
        o start_time mais recente.

        Returns:
            ExecutionSummary da execução mais recente ou None se não houver.
        """
        logger.info("Buscando execução mais recente")

        items: list[dict[str, Any]] = []
        last_key = None

        while True:
            scan_kwargs: dict[str, Any] = {}
            if last_key:
                scan_kwargs["ExclusiveStartKey"] = last_key

            response = self._executions_table.scan(**scan_kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")

            if not last_key:
                break

        if not items:
            logger.info("Nenhuma execução encontrada")
            return None

        # Ordena por start_time descending e retorna a mais recente
        items.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        latest = items[0]

        execution = ExecutionSummary(
            execution_id=latest["execution_id"],
            start_time=latest["start_time"],
            end_time=latest["end_time"],
            mode=latest["mode"],
            model_version=latest["model_version"],
            users_processed=int(latest["users_processed"]),
            users_failed=int(latest["users_failed"]),
            status=latest["status"],
            output_s3_path=latest["output_s3_path"],
        )

        logger.info(
            f"Execução mais recente: {execution.execution_id}",
            extra={"execution_id": execution.execution_id},
        )
        return execution

    def get_predictions(
        self,
        risk_level: str | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> list[dict]:
        """Retorna lista de predições com suporte a filtros.

        Args:
            risk_level: Filtro por nível de risco (Low, Medium, High).
                       None ou "All" retorna todos.
            period_start: Data início do filtro (ISO 8601).
            period_end: Data fim do filtro (ISO 8601).

        Returns:
            Lista de dicts com dados das predições filtradas.
        """
        logger.info(
            "Buscando predições",
            extra={
                "risk_level": risk_level,
                "period_start": period_start,
                "period_end": period_end,
            },
        )

        # Monta FilterExpression dinâmico
        filter_expressions = []

        if risk_level and risk_level != "All":
            filter_expressions.append(Attr("risk_tier").eq(risk_level))

        if period_start:
            filter_expressions.append(Attr("timestamp").gte(period_start))

        if period_end:
            filter_expressions.append(Attr("timestamp").lte(period_end))

        scan_kwargs: dict[str, Any] = {}
        if filter_expressions:
            combined = filter_expressions[0]
            for expr in filter_expressions[1:]:
                combined = combined & expr
            scan_kwargs["FilterExpression"] = combined

        items: list[dict[str, Any]] = []
        last_key = None

        while True:
            if last_key:
                scan_kwargs["ExclusiveStartKey"] = last_key

            response = self._predictions_table.scan(**scan_kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")

            if not last_key:
                break

        predictions = [self._deserialize_prediction(item) for item in items]

        logger.info(
            f"Predições retornadas: {len(predictions)}",
            extra={"count": len(predictions)},
        )
        return predictions

    def get_subscriber_detail(self, user_id: str) -> dict | None:
        """Retorna detalhes completos de um assinante específico.

        Combina dados da predição mais recente com features mais recentes
        e explicação do Bedrock.

        Args:
            user_id: ID do assinante para buscar.

        Returns:
            Dict com dados completos do assinante ou None se não encontrado.
        """
        logger.info(
            f"Buscando detalhe do assinante: {user_id}",
            extra={"user_id": user_id},
        )

        # Busca predição mais recente para este user
        prediction = self._get_latest_prediction_for_user(user_id)
        if not prediction:
            logger.info(
                f"Nenhuma predição encontrada para user_id={user_id}",
                extra={"user_id": user_id},
            )
            return None

        # Busca features mais recentes
        features = self._get_latest_features_for_user(user_id)

        detail = {
            "user_id": user_id,
            "churn_probability": float(prediction.get("churn_probability", 0)),
            "confidence": float(prediction.get("confidence", 0)),
            "risk_tier": prediction.get("risk_tier", ""),
            "model_version": prediction.get("model_version", ""),
            "timestamp": prediction.get("timestamp", ""),
            "shap_results": prediction.get("shap_results", {}),
            "bedrock_explanation": prediction.get("bedrock_explanation"),
            "explanation_status": prediction.get(
                "explanation_status", "unavailable"
            ),
            "features": features,
        }

        logger.info(
            f"Detalhe encontrado para user_id={user_id}",
            extra={"user_id": user_id, "risk_tier": detail["risk_tier"]},
        )
        return detail

    def get_history(self, user_id: str) -> list[dict]:
        """Retorna histórico de predições e features de um assinante.

        Busca todas as predições feitas para o user ao longo do tempo,
        permitindo visualizar a evolução temporal.

        Args:
            user_id: ID do assinante.

        Returns:
            Lista de dicts com histórico ordenado por timestamp.
        """
        logger.info(
            f"Buscando histórico do assinante: {user_id}",
            extra={"user_id": user_id},
        )

        # Busca todas as predições para este user (scan com filtro)
        filter_expr = Attr("user_id").eq(user_id)

        items: list[dict[str, Any]] = []
        last_key = None

        while True:
            scan_kwargs: dict[str, Any] = {
                "FilterExpression": filter_expr,
            }
            if last_key:
                scan_kwargs["ExclusiveStartKey"] = last_key

            response = self._predictions_table.scan(**scan_kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")

            if not last_key:
                break

        # Ordena por timestamp ascendente
        items.sort(key=lambda x: x.get("timestamp", ""))

        history = [self._deserialize_prediction(item) for item in items]

        logger.info(
            f"Histórico retornado: {len(history)} registros",
            extra={"user_id": user_id, "count": len(history)},
        )
        return history

    def _get_latest_prediction_for_user(
        self, user_id: str
    ) -> dict[str, Any] | None:
        """Busca a predição mais recente para um user_id.

        Como a tabela churn_predictions tem PK=execution_id e SK=user_id,
        precisamos fazer scan com filtro para encontrar todas as predições
        de um user e selecionar a mais recente.
        """
        filter_expr = Attr("user_id").eq(user_id)

        items: list[dict[str, Any]] = []
        last_key = None

        while True:
            scan_kwargs: dict[str, Any] = {
                "FilterExpression": filter_expr,
            }
            if last_key:
                scan_kwargs["ExclusiveStartKey"] = last_key

            response = self._predictions_table.scan(**scan_kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")

            if not last_key:
                break

        if not items:
            return None

        # Retorna a mais recente por timestamp
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return items[0]

    def _get_latest_features_for_user(
        self, user_id: str
    ) -> dict[str, Any] | None:
        """Busca as features mais recentes de um user no Feature Store."""
        response = self._feature_store_table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,
            Limit=1,
        )

        items = response.get("Items", [])
        if not items:
            return None

        item = items[0]
        features = item.get("features", {})

        # Converte Decimal para float
        return {
            k: float(v) if isinstance(v, Decimal) else v
            for k, v in features.items()
        }

    def _deserialize_prediction(self, item: dict[str, Any]) -> dict:
        """Desserializa um item de predição do DynamoDB para dict."""
        shap_results = item.get("shap_results", {})
        # Converte Decimals em shap_results
        if shap_results and isinstance(shap_results, dict):
            shap_results = {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in shap_results.items()
            }

        return {
            "execution_id": item.get("execution_id", ""),
            "user_id": item.get("user_id", ""),
            "churn_probability": float(item.get("churn_probability", 0)),
            "confidence": float(item.get("confidence", 0)),
            "risk_tier": item.get("risk_tier", ""),
            "model_version": item.get("model_version", ""),
            "feature_version": int(item.get("feature_version", 0)),
            "timestamp": item.get("timestamp", ""),
            "bedrock_explanation": item.get("bedrock_explanation"),
            "explanation_status": item.get(
                "explanation_status", "unavailable"
            ),
        }
