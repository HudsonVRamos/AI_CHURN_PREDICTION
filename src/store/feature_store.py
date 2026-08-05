"""Feature Store para armazenamento versionado de Feature Vectors no DynamoDB.

Implementa armazenamento imutável com versionamento auto-increment por usuário.
Cada Feature Vector recebe uma versão única e nunca é sobrescrito.

Schema DynamoDB:
    PK: user_id (String)
    SK: version (Number) - auto-increment por user
    Attributes: generated_at, observation_start, observation_end, features (Map)

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key, Attr

from src.common.logging import get_logger
from src.common.models import FeatureVector

logger = get_logger("feature-engineering")


class FeatureStore:
    """Armazenamento versionado de Feature Vectors no DynamoDB.

    Garante imutabilidade: versões armazenadas nunca são sobrescritas.
    Versionamento auto-increment por user_id.

    Args:
        table_name: Nome da tabela DynamoDB (default: churn_feature_store).
        dynamodb_resource: Recurso boto3 DynamoDB (para injeção em testes).
    """

    TABLE_NAME = "churn_feature_store"

    def __init__(
        self,
        table_name: str | None = None,
        dynamodb_resource: Any = None,
    ) -> None:
        self._table_name = table_name or self.TABLE_NAME
        self._dynamodb = dynamodb_resource or boto3.resource("dynamodb")
        self._table = self._dynamodb.Table(self._table_name)

    def store(self, feature_vector: FeatureVector) -> int:
        """Armazena feature vector e retorna a versão atribuída.

        A versão é auto-incrementada por user_id. Usa ConditionExpression
        para garantir imutabilidade (nunca sobrescreve versão existente).

        Args:
            feature_vector: FeatureVector a ser armazenado.

        Returns:
            Versão atribuída ao feature vector armazenado.

        Raises:
            RuntimeError: Se não conseguir armazenar após retries de conflito.
        """
        user_id = feature_vector.user_id
        max_retries = 3

        for attempt in range(max_retries):
            next_version = self._get_next_version(user_id)
            item = self._serialize(feature_vector, next_version)

            try:
                self._table.put_item(
                    Item=item,
                    ConditionExpression="attribute_not_exists(user_id) AND attribute_not_exists(version)",
                )
                logger.info(
                    f"Feature vector armazenado: user_id={user_id}, version={next_version}",
                    extra={"user_id": user_id, "version": next_version},
                )
                return next_version
            except self._dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
                logger.warning(
                    f"Conflito de versão (tentativa {attempt + 1}/{max_retries}): "
                    f"user_id={user_id}, version={next_version}",
                    extra={"user_id": user_id, "version": next_version},
                )
                continue

        raise RuntimeError(
            f"Falha ao armazenar feature vector após {max_retries} tentativas "
            f"para user_id={user_id}"
        )

    def get_latest(self, user_id: str) -> FeatureVector | None:
        """Retorna a versão mais recente das features de um user.

        Args:
            user_id: ID do assinante.

        Returns:
            FeatureVector mais recente ou None se não existir.
        """
        response = self._table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,
            Limit=1,
        )

        items = response.get("Items", [])
        if not items:
            logger.info(
                f"Nenhum feature vector encontrado para user_id={user_id}",
                extra={"user_id": user_id},
            )
            return None

        return self._deserialize(items[0])

    def get_version(self, user_id: str, version: int) -> FeatureVector | None:
        """Retorna uma versão específica do feature vector.

        Args:
            user_id: ID do assinante.
            version: Número da versão desejada.

        Returns:
            FeatureVector da versão especificada ou None se não existir.
        """
        response = self._table.get_item(
            Key={"user_id": user_id, "version": version}
        )

        item = response.get("Item")
        if not item:
            logger.info(
                f"Versão não encontrada: user_id={user_id}, version={version}",
                extra={"user_id": user_id, "version": version},
            )
            return None

        return self._deserialize(item)

    def get_history(
        self, user_id: str, from_date: str | None = None
    ) -> list[FeatureVector]:
        """Retorna todas as versões de um user com filtro de data opcional.

        Args:
            user_id: ID do assinante.
            from_date: Data mínima (ISO 8601) para filtrar resultados.
                       Retorna apenas versões geradas a partir desta data.

        Returns:
            Lista de FeatureVectors ordenada por versão ascendente.
        """
        query_kwargs: dict[str, Any] = {
            "KeyConditionExpression": Key("user_id").eq(user_id),
            "ScanIndexForward": True,
        }

        if from_date:
            query_kwargs["FilterExpression"] = Attr("generated_at").gte(from_date)

        items: list[dict[str, Any]] = []
        last_evaluated_key = None

        while True:
            if last_evaluated_key:
                query_kwargs["ExclusiveStartKey"] = last_evaluated_key

            response = self._table.query(**query_kwargs)
            items.extend(response.get("Items", []))
            last_evaluated_key = response.get("LastEvaluatedKey")

            if not last_evaluated_key:
                break

        logger.info(
            f"Histórico recuperado: user_id={user_id}, total={len(items)}",
            extra={"user_id": user_id, "total": len(items), "from_date": from_date},
        )

        return [self._deserialize(item) for item in items]

    def _get_next_version(self, user_id: str) -> int:
        """Obtém a próxima versão disponível para um user_id.

        Consulta a versão mais recente e incrementa em 1.
        Se não existir nenhuma versão, retorna 1.
        """
        response = self._table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,
            Limit=1,
            ProjectionExpression="version",
        )

        items = response.get("Items", [])
        if not items:
            return 1

        current_max = int(items[0]["version"])
        return current_max + 1

    def _serialize(self, fv: FeatureVector, version: int) -> dict[str, Any]:
        """Serializa um FeatureVector para formato compatível com DynamoDB.

        Armazena as features comportamentais em um Map separado
        para facilitar queries e manter o schema limpo.
        """
        features = {
            "total_sessions": fv.total_sessions,
            "total_viewing_hours": Decimal(str(fv.total_viewing_hours)),
            "avg_session_duration_min": Decimal(str(fv.avg_session_duration_min)),
            "sessions_per_week": Decimal(str(fv.sessions_per_week)),
            "distinct_channels": fv.distinct_channels,
            "avg_happiness_score": Decimal(str(fv.avg_happiness_score)),
            "avg_buffer_ratio": Decimal(str(fv.avg_buffer_ratio)),
            "error_rate": Decimal(str(fv.error_rate)),
            "avg_bitrate": Decimal(str(fv.avg_bitrate)),
            "pct_episode": Decimal(str(fv.pct_episode)),
            "pct_sport": Decimal(str(fv.pct_sport)),
            "pct_live": Decimal(str(fv.pct_live)),
            "pct_show": Decimal(str(fv.pct_show)),
            "distinct_devices": fv.distinct_devices,
            "avg_pause_count": Decimal(str(fv.avg_pause_count)),
            "avg_seek_count": Decimal(str(fv.avg_seek_count)),
        }

        # Trends podem ser None
        if fv.viewing_time_trend is not None:
            features["viewing_time_trend"] = Decimal(str(fv.viewing_time_trend))
        if fv.error_rate_trend is not None:
            features["error_rate_trend"] = Decimal(str(fv.error_rate_trend))
        if fv.session_frequency_trend is not None:
            features["session_frequency_trend"] = Decimal(
                str(fv.session_frequency_trend)
            )

        return {
            "user_id": fv.user_id,
            "version": version,
            "generated_at": fv.generated_at,
            "observation_start": fv.observation_start,
            "observation_end": fv.observation_end,
            "features": features,
        }

    def _deserialize(self, item: dict[str, Any]) -> FeatureVector:
        """Desserializa um item DynamoDB para FeatureVector.

        Converte Decimal para float/int conforme o campo.
        """
        features = item.get("features", {})

        return FeatureVector(
            user_id=item["user_id"],
            version=int(item["version"]),
            generated_at=item["generated_at"],
            observation_start=item["observation_start"],
            observation_end=item["observation_end"],
            # Engagement
            total_sessions=int(features.get("total_sessions", 0)),
            total_viewing_hours=float(features.get("total_viewing_hours", 0)),
            avg_session_duration_min=float(
                features.get("avg_session_duration_min", 0)
            ),
            sessions_per_week=float(features.get("sessions_per_week", 0)),
            distinct_channels=int(features.get("distinct_channels", 0)),
            # Quality
            avg_happiness_score=float(features.get("avg_happiness_score", 0)),
            avg_buffer_ratio=float(features.get("avg_buffer_ratio", 0)),
            error_rate=float(features.get("error_rate", 0)),
            avg_bitrate=float(features.get("avg_bitrate", 0)),
            # Behavioral
            pct_episode=float(features.get("pct_episode", 0)),
            pct_sport=float(features.get("pct_sport", 0)),
            pct_live=float(features.get("pct_live", 0)),
            pct_show=float(features.get("pct_show", 0)),
            distinct_devices=int(features.get("distinct_devices", 0)),
            avg_pause_count=float(features.get("avg_pause_count", 0)),
            avg_seek_count=float(features.get("avg_seek_count", 0)),
            # Trends
            viewing_time_trend=_decimal_to_float_or_none(
                features.get("viewing_time_trend")
            ),
            error_rate_trend=_decimal_to_float_or_none(
                features.get("error_rate_trend")
            ),
            session_frequency_trend=_decimal_to_float_or_none(
                features.get("session_frequency_trend")
            ),
        )


def _decimal_to_float_or_none(value: Any) -> float | None:
    """Converte Decimal para float, retornando None se o valor for None."""
    if value is None:
        return None
    return float(value)
