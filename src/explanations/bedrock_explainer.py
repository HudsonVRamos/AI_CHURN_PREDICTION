"""Gerador de explicações em linguagem natural via AWS Bedrock.

Utiliza Claude 3 Haiku para gerar explicações em PT-BR sobre predições
de churn, com base nos dados fornecidos pelo Explainability_Engine.

O modelo NÃO calcula probabilidade de churn — apenas explica os fatores
que contribuíram para a classificação já realizada pelo modelo de ML.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""

from __future__ import annotations

import json
import time
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ReadTimeoutError,
)

from src.common.logging import get_logger
from src.common.models import FeatureContribution

logger = get_logger("bedrock-explanation")


class BedrockExplainer:
    """Gera explicações em linguagem natural via AWS Bedrock.

    Utiliza Claude 3 Haiku para produzir explicações objetivas e factuais
    em português brasileiro sobre predições de churn.

    Attributes:
        MODEL_ID: ID do modelo Claude 3 Haiku no Bedrock.
        TIMEOUT_SECONDS: Timeout máximo para chamada ao Bedrock (60s).
        MAX_RETRIES: Número máximo de retentativas em caso de falha.
        RETRY_INTERVAL_SECONDS: Intervalo entre retentativas (5s).
    """

    MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
    TIMEOUT_SECONDS = 60
    MAX_RETRIES = 2
    RETRY_INTERVAL_SECONDS = 5

    def __init__(
        self,
        region_name: str = "us-east-1",
        client: Any | None = None,
    ) -> None:
        """Inicializa o explainer com cliente Bedrock.

        Args:
            region_name: Região AWS onde o Bedrock está disponível.
            client: Cliente boto3 bedrock-runtime opcional (para testes).
        """
        if client is not None:
            self._client = client
        else:
            config = Config(
                read_timeout=self.TIMEOUT_SECONDS,
                connect_timeout=self.TIMEOUT_SECONDS,
                retries={"max_attempts": 0},
            )
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=region_name,
                config=config,
            )

    def generate_explanation(
        self,
        user_id: str,
        churn_probability: float,
        confidence: float,
        top_features: list[FeatureContribution],
        user_feature_values: dict,
        population_stats: dict,
    ) -> str | None:
        """Gera explicação em PT-BR sobre a predição de churn.

        Envia prompt estruturado ao Bedrock com os dados da predição
        e retorna a explicação gerada. Em caso de falha após retentativas,
        retorna None (graceful degradation).

        Args:
            user_id: ID do assinante.
            churn_probability: Probabilidade de churn (0.0-1.0).
            confidence: Grau de confiança da predição (0.0-1.0).
            top_features: Lista de features com maior contribuição.
            user_feature_values: Valores das features do assinante.
            population_stats: Estatísticas populacionais por feature
                (mean, std para cada feature relevante).

        Returns:
            Explicação em português brasileiro ou None se indisponível.

        Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
        """
        prompt = self._build_prompt(
            user_id=user_id,
            churn_probability=churn_probability,
            confidence=confidence,
            top_features=top_features,
            user_feature_values=user_feature_values,
            population_stats=population_stats,
        )

        logger.info(
            "Gerando explicação via Bedrock",
            extra={"user_id": user_id},
        )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self._invoke_model(prompt)
                logger.info(
                    "Explicação gerada com sucesso",
                    extra={
                        "user_id": user_id,
                        "attempt": attempt,
                    },
                )
                return response

            except (ReadTimeoutError, ClientError, BotoCoreError) as e:
                msg = (
                    f"Falha ao invocar Bedrock "
                    f"(tentativa {attempt}/{self.MAX_RETRIES})"
                )
                logger.warning(
                    msg,
                    extra={
                        "user_id": user_id,
                        "attempt": attempt,
                        "error": str(e),
                    },
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_INTERVAL_SECONDS)

            except Exception as e:
                logger.error(
                    "Erro inesperado ao invocar Bedrock",
                    extra={
                        "user_id": user_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_INTERVAL_SECONDS)

        logger.error(
            "Bedrock indisponível após todas as tentativas",
            extra={
                "user_id": user_id,
                "max_retries": self.MAX_RETRIES,
            },
        )
        return None

    def _invoke_model(self, prompt: str) -> str:
        """Invoca o modelo Claude 3 Haiku no Bedrock.

        Args:
            prompt: Prompt formatado para o modelo.

        Returns:
            Texto da resposta gerada pelo modelo.

        Raises:
            ClientError: Erro na chamada à API do Bedrock.
            ReadTimeoutError: Timeout na leitura da resposta.
            BotoCoreError: Erro genérico do boto3.
        """
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.3,
        })

        response = self._client.invoke_model(
            modelId=self.MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]

    def _build_prompt(
        self,
        user_id: str,
        churn_probability: float,
        confidence: float,
        top_features: list[FeatureContribution],
        user_feature_values: dict,
        population_stats: dict,
    ) -> str:
        """Monta o prompt estruturado em PT-BR para o Bedrock.

        O prompt instrui o modelo a:
        1. Explicar em português brasileiro
        2. Usar APENAS os dados fornecidos
        3. NÃO calcular probabilidade de churn
        4. Comparar o assinante com a população
        5. Ser objetivo e factual

        Args:
            user_id: ID do assinante.
            churn_probability: Probabilidade de churn.
            confidence: Grau de confiança.
            top_features: Features com maior contribuição.
            user_feature_values: Valores das features do assinante.
            population_stats: Estatísticas populacionais.

        Returns:
            Prompt formatado para envio ao Bedrock.
        """
        risk_tier = self._classify_risk(churn_probability)

        # Formatar contribuições das features
        features_text = self._format_features(
            top_features, user_feature_values, population_stats
        )

        prompt = (
            "Você é um analista de dados especializado em análise "
            "de churn de assinantes de streaming. Sua tarefa é gerar "
            "uma explicação clara e objetiva sobre os fatores que "
            "contribuíram para a classificação de risco de um "
            "assinante.\n\n"
            "REGRAS OBRIGATÓRIAS:\n"
            "- Responda EXCLUSIVAMENTE em português brasileiro "
            "(PT-BR)\n"
            "- Use APENAS as informações fornecidas abaixo\n"
            "- NÃO calcule, recalcule ou questione a probabilidade "
            "de churn\n"
            "- NÃO altere ou questione o resultado do modelo de "
            "Machine Learning\n"
            "- A probabilidade de churn foi calculada por um modelo "
            "de ML e é DEFINITIVA\n"
            "- Seja objetivo, factual e conciso\n"
            "- Compare os valores do assinante com a média da "
            "população\n\n"
            "DADOS DA PREDIÇÃO:\n"
            f"- ID do Assinante: {user_id}\n"
            f"- Probabilidade de Churn: {churn_probability:.1%}\n"
            f"- Grau de Confiança: {confidence:.1%}\n"
            f"- Classificação de Risco: {risk_tier}\n\n"
            "FATORES QUE CONTRIBUÍRAM PARA A CLASSIFICAÇÃO:\n"
            f"{features_text}\n\n"
            "INSTRUÇÕES DE FORMATO:\n"
            "- Gere um resumo executivo de 2 a 4 parágrafos\n"
            "- Primeiro parágrafo: visão geral da situação do "
            "assinante\n"
            "- Parágrafos seguintes: detalhamento dos fatores mais "
            "relevantes, comparando com a população\n"
            "- Não use listas com marcadores; use texto corrido\n"
            "- Não inclua cabeçalhos ou títulos"
        )

        return prompt

    def _format_features(
        self,
        top_features: list[FeatureContribution],
        user_feature_values: dict,
        population_stats: dict,
    ) -> str:
        """Formata as features para inclusão no prompt.

        Para cada feature, mostra: nome, contribuição, valor do
        assinante e comparação com a população (média e desvio padrão).

        Args:
            top_features: Features com maior contribuição.
            user_feature_values: Valores das features do assinante.
            population_stats: Estatísticas populacionais.

        Returns:
            Texto formatado com as features.
        """
        lines = []
        for feature in top_features:
            name = feature.feature_name
            weight = feature.contribution_weight
            impact = feature.normalized_impact

            user_value = user_feature_values.get(name, "N/A")
            stats = population_stats.get(name, {})
            pop_mean = stats.get("mean", "N/A")
            pop_std = stats.get("std", "N/A")

            direction = "aumenta" if weight > 0 else "diminui"

            line = (
                f"- {name}: contribuição {direction} risco de churn "
                f"(peso: {weight:.4f}, impacto normalizado: {impact:.4f})\n"
                f"  Valor do assinante: {user_value}\n"
                f"  Média da população: {pop_mean}"
            )
            if pop_std != "N/A":
                line += f" (desvio padrão: {pop_std})"

            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _classify_risk(churn_probability: float) -> str:
        """Classifica o risco com base na probabilidade de churn.

        Args:
            churn_probability: Probabilidade de churn (0.0-1.0).

        Returns:
            Classificação: 'Baixo', 'Médio' ou 'Alto'.
        """
        if churn_probability <= 0.30:
            return "Baixo"
        elif churn_probability <= 0.60:
            return "Médio"
        else:
            return "Alto"
