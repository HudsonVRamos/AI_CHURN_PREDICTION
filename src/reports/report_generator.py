"""Gerador de relatórios individuais e executivos de churn.

Produz relatórios em JSON e Markdown, com upload para S3.
Inclui metadados de versão, período de análise e timestamp ISO 8601.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 13.1, 13.2, 13.3, 13.4, 13.5
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from src.common.logging import get_logger
from src.common.models import ExplainabilityResult, PredictionResult

logger = get_logger("report-generation")


class ReportGenerator:
    """Gera relatórios individuais e executivos em JSON e Markdown.

    Parâmetros do construtor permitem injeção de dependências
    para facilitar testes (S3 client e bucket name).
    """

    def __init__(
        self,
        s3_client: Any | None = None,
        bucket: str = "sky-brazil-churn-prediction",
        model_version: str = "unknown",
        feature_version: str = "unknown",
        analysis_period_start: str | None = None,
        analysis_period_end: str | None = None,
    ) -> None:
        """Inicializa o ReportGenerator.

        Args:
            s3_client: Cliente boto3 S3 (opcional, None desabilita upload).
            bucket: Nome do bucket S3 para upload dos relatórios.
            model_version: Versão do modelo ML utilizado.
            feature_version: Versão das features utilizadas.
            analysis_period_start: Início do período de análise (ISO 8601).
            analysis_period_end: Fim do período de análise (ISO 8601).
        """
        self._s3_client = s3_client
        self._bucket = bucket
        self._model_version = model_version
        self._feature_version = feature_version
        self._analysis_period_start = analysis_period_start
        self._analysis_period_end = analysis_period_end

    def _build_metadata(self) -> dict[str, str]:
        """Constrói metadados padrão para inclusão em todos os relatórios."""
        return {
            "model_version": self._model_version,
            "feature_version": self._feature_version,
            "analysis_period_start": self._analysis_period_start or "",
            "analysis_period_end": self._analysis_period_end or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def generate_individual_report(
        self,
        user_id: str,
        prediction: PredictionResult,
        explainability: ExplainabilityResult | None,
        explanation: str | None,
    ) -> dict:
        """Gera relatório individual para um assinante.

        Validates: Requirements 13.1, 13.4, 13.5

        Args:
            user_id: ID do assinante.
            prediction: Resultado da predição de churn.
            explainability: Resultado do SHAP (pode ser None).
            explanation: Explicação em linguagem natural do Bedrock.

        Returns:
            Dicionário com o relatório individual completo.
        """
        logger.info(
            f"Gerando relatório individual para user_id={user_id}"
        )

        top_features = []
        if explainability is not None:
            top_features = [
                {
                    "feature_name": fc.feature_name,
                    "contribution_weight": fc.contribution_weight,
                    "normalized_impact": fc.normalized_impact,
                }
                for fc in explainability.top_features
            ]

        # Determinar status da explicação (R13.5)
        explanation_status = "available" if explanation else "unavailable"

        report = {
            "report_type": "individual",
            "user_id": user_id,
            "churn_probability": prediction.churn_probability,
            "confidence": prediction.confidence,
            "risk_tier": prediction.risk_tier,
            "top_features": top_features,
            "explanation": explanation,
            "explanation_status": explanation_status,
            "metadata": self._build_metadata(),
        }

        logger.info(
            f"Relatório individual gerado: user_id={user_id}, "
            f"risk_tier={prediction.risk_tier}"
        )
        return report

    def generate_executive_report(
        self,
        predictions: list[PredictionResult],
        explainabilities: list[ExplainabilityResult],
    ) -> dict:
        """Gera relatório executivo agregado.

        Validates: Requirements 6.1, 6.2, 6.5, 6.6, 13.2, 13.4

        Args:
            predictions: Lista de resultados de predição.
            explainabilities: Lista de resultados de explicabilidade.

        Returns:
            Dicionário com o relatório executivo completo.
        """
        logger.info(
            f"Gerando relatório executivo para {len(predictions)} usuários"
        )

        total_analyzed = len(predictions)

        # Distribuição por risk tier (R6.1, R13.2)
        tier_counter: Counter[str] = Counter()
        for pred in predictions:
            tier_counter[pred.risk_tier] += 1

        distribution = {}
        for tier in ["Low", "Medium", "High"]:
            count = tier_counter.get(tier, 0)
            pct = (count / total_analyzed * 100) if total_analyzed > 0 else 0.0
            distribution[tier] = {
                "count": count,
                "percentage": round(pct, 2),
            }

        # Média de churn probability (R13.2)
        avg_churn = 0.0
        if total_analyzed > 0:
            avg_churn = sum(
                p.churn_probability for p in predictions
            ) / total_analyzed

        # Top fatores populacionais agregados (R6.1, R13.2)
        feature_importance: Counter[str] = Counter()
        feature_weight_sum: dict[str, float] = {}
        for expl in explainabilities:
            for fc in expl.top_features:
                feature_importance[fc.feature_name] += 1
                feature_weight_sum[fc.feature_name] = (
                    feature_weight_sum.get(fc.feature_name, 0.0)
                    + abs(fc.contribution_weight)
                )

        # Ordenar por frequência e depois por peso total
        top_factors = sorted(
            feature_importance.keys(),
            key=lambda f: (
                feature_importance[f],
                feature_weight_sum.get(f, 0.0),
            ),
            reverse=True,
        )[:10]

        top_factors_detail = [
            {
                "feature_name": f,
                "occurrence_count": feature_importance[f],
                "total_weight": round(feature_weight_sum.get(f, 0.0), 4),
            }
            for f in top_factors
        ]

        # Lista de high risk users ordenada (R6.2, R6.6)
        high_risk_users = []
        high_risk_predictions = [
            p for p in predictions if p.risk_tier == "High"
        ]
        if high_risk_predictions:
            high_risk_sorted = sorted(
                high_risk_predictions,
                key=lambda p: p.churn_probability,
                reverse=True,
            )
            # Buscar explainability por user_id
            expl_map = {e.user_id: e for e in explainabilities}
            for pred in high_risk_sorted:
                user_entry: dict[str, Any] = {
                    "user_id": pred.user_id,
                    "churn_probability": pred.churn_probability,
                    "confidence": pred.confidence,
                    "risk_tier": pred.risk_tier,
                }
                expl = expl_map.get(pred.user_id)
                if expl:
                    user_entry["top_features"] = [
                        {
                            "feature_name": fc.feature_name,
                            "contribution_weight": fc.contribution_weight,
                            "normalized_impact": fc.normalized_impact,
                        }
                        for fc in expl.top_features[:5]
                    ]
                high_risk_users.append(user_entry)

        report = {
            "report_type": "executive",
            "total_analyzed": total_analyzed,
            "distribution": distribution,
            "average_churn_probability": round(avg_churn, 4),
            "top_population_factors": top_factors_detail,
            "high_risk_users": high_risk_users,
            "metadata": self._build_metadata(),
        }

        logger.info(
            f"Relatório executivo gerado: total={total_analyzed}, "
            f"high_risk={distribution['High']['count']}"
        )
        return report

    def export_json(self, report: dict, output_path: str) -> None:
        """Exporta relatório em formato JSON.

        Validates: Requirements 6.3, 13.3

        Args:
            report: Dicionário do relatório a ser exportado.
            output_path: Caminho local para salvar o arquivo JSON.
        """
        logger.info(f"Exportando relatório JSON: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Relatório JSON exportado: {output_path}")

    def export_markdown(self, report: dict, output_path: str) -> None:
        """Exporta relatório em formato Markdown legível.

        Validates: Requirements 6.4, 13.3

        Args:
            report: Dicionário do relatório a ser exportado.
            output_path: Caminho local para salvar o arquivo Markdown.
        """
        logger.info(f"Exportando relatório Markdown: {output_path}")

        report_type = report.get("report_type", "unknown")
        if report_type == "individual":
            content = self._render_individual_markdown(report)
        elif report_type == "executive":
            content = self._render_executive_markdown(report)
        else:
            content = self._render_generic_markdown(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Relatório Markdown exportado: {output_path}")

    def upload_to_s3(
        self,
        local_path: str,
        execution_id: str,
        filename: str,
    ) -> str | None:
        """Faz upload de arquivo para S3 na estrutura definida.

        Estrutura: s3://bucket/reports/{execution_id}/{filename}

        Args:
            local_path: Caminho local do arquivo a ser enviado.
            execution_id: ID da execução do pipeline.
            filename: Nome do arquivo no S3.

        Returns:
            S3 URI completo ou None se S3 não configurado.
        """
        if self._s3_client is None:
            logger.warning("S3 client não configurado, upload ignorado")
            return None

        s3_key = f"reports/{execution_id}/{filename}"
        logger.info(
            f"Uploading para s3://{self._bucket}/{s3_key}"
        )

        try:
            with open(local_path, "rb") as f:
                self._s3_client.put_object(
                    Bucket=self._bucket,
                    Key=s3_key,
                    Body=f.read(),
                )
            s3_uri = f"s3://{self._bucket}/{s3_key}"
            logger.info(f"Upload concluído: {s3_uri}")
            return s3_uri
        except Exception as e:
            logger.error(
                f"Falha no upload para S3: {e}",
                exc_info=True,
            )
            raise

    def _render_individual_markdown(self, report: dict) -> str:
        """Renderiza relatório individual em Markdown."""
        metadata = report.get("metadata", {})
        lines = [
            "# Relatório Individual de Churn",
            "",
            "## Informações do Assinante",
            "",
            "| Campo | Valor |",
            "|-------|-------|",
            f"| User ID | `{report.get('user_id', '')}` |",
            f"| Probabilidade de Churn | "
            f"{report.get('churn_probability', 0):.2%} |",
            f"| Confiança | {report.get('confidence', 0):.2%} |",
            f"| Classificação de Risco | "
            f"**{report.get('risk_tier', '')}** |",
            "",
        ]

        # Top features
        top_features = report.get("top_features", [])
        if top_features:
            lines.extend([
                "## Principais Fatores",
                "",
                "| Feature | Peso | Impacto Normalizado |",
                "|---------|------|---------------------|",
            ])
            for feat in top_features:
                name = feat.get("feature_name", "")
                weight = feat.get("contribution_weight", 0)
                impact = feat.get("normalized_impact", 0)
                direction = "↑ churn" if weight > 0 else "↓ churn"
                lines.append(
                    f"| {name} | {weight:+.4f} ({direction}) "
                    f"| {impact:+.4f} |"
                )
            lines.append("")

        # Explicação
        explanation = report.get("explanation")
        explanation_status = report.get("explanation_status", "unavailable")
        lines.append("## Explicação")
        lines.append("")
        if explanation_status == "available" and explanation:
            lines.append(explanation)
        else:
            lines.append(
                "*Explicação em linguagem natural indisponível "
                "para este assinante.*"
            )
        lines.append("")

        # Metadados
        lines.extend(self._render_metadata_section(metadata))

        return "\n".join(lines)

    def _render_executive_markdown(self, report: dict) -> str:
        """Renderiza relatório executivo em Markdown."""
        metadata = report.get("metadata", {})
        distribution = report.get("distribution", {})
        avg_churn = report.get("average_churn_probability", 0)
        total = report.get("total_analyzed", 0)

        lines = [
            "# Relatório Executivo de Churn",
            "",
            "## Resumo",
            "",
            f"- **Total de assinantes analisados:** {total}",
            f"- **Probabilidade média de churn:** {avg_churn:.2%}",
            "",
            "## Distribuição por Nível de Risco",
            "",
            "| Nível | Quantidade | Percentual |",
            "|-------|-----------|------------|",
        ]

        for tier in ["High", "Medium", "Low"]:
            tier_data = distribution.get(tier, {})
            count = tier_data.get("count", 0)
            pct = tier_data.get("percentage", 0)
            lines.append(f"| {tier} | {count} | {pct:.1f}% |")

        lines.append("")

        # Top fatores populacionais
        top_factors = report.get("top_population_factors", [])
        if top_factors:
            lines.extend([
                "## Principais Fatores na População",
                "",
                "| # | Feature | Ocorrências | Peso Total |",
                "|---|---------|-------------|------------|",
            ])
            for i, factor in enumerate(top_factors, 1):
                name = factor.get("feature_name", "")
                occ = factor.get("occurrence_count", 0)
                weight = factor.get("total_weight", 0)
                lines.append(f"| {i} | {name} | {occ} | {weight:.4f} |")
            lines.append("")

        # Lista de high risk
        high_risk = report.get("high_risk_users", [])
        if high_risk:
            lines.extend([
                "## Assinantes de Alto Risco",
                "",
                "| User ID | Prob. Churn | Confiança |",
                "|---------|-------------|-----------|",
            ])
            for user in high_risk:
                uid = user.get("user_id", "")
                prob = user.get("churn_probability", 0)
                conf = user.get("confidence", 0)
                lines.append(f"| `{uid}` | {prob:.2%} | {conf:.2%} |")
            lines.append("")
        else:
            lines.extend([
                "## Assinantes de Alto Risco",
                "",
                "*Nenhum assinante classificado como Alto Risco "
                "nesta execução.*",
                "",
            ])

        # Metadados
        lines.extend(self._render_metadata_section(metadata))

        return "\n".join(lines)

    def _render_metadata_section(self, metadata: dict) -> list[str]:
        """Renderiza seção de metadados comum a todos os relatórios."""
        return [
            "---",
            "",
            "## Metadados",
            "",
            f"- **Versão do Modelo:** {metadata.get('model_version', '')}",
            f"- **Versão das Features:** "
            f"{metadata.get('feature_version', '')}",
            f"- **Período:** {metadata.get('analysis_period_start', '')} "
            f"a {metadata.get('analysis_period_end', '')}",
            f"- **Timestamp:** {metadata.get('timestamp', '')}",
            "",
        ]

    def _render_generic_markdown(self, report: dict) -> str:
        """Renderiza relatório genérico como JSON formatado em Markdown."""
        lines = [
            "# Relatório",
            "",
            "```json",
            json.dumps(report, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
        return "\n".join(lines)
