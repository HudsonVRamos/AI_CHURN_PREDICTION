"""Lambda handler para o estágio de Geração de Relatórios do pipeline.

Responsável por gerar relatórios individuais e executivos em JSON
e Markdown, com upload para S3.

Requirements: 8.1, 8.3, 8.4, 17.3, 17.4
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from typing import Any

import boto3

from src.common.logging import get_logger, set_execution_id
from src.common.models import (
    ExplainabilityResult,
    FeatureContribution,
    PredictionResult,
)
from src.reports.report_generator import ReportGenerator


def handler(event: dict, context: Any) -> dict:
    """Lambda handler para o estágio de geração de relatórios.

    Gera relatórios executivos e individuais em JSON e Markdown.
    Faz upload dos relatórios para S3.

    Os dados de predição e explicabilidade já devem estar persistidos
    em estágios anteriores (R17.4). Este estágio é o último e sua
    falha não afeta os dados já armazenados.

    Args:
        event: Dicionário com:
            - execution_id: UUID da execução do pipeline.
            - predictions: Lista de PredictionResult serializados.
            - explainability_results: Lista de ExplainabilityResult.
            - explanations: Dict user_id -> texto de explicação.
            - model_version_used: Versão do modelo utilizada.
            - bucket: Nome do bucket S3.
        context: Contexto Lambda.

    Returns:
        Dicionário com event enriquecido com:
            - report_s3_paths: Dict com paths dos relatórios no S3.
            - stage_completed: "report-generation"
    """
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    set_execution_id(execution_id)
    logger = get_logger("report-generation")
    logger.log_stage_start()
    start_time = time.time()

    try:
        predictions_data = event.get("predictions", [])
        explainability_data = event.get("explainability_results", [])
        explanations = event.get("explanations", {})
        model_version = event.get("model_version_used", "unknown")
        bucket = event.get("bucket", "sky-brazil-churn-prediction")

        # Reconstruir objetos tipados
        predictions = [
            PredictionResult(**pred) for pred in predictions_data
        ]

        explainabilities = []
        for expl_data in explainability_data:
            if expl_data is not None:
                explainabilities.append(
                    ExplainabilityResult(**expl_data)
                )

        # Inicializar gerador com S3 client
        s3_client = boto3.client("s3")
        generator = ReportGenerator(
            s3_client=s3_client,
            bucket=bucket,
            model_version=model_version,
        )

        report_s3_paths: dict[str, str] = {}

        # Gerar relatório executivo
        executive_report = generator.generate_executive_report(
            predictions=predictions,
            explainabilities=explainabilities,
        )

        # Exportar e fazer upload dos relatórios
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Executive JSON
            exec_json_path = os.path.join(tmp_dir, "executive_report.json")
            generator.export_json(executive_report, exec_json_path)
            s3_uri = generator.upload_to_s3(
                exec_json_path, execution_id, "executive_report.json"
            )
            if s3_uri:
                report_s3_paths["executive_json"] = s3_uri

            # Executive Markdown
            exec_md_path = os.path.join(tmp_dir, "executive_report.md")
            generator.export_markdown(executive_report, exec_md_path)
            s3_uri = generator.upload_to_s3(
                exec_md_path, execution_id, "executive_report.md"
            )
            if s3_uri:
                report_s3_paths["executive_markdown"] = s3_uri

            # High risk users JSON
            high_risk_report = {
                "report_type": "high_risk",
                "users": executive_report.get("high_risk_users", []),
                "metadata": executive_report.get("metadata", {}),
            }
            hr_json_path = os.path.join(tmp_dir, "high_risk_users.json")
            generator.export_json(high_risk_report, hr_json_path)
            s3_uri = generator.upload_to_s3(
                hr_json_path, execution_id, "high_risk_users.json"
            )
            if s3_uri:
                report_s3_paths["high_risk_json"] = s3_uri

        output = {
            **event,
            "execution_id": execution_id,
            "report_s3_paths": report_s3_paths,
            "stage_completed": "report-generation",
        }

        duration = time.time() - start_time
        logger.log_stage_completion(duration_seconds=duration)
        return output

    except Exception as e:
        logger.error(
            f"Falha no estágio report-generation: {e}",
            extra={"error_type": type(e).__name__},
        )
        raise
