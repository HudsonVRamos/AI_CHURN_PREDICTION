"""Pipeline runner local para execução integrada de todos os estágios.

Módulo de integração que documenta o contrato de dados entre estágios,
permite execução local (sem Step Functions) e valida a propagação
correta de campos entre handlers.

Sequência do pipeline:
    ingest → extract → feature → store → (train | predict) → shap → bedrock → report

Requirements: 8.1, 9.1, 17.3, 17.4
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("churn-prediction.pipeline")


# ──────────────────────────────────────────────────────────────────────
# Contrato de dados entre estágios
# ──────────────────────────────────────────────────────────────────────

STAGE_DATA_CONTRACT = {
    "ingestion": {
        "requires": ["source"],
        "produces": ["valid_user_ids", "execution_id", "stage_completed"],
    },
    "extraction": {
        "requires": ["valid_user_ids", "execution_id"],
        "produces": [
            "extracted_sessions",
            "extracted_data_s3_prefix",
            "users_extracted",
            "stage_completed",
        ],
    },
    "feature-engineering": {
        "requires": ["extracted_sessions", "execution_id"],
        "produces": [
            "feature_vectors",
            "users_with_features",
            "stage_completed",
        ],
    },
    "store-features": {
        "requires": ["feature_vectors", "execution_id"],
        "produces": [
            "stored_feature_versions",
            "users_stored",
            "stage_completed",
        ],
    },
    "ml-inference": {
        "requires": ["feature_vectors", "execution_id"],
        "produces": [
            "predictions",
            "predictions_count",
            "model_version_used",
            "stage_completed",
        ],
    },
    "explainability": {
        "requires": ["predictions", "feature_vectors", "execution_id"],
        "produces": [
            "explainability_results",
            "shap_success_count",
            "stage_completed",
        ],
    },
    "bedrock-explanation": {
        "requires": [
            "predictions",
            "explainability_results",
            "feature_vectors",
            "execution_id",
        ],
        "produces": [
            "explanations",
            "explanation_success_count",
            "stage_completed",
        ],
    },
    "report-generation": {
        "requires": [
            "predictions",
            "explainability_results",
            "explanations",
            "execution_id",
        ],
        "produces": ["report_s3_paths", "stage_completed"],
    },
}


# ──────────────────────────────────────────────────────────────────────
# Exceções
# ──────────────────────────────────────────────────────────────────────


class PipelineError(Exception):
    """Erro genérico do pipeline."""


class StageValidationError(PipelineError):
    """Falha na validação de pré-condições de um estágio."""


class DataPropagationError(PipelineError):
    """Dados obrigatórios não foram propagados entre estágios."""


# ──────────────────────────────────────────────────────────────────────
# Resultado de estágio
# ──────────────────────────────────────────────────────────────────────


@dataclass
class StageResult:
    """Resultado da execução de um estágio individual."""

    stage_name: str
    success: bool
    duration_seconds: float
    output: dict = field(default_factory=dict)
    error: str | None = None


# ──────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ──────────────────────────────────────────────────────────────────────


class PipelineRunner:
    """Executa o pipeline completo localmente (para testes sem Step Functions).

    Invoca cada handler em sequência, passando o output de um como input
    do próximo. Valida que dados são persistidos entre estágios (R17.4).

    Atributos:
        stages: Lista de resultados de cada estágio executado.
    """

    def __init__(self, handlers: dict[str, Callable] | None = None):
        """Inicializa o runner com handlers opcionais (injeção para testes).

        Args:
            handlers: Dict mapeando nome_do_estágio -> callable(event, context).
                      Se None, importa os handlers reais do módulo handlers.
        """
        self._handlers = handlers or self._load_default_handlers()
        self.stages: list[StageResult] = []

    def run(self, mode: str, input_data: dict) -> dict:
        """Executa pipeline completo em modo train ou predict.

        Args:
            mode: "train" ou "predict". Determina se após store-features
                  o pipeline segue para treino ou inferência+explicação+relatório.
            input_data: Dados iniciais (source, from_date, npaw_account_code, etc.).

        Returns:
            Output final do último estágio executado.

        Raises:
            PipelineError: Se um estágio falhar.
            StageValidationError: Se pré-condições de um estágio não forem atendidas.
            ValueError: Se mode não for 'train' nem 'predict'.
        """
        if mode not in ("train", "predict"):
            raise ValueError(f"Mode deve ser 'train' ou 'predict', recebido: '{mode}'")

        self.stages = []
        execution_id = input_data.get("execution_id", str(uuid.uuid4()))
        input_data["execution_id"] = execution_id
        input_data["mode"] = mode

        logger.info(
            "Iniciando pipeline mode=%s execution_id=%s", mode, execution_id
        )

        # Estágios comuns (ingest → extract → feature → store)
        common_stages = ["ingestion", "extraction", "feature-engineering", "store-features"]

        # Estágios após store dependem do modo
        if mode == "train":
            # No modo treino, para após armazenar features (treino é no SageMaker)
            post_stages: list[str] = []
        else:
            # Modo predict: inferência → shap → bedrock → report
            post_stages = [
                "ml-inference",
                "explainability",
                "bedrock-explanation",
                "report-generation",
            ]

        all_stages = common_stages + post_stages
        current_event = dict(input_data)

        for stage_name in all_stages:
            current_event = self._execute_stage(stage_name, current_event)

        logger.info(
            "Pipeline finalizado. Estágios executados: %d",
            len(self.stages),
        )
        return current_event

    def _execute_stage(self, stage_name: str, event: dict) -> dict:
        """Executa um estágio individual com validação e logging.

        Args:
            stage_name: Nome do estágio (chave em STAGE_DATA_CONTRACT).
            event: Evento de entrada para o estágio.

        Returns:
            Output do estágio (input para o próximo).

        Raises:
            StageValidationError: Se pré-condições falharem.
            PipelineError: Se o handler falhar.
        """
        # Validar pré-condições (R17.4 - dados persistidos antes do próximo estágio)
        self._validate_preconditions(stage_name, event)

        handler_fn = self._handlers.get(stage_name)
        if handler_fn is None:
            raise PipelineError(
                f"Handler não encontrado para estágio '{stage_name}'"
            )

        logger.info("▶ Executando estágio: %s", stage_name)
        start_time = time.time()

        try:
            output = handler_fn(event, None)
            duration = time.time() - start_time

            # Validar que campos obrigatórios foram produzidos
            self._validate_output(stage_name, output)

            result = StageResult(
                stage_name=stage_name,
                success=True,
                duration_seconds=duration,
                output=output,
            )
            self.stages.append(result)

            logger.info(
                "✓ Estágio '%s' concluído em %.2fs",
                stage_name,
                duration,
            )
            return output

        except Exception as e:
            duration = time.time() - start_time
            result = StageResult(
                stage_name=stage_name,
                success=False,
                duration_seconds=duration,
                error=str(e),
            )
            self.stages.append(result)

            logger.error(
                "✗ Estágio '%s' falhou após %.2fs: %s",
                stage_name,
                duration,
                e,
            )
            raise PipelineError(
                f"Falha no estágio '{stage_name}': {e}"
            ) from e

    def _validate_preconditions(self, stage_name: str, event: dict) -> None:
        """Valida que campos obrigatórios existem no evento de entrada.

        Garante que dados foram persistidos pelo estágio anterior antes
        de iniciar o próximo (R17.4 - Data Integrity).

        Args:
            stage_name: Nome do estágio a validar.
            event: Evento de entrada.

        Raises:
            StageValidationError: Se campos obrigatórios estiverem ausentes.
        """
        contract = STAGE_DATA_CONTRACT.get(stage_name)
        if contract is None:
            return

        required_fields = contract["requires"]
        missing = [f for f in required_fields if f not in event]

        if missing:
            raise StageValidationError(
                f"Estágio '{stage_name}' requer campos ausentes: {missing}. "
                f"Verifique se o estágio anterior persistiu os dados (R17.4)."
            )

    def _validate_output(self, stage_name: str, output: dict) -> None:
        """Valida que o estágio produziu os campos esperados.

        Args:
            stage_name: Nome do estágio executado.
            output: Output retornado pelo handler.

        Raises:
            DataPropagationError: Se campos obrigatórios não foram produzidos.
        """
        contract = STAGE_DATA_CONTRACT.get(stage_name)
        if contract is None:
            return

        produced_fields = contract["produces"]
        missing = [f for f in produced_fields if f not in output]

        if missing:
            raise DataPropagationError(
                f"Estágio '{stage_name}' não produziu campos esperados: {missing}. "
                f"O contrato de dados entre estágios foi violado."
            )

    def _load_default_handlers(self) -> dict[str, Callable]:
        """Carrega os handlers reais do módulo handlers.

        Returns:
            Dict mapeando nome_do_estágio -> handler function.
        """
        from src.orchestrator.handlers.ingest_handler import (
            handler as ingest_handler,
        )
        from src.orchestrator.handlers.extract_handler import (
            handler as extract_handler,
        )
        from src.orchestrator.handlers.feature_handler import (
            handler as feature_handler,
        )
        from src.orchestrator.handlers.store_handler import (
            handler as store_handler,
        )
        from src.orchestrator.handlers.predict_handler import (
            handler as predict_handler,
        )
        from src.orchestrator.handlers.shap_handler import (
            handler as shap_handler,
        )
        from src.orchestrator.handlers.bedrock_handler import (
            handler as bedrock_handler,
        )
        from src.orchestrator.handlers.report_handler import (
            handler as report_handler,
        )

        return {
            "ingestion": ingest_handler,
            "extraction": extract_handler,
            "feature-engineering": feature_handler,
            "store-features": store_handler,
            "ml-inference": predict_handler,
            "explainability": shap_handler,
            "bedrock-explanation": bedrock_handler,
            "report-generation": report_handler,
        }

    def get_summary(self) -> dict:
        """Retorna resumo da execução do pipeline.

        Returns:
            Dict com total de estágios, sucesso/falha, e duração total.
        """
        total_duration = sum(s.duration_seconds for s in self.stages)
        successful = [s for s in self.stages if s.success]
        failed = [s for s in self.stages if not s.success]

        return {
            "total_stages": len(self.stages),
            "successful": len(successful),
            "failed": len(failed),
            "total_duration_seconds": round(total_duration, 2),
            "stages": [
                {
                    "name": s.stage_name,
                    "success": s.success,
                    "duration_seconds": round(s.duration_seconds, 2),
                    "error": s.error,
                }
                for s in self.stages
            ],
        }


# ──────────────────────────────────────────────────────────────────────
# Funções utilitárias
# ──────────────────────────────────────────────────────────────────────


def validate_stage_wiring() -> list[str]:
    """Valida que o contrato de dados entre estágios é consistente.

    Verifica que cada estágio produz os campos que o estágio seguinte
    requer (na sequência predict).

    Returns:
        Lista de problemas encontrados (vazia se tudo OK).
    """
    predict_sequence = [
        "ingestion",
        "extraction",
        "feature-engineering",
        "store-features",
        "ml-inference",
        "explainability",
        "bedrock-explanation",
        "report-generation",
    ]

    problems: list[str] = []
    available_fields: set[str] = set()

    for i, stage_name in enumerate(predict_sequence):
        contract = STAGE_DATA_CONTRACT.get(stage_name)
        if contract is None:
            problems.append(f"Estágio '{stage_name}' não tem contrato definido")
            continue

        # Verificar se campos requeridos estão disponíveis
        # (exceto o primeiro estágio, que recebe input externo)
        if i > 0:
            for req_field in contract["requires"]:
                if req_field not in available_fields:
                    problems.append(
                        f"Estágio '{stage_name}' requer '{req_field}' "
                        f"mas nenhum estágio anterior produz esse campo"
                    )

        # Adicionar campos produzidos ao conjunto disponível
        available_fields.update(contract["produces"])

    return problems


def get_environment_variables() -> dict[str, str]:
    """Retorna mapeamento de variáveis de ambiente necessárias por Lambda.

    Documentação das variáveis de ambiente que cada handler espera,
    útil para configuração do CDK/CloudFormation.

    Returns:
        Dict com nome_lambda -> dict de variáveis necessárias.
    """
    return {
        "ingest-user-lists": {
            "LOG_LEVEL": "INFO",
            "EXECUTION_TABLE": "churn_executions",
        },
        "extract-npaw-data": {
            "LOG_LEVEL": "INFO",
            "S3_BUCKET": "sky-brazil-churn-prediction",
            "NPAW_SECRET_ARN": "churn-prediction/npaw-api-key",
            "NPAW_ACCOUNT_CODE": "sky_brazil",
            "NPAW_RATE_LIMIT": "1.0",
            "NPAW_MAX_CONCURRENT": "5",
        },
        "compute-features": {
            "LOG_LEVEL": "INFO",
            "MIN_SESSIONS": "5",
            "MIN_WEEKS_TRENDS": "4",
        },
        "store-features": {
            "LOG_LEVEL": "INFO",
            "FEATURE_STORE_TABLE": "churn_feature_store",
            "AWS_REGION": "us-east-1",
        },
        "batch-predict": {
            "LOG_LEVEL": "INFO",
            "S3_BUCKET": "sky-brazil-churn-prediction",
            "MODEL_PACKAGE_GROUP": "churn-prediction-models",
            "PREDICTIONS_TABLE": "churn_predictions",
            "AWS_REGION": "us-east-1",
        },
        "compute-shap": {
            "LOG_LEVEL": "INFO",
            "S3_BUCKET": "sky-brazil-churn-prediction",
            "TOP_FEATURES": "10",
        },
        "generate-explanations": {
            "LOG_LEVEL": "INFO",
            "S3_BUCKET": "sky-brazil-churn-prediction",
            "BEDROCK_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
            "BEDROCK_TIMEOUT": "60",
            "BEDROCK_MAX_RETRIES": "2",
            "AWS_REGION": "us-east-1",
        },
        "generate-reports": {
            "LOG_LEVEL": "INFO",
            "S3_BUCKET": "sky-brazil-churn-prediction",
            "REPORTS_PREFIX": "reports",
        },
    }
