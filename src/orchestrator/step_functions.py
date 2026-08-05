"""Orquestrador do pipeline via AWS Step Functions.

Define a state machine ASL (Amazon States Language) com estados para cada etapa
do pipeline de predição de churn, incluindo Choice state para modo train/predict,
retry e error handling por estado.

Requirements: 8.1, 8.4, 16.1, 17.1, 17.3
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from src.common.logging import get_logger

logger = get_logger("extraction")  # orchestration usa stage "extraction" conforme design


# Configuração padrão de retry por estado (exponential backoff)
DEFAULT_RETRY_CONFIG = [
    {
        "ErrorEquals": ["States.TaskFailed", "States.Timeout"],
        "IntervalSeconds": 5,
        "MaxAttempts": 3,
        "BackoffRate": 2.0,
    }
]

# Configuração de retry para estados de ML (mais conservador)
ML_RETRY_CONFIG = [
    {
        "ErrorEquals": ["States.TaskFailed"],
        "IntervalSeconds": 30,
        "MaxAttempts": 1,
        "BackoffRate": 2.0,
    }
]

# Configuração de retry para Bedrock (timeout tolerante)
BEDROCK_RETRY_CONFIG = [
    {
        "ErrorEquals": ["States.TaskFailed", "States.Timeout"],
        "IntervalSeconds": 5,
        "MaxAttempts": 2,
        "BackoffRate": 1.5,
    }
]

# Configuração de retry para DynamoDB (throttling)
DYNAMODB_RETRY_CONFIG = [
    {
        "ErrorEquals": ["States.TaskFailed", "DynamoDB.ProvisionedThroughputExceededException"],
        "IntervalSeconds": 2,
        "MaxAttempts": 5,
        "BackoffRate": 2.0,
    }
]

# Catch padrão — envia para estado de falha
DEFAULT_CATCH_CONFIG = [
    {
        "ErrorEquals": ["States.ALL"],
        "ResultPath": "$.error",
        "Next": "PipelineFailed",
    }
]


# Definição estática da state machine (template)
PIPELINE_DEFINITION: dict[str, Any] = {
    "Comment": "Pipeline de predição de churn - Sky Brazil",
    "StartAt": "Ingestion",
    "States": {
        "Ingestion": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:ingest-user-lists",
            "Next": "Extraction",
            "Retry": DEFAULT_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "Extraction": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:extract-npaw-data",
            "Next": "FeatureEngineering",
            "Retry": DEFAULT_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "FeatureEngineering": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:compute-features",
            "Next": "StoreFeatures",
            "Retry": DEFAULT_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "StoreFeatures": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:store-features",
            "Next": "ChooseMode",
            "Retry": DYNAMODB_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "ChooseMode": {
            "Type": "Choice",
            "Choices": [
                {
                    "Variable": "$.mode",
                    "StringEquals": "train",
                    "Next": "Training",
                },
                {
                    "Variable": "$.mode",
                    "StringEquals": "predict",
                    "Next": "BatchPredict",
                },
            ],
            "Default": "BatchPredict",
        },
        "Training": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:churn-pipeline-predict",
            "Next": "EvaluateModel",
            "Retry": ML_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "EvaluateModel": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:evaluate-model",
            "Next": "RegisterModel",
            "Retry": DEFAULT_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "RegisterModel": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:churn-pipeline-predict",
            "End": True,
            "Retry": ML_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "BatchPredict": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:churn-pipeline-predict",
            "Next": "Explainability",
            "Retry": ML_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "Explainability": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:compute-shap",
            "Next": "BedrockExplanations",
            "Retry": DEFAULT_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "BedrockExplanations": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:generate-explanations",
            "Next": "GenerateReports",
            "Retry": BEDROCK_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "GenerateReports": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:generate-reports",
            "End": True,
            "Retry": DEFAULT_RETRY_CONFIG,
            "Catch": DEFAULT_CATCH_CONFIG,
        },
        "PipelineFailed": {
            "Type": "Fail",
            "Cause": "Pipeline execution failed",
            "Error": "PipelineError",
        },
    },
}


def _add_retry_and_catch(
    state: dict[str, Any],
    retry: list[dict[str, Any]] | None = None,
    catch: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Adiciona configuração de Retry e Catch a um estado Task.

    Args:
        state: Definição do estado ASL.
        retry: Lista de políticas de retry. Se None, usa DEFAULT_RETRY_CONFIG.
        catch: Lista de catchers. Se None, usa DEFAULT_CATCH_CONFIG.

    Returns:
        Estado com retry e catch configurados.
    """
    if state.get("Type") != "Task":
        return state

    updated = dict(state)
    if retry is not None:
        updated["Retry"] = retry
    elif "Retry" not in updated:
        updated["Retry"] = DEFAULT_RETRY_CONFIG

    if catch is not None:
        updated["Catch"] = catch
    elif "Catch" not in updated:
        updated["Catch"] = DEFAULT_CATCH_CONFIG

    return updated


class PipelineOrchestrator:
    """Orquestrador do pipeline de churn via AWS Step Functions.

    Responsável por construir a definição da state machine, iniciar execuções,
    consultar status e aguardar conclusão.

    Args:
        sfn_client: Cliente boto3 para Step Functions (injetável para testes).
    """

    # Mapeamento de estado -> configuração de retry específica
    _STATE_RETRY_MAP: dict[str, list[dict[str, Any]]] = {
        "StoreFeatures": DYNAMODB_RETRY_CONFIG,
        "Training": ML_RETRY_CONFIG,
        "EvaluateModel": DEFAULT_RETRY_CONFIG,
        "RegisterModel": ML_RETRY_CONFIG,
        "BatchPredict": ML_RETRY_CONFIG,
        "BedrockExplanations": BEDROCK_RETRY_CONFIG,
    }

    def __init__(self, sfn_client: Any) -> None:
        self._sfn_client = sfn_client
        self._logger = get_logger("extraction")

    def build_state_machine_definition(self, lambda_arns: dict[str, str]) -> dict[str, Any]:
        """Constrói a definição completa da state machine ASL com ARNs reais.

        Args:
            lambda_arns: Dicionário mapeando nome do estado para ARN do recurso.
                Exemplo: {"Ingestion": "arn:aws:lambda:us-east-1:123:function:ingest", ...}

        Returns:
            Definição ASL completa pronta para CreateStateMachine.
        """
        import copy

        definition = copy.deepcopy(PIPELINE_DEFINITION)

        for state_name, arn in lambda_arns.items():
            if state_name in definition["States"]:
                state = definition["States"][state_name]
                if state.get("Type") == "Task":
                    state["Resource"] = arn

        # Garante que cada Task state tem retry e catch configurados
        for state_name, state in definition["States"].items():
            if state.get("Type") == "Task":
                retry_config = self._STATE_RETRY_MAP.get(state_name, DEFAULT_RETRY_CONFIG)
                definition["States"][state_name] = _add_retry_and_catch(
                    state, retry=retry_config, catch=DEFAULT_CATCH_CONFIG
                )

        self._logger.info(
            "State machine definition construída",
            extra={"states_count": len(definition["States"])},
        )

        return definition

    def start_execution(
        self, state_machine_arn: str, input_data: dict[str, Any]
    ) -> str:
        """Inicia uma execução da state machine.

        Args:
            state_machine_arn: ARN da state machine no AWS Step Functions.
            input_data: Dados de entrada para a execução (serão serializados como JSON).

        Returns:
            execution_arn da execução iniciada.

        Raises:
            RuntimeError: Se a chamada à API falhar.
        """
        execution_name = f"churn-pipeline-{uuid.uuid4().hex[:12]}"

        self._logger.info(
            "Iniciando execução da state machine",
            extra={
                "state_machine_arn": state_machine_arn,
                "execution_name": execution_name,
                "mode": input_data.get("mode", "unknown"),
            },
        )

        try:
            response = self._sfn_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=execution_name,
                input=json.dumps(input_data, default=str),
            )
            execution_arn = response["executionArn"]

            self._logger.info(
                "Execução iniciada com sucesso",
                extra={"execution_arn": execution_arn},
            )

            return execution_arn

        except Exception as e:
            self._logger.error(
                "Falha ao iniciar execução da state machine",
                extra={
                    "state_machine_arn": state_machine_arn,
                    "error": str(e),
                },
            )
            raise RuntimeError(f"Falha ao iniciar execução: {e}") from e

    def get_execution_status(self, execution_arn: str) -> dict[str, Any]:
        """Consulta o status atual de uma execução.

        Args:
            execution_arn: ARN da execução a ser consultada.

        Returns:
            Dicionário com status, startDate, stopDate (se concluída),
            input e output (se disponível).

        Raises:
            RuntimeError: Se a consulta falhar.
        """
        try:
            response = self._sfn_client.describe_execution(
                executionArn=execution_arn
            )

            status_info: dict[str, Any] = {
                "status": response["status"],
                "startDate": response.get("startDate"),
                "stopDate": response.get("stopDate"),
            }

            if "output" in response:
                try:
                    status_info["output"] = json.loads(response["output"])
                except (json.JSONDecodeError, TypeError):
                    status_info["output"] = response["output"]

            if "error" in response:
                status_info["error"] = response["error"]
                status_info["cause"] = response.get("cause")

            return status_info

        except Exception as e:
            self._logger.error(
                "Falha ao consultar status da execução",
                extra={"execution_arn": execution_arn, "error": str(e)},
            )
            raise RuntimeError(f"Falha ao consultar execução: {e}") from e

    def wait_for_completion(
        self, execution_arn: str, timeout: int = 3600, poll_interval: int = 10
    ) -> dict[str, Any]:
        """Aguarda a conclusão de uma execução com polling.

        Args:
            execution_arn: ARN da execução a aguardar.
            timeout: Tempo máximo de espera em segundos (padrão: 3600 = 1h).
            poll_interval: Intervalo entre consultas em segundos (padrão: 10).

        Returns:
            Status final da execução (mesmo formato de get_execution_status).

        Raises:
            TimeoutError: Se a execução não concluir dentro do timeout.
            RuntimeError: Se ocorrer erro na consulta.
        """
        self._logger.info(
            "Aguardando conclusão da execução",
            extra={
                "execution_arn": execution_arn,
                "timeout_seconds": timeout,
            },
        )

        start_time = time.time()
        terminal_statuses = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self._logger.error(
                    "Timeout aguardando conclusão da execução",
                    extra={
                        "execution_arn": execution_arn,
                        "elapsed_seconds": round(elapsed, 1),
                        "timeout_seconds": timeout,
                    },
                )
                raise TimeoutError(
                    f"Execução não concluiu em {timeout}s: {execution_arn}"
                )

            status = self.get_execution_status(execution_arn)

            if status["status"] in terminal_statuses:
                self._logger.info(
                    "Execução concluída",
                    extra={
                        "execution_arn": execution_arn,
                        "final_status": status["status"],
                        "elapsed_seconds": round(elapsed, 1),
                    },
                )
                return status

            time.sleep(poll_interval)
