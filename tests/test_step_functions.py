"""Testes unitários para o orquestrador Step Functions.

Verifica: construção da state machine, start/status/wait de execuções,
error handling, retry configs e Choice state.

Requirements: 8.1, 8.4, 16.1, 17.1, 17.3
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.orchestrator.step_functions import (
    BEDROCK_RETRY_CONFIG,
    DEFAULT_CATCH_CONFIG,
    DEFAULT_RETRY_CONFIG,
    DYNAMODB_RETRY_CONFIG,
    ML_RETRY_CONFIG,
    PIPELINE_DEFINITION,
    PipelineOrchestrator,
    _add_retry_and_catch,
)


@pytest.fixture
def mock_sfn_client() -> MagicMock:
    """Cria um mock do cliente Step Functions."""
    return MagicMock()


@pytest.fixture
def orchestrator(mock_sfn_client: MagicMock) -> PipelineOrchestrator:
    """Cria uma instância do PipelineOrchestrator com cliente mockado."""
    return PipelineOrchestrator(sfn_client=mock_sfn_client)


@pytest.fixture
def sample_lambda_arns() -> dict[str, str]:
    """ARNs de exemplo para construção da state machine."""
    return {
        "Ingestion": "arn:aws:lambda:us-east-1:123456789:function:ingest-user-lists",
        "Extraction": "arn:aws:lambda:us-east-1:123456789:function:extract-npaw-data",
        "FeatureEngineering": "arn:aws:lambda:us-east-1:123456789:function:compute-features",
        "StoreFeatures": "arn:aws:lambda:us-east-1:123456789:function:store-features",
        "Training": "arn:aws:states:::sagemaker:createTrainingJob.sync",
        "EvaluateModel": "arn:aws:lambda:us-east-1:123456789:function:evaluate-model",
        "RegisterModel": "arn:aws:states:::sagemaker:createModel",
        "BatchPredict": "arn:aws:states:::sagemaker:createTransformJob.sync",
        "Explainability": "arn:aws:lambda:us-east-1:123456789:function:compute-shap",
        "BedrockExplanations": "arn:aws:lambda:us-east-1:123456789:function:generate-explanations",
        "GenerateReports": "arn:aws:lambda:us-east-1:123456789:function:generate-reports",
    }


class TestPipelineDefinition:
    """Testes para a definição estática PIPELINE_DEFINITION."""

    def test_starts_at_ingestion(self):
        """A state machine deve começar no estado Ingestion."""
        assert PIPELINE_DEFINITION["StartAt"] == "Ingestion"

    def test_all_required_states_present(self):
        """Deve conter todos os estados do pipeline."""
        expected_states = {
            "Ingestion",
            "Extraction",
            "FeatureEngineering",
            "StoreFeatures",
            "ChooseMode",
            "Training",
            "EvaluateModel",
            "RegisterModel",
            "BatchPredict",
            "Explainability",
            "BedrockExplanations",
            "GenerateReports",
            "PipelineFailed",
        }
        actual_states = set(PIPELINE_DEFINITION["States"].keys())
        assert expected_states == actual_states

    def test_choose_mode_is_choice_state(self):
        """ChooseMode deve ser um Choice state com opções train e predict."""
        state = PIPELINE_DEFINITION["States"]["ChooseMode"]
        assert state["Type"] == "Choice"
        assert len(state["Choices"]) == 2

        choices_map = {c["StringEquals"]: c["Next"] for c in state["Choices"]}
        assert choices_map["train"] == "Training"
        assert choices_map["predict"] == "BatchPredict"

    def test_choose_mode_has_default(self):
        """ChooseMode deve ter um Default caso o modo não seja reconhecido."""
        state = PIPELINE_DEFINITION["States"]["ChooseMode"]
        assert "Default" in state
        assert state["Default"] == "BatchPredict"

    def test_train_path_ends_at_register_model(self):
        """O caminho de treino deve terminar em RegisterModel."""
        assert PIPELINE_DEFINITION["States"]["RegisterModel"].get("End") is True

    def test_predict_path_ends_at_generate_reports(self):
        """O caminho de predição deve terminar em GenerateReports."""
        assert PIPELINE_DEFINITION["States"]["GenerateReports"].get("End") is True

    def test_predict_path_sequence(self):
        """Verifica a sequência: BatchPredict → Explainability → BedrockExplanations → GenerateReports."""
        states = PIPELINE_DEFINITION["States"]
        assert states["BatchPredict"]["Next"] == "Explainability"
        assert states["Explainability"]["Next"] == "BedrockExplanations"
        assert states["BedrockExplanations"]["Next"] == "GenerateReports"

    def test_train_path_sequence(self):
        """Verifica a sequência: Training → EvaluateModel → RegisterModel."""
        states = PIPELINE_DEFINITION["States"]
        assert states["Training"]["Next"] == "EvaluateModel"
        assert states["EvaluateModel"]["Next"] == "RegisterModel"

    def test_ingestion_to_store_features_sequence(self):
        """Verifica: Ingestion → Extraction → FeatureEngineering → StoreFeatures → ChooseMode."""
        states = PIPELINE_DEFINITION["States"]
        assert states["Ingestion"]["Next"] == "Extraction"
        assert states["Extraction"]["Next"] == "FeatureEngineering"
        assert states["FeatureEngineering"]["Next"] == "StoreFeatures"
        assert states["StoreFeatures"]["Next"] == "ChooseMode"

    def test_all_task_states_have_retry(self):
        """Todos os estados Task devem ter configuração de Retry."""
        for name, state in PIPELINE_DEFINITION["States"].items():
            if state.get("Type") == "Task":
                assert "Retry" in state, f"Estado '{name}' não possui Retry"

    def test_all_task_states_have_catch(self):
        """Todos os estados Task devem ter configuração de Catch."""
        for name, state in PIPELINE_DEFINITION["States"].items():
            if state.get("Type") == "Task":
                assert "Catch" in state, f"Estado '{name}' não possui Catch"

    def test_pipeline_failed_state(self):
        """PipelineFailed deve ser um estado Fail com Error e Cause."""
        state = PIPELINE_DEFINITION["States"]["PipelineFailed"]
        assert state["Type"] == "Fail"
        assert "Error" in state
        assert "Cause" in state


class TestAddRetryAndCatch:
    """Testes para a função helper _add_retry_and_catch."""

    def test_adds_retry_to_task_state(self):
        """Deve adicionar retry a um estado Task."""
        state = {"Type": "Task", "Resource": "arn:...", "Next": "NextState"}
        result = _add_retry_and_catch(state, retry=ML_RETRY_CONFIG)
        assert result["Retry"] == ML_RETRY_CONFIG

    def test_adds_catch_to_task_state(self):
        """Deve adicionar catch a um estado Task."""
        state = {"Type": "Task", "Resource": "arn:...", "Next": "NextState"}
        result = _add_retry_and_catch(state, catch=DEFAULT_CATCH_CONFIG)
        assert result["Catch"] == DEFAULT_CATCH_CONFIG

    def test_does_not_modify_non_task_state(self):
        """Não deve modificar estados que não são Task."""
        choice_state = {"Type": "Choice", "Choices": []}
        result = _add_retry_and_catch(choice_state, retry=DEFAULT_RETRY_CONFIG)
        assert "Retry" not in result

    def test_uses_defaults_when_none_provided(self):
        """Deve usar configs padrão quando retry/catch são None."""
        state = {"Type": "Task", "Resource": "arn:...", "Next": "X"}
        result = _add_retry_and_catch(state)
        assert result["Retry"] == DEFAULT_RETRY_CONFIG
        assert result["Catch"] == DEFAULT_CATCH_CONFIG

    def test_does_not_overwrite_existing_retry(self):
        """Não deve sobrescrever retry existente quando nenhum é fornecido."""
        custom_retry = [{"ErrorEquals": ["Custom"], "MaxAttempts": 1}]
        state = {"Type": "Task", "Resource": "arn:...", "Next": "X", "Retry": custom_retry}
        result = _add_retry_and_catch(state)
        assert result["Retry"] == custom_retry


class TestRetryConfigs:
    """Testes para as configurações de retry."""

    def test_default_retry_has_backoff(self):
        """DEFAULT_RETRY_CONFIG deve usar exponential backoff."""
        config = DEFAULT_RETRY_CONFIG[0]
        assert config["BackoffRate"] == 2.0
        assert config["MaxAttempts"] == 3
        assert config["IntervalSeconds"] == 5

    def test_ml_retry_is_conservative(self):
        """ML_RETRY_CONFIG deve ser mais conservador (menos tentativas)."""
        config = ML_RETRY_CONFIG[0]
        assert config["MaxAttempts"] == 1
        assert config["IntervalSeconds"] >= 30

    def test_bedrock_retry_has_limited_attempts(self):
        """BEDROCK_RETRY_CONFIG deve ter no máximo 2 tentativas."""
        config = BEDROCK_RETRY_CONFIG[0]
        assert config["MaxAttempts"] == 2

    def test_dynamodb_retry_handles_throttling(self):
        """DYNAMODB_RETRY_CONFIG deve tratar throttling com mais tentativas."""
        config = DYNAMODB_RETRY_CONFIG[0]
        assert config["MaxAttempts"] == 5
        assert "DynamoDB.ProvisionedThroughputExceededException" in config["ErrorEquals"]


class TestBuildStateMachineDefinition:
    """Testes para build_state_machine_definition."""

    def test_replaces_arns(self, orchestrator: PipelineOrchestrator, sample_lambda_arns: dict):
        """Deve substituir os ARNs placeholder pelos ARNs reais."""
        definition = orchestrator.build_state_machine_definition(sample_lambda_arns)

        assert definition["States"]["Ingestion"]["Resource"] == sample_lambda_arns["Ingestion"]
        assert definition["States"]["Extraction"]["Resource"] == sample_lambda_arns["Extraction"]

    def test_preserves_state_machine_structure(
        self, orchestrator: PipelineOrchestrator, sample_lambda_arns: dict
    ):
        """Deve preservar a estrutura geral da state machine."""
        definition = orchestrator.build_state_machine_definition(sample_lambda_arns)

        assert definition["StartAt"] == "Ingestion"
        assert "States" in definition
        assert definition["States"]["ChooseMode"]["Type"] == "Choice"

    def test_all_task_states_have_retry_after_build(
        self, orchestrator: PipelineOrchestrator, sample_lambda_arns: dict
    ):
        """Após build, todos os Task states devem ter Retry configurado."""
        definition = orchestrator.build_state_machine_definition(sample_lambda_arns)

        for name, state in definition["States"].items():
            if state.get("Type") == "Task":
                assert "Retry" in state, f"Estado '{name}' sem Retry após build"

    def test_does_not_mutate_original(
        self, orchestrator: PipelineOrchestrator, sample_lambda_arns: dict
    ):
        """Não deve alterar a definição original PIPELINE_DEFINITION."""
        original_resource = PIPELINE_DEFINITION["States"]["Ingestion"]["Resource"]
        orchestrator.build_state_machine_definition(sample_lambda_arns)
        assert PIPELINE_DEFINITION["States"]["Ingestion"]["Resource"] == original_resource

    def test_ignores_unknown_arns(self, orchestrator: PipelineOrchestrator):
        """Deve ignorar ARNs para estados que não existem."""
        arns = {"NonExistentState": "arn:aws:lambda:us-east-1:123:function:unknown"}
        definition = orchestrator.build_state_machine_definition(arns)
        assert "NonExistentState" not in definition["States"]

    def test_store_features_uses_dynamodb_retry(
        self, orchestrator: PipelineOrchestrator, sample_lambda_arns: dict
    ):
        """StoreFeatures deve usar DYNAMODB_RETRY_CONFIG."""
        definition = orchestrator.build_state_machine_definition(sample_lambda_arns)
        assert definition["States"]["StoreFeatures"]["Retry"] == DYNAMODB_RETRY_CONFIG

    def test_bedrock_uses_bedrock_retry(
        self, orchestrator: PipelineOrchestrator, sample_lambda_arns: dict
    ):
        """BedrockExplanations deve usar BEDROCK_RETRY_CONFIG."""
        definition = orchestrator.build_state_machine_definition(sample_lambda_arns)
        assert definition["States"]["BedrockExplanations"]["Retry"] == BEDROCK_RETRY_CONFIG


class TestStartExecution:
    """Testes para start_execution."""

    def test_returns_execution_arn(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve retornar o execution_arn da resposta."""
        mock_sfn_client.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:my-sm:run-001",
            "startDate": "2024-01-01T00:00:00Z",
        }

        result = orchestrator.start_execution(
            state_machine_arn="arn:aws:states:us-east-1:123:stateMachine:my-sm",
            input_data={"mode": "predict", "user_ids": ["u-001"]},
        )

        assert result == "arn:aws:states:us-east-1:123:execution:my-sm:run-001"

    def test_passes_serialized_input(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve serializar input_data como JSON na chamada."""
        mock_sfn_client.start_execution.return_value = {
            "executionArn": "arn:aws:states:us-east-1:123:execution:sm:run",
            "startDate": "2024-01-01T00:00:00Z",
        }
        input_data = {"mode": "train", "dataset_version": 5}

        orchestrator.start_execution("arn:aws:states:...:sm", input_data)

        call_kwargs = mock_sfn_client.start_execution.call_args[1]
        parsed_input = json.loads(call_kwargs["input"])
        assert parsed_input["mode"] == "train"
        assert parsed_input["dataset_version"] == 5

    def test_raises_runtime_error_on_failure(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve lançar RuntimeError se a API falhar."""
        mock_sfn_client.start_execution.side_effect = Exception("AccessDenied")

        with pytest.raises(RuntimeError, match="Falha ao iniciar execução"):
            orchestrator.start_execution("arn:...", {"mode": "predict"})

    def test_execution_name_is_unique(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve gerar nomes de execução únicos."""
        mock_sfn_client.start_execution.return_value = {
            "executionArn": "arn:...",
            "startDate": "2024-01-01T00:00:00Z",
        }

        orchestrator.start_execution("arn:...", {"mode": "predict"})
        name1 = mock_sfn_client.start_execution.call_args[1]["name"]

        orchestrator.start_execution("arn:...", {"mode": "predict"})
        name2 = mock_sfn_client.start_execution.call_args[1]["name"]

        assert name1 != name2


class TestGetExecutionStatus:
    """Testes para get_execution_status."""

    def test_returns_running_status(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve retornar status RUNNING quando execução está em andamento."""
        mock_sfn_client.describe_execution.return_value = {
            "status": "RUNNING",
            "startDate": "2024-01-01T00:00:00Z",
        }

        result = orchestrator.get_execution_status("arn:aws:states:...:exec")
        assert result["status"] == "RUNNING"
        assert result["startDate"] == "2024-01-01T00:00:00Z"

    def test_returns_succeeded_with_output(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve retornar output deserializado quando a execução é bem-sucedida."""
        mock_sfn_client.describe_execution.return_value = {
            "status": "SUCCEEDED",
            "startDate": "2024-01-01T00:00:00Z",
            "stopDate": "2024-01-01T01:00:00Z",
            "output": json.dumps({"predictions_count": 150}),
        }

        result = orchestrator.get_execution_status("arn:...")
        assert result["status"] == "SUCCEEDED"
        assert result["output"]["predictions_count"] == 150

    def test_returns_failed_with_error(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve retornar error e cause quando execução falha."""
        mock_sfn_client.describe_execution.return_value = {
            "status": "FAILED",
            "startDate": "2024-01-01T00:00:00Z",
            "stopDate": "2024-01-01T00:05:00Z",
            "error": "PipelineError",
            "cause": "Extraction failed due to API timeout",
        }

        result = orchestrator.get_execution_status("arn:...")
        assert result["status"] == "FAILED"
        assert result["error"] == "PipelineError"
        assert "API timeout" in result["cause"]

    def test_raises_runtime_error_on_api_failure(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve lançar RuntimeError se describe_execution falhar."""
        mock_sfn_client.describe_execution.side_effect = Exception("Not found")

        with pytest.raises(RuntimeError, match="Falha ao consultar execução"):
            orchestrator.get_execution_status("arn:...")


class TestWaitForCompletion:
    """Testes para wait_for_completion."""

    def test_returns_on_succeeded(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve retornar imediatamente quando status é SUCCEEDED."""
        mock_sfn_client.describe_execution.return_value = {
            "status": "SUCCEEDED",
            "startDate": "2024-01-01T00:00:00Z",
            "stopDate": "2024-01-01T01:00:00Z",
            "output": json.dumps({"done": True}),
        }

        result = orchestrator.wait_for_completion("arn:...", timeout=60, poll_interval=1)
        assert result["status"] == "SUCCEEDED"

    def test_returns_on_failed(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve retornar quando status é FAILED (estado terminal)."""
        mock_sfn_client.describe_execution.return_value = {
            "status": "FAILED",
            "startDate": "2024-01-01T00:00:00Z",
            "stopDate": "2024-01-01T00:05:00Z",
            "error": "PipelineError",
        }

        result = orchestrator.wait_for_completion("arn:...", timeout=60, poll_interval=1)
        assert result["status"] == "FAILED"

    @patch("src.orchestrator.step_functions.time.sleep")
    def test_polls_until_complete(
        self,
        mock_sleep: MagicMock,
        orchestrator: PipelineOrchestrator,
        mock_sfn_client: MagicMock,
    ):
        """Deve fazer polling até status terminal."""
        mock_sfn_client.describe_execution.side_effect = [
            {"status": "RUNNING", "startDate": "2024-01-01T00:00:00Z"},
            {"status": "RUNNING", "startDate": "2024-01-01T00:00:00Z"},
            {
                "status": "SUCCEEDED",
                "startDate": "2024-01-01T00:00:00Z",
                "stopDate": "2024-01-01T00:01:00Z",
                "output": "{}",
            },
        ]

        result = orchestrator.wait_for_completion("arn:...", timeout=60, poll_interval=1)
        assert result["status"] == "SUCCEEDED"
        assert mock_sfn_client.describe_execution.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("src.orchestrator.step_functions.time.sleep")
    def test_raises_timeout_error(
        self,
        mock_sleep: MagicMock,
        orchestrator: PipelineOrchestrator,
        mock_sfn_client: MagicMock,
    ):
        """Deve lançar TimeoutError quando timeout é excedido."""
        mock_sfn_client.describe_execution.return_value = {
            "status": "RUNNING",
            "startDate": "2024-01-01T00:00:00Z",
        }

        # Timeout de 0 segundos garante que o TimeoutError é levantado imediatamente
        with pytest.raises(TimeoutError, match="não concluiu em 0s"):
            orchestrator.wait_for_completion("arn:...", timeout=0, poll_interval=1)

    def test_returns_on_aborted(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve retornar quando status é ABORTED (estado terminal)."""
        mock_sfn_client.describe_execution.return_value = {
            "status": "ABORTED",
            "startDate": "2024-01-01T00:00:00Z",
            "stopDate": "2024-01-01T00:02:00Z",
        }

        result = orchestrator.wait_for_completion("arn:...", timeout=60, poll_interval=1)
        assert result["status"] == "ABORTED"

    def test_returns_on_timed_out(
        self, orchestrator: PipelineOrchestrator, mock_sfn_client: MagicMock
    ):
        """Deve retornar quando status é TIMED_OUT (estado terminal do Step Functions)."""
        mock_sfn_client.describe_execution.return_value = {
            "status": "TIMED_OUT",
            "startDate": "2024-01-01T00:00:00Z",
            "stopDate": "2024-01-01T02:00:00Z",
        }

        result = orchestrator.wait_for_completion("arn:...", timeout=60, poll_interval=1)
        assert result["status"] == "TIMED_OUT"
