"""Testes end-to-end do pipeline de churn prediction.

Utiliza o PipelineRunner com mock handlers para validar o fluxo completo
sem dependência de serviços AWS reais.

Verifica:
- Pipeline completo com dados sintéticos (Step Functions local)
- Todos os artefatos são gerados corretamente
- Propagação de dados entre estágios
- Modos train e predict
- Validação de falhas (campos ausentes)

Requirements: 8.1, 8.6, 17.1, 17.4
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.orchestrator.pipeline import (
    PipelineError,
    PipelineRunner,
    STAGE_DATA_CONTRACT,
)
from tests.fixtures.synthetic_users import (
    ACTIVE_USER_IDS,
    CHURNED_USER_IDS,
)
from tests.fixtures.mock_responses import (
    MockBedrockResponses,
    MockSageMakerResponses,
)


# ──────────────────────────────────────────────────────────────────────
# Mock handlers que simulam cada estágio do pipeline
# ──────────────────────────────────────────────────────────────────────


def _mock_ingestion_handler(event: dict, context: Any) -> dict:
    """Simula ingestão: valida source e retorna IDs válidos."""
    execution_id = event.get("execution_id", str(uuid.uuid4()))
    user_ids = CHURNED_USER_IDS[:5] + ACTIVE_USER_IDS[:5]

    return {
        **event,
        "valid_user_ids": user_ids,
        "execution_id": execution_id,
        "stage_completed": "ingestion",
    }


def _mock_extraction_handler(event: dict, context: Any) -> dict:
    """Simula extração NPAW: retorna sessões sintéticas."""
    user_ids = event["valid_user_ids"]

    # Sessões simplificadas por user
    extracted = {
        uid: [
            {"session_id": f"sess-{i}", "effective_time": 3600000}
            for i in range(10)
        ]
        for uid in user_ids
    }

    return {
        **event,
        "extracted_sessions": extracted,
        "extracted_data_s3_prefix": (
            f"s3://bucket/raw/{event['execution_id']}/"
        ),
        "users_extracted": len(user_ids),
        "stage_completed": "extraction",
    }


def _mock_feature_handler(event: dict, context: Any) -> dict:
    """Simula feature engineering: gera vetores de features."""
    sessions = event["extracted_sessions"]

    feature_vectors = []
    for uid in sessions:
        feature_vectors.append({
            "user_id": uid,
            "total_sessions": 10,
            "total_viewing_hours": 25.5,
            "avg_happiness_score": 7.2,
            "error_rate": 0.05,
            "sessions_per_week": 3.5,
        })

    return {
        **event,
        "feature_vectors": feature_vectors,
        "users_with_features": len(feature_vectors),
        "stage_completed": "feature-engineering",
    }


def _mock_store_handler(event: dict, context: Any) -> dict:
    """Simula armazenamento no Feature Store."""
    feature_vectors = event["feature_vectors"]

    stored_versions = {
        fv["user_id"]: 1 for fv in feature_vectors
    }

    return {
        **event,
        "stored_feature_versions": stored_versions,
        "users_stored": len(stored_versions),
        "stage_completed": "store-features",
    }


def _mock_predict_handler(event: dict, context: Any) -> dict:
    """Simula inferência ML (SageMaker Batch Transform)."""
    feature_vectors = event["feature_vectors"]

    mock_sm = MockSageMakerResponses()
    user_ids = [fv["user_id"] for fv in feature_vectors]
    predictions = mock_sm.predict_batch(user_ids)

    return {
        **event,
        "predictions": predictions,
        "predictions_count": len(predictions),
        "model_version_used": (
            MockSageMakerResponses.MODEL_VERSION
        ),
        "stage_completed": "ml-inference",
    }


def _mock_shap_handler(event: dict, context: Any) -> dict:
    """Simula SHAP explainability."""
    predictions = event["predictions"]

    mock_sm = MockSageMakerResponses()
    results = []
    for pred in predictions:
        shap_result = mock_sm.get_shap_values(pred["user_id"])
        results.append(shap_result)

    return {
        **event,
        "explainability_results": results,
        "shap_success_count": len(results),
        "stage_completed": "explainability",
    }


def _mock_bedrock_handler(event: dict, context: Any) -> dict:
    """Simula geração de explicações via Bedrock."""
    predictions = event["predictions"]

    mock_bedrock = MockBedrockResponses()
    explanations = []
    for pred in predictions:
        text = mock_bedrock.get_explanation_text(pred["risk_tier"])
        explanations.append({
            "user_id": pred["user_id"],
            "explanation": text,
            "status": "available",
        })

    return {
        **event,
        "explanations": explanations,
        "explanation_success_count": len(explanations),
        "stage_completed": "bedrock-explanation",
    }


def _mock_report_handler(event: dict, context: Any) -> dict:
    """Simula geração de relatórios."""
    exec_id = event["execution_id"]

    return {
        **event,
        "report_s3_paths": {
            "executive_json": (
                f"s3://bucket/reports/{exec_id}/executive.json"
            ),
            "executive_md": (
                f"s3://bucket/reports/{exec_id}/executive.md"
            ),
            "high_risk_json": (
                f"s3://bucket/reports/{exec_id}/high_risk.json"
            ),
        },
        "stage_completed": "report-generation",
    }


def _build_mock_handlers() -> dict[str, Any]:
    """Constrói dict com todos os mock handlers."""
    return {
        "ingestion": _mock_ingestion_handler,
        "extraction": _mock_extraction_handler,
        "feature-engineering": _mock_feature_handler,
        "store-features": _mock_store_handler,
        "ml-inference": _mock_predict_handler,
        "explainability": _mock_shap_handler,
        "bedrock-explanation": _mock_bedrock_handler,
        "report-generation": _mock_report_handler,
    }


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_handlers() -> dict[str, Any]:
    """Retorna handlers mock para todos os estágios."""
    return _build_mock_handlers()


@pytest.fixture
def pipeline(mock_handlers: dict) -> PipelineRunner:
    """Retorna PipelineRunner configurado com mock handlers."""
    return PipelineRunner(handlers=mock_handlers)


@pytest.fixture
def predict_input() -> dict:
    """Dados de entrada para o pipeline em modo predict."""
    return {
        "source": "test-input",
        "from_date": "2024-01-01",
        "to_date": "2024-06-01",
        "npaw_account_code": "sky_brazil",
    }


@pytest.fixture
def train_input() -> dict:
    """Dados de entrada para o pipeline em modo train."""
    return {
        "source": "training-dataset-v1",
        "from_date": "2023-06-01",
        "to_date": "2024-06-01",
        "npaw_account_code": "sky_brazil",
    }


# ──────────────────────────────────────────────────────────────────────
# Testes: Pipeline modo predict (fluxo completo)
# ──────────────────────────────────────────────────────────────────────


class TestPipelinePredictMode:
    """Testes end-to-end do pipeline em modo predict."""

    def test_pipeline_completes_without_error(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """Pipeline predict deve completar sem lançar exceções."""
        result = pipeline.run(mode="predict", input_data=predict_input)
        assert result is not None

    def test_all_stages_executed_in_order(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """Todos os 8 estágios devem executar na sequência correta."""
        pipeline.run(mode="predict", input_data=predict_input)

        expected_order = [
            "ingestion",
            "extraction",
            "feature-engineering",
            "store-features",
            "ml-inference",
            "explainability",
            "bedrock-explanation",
            "report-generation",
        ]

        actual_order = [s.stage_name for s in pipeline.stages]
        assert actual_order == expected_order

    def test_all_stages_marked_successful(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """Todos os estágios devem ser marcados como sucesso."""
        pipeline.run(mode="predict", input_data=predict_input)

        for stage in pipeline.stages:
            assert stage.success is True, (
                f"Estágio '{stage.stage_name}' falhou: {stage.error}"
            )

    def test_execution_id_propagated(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """O execution_id deve ser propagado para todos os estágios."""
        result = pipeline.run(mode="predict", input_data=predict_input)

        assert "execution_id" in result
        # Verifica que é um UUID válido
        uuid.UUID(result["execution_id"])

    def test_data_propagation_between_stages(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """Dados devem ser propagados corretamente entre estágios."""
        result = pipeline.run(mode="predict", input_data=predict_input)

        # Campos produzidos pelo pipeline completo
        assert "valid_user_ids" in result
        assert "extracted_sessions" in result
        assert "feature_vectors" in result
        assert "predictions" in result
        assert "explainability_results" in result
        assert "explanations" in result
        assert "report_s3_paths" in result

    def test_final_output_contains_expected_fields(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """Output final deve conter campos esperados do relatório."""
        result = pipeline.run(mode="predict", input_data=predict_input)

        # Campos do report-generation
        assert "report_s3_paths" in result
        assert "executive_json" in result["report_s3_paths"]
        assert "executive_md" in result["report_s3_paths"]

        # Campos de contagem
        assert "predictions_count" in result
        assert result["predictions_count"] == 10  # 5 churned + 5 active

    def test_predictions_contain_required_fields(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """Cada predição deve ter os campos obrigatórios (R10.4)."""
        result = pipeline.run(mode="predict", input_data=predict_input)

        for pred in result["predictions"]:
            assert "user_id" in pred
            assert "churn_probability" in pred
            assert "confidence" in pred
            assert "risk_tier" in pred
            assert "model_version" in pred
            assert 0.0 <= pred["churn_probability"] <= 1.0
            assert 0.0 <= pred["confidence"] <= 1.0
            assert pred["risk_tier"] in ("Low", "Medium", "High")

    def test_model_version_recorded(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """A versão do modelo deve ser registrada no output (R17.1)."""
        result = pipeline.run(mode="predict", input_data=predict_input)

        assert result["model_version_used"] == (
            MockSageMakerResponses.MODEL_VERSION
        )

    def test_summary_reflects_execution(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """O resumo deve refletir a execução completa."""
        pipeline.run(mode="predict", input_data=predict_input)
        summary = pipeline.get_summary()

        assert summary["total_stages"] == 8
        assert summary["successful"] == 8
        assert summary["failed"] == 0
        assert summary["total_duration_seconds"] >= 0


# ──────────────────────────────────────────────────────────────────────
# Testes: Pipeline modo train (para após store-features)
# ──────────────────────────────────────────────────────────────────────


class TestPipelineTrainMode:
    """Testes end-to-end do pipeline em modo train."""

    def test_train_mode_stops_after_store_features(
        self, pipeline: PipelineRunner, train_input: dict
    ):
        """Modo train deve parar após store-features (4 estágios)."""
        pipeline.run(mode="train", input_data=train_input)

        expected_stages = [
            "ingestion",
            "extraction",
            "feature-engineering",
            "store-features",
        ]
        actual_stages = [s.stage_name for s in pipeline.stages]
        assert actual_stages == expected_stages

    def test_train_mode_does_not_run_inference(
        self, pipeline: PipelineRunner, train_input: dict
    ):
        """Modo train NÃO deve executar ml-inference."""
        pipeline.run(mode="train", input_data=train_input)

        stage_names = {s.stage_name for s in pipeline.stages}
        assert "ml-inference" not in stage_names
        assert "explainability" not in stage_names
        assert "bedrock-explanation" not in stage_names
        assert "report-generation" not in stage_names

    def test_train_mode_produces_stored_features(
        self, pipeline: PipelineRunner, train_input: dict
    ):
        """Modo train deve produzir features armazenadas."""
        result = pipeline.run(mode="train", input_data=train_input)

        assert "stored_feature_versions" in result
        assert "users_stored" in result
        assert result["users_stored"] == 10

    def test_train_summary_has_4_stages(
        self, pipeline: PipelineRunner, train_input: dict
    ):
        """Resumo do modo train deve ter 4 estágios."""
        pipeline.run(mode="train", input_data=train_input)
        summary = pipeline.get_summary()

        assert summary["total_stages"] == 4
        assert summary["successful"] == 4
        assert summary["failed"] == 0


# ──────────────────────────────────────────────────────────────────────
# Testes: Validações e falhas
# ──────────────────────────────────────────────────────────────────────


class TestPipelineValidation:
    """Testes de validação de pré-condições e propagação de dados."""

    def test_invalid_mode_raises_value_error(self, mock_handlers: dict):
        """Mode inválido deve lançar ValueError."""
        runner = PipelineRunner(handlers=mock_handlers)

        with pytest.raises(ValueError, match="Mode deve ser"):
            runner.run(mode="invalid", input_data={"source": "test"})

    def test_missing_source_raises_stage_validation_error(
        self, mock_handlers: dict
    ):
        """Ingestion sem 'source' deve falhar."""
        runner = PipelineRunner(handlers=mock_handlers)

        with pytest.raises(PipelineError):
            runner.run(mode="predict", input_data={})

    def test_missing_handler_raises_pipeline_error(self):
        """Se um handler não está registrado, deve lançar PipelineError."""
        incomplete_handlers = {
            "ingestion": _mock_ingestion_handler,
            # extraction está faltando
        }
        runner = PipelineRunner(handlers=incomplete_handlers)

        with pytest.raises(PipelineError, match="Handler não encontrado"):
            runner.run(
                mode="predict",
                input_data={"source": "test"},
            )

    def test_handler_that_omits_required_output_raises_error(
        self, mock_handlers: dict
    ):
        """Handler que não produz campos obrigatórios deve falhar."""
        def bad_ingestion(event, context):
            # Não retorna 'valid_user_ids' nem 'stage_completed'
            return {"execution_id": "abc"}

        mock_handlers["ingestion"] = bad_ingestion
        runner = PipelineRunner(handlers=mock_handlers)

        with pytest.raises(PipelineError):
            runner.run(mode="predict", input_data={"source": "test"})

    def test_extraction_without_valid_user_ids_raises_error(
        self, mock_handlers: dict
    ):
        """Extraction sem valid_user_ids deve falhar na validação."""
        def ingestion_no_ids(event, context):
            return {
                **event,
                "execution_id": event.get("execution_id", "test-id"),
                "stage_completed": "ingestion",
                # 'valid_user_ids' ausente propositalmente
            }

        mock_handlers["ingestion"] = ingestion_no_ids
        runner = PipelineRunner(handlers=mock_handlers)

        with pytest.raises(PipelineError):
            runner.run(mode="predict", input_data={"source": "test"})

    def test_stage_data_contract_consistency(self):
        """Contrato de dados entre estágios deve ser consistente."""
        # Verifica que estágios do predict estão no contrato
        predict_stages = [
            "ingestion",
            "extraction",
            "feature-engineering",
            "store-features",
            "ml-inference",
            "explainability",
            "bedrock-explanation",
            "report-generation",
        ]

        for stage in predict_stages:
            assert stage in STAGE_DATA_CONTRACT, (
                f"Estágio '{stage}' não definido no STAGE_DATA_CONTRACT"
            )
            contract = STAGE_DATA_CONTRACT[stage]
            assert "requires" in contract
            assert "produces" in contract
            assert isinstance(contract["requires"], list)
            assert isinstance(contract["produces"], list)

    def test_stage_failure_preserves_previous_results(
        self, mock_handlers: dict
    ):
        """Se um estágio falha, resultados anteriores são preservados."""
        def failing_shap(event, context):
            raise RuntimeError("SHAP computation failed")

        mock_handlers["explainability"] = failing_shap
        runner = PipelineRunner(handlers=mock_handlers)

        with pytest.raises(PipelineError, match="explainability"):
            runner.run(mode="predict", input_data={"source": "test"})

        # Estágios anteriores devem ter sido executados com sucesso
        successful_stages = [
            s.stage_name for s in runner.stages if s.success
        ]
        assert "ingestion" in successful_stages
        assert "extraction" in successful_stages
        assert "feature-engineering" in successful_stages
        assert "store-features" in successful_stages
        assert "ml-inference" in successful_stages

        # O estágio que falhou deve estar registrado
        failed_stages = [
            s.stage_name for s in runner.stages if not s.success
        ]
        assert "explainability" in failed_stages


# ──────────────────────────────────────────────────────────────────────
# Testes: Integridade de dados entre estágios (R17.4)
# ──────────────────────────────────────────────────────────────────────


class TestDataIntegrity:
    """Verifica persistência de dados entre estágios (R17.4)."""

    def test_features_stored_before_inference(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """Features devem ser armazenadas antes de ml-inference."""
        pipeline.run(mode="predict", input_data=predict_input)

        stage_names = [s.stage_name for s in pipeline.stages]
        store_idx = stage_names.index("store-features")
        inference_idx = stage_names.index("ml-inference")
        assert store_idx < inference_idx

    def test_predictions_available_before_report(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """Predictions devem existir antes da geração de relatório."""
        result = pipeline.run(mode="predict", input_data=predict_input)

        # report-generation recebeu predictions e explainability_results
        assert len(result["predictions"]) > 0
        assert len(result["explainability_results"]) > 0

    def test_execution_id_consistent_across_all_stages(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """O mesmo execution_id deve estar presente em todos os outputs."""
        result = pipeline.run(mode="predict", input_data=predict_input)
        exec_id = result["execution_id"]

        # Verifica que todos os estágios receberam o mesmo execution_id
        for stage in pipeline.stages:
            assert stage.output.get("execution_id") == exec_id, (
                f"Estágio '{stage.stage_name}' tem execution_id diferente"
            )

    def test_user_count_consistent_through_pipeline(
        self, pipeline: PipelineRunner, predict_input: dict
    ):
        """Número de usuários deve ser consistente entre estágios."""
        result = pipeline.run(mode="predict", input_data=predict_input)

        num_users = len(result["valid_user_ids"])
        assert result["users_extracted"] == num_users
        assert result["users_with_features"] == num_users
        assert result["users_stored"] == num_users
        assert result["predictions_count"] == num_users
        assert result["shap_success_count"] == num_users
        assert result["explanation_success_count"] == num_users
