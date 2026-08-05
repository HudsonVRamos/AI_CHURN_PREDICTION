"""Testes unitários para o módulo de logging estruturado.

Verifica: formato JSON, campos obrigatórios, propagação de execution_id,
níveis de log, e logging de início/conclusão de estágios.
"""

import json
import logging
import uuid

import pytest

from src.common.logging import (
    JSONFormatter,
    StageLogger,
    get_execution_id,
    get_logger,
    reset_execution_id,
    set_execution_id,
)


@pytest.fixture(autouse=True)
def _reset_context():
    """Reseta o contexto de execução antes e após cada teste."""
    reset_execution_id()
    yield
    reset_execution_id()


class TestExecutionContext:
    """Testes para gerenciamento de contexto de execução."""

    def test_set_and_get_execution_id(self):
        """Deve definir e recuperar o execution_id corretamente."""
        test_id = "abc-123-def-456"
        set_execution_id(test_id)
        assert get_execution_id() == test_id

    def test_get_execution_id_generates_uuid_when_not_set(self):
        """Deve gerar um UUID v4 se nenhum execution_id foi definido."""
        exec_id = get_execution_id()
        # Valida formato UUID v4
        parsed = uuid.UUID(exec_id, version=4)
        assert str(parsed) == exec_id

    def test_get_execution_id_returns_same_value_on_repeated_calls(self):
        """Deve retornar o mesmo ID em chamadas consecutivas."""
        first = get_execution_id()
        second = get_execution_id()
        assert first == second

    def test_reset_execution_id(self):
        """Deve limpar o execution_id, gerando um novo na próxima chamada."""
        set_execution_id("old-id")
        reset_execution_id()
        new_id = get_execution_id()
        assert new_id != "old-id"

    def test_set_execution_id_overwrites_previous(self):
        """Deve sobrescrever um execution_id previamente definido."""
        set_execution_id("first")
        set_execution_id("second")
        assert get_execution_id() == "second"


class TestJSONFormatter:
    """Testes para o JSONFormatter compatível com CloudWatch."""

    def _make_record(
        self, message: str, level: int = logging.INFO, stage: str = "extraction"
    ) -> logging.LogRecord:
        """Helper para criar um LogRecord com stage."""
        record = logging.LogRecord(
            name="test",
            level=level,
            pathname="test.py",
            lineno=1,
            msg=message,
            args=None,
            exc_info=None,
        )
        record.stage = stage
        return record

    def test_output_is_valid_json(self):
        """Deve produzir saída JSON válida."""
        formatter = JSONFormatter()
        record = self._make_record("test message")
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_contains_required_fields(self):
        """Deve conter todos os campos obrigatórios: timestamp, execution_id, level, stage, message."""
        set_execution_id("test-exec-id")
        formatter = JSONFormatter()
        record = self._make_record("hello world", stage="ml-inference")
        output = json.loads(formatter.format(record))

        assert "timestamp" in output
        assert output["execution_id"] == "test-exec-id"
        assert output["level"] == "INFO"
        assert output["stage"] == "ml-inference"
        assert output["message"] == "hello world"

    def test_timestamp_is_iso8601(self):
        """Deve produzir timestamp em formato ISO 8601."""
        formatter = JSONFormatter()
        record = self._make_record("test")
        output = json.loads(formatter.format(record))

        # ISO 8601 com timezone
        timestamp = output["timestamp"]
        assert "T" in timestamp
        assert "+" in timestamp or "Z" in timestamp

    def test_all_log_levels(self):
        """Deve registrar corretamente todos os níveis de log."""
        formatter = JSONFormatter()
        levels = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL",
        }
        for level_num, level_name in levels.items():
            record = self._make_record("msg", level=level_num)
            output = json.loads(formatter.format(record))
            assert output["level"] == level_name

    def test_extra_fields_included(self):
        """Deve incluir campos extras no JSON de saída."""
        formatter = JSONFormatter()
        record = self._make_record("with extras")
        record.user_id = "user-123"
        record.duration_ms = 450
        output = json.loads(formatter.format(record))

        assert output["user_id"] == "user-123"
        assert output["duration_ms"] == 450

    def test_exception_info_included(self):
        """Deve incluir informação de exceção quando presente."""
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="error occurred",
                args=None,
                exc_info=sys.exc_info(),
            )
            record.stage = "extraction"

        output = json.loads(formatter.format(record))
        assert "exception" in output
        assert "ValueError: test error" in output["exception"]

    def test_stage_defaults_to_unknown(self):
        """Deve usar 'unknown' se stage não estiver definido no record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="no stage",
            args=None,
            exc_info=None,
        )
        output = json.loads(formatter.format(record))
        assert output["stage"] == "unknown"


class TestStageLogger:
    """Testes para o StageLogger."""

    def _create_stage_logger(self, stage: str = "extraction") -> tuple[StageLogger, logging.Logger]:
        """Cria um StageLogger com logger base configurado."""
        logger = logging.getLogger(f"test.{stage}.{uuid.uuid4().hex[:8]}")
        logger.setLevel(logging.DEBUG)
        return StageLogger(logger=logger, stage=stage), logger

    def test_debug_log(self, capfd):
        """Deve emitir log DEBUG com stage correto."""
        stage_logger, base_logger = self._create_stage_logger("extraction")
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)

        stage_logger.debug("debug message")
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert output["level"] == "DEBUG"
        assert output["stage"] == "extraction"
        assert output["message"] == "debug message"

    def test_info_log(self, capfd):
        """Deve emitir log INFO com stage correto."""
        stage_logger, base_logger = self._create_stage_logger("ml-inference")
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)

        stage_logger.info("info message")
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert output["level"] == "INFO"
        assert output["stage"] == "ml-inference"

    def test_warning_log(self, capfd):
        """Deve emitir log WARNING com stage correto."""
        stage_logger, base_logger = self._create_stage_logger("feature-engineering")
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)

        stage_logger.warning("warning message")
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert output["level"] == "WARNING"
        assert output["stage"] == "feature-engineering"

    def test_error_log(self, capfd):
        """Deve emitir log ERROR com stage correto."""
        stage_logger, base_logger = self._create_stage_logger("bedrock-explanation")
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)

        stage_logger.error("error message")
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert output["level"] == "ERROR"
        assert output["stage"] == "bedrock-explanation"

    def test_critical_log(self, capfd):
        """Deve emitir log CRITICAL com stage correto."""
        stage_logger, base_logger = self._create_stage_logger("report-generation")
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)

        stage_logger.critical("critical message")
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert output["level"] == "CRITICAL"
        assert output["stage"] == "report-generation"

    def test_extra_fields_propagated(self, capfd):
        """Deve propagar campos extras para o JSON."""
        stage_logger, base_logger = self._create_stage_logger("extraction")
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)

        stage_logger.info("processing", extra={"user_id": "u-001", "batch_size": 100})
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert output["user_id"] == "u-001"
        assert output["batch_size"] == 100

    def test_log_stage_start(self, capfd):
        """Deve logar início do estágio."""
        stage_logger, base_logger = self._create_stage_logger("explainability")
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)

        stage_logger.log_stage_start()
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert output["level"] == "INFO"
        assert "explainability" in output["message"]
        assert output["stage"] == "explainability"

    def test_log_stage_completion(self, capfd):
        """Deve logar conclusão do estágio com duração."""
        stage_logger, base_logger = self._create_stage_logger("ml-inference")
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)

        stage_logger.log_stage_completion(duration_seconds=12.345)
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert output["level"] == "INFO"
        assert "ml-inference" in output["message"]
        assert output["duration_seconds"] == 12.345

    def test_log_stage_completion_without_duration(self, capfd):
        """Deve logar conclusão sem duração quando não fornecida."""
        stage_logger, base_logger = self._create_stage_logger("dashboard")
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        base_logger.addHandler(handler)

        stage_logger.log_stage_completion()
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert "duration_seconds" not in output

    def test_stage_property(self):
        """Deve expor o nome do estágio via property."""
        stage_logger, _ = self._create_stage_logger("extraction")
        assert stage_logger.stage == "extraction"


class TestGetLogger:
    """Testes para a factory function get_logger."""

    def test_returns_stage_logger(self):
        """Deve retornar uma instância de StageLogger."""
        logger = get_logger("extraction")
        assert isinstance(logger, StageLogger)

    def test_stage_is_set_correctly(self):
        """Deve configurar o stage corretamente."""
        logger = get_logger("ml-inference")
        assert logger.stage == "ml-inference"

    def test_custom_log_level(self, capfd):
        """Deve respeitar o nível de log configurado."""
        # Usar stage único para evitar conflito com handlers de outros testes
        unique_stage = f"level-test-{uuid.uuid4().hex[:8]}"
        logger = get_logger(unique_stage, level=logging.WARNING)
        logger.info("should not appear")
        logger.warning("should appear")
        captured = capfd.readouterr()
        # INFO não deve aparecer, WARNING sim
        lines = [line for line in captured.err.strip().split("\n") if line]
        assert len(lines) == 1
        output = json.loads(lines[0])
        assert output["level"] == "WARNING"

    def test_execution_id_propagated(self, capfd):
        """Deve propagar o execution_id no JSON de saída."""
        set_execution_id("pipeline-run-001")
        logger = get_logger("feature-engineering")
        logger.info("test propagation")
        captured = capfd.readouterr()
        output = json.loads(captured.err)
        assert output["execution_id"] == "pipeline-run-001"

    def test_no_duplicate_handlers(self):
        """Não deve adicionar handlers duplicados ao mesmo logger."""
        # Chamar get_logger múltiplas vezes para o mesmo stage
        logger1 = get_logger("extraction")
        logger2 = get_logger("extraction")
        # O logger base é o mesmo (mesmo name)
        assert logger1._logger is logger2._logger
        assert len(logger1._logger.handlers) == 1
