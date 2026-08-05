"""Testes para o módulo de configuração src/common/config.py.

Verifica:
- Carregamento correto do YAML
- Override de variáveis de ambiente
- Validação de valores obrigatórios
- Validação de ranges
- Fail-fast com mensagens descritivas
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.common.config import (
    ConfigurationError,
    Settings,
    get_settings,
    reset_settings_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_valid_config() -> dict[str, Any]:
    """Configuração mínima válida para testes."""
    return {
        "npaw": {
            "account_code": "sky_brazil",
            "api_key": "test-api-key-123",
            "base_url": "https://api.npaw.com",
        },
        "observation": {
            "time_window_months": 6,
            "min_sessions": 5,
            "min_weeks_for_trends": 4,
        },
        "sagemaker": {
            "region": "us-east-1",
            "algorithm": "xgboost",
        },
        "bedrock": {
            "region": "us-east-1",
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        },
        "reports": {
            "formats": ["json", "markdown"],
            "output_bucket": "sky-brazil-churn-prediction",
            "output_prefix": "reports",
        },
    }


def _write_yaml_config(data: dict[str, Any], dir_path: Path) -> Path:
    """Escreve um arquivo settings.yaml temporário e retorna o caminho."""
    config_dir = dir_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "settings.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
    return config_file


@pytest.fixture(autouse=True)
def _clean_env_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limpa variáveis de ambiente CHURN_* e cache entre testes."""
    # Remove qualquer variável CHURN_ que possa existir
    for key in list(os.environ.keys()):
        if key.startswith("CHURN_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()


# ---------------------------------------------------------------------------
# Testes de carregamento YAML
# ---------------------------------------------------------------------------


class TestYAMLLoading:
    """Testes de carregamento do arquivo YAML."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Carrega configuração válida do YAML com sucesso."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        settings = Settings.load(config_file)

        assert settings.npaw.account_code == "sky_brazil"
        assert settings.npaw.api_key == "test-api-key-123"
        assert settings.observation.time_window_months == 6
        assert settings.observation.min_sessions == 5
        assert settings.bedrock.region == "us-east-1"
        assert settings.bedrock.model_id == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_load_full_config(self, tmp_path: Path) -> None:
        """Carrega configuração completa com todas as seções."""
        data = _minimal_valid_config()
        data["explainability"] = {"top_features": 10, "method": "shap"}
        data["prediction"] = {"risk_thresholds": {"low_max": 30, "medium_max": 60}, "batch_size": 1000}
        data["monitoring"] = {"drift_threshold_std": 2.0, "max_inference_time_ms": 5000, "max_failure_rate_pct": 5.0}
        data["dashboard"] = {"port": 8501, "refresh_on_new_execution": True}
        config_file = _write_yaml_config(data, tmp_path)

        settings = Settings.load(config_file)

        assert settings.explainability.top_features == 10
        assert settings.prediction.batch_size == 1000
        assert settings.monitoring.drift_threshold_std == 2.0
        assert settings.dashboard.port == 8501

    def test_file_not_found_raises_config_error(self) -> None:
        """Arquivo inexistente gera ConfigurationError."""
        with pytest.raises(ConfigurationError, match="não encontrado"):
            Settings.load("/caminho/inexistente/settings.yaml")

    def test_invalid_yaml_raises_config_error(self, tmp_path: Path) -> None:
        """YAML malformado gera ConfigurationError."""
        config_file = tmp_path / "config" / "settings.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("invalid: yaml: [broken", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="parsear"):
            Settings.load(config_file)

    def test_non_dict_yaml_raises_config_error(self, tmp_path: Path) -> None:
        """YAML que não é um dicionário gera ConfigurationError."""
        config_file = tmp_path / "config" / "settings.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="mapeamento YAML"):
            Settings.load(config_file)


# ---------------------------------------------------------------------------
# Testes de override via variáveis de ambiente
# ---------------------------------------------------------------------------


class TestEnvironmentOverrides:
    """Testes de override de configuração via variáveis de ambiente."""

    def test_env_var_overrides_yaml_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Variável de ambiente sobrescreve valor do YAML."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        monkeypatch.setenv("CHURN_NPAW_API_KEY", "env-api-key-override")

        settings = Settings.load(config_file)
        assert settings.npaw.api_key == "env-api-key-override"

    def test_env_var_overrides_bedrock_region(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CHURN_BEDROCK_REGION sobrescreve o region do Bedrock."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        monkeypatch.setenv("CHURN_BEDROCK_REGION", "eu-west-1")

        settings = Settings.load(config_file)
        assert settings.bedrock.region == "eu-west-1"

    def test_env_var_overrides_observation_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CHURN_OBSERVATION_TIME_WINDOW_MONTHS sobrescreve janela de observação."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        monkeypatch.setenv("CHURN_OBSERVATION_TIME_WINDOW_MONTHS", "12")

        settings = Settings.load(config_file)
        assert settings.observation.time_window_months == 12

    def test_env_var_overrides_min_sessions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CHURN_OBSERVATION_MIN_SESSIONS sobrescreve mínimo de sessões."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        monkeypatch.setenv("CHURN_OBSERVATION_MIN_SESSIONS", "10")

        settings = Settings.load(config_file)
        assert settings.observation.min_sessions == 10

    def test_env_var_provides_missing_yaml_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Variável de ambiente fornece valor ausente no YAML."""
        data = _minimal_valid_config()
        del data["npaw"]["api_key"]  # Remover do YAML
        config_file = _write_yaml_config(data, tmp_path)

        monkeypatch.setenv("CHURN_NPAW_API_KEY", "from-env")

        settings = Settings.load(config_file)
        assert settings.npaw.api_key == "from-env"

    def test_env_var_not_set_uses_yaml(self, tmp_path: Path) -> None:
        """Sem variável de ambiente, usa valor do YAML."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        settings = Settings.load(config_file)
        assert settings.npaw.api_key == "test-api-key-123"


# ---------------------------------------------------------------------------
# Testes de validação de valores obrigatórios (fail-fast)
# ---------------------------------------------------------------------------


class TestRequiredFieldValidation:
    """Testes de validação de campos obrigatórios com fail-fast."""

    def test_missing_npaw_api_key_fails(self, tmp_path: Path) -> None:
        """Ausência de api_key gera erro com nome do parâmetro e fonte."""
        data = _minimal_valid_config()
        del data["npaw"]["api_key"]
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "npaw.api_key" in error_msg
        assert "CHURN_NPAW_API_KEY" in error_msg

    def test_missing_bedrock_region_fails(self, tmp_path: Path) -> None:
        """Ausência de bedrock.region gera erro com nome do parâmetro."""
        data = _minimal_valid_config()
        del data["bedrock"]["region"]
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "bedrock.region" in error_msg

    def test_missing_bedrock_model_id_fails(self, tmp_path: Path) -> None:
        """Ausência de bedrock.model_id gera erro com nome do parâmetro."""
        data = _minimal_valid_config()
        del data["bedrock"]["model_id"]
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "bedrock.model_id" in error_msg

    def test_missing_reports_output_bucket_fails(self, tmp_path: Path) -> None:
        """Ausência de reports.output_bucket gera erro descritivo."""
        data = _minimal_valid_config()
        del data["reports"]["output_bucket"]
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "reports.output_bucket" in error_msg

    def test_empty_api_key_fails(self, tmp_path: Path) -> None:
        """api_key vazia gera erro de validação."""
        data = _minimal_valid_config()
        data["npaw"]["api_key"] = ""
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "npaw.api_key" in error_msg


# ---------------------------------------------------------------------------
# Testes de validação de ranges
# ---------------------------------------------------------------------------


class TestRangeValidation:
    """Testes de validação de ranges e valores inválidos."""

    def test_time_window_below_range_fails(self, tmp_path: Path) -> None:
        """time_window_months < 1 gera erro com range."""
        data = _minimal_valid_config()
        data["observation"]["time_window_months"] = 0
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "observation.time_window_months" in error_msg

    def test_time_window_above_range_fails(self, tmp_path: Path) -> None:
        """time_window_months > 24 gera erro com range."""
        data = _minimal_valid_config()
        data["observation"]["time_window_months"] = 25
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "observation.time_window_months" in error_msg

    def test_min_sessions_below_range_fails(self, tmp_path: Path) -> None:
        """min_sessions < 1 gera erro."""
        data = _minimal_valid_config()
        data["observation"]["min_sessions"] = 0
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "observation.min_sessions" in error_msg

    def test_min_sessions_above_range_fails(self, tmp_path: Path) -> None:
        """min_sessions > 10000 gera erro."""
        data = _minimal_valid_config()
        data["observation"]["min_sessions"] = 10001
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "observation.min_sessions" in error_msg

    def test_invalid_algorithm_fails(self, tmp_path: Path) -> None:
        """Algoritmo inválido gera erro com valores aceitos."""
        data = _minimal_valid_config()
        data["sagemaker"]["algorithm"] = "random_forest"
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "sagemaker.algorithm" in error_msg
        assert "xgboost" in error_msg

    def test_invalid_explainability_method_fails(self, tmp_path: Path) -> None:
        """Método de explicabilidade inválido gera erro."""
        data = _minimal_valid_config()
        data["explainability"] = {"method": "invalid_method", "top_features": 10}
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "explainability.method" in error_msg

    def test_risk_thresholds_invalid_order_fails(self, tmp_path: Path) -> None:
        """low_max >= medium_max gera erro."""
        data = _minimal_valid_config()
        data["prediction"] = {
            "risk_thresholds": {"low_max": 70, "medium_max": 60},
            "batch_size": 1000,
        }
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "low_max" in error_msg or "medium_max" in error_msg

    def test_valid_boundary_values_accepted(self, tmp_path: Path) -> None:
        """Valores nos limites do range são aceitos."""
        data = _minimal_valid_config()
        data["observation"]["time_window_months"] = 1  # mínimo
        data["observation"]["min_sessions"] = 10000  # máximo
        config_file = _write_yaml_config(data, tmp_path)

        settings = Settings.load(config_file)
        assert settings.observation.time_window_months == 1
        assert settings.observation.min_sessions == 10000


# ---------------------------------------------------------------------------
# Testes da função get_settings (singleton/cache)
# ---------------------------------------------------------------------------


class TestGetSettings:
    """Testes da função get_settings com cache."""

    def test_get_settings_returns_same_instance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_settings retorna a mesma instância em chamadas subsequentes."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        reset_settings_cache()
        s1 = get_settings(str(config_file))
        s2 = get_settings(str(config_file))
        assert s1 is s2

    def test_reset_cache_allows_reload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reset_settings_cache permite recarregar configurações."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        s1 = get_settings(str(config_file))

        reset_settings_cache()
        monkeypatch.setenv("CHURN_NPAW_API_KEY", "new-key-after-reset")

        s2 = get_settings(str(config_file))
        assert s2.npaw.api_key == "new-key-after-reset"
        assert s1 is not s2


# ---------------------------------------------------------------------------
# Testes de load_from_dict
# ---------------------------------------------------------------------------


class TestLoadFromDict:
    """Testes de carregamento a partir de dicionário."""

    def test_load_from_dict_valid(self) -> None:
        """Carrega configuração válida de um dicionário."""
        data = _minimal_valid_config()
        settings = Settings.load_from_dict(data)
        assert settings.npaw.account_code == "sky_brazil"

    def test_load_from_dict_invalid_raises(self) -> None:
        """Dicionário inválido gera ConfigurationError."""
        data = {"npaw": {"account_code": "test"}}  # Faltam campos obrigatórios

        with pytest.raises(ConfigurationError):
            Settings.load_from_dict(data)


# ---------------------------------------------------------------------------
# Testes adicionais: valor não numérico para campo numérico (Req 7.6)
# ---------------------------------------------------------------------------


class TestInvalidTypeValidation:
    """Testes de validação quando tipo é incompatível (Req 7.6)."""

    def test_non_numeric_time_window_fails(self, tmp_path: Path) -> None:
        """Valor não numérico para time_window_months gera erro."""
        data = _minimal_valid_config()
        data["observation"]["time_window_months"] = "abc"
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "observation.time_window_months" in error_msg

    def test_non_numeric_min_sessions_fails(self, tmp_path: Path) -> None:
        """Valor não numérico para min_sessions gera erro."""
        data = _minimal_valid_config()
        data["observation"]["min_sessions"] = "muitas"
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "observation.min_sessions" in error_msg

    def test_non_numeric_batch_size_fails(self, tmp_path: Path) -> None:
        """Valor não numérico para batch_size gera erro."""
        data = _minimal_valid_config()
        data["npaw"]["batch_size"] = "grande"
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "npaw.batch_size" in error_msg


# ---------------------------------------------------------------------------
# Testes adicionais: mensagem de erro inclui valor e range (Req 7.6)
# ---------------------------------------------------------------------------


class TestErrorMessageContent:
    """Verifica que as mensagens de erro contêm nome, fonte e range."""

    def test_error_includes_param_name_and_source(
        self, tmp_path: Path
    ) -> None:
        """Erro de parâmetro ausente inclui nome do parâmetro e fonte."""
        data = _minimal_valid_config()
        del data["npaw"]["api_key"]
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        # Req 7.5: mensagem deve incluir nome do parâmetro
        assert "npaw.api_key" in error_msg
        # Req 7.5: mensagem deve incluir fonte esperada
        assert "CHURN_NPAW_API_KEY" in error_msg or "config" in error_msg.lower()

    def test_range_error_includes_value_and_constraint(
        self, tmp_path: Path
    ) -> None:
        """Erro de range inclui valor inválido fornecido."""
        data = _minimal_valid_config()
        data["observation"]["time_window_months"] = 99
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        # Req 7.6: inclui nome do parâmetro
        assert "observation.time_window_months" in error_msg
        # Req 7.6: inclui valor fornecido
        assert "99" in error_msg

    def test_invalid_algorithm_error_shows_accepted_values(
        self, tmp_path: Path
    ) -> None:
        """Erro de algoritmo inválido mostra valores aceitos."""
        data = _minimal_valid_config()
        data["sagemaker"]["algorithm"] = "neural_network"
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "sagemaker.algorithm" in error_msg
        # Req 7.6: mostra formato/range aceitável
        assert "xgboost" in error_msg
        assert "lightgbm" in error_msg
        assert "catboost" in error_msg

    def test_missing_section_error_includes_param_names(
        self, tmp_path: Path
    ) -> None:
        """Seção inteira ausente gera erro com nomes dos parâmetros."""
        data = _minimal_valid_config()
        del data["npaw"]
        config_file = _write_yaml_config(data, tmp_path)

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        # Deve mencionar a seção/campo ausente
        assert "npaw" in error_msg


# ---------------------------------------------------------------------------
# Testes adicionais: env var com valor inválido (Req 7.6)
# ---------------------------------------------------------------------------


class TestEnvVarInvalidValues:
    """Testes de variáveis de ambiente com valores inválidos."""

    def test_env_var_with_invalid_range_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var com valor fora do range gera ConfigurationError."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        # time_window_months deve ser entre 1-24
        monkeypatch.setenv("CHURN_OBSERVATION_TIME_WINDOW_MONTHS", "50")

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "observation.time_window_months" in error_msg

    def test_env_var_with_empty_string_for_required_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var com string vazia para campo obrigatório gera erro."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        monkeypatch.setenv("CHURN_NPAW_API_KEY", "")

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "npaw.api_key" in error_msg

    def test_env_var_with_invalid_algorithm_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var CHURN_SAGEMAKER_ALGORITHM com valor inválido gera erro."""
        data = _minimal_valid_config()
        config_file = _write_yaml_config(data, tmp_path)

        monkeypatch.setenv("CHURN_SAGEMAKER_ALGORITHM", "invalid_algo")

        with pytest.raises(ConfigurationError) as exc_info:
            Settings.load(config_file)

        error_msg = str(exc_info.value)
        assert "sagemaker.algorithm" in error_msg
        assert "xgboost" in error_msg
