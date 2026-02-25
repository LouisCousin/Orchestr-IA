"""Tests unitaires pour l'isolation du config dans l'Orchestrator (Bug #1)."""

import copy
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.core.orchestrator import _normalize_config, Orchestrator


class TestNormalizeConfigNoMutation:
    """Vérifie que _normalize_config ne mute pas le dict d'entrée."""

    def test_does_not_mutate_input(self):
        original = {
            "conditional_generation": {"enabled": True, "sufficient_threshold": 0.7},
            "anti_hallucination": {"enabled": False},
        }
        frozen = copy.deepcopy(original)
        result = _normalize_config(original)
        # The original dict must be unchanged
        assert original == frozen
        # The result must have the flattened keys
        assert result["conditional_generation_enabled"] is True
        assert result["coverage_sufficient_threshold"] == 0.7
        assert result["anti_hallucination_enabled"] is False

    def test_flat_keys_already_present_take_precedence(self):
        original = {
            "conditional_generation": {"enabled": True},
            "conditional_generation_enabled": False,  # explicit flat key
        }
        frozen = copy.deepcopy(original)
        result = _normalize_config(original)
        assert original == frozen
        # Flat key already present → must NOT be overwritten
        assert result["conditional_generation_enabled"] is False

    def test_empty_config(self):
        original = {}
        result = _normalize_config(original)
        assert original == {}
        assert isinstance(result, dict)

    def test_no_nested_keys(self):
        original = {"model": "gpt-4o", "temperature": 0.5}
        frozen = copy.deepcopy(original)
        result = _normalize_config(original)
        assert original == frozen
        assert result["model"] == "gpt-4o"


class TestOrchestratorConfigIsolation:
    """Vérifie que l'Orchestrator deepcopy son config."""

    def test_init_does_not_share_config_reference(self):
        provider = MagicMock()
        provider.get_default_model.return_value = "gpt-4o"
        original_config = {
            "model": "gpt-4o",
            "temperature": 0.7,
            "conditional_generation": {"enabled": True},
        }
        frozen = copy.deepcopy(original_config)

        orch = Orchestrator(
            provider=provider,
            project_dir=Path("/tmp/test_orch_config"),
            config=original_config,
        )

        # Mutate the orchestrator's config
        orch.config["model"] = "gpt-3.5-turbo"
        orch.config["new_key"] = "injected"

        # Original must be untouched
        assert original_config == frozen

    def test_nested_dict_isolation(self):
        provider = MagicMock()
        provider.get_default_model.return_value = "gpt-4o"
        original_config = {
            "rag": {"top_k": 10, "chunking": {"strategy": "fixed"}},
            "quality_evaluation": {"weights": {"C1": 0.3}},
        }
        frozen = copy.deepcopy(original_config)

        orch = Orchestrator(
            provider=provider,
            project_dir=Path("/tmp/test_orch_nested"),
            config=original_config,
        )

        # Mutate a nested dict inside the orchestrator
        orch.config["rag"]["top_k"] = 999
        orch.config["quality_evaluation"]["weights"]["C1"] = 0.0

        # Original nested dicts must be untouched
        assert original_config == frozen
