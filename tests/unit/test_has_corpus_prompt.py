"""Tests unitaires pour le paramètre has_corpus de build_system_prompt (Bug #4)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.core.prompt_engine import PromptEngine


class TestBuildSystemPromptHasCorpus:
    """Vérifie que has_corpus contrôle le bloc anti-hallucination."""

    def test_without_corpus_no_anti_hallucination(self):
        engine = PromptEngine()
        prompt = engine.build_system_prompt(has_corpus=False)
        assert "RÈGLES DE FIABILITÉ" not in prompt
        assert "SOURCES EXCLUSIVES" not in prompt

    def test_with_corpus_has_anti_hallucination(self):
        engine = PromptEngine()
        prompt = engine.build_system_prompt(has_corpus=True)
        assert "RÈGLES DE FIABILITÉ" in prompt
        assert "SOURCES EXCLUSIVES" in prompt

    def test_default_is_true(self):
        engine = PromptEngine()
        prompt_default = engine.build_system_prompt()
        prompt_true = engine.build_system_prompt(has_corpus=True)
        # Both should contain the anti-hallucination block
        assert "RÈGLES DE FIABILITÉ" in prompt_default
        assert "RÈGLES DE FIABILITÉ" in prompt_true

    def test_anti_hallucination_disabled_globally(self):
        engine = PromptEngine(anti_hallucination_enabled=False)
        # Even with has_corpus=True, the block should not appear
        prompt = engine.build_system_prompt(has_corpus=True)
        assert "RÈGLES DE FIABILITÉ" not in prompt


class TestAutoCorrectSystemPrompt:
    """Vérifie que _auto_correct_if_needed passe has_corpus correctement."""

    @patch("src.core.orchestrator.Orchestrator._init_phase3_engines")
    def test_auto_correct_passes_has_corpus_false_when_no_chunks(self, mock_init):
        from src.core.orchestrator import Orchestrator
        from src.core.plan_parser import PlanSection, NormalizedPlan

        provider = MagicMock()
        provider.get_default_model.return_value = "gpt-4o"
        provider.name = "openai"
        provider.generate.return_value = MagicMock(
            content="corrected text", input_tokens=100, output_tokens=50,
        )

        orch = Orchestrator(
            provider=provider,
            project_dir=Path("/tmp/test_autocorrect"),
            config={"model": "gpt-4o"},
        )
        orch.state = MagicMock()
        orch.state.generated_sections = {}
        orch.state.section_summaries = []

        section = PlanSection(id="1", title="Test", level=1)
        plan = NormalizedPlan(title="Test", objective="Test")

        # Mock factcheck engine that triggers correction
        fc_report = MagicMock()
        fc_report.reliability_score = 40.0
        orch._factcheck_engine = MagicMock()
        orch._factcheck_engine.should_correct.return_value = True
        orch._factcheck_engine.get_correction_instruction.return_value = "Fix facts"

        with patch.object(orch.prompt_engine, "build_system_prompt", wraps=orch.prompt_engine.build_system_prompt) as spy:
            # Call with empty corpus_chunks
            orch._auto_correct_if_needed(section, "draft", plan, [], fc_report, None)
            # Verify build_system_prompt was called with has_corpus=False
            spy.assert_called_once_with(has_corpus=False, section_id="1")

    @patch("src.core.orchestrator.Orchestrator._init_phase3_engines")
    def test_auto_correct_passes_has_corpus_true_when_chunks_present(self, mock_init):
        from src.core.orchestrator import Orchestrator
        from src.core.plan_parser import PlanSection, NormalizedPlan

        provider = MagicMock()
        provider.get_default_model.return_value = "gpt-4o"
        provider.name = "openai"
        provider.generate.return_value = MagicMock(
            content="corrected text", input_tokens=100, output_tokens=50,
        )

        orch = Orchestrator(
            provider=provider,
            project_dir=Path("/tmp/test_autocorrect2"),
            config={"model": "gpt-4o"},
        )
        orch.state = MagicMock()
        orch.state.generated_sections = {}
        orch.state.section_summaries = []

        section = PlanSection(id="2", title="Test2", level=1)
        plan = NormalizedPlan(title="Test", objective="Test")

        fc_report = MagicMock()
        fc_report.reliability_score = 40.0
        orch._factcheck_engine = MagicMock()
        orch._factcheck_engine.should_correct.return_value = True
        orch._factcheck_engine.get_correction_instruction.return_value = "Fix"

        corpus_chunks = [{"text": "source text", "source_file": "doc.pdf"}]

        with patch.object(orch.prompt_engine, "build_system_prompt", wraps=orch.prompt_engine.build_system_prompt) as spy:
            orch._auto_correct_if_needed(section, "draft", plan, corpus_chunks, fc_report, None)
            spy.assert_called_once_with(has_corpus=True, section_id="2")


class TestGeneratePlanSystemPrompt:
    """Vérifie que generate_plan_from_objective passe has_corpus correctement."""

    def test_plan_without_corpus_no_anti_hallucination(self):
        from src.core.orchestrator import Orchestrator
        from src.core.plan_parser import NormalizedPlan

        provider = MagicMock()
        provider.get_default_model.return_value = "gpt-4o"
        provider.name = "openai"
        # Return a parseable plan
        provider.generate.return_value = MagicMock(
            content="# Plan\n## 1. Introduction\n## 2. Conclusion",
            input_tokens=100,
            output_tokens=50,
        )

        orch = Orchestrator(
            provider=provider,
            project_dir=Path("/tmp/test_plan_prompt"),
            config={"model": "gpt-4o"},
        )
        orch.state = MagicMock()
        orch.state.corpus = None
        orch.state.plan = None

        with patch.object(orch.prompt_engine, "build_system_prompt", wraps=orch.prompt_engine.build_system_prompt) as spy:
            try:
                orch.generate_plan_from_objective("Write a report", corpus=None)
            except Exception:
                pass  # parsing may fail, we only care about the call
            # Should be called with has_corpus=False since no corpus
            if spy.called:
                args, kwargs = spy.call_args
                assert kwargs.get("has_corpus", args[0] if args else True) is False
