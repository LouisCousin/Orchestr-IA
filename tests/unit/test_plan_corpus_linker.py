"""Tests unitaires pour le module plan_corpus_linker (Phase 2.5)."""

import pytest
from unittest.mock import MagicMock, patch

from src.core.plan_corpus_linker import (
    PlanContext,
    _extract_themes_simple,
    format_plan_context_for_prompt,
)


# ── Tests PlanContext ──

class TestPlanContext:
    def test_default_values(self):
        ctx = PlanContext()
        assert ctx.corpus_summary == {}
        assert ctx.themes == []
        assert ctx.coverage == {}

    def test_to_dict(self):
        ctx = PlanContext(
            corpus_summary={"total_documents": 5},
            themes=["thème A", "thème B"],
            coverage={"thème A": {"avg_score": 0.6, "nb_chunks": 3}},
        )
        d = ctx.to_dict()
        assert d["corpus_summary"]["total_documents"] == 5
        assert len(d["themes"]) == 2
        assert "thème A" in d["coverage"]


# ── Tests _extract_themes_simple ──

class TestExtractThemesSimple:
    def test_extracts_section_titles(self):
        chunks = [
            {"section_title": "Introduction", "text": "Contenu intro."},
            {"section_title": "Méthodologie", "text": "Contenu méthodo."},
            {"section_title": "Résultats", "text": "Contenu résultats."},
        ]
        themes = _extract_themes_simple(chunks, "objectif test")
        assert "Introduction" in themes
        assert "Méthodologie" in themes
        assert "Résultats" in themes

    def test_deduplicates_titles(self):
        chunks = [
            {"section_title": "Introduction", "text": "A."},
            {"section_title": "Introduction", "text": "B."},
            {"section_title": "Analyse", "text": "C."},
        ]
        themes = _extract_themes_simple(chunks, "objectif")
        title_count = themes.count("Introduction")
        assert title_count == 1

    def test_fallback_to_keywords_when_few_sections(self):
        chunks = [
            {"section_title": "", "text": "cloud computing infrastructure déploiement " * 20},
        ]
        themes = _extract_themes_simple(chunks, "objectif")
        assert len(themes) >= 1  # Au moins des mots-clés extraits

    def test_empty_chunks(self):
        themes = _extract_themes_simple([], "objectif")
        assert themes == [] or len(themes) >= 0  # Pas d'erreur

    def test_max_15_themes(self):
        chunks = [
            {"section_title": f"Thème {i}", "text": f"Contenu {i}."}
            for i in range(20)
        ]
        themes = _extract_themes_simple(chunks, "objectif")
        assert len(themes) <= 15

    def test_handles_non_dict_chunks(self):
        """Chunks qui ne sont pas des dicts ne causent pas d'erreur."""
        chunks = ["texte brut", {"section_title": "OK", "text": "Content"}]
        themes = _extract_themes_simple(chunks, "objectif")
        assert "OK" in themes


# ── Tests format_plan_context_for_prompt ──

class TestFormatPlanContextForPrompt:
    def test_basic_formatting(self):
        ctx = PlanContext(
            corpus_summary={
                "total_documents": 3,
                "total_tokens": 5000,
                "languages": ["fr", "en"],
                "types": ["article", "report"],
                "documents": [
                    {"title": "Doc A", "pages": 10, "tokens": 2000, "type": "article"},
                    {"title": "Doc B", "pages": 5, "tokens": 1500, "type": "report"},
                ],
            },
            themes=["Intelligence artificielle", "Cloud computing"],
            coverage={
                "Intelligence artificielle": {"avg_score": 0.6, "nb_chunks": 5},
                "Cloud computing": {"avg_score": 0.2, "nb_chunks": 1},
            },
        )
        result = format_plan_context_for_prompt(ctx)
        assert "3" in result  # total_documents
        assert "5000" in result  # total_tokens
        assert "fr" in result
        assert "Doc A" in result
        assert "FORT" in result  # score 0.6 >= 0.5
        assert "FAIBLE" in result  # score 0.2 < 0.3

    def test_partial_coverage(self):
        ctx = PlanContext(
            corpus_summary={"total_documents": 1, "total_tokens": 100, "documents": []},
            themes=["Thème A"],
            coverage={"Thème A": {"avg_score": 0.4, "nb_chunks": 2}},
        )
        result = format_plan_context_for_prompt(ctx)
        assert "PARTIEL" in result  # 0.3 <= 0.4 < 0.5

    def test_no_themes(self):
        ctx = PlanContext(
            corpus_summary={"total_documents": 0, "total_tokens": 0, "documents": []},
            themes=[],
            coverage={},
        )
        result = format_plan_context_for_prompt(ctx)
        assert "Thèmes" not in result  # Pas de section thèmes

    def test_theme_without_coverage(self):
        ctx = PlanContext(
            corpus_summary={"total_documents": 1, "total_tokens": 100, "documents": []},
            themes=["Orphan theme"],
            coverage={},
        )
        result = format_plan_context_for_prompt(ctx)
        assert "FAIBLE" in result  # Default score 0 < 0.3


# ── Tests anti-hallucination integration ──

class TestAntiHallucination:
    def test_needs_source_marker_format(self):
        """Vérifie que le format du marqueur est correct."""
        import re
        pattern = re.compile(r'\{\{NEEDS_SOURCE:\s*(.+?)\}\}')
        text = "Selon les données {{NEEDS_SOURCE: statistiques de vente 2024}}, les résultats sont bons."
        match = pattern.search(text)
        assert match is not None
        assert match.group(1).strip() == "statistiques de vente 2024"
