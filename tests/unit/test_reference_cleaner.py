"""Tests unitaires pour le module reference_cleaner."""

import pytest

from src.utils.reference_cleaner import clean_source_references, has_source_references


class TestCleanSourceReferences:
    def test_simple_source_reference(self):
        """CA3-5 : Suppression de [Source 5]."""
        result = clean_source_references("Selon [Source 5], la tendance...")
        assert "[Source 5]" not in result
        assert "tendance" in result

    def test_source_with_colon(self):
        """Suppression de [Source 3: rapport.pdf]."""
        result = clean_source_references("D'après [Source 3: rapport.pdf], les données...")
        assert "[Source 3: rapport.pdf]" not in result
        assert "données" in result

    def test_multiple_references(self):
        text = "[Source 1] indique et [Source 2] confirme."
        result = clean_source_references(text)
        assert "[Source 1]" not in result
        assert "[Source 2]" not in result

    def test_case_insensitive(self):
        result = clean_source_references("Voir [source 3] pour détails.")
        assert "[source 3]" not in result

    def test_no_references(self):
        text = "Un texte sans références."
        result = clean_source_references(text)
        assert result.strip() == text

    def test_source_with_space_before_number(self):
        result = clean_source_references("Voir [Source 5] pour plus.")
        assert "[Source 5]" not in result

    def test_with_source_map(self):
        source_map = {5: "rapport_2024.pdf"}
        result = clean_source_references("Selon [Source 5], les résultats...", source_map=source_map)
        assert "rapport_2024.pdf" in result
        assert "[Source 5]" not in result


class TestHasSourceReferences:
    def test_has_references(self):
        assert has_source_references("Voir [Source 3] pour détails.")

    def test_no_references(self):
        assert not has_source_references("Un texte normal.")

    def test_source_with_colon(self):
        assert has_source_references("D'après [Source 1: doc.pdf]...")
