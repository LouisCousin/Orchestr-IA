"""Tests unitaires pour le module content_validator."""

import pytest

from src.utils.content_validator import is_antibot_page, is_valid_pdf_content


class TestIsAntibotPage:
    def test_anubis_page(self):
        """CA4-2 : Page Anubis détectée comme anti-bot."""
        text = "Making sure you're not a bot! Please wait while proof-of-work is computed."
        assert is_antibot_page(text) is True

    def test_cloudflare_page(self):
        """Page Cloudflare détectée comme anti-bot."""
        text = "Attention Required! Cloudflare. Enable JavaScript and cookies to continue. Ray ID: abc123"
        assert is_antibot_page(text) is True

    def test_normal_content(self):
        """CA4-3 : Contenu normal retourne False."""
        text = (
            "Ceci est un rapport d'analyse détaillé sur l'économie mondiale. "
            "Les indicateurs macroéconomiques montrent une reprise progressive "
            "dans les principaux marchés développés. " * 10
        )
        assert is_antibot_page(text) is False

    def test_empty_text(self):
        """Empty/None text is not an anti-bot page (no keywords matched)."""
        assert is_antibot_page("") is False
        assert is_antibot_page(None) is False

    def test_short_text_with_keyword(self):
        text = "Checking your browser. Please wait."
        assert is_antibot_page(text) is True

    def test_long_text_with_single_keyword(self):
        """Long text with one keyword should not trigger."""
        text = "This is a real page about cloudflare as a technology company. " * 20
        assert is_antibot_page(text) is False


class TestIsValidPdfContent:
    def test_valid_pdf(self):
        """CA4-5 : Vrai PDF reconnu."""
        content = b"%PDF-1.4 some content here..."
        assert is_valid_pdf_content(content) is True

    def test_html_as_pdf(self):
        """CA4-4 : Faux PDF (HTML) détecté."""
        content = b"<html><body>Not a real PDF</body></html>"
        assert is_valid_pdf_content(content) is False

    def test_empty_content(self):
        assert is_valid_pdf_content(b"") is False
