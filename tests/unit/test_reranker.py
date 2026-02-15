"""Tests unitaires pour le module reranker (Phase 2.5)."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.core.reranker import Reranker, ScoredChunk, build_context, DEFAULT_RERANKER_MODEL


@pytest.fixture(autouse=True)
def reset_singleton():
    """Réinitialise le singleton avant chaque test."""
    Reranker.reset_instance()
    yield
    Reranker.reset_instance()


def _make_scored_chunk(
    chunk_id="c1", doc_id="d1", text="Texte du chunk.",
    page_number=1, section_title="Intro", cosine_score=0.8,
    rerank_score=0.0, doc_title="", doc_authors="", apa_reference="",
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, doc_id=doc_id, text=text,
        page_number=page_number, section_title=section_title,
        cosine_score=cosine_score, rerank_score=rerank_score,
        doc_title=doc_title, doc_authors=doc_authors,
        apa_reference=apa_reference,
    )


# ── Tests ScoredChunk ──

class TestScoredChunk:
    def test_to_dict(self):
        chunk = _make_scored_chunk(cosine_score=0.85123, rerank_score=0.92456)
        d = chunk.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["doc_id"] == "d1"
        assert d["cosine_score"] == 0.8512
        assert d["rerank_score"] == 0.9246

    def test_default_values(self):
        chunk = ScoredChunk(
            chunk_id="c1", doc_id="d1", text="test",
            page_number=1, section_title="", cosine_score=0.5,
        )
        assert chunk.rerank_score == 0.0
        assert chunk.doc_title == ""
        assert chunk.doc_authors == ""
        assert chunk.apa_reference == ""

    def test_enriched_metadata(self):
        chunk = _make_scored_chunk(
            doc_title="Rapport 2024",
            doc_authors="Dupont, J.",
            apa_reference="Dupont, J. (2024). Rapport.",
        )
        assert chunk.doc_title == "Rapport 2024"
        assert chunk.doc_authors == "Dupont, J."
        assert chunk.apa_reference == "Dupont, J. (2024). Rapport."


# ── Tests Reranker singleton ──

class TestRerankerSingleton:
    def test_get_instance_returns_same(self):
        r1 = Reranker.get_instance()
        r2 = Reranker.get_instance()
        assert r1 is r2

    def test_reset_instance(self):
        r1 = Reranker.get_instance()
        Reranker.reset_instance()
        r2 = Reranker.get_instance()
        assert r1 is not r2

    def test_default_model_name(self):
        r = Reranker()
        assert r._model_name == DEFAULT_RERANKER_MODEL

    def test_custom_model_name(self):
        r = Reranker(model_name="custom/reranker")
        assert r._model_name == "custom/reranker"


# ── Tests Reranker.rerank ──

class TestRerank:
    def test_empty_candidates(self):
        r = Reranker()
        result = r.rerank("query", [])
        assert result == []

    @patch("src.core.reranker.Reranker._load_model")
    def test_rerank_assigns_scores(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.7])
        mock_load.return_value = mock_model

        candidates = [
            _make_scored_chunk(chunk_id="c1", cosine_score=0.8),
            _make_scored_chunk(chunk_id="c2", cosine_score=0.7),
            _make_scored_chunk(chunk_id="c3", cosine_score=0.6),
        ]

        r = Reranker()
        result = r.rerank("ma requête", candidates, top_k=10)

        assert len(result) == 3
        # Trié par rerank_score décroissant
        assert result[0].rerank_score == 0.9
        assert result[1].rerank_score == 0.7
        assert result[2].rerank_score == 0.3

    @patch("src.core.reranker.Reranker._load_model")
    def test_rerank_respects_top_k(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        mock_load.return_value = mock_model

        candidates = [
            _make_scored_chunk(chunk_id=f"c{i}") for i in range(5)
        ]

        r = Reranker()
        result = r.rerank("query", candidates, top_k=2)

        assert len(result) == 2

    @patch("src.core.reranker.Reranker._load_model")
    def test_rerank_creates_correct_pairs(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.6])
        mock_load.return_value = mock_model

        candidates = [
            _make_scored_chunk(chunk_id="c1", text="Texte un"),
            _make_scored_chunk(chunk_id="c2", text="Texte deux"),
        ]

        r = Reranker()
        r.rerank("ma question", candidates)

        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [("ma question", "Texte un"), ("ma question", "Texte deux")]

    @patch("src.core.reranker.Reranker._load_model")
    def test_rerank_preserves_original_fields(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9])
        mock_load.return_value = mock_model

        candidates = [
            _make_scored_chunk(
                chunk_id="c1", doc_id="doc42", text="Contenu",
                page_number=5, section_title="Résultats",
                cosine_score=0.75, doc_title="Rapport",
            ),
        ]

        r = Reranker()
        result = r.rerank("query", candidates)

        assert result[0].chunk_id == "c1"
        assert result[0].doc_id == "doc42"
        assert result[0].page_number == 5
        assert result[0].section_title == "Résultats"
        assert result[0].cosine_score == 0.75
        assert result[0].doc_title == "Rapport"


class TestLazyLoading:
    def test_model_not_loaded_on_init(self):
        r = Reranker()
        assert r._model is None


# ── Tests build_context ──

class TestBuildContext:
    def test_empty_chunks(self):
        result = build_context([])
        assert "Aucun" in result

    def test_single_chunk(self):
        chunk = _make_scored_chunk(
            text="Contenu du bloc source.",
            page_number=3,
            section_title="Analyse",
            apa_reference="Ref APA",
        )
        result = build_context([chunk])
        assert "SOURCE 1" in result
        assert "FIN SOURCE 1" in result
        assert "Ref APA" in result
        assert "Page : 3" in result
        assert "Section : Analyse" in result
        assert "Contenu du bloc source." in result

    def test_multiple_chunks(self):
        chunks = [
            _make_scored_chunk(chunk_id="c1", text="Premier bloc."),
            _make_scored_chunk(chunk_id="c2", text="Deuxième bloc."),
            _make_scored_chunk(chunk_id="c3", text="Troisième bloc."),
        ]
        result = build_context(chunks)
        assert "SOURCE 1" in result
        assert "SOURCE 2" in result
        assert "SOURCE 3" in result
        assert "Premier bloc." in result
        assert "Troisième bloc." in result

    def test_fallback_to_doc_title(self):
        chunk = _make_scored_chunk(
            doc_title="Mon Rapport",
            apa_reference="",
        )
        result = build_context([chunk])
        assert "Mon Rapport" in result

    def test_fallback_to_doc_id(self):
        chunk = _make_scored_chunk(
            doc_id="doc_xyz",
            doc_title="",
            apa_reference="",
        )
        result = build_context([chunk])
        assert "doc_xyz" in result
