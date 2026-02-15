"""Tests d'intégration pour le pipeline RAG Phase 2.5.

Vérifie l'intégration entre les composants :
  - semantic_chunker → metadata_store → (mock embedder) → reranker → build_context
  - export_engine NEEDS_SOURCE marker detection
  - prompt_engine anti-hallucination block
"""

import pytest
import os
from unittest.mock import patch, MagicMock
import numpy as np

from src.core.semantic_chunker import Chunk, chunk_document
from src.core.text_extractor import ExtractionResult
from src.core.metadata_store import MetadataStore, DocumentMetadata
from src.core.reranker import ScoredChunk, build_context
from src.core.export_engine import detect_needs_source_markers, scan_all_sections_for_markers
from src.core.prompt_engine import PromptEngine, ANTI_HALLUCINATION_BLOCK


def _make_extraction(text: str, structure=None) -> ExtractionResult:
    return ExtractionResult(
        text=text,
        page_count=1,
        char_count=len(text),
        word_count=len(text.split()),
        extraction_method="test",
        status="success",
        source_filename="test.pdf",
        source_size_bytes=0,
        structure=structure,
    )


# ── Test 1 : Pipeline chunking → metadata ──

class TestChunkingToMetadata:
    """Vérifie que les chunks produits s'intègrent avec le MetadataStore."""

    def test_chunk_and_store(self, tmp_path):
        structure = [
            {"text": "Introduction", "type": "title", "page": 1, "level": 1},
            {"text": "Le contenu d'introduction détaillé avec assez de mots. " * 10,
             "type": "paragraph", "page": 1, "level": 0},
            {"text": "Méthodologie", "type": "title", "page": 2, "level": 1},
            {"text": "La méthodologie utilisée est décrite ici avec des détails. " * 10,
             "type": "paragraph", "page": 2, "level": 0},
        ]
        extraction = _make_extraction("test", structure=structure)
        chunks = chunk_document(extraction, doc_id="integration_doc")

        assert len(chunks) >= 2

        store = MetadataStore(str(tmp_path))
        doc = DocumentMetadata(
            doc_id="integration_doc",
            filepath="/test/doc.pdf",
            filename="doc.pdf",
            title="Document de test",
            language="fr",
            doc_type="report",
        )
        store.add_document(doc)
        store.add_chunks(chunks)

        stored_chunks = store.get_chunks_by_doc("integration_doc")
        assert len(stored_chunks) == len(chunks)

        for i, sc in enumerate(stored_chunks):
            assert sc["chunk_id"] == chunks[i].chunk_id
            assert sc["text"] == chunks[i].text
            assert sc["page_number"] == chunks[i].page_number
            assert sc["section_title"] == chunks[i].section_title

        assert store.count_chunks() == len(chunks)
        store.close()

    def test_multiple_docs_independence(self, tmp_path):
        store = MetadataStore(str(tmp_path))

        for doc_num in range(3):
            doc_id = f"doc_{doc_num}"
            structure = [
                {"text": f"Titre doc {doc_num}", "type": "title", "page": 1, "level": 1},
                {"text": f"Contenu du document numéro {doc_num}. " * 15,
                 "type": "paragraph", "page": 1, "level": 0},
            ]
            extraction = _make_extraction("test", structure=structure)
            chunks = chunk_document(extraction, doc_id=doc_id)

            store.add_document(DocumentMetadata(
                doc_id=doc_id, filepath=f"/test/{doc_num}.pdf",
                filename=f"{doc_num}.pdf",
            ))
            store.add_chunks(chunks)

        all_docs = store.get_all_documents()
        assert len(all_docs) == 3

        total_chunks = store.count_chunks()
        assert total_chunks >= 3  # At least 1 chunk per doc

        store.close()


# ── Test 2 : ScoredChunk → build_context ──

class TestScoredChunkToContext:
    def test_full_pipeline_context_building(self):
        """Simule le flux : recherche → reranking → contexte prompt."""
        scored = [
            ScoredChunk(
                chunk_id="doc1_0001", doc_id="doc1",
                text="Les résultats montrent une augmentation de 15% du CA.",
                page_number=5, section_title="Résultats financiers",
                cosine_score=0.85, rerank_score=0.92,
                doc_title="Rapport annuel 2024",
                doc_authors="Dupont, J.",
                apa_reference="Dupont, J. (2024). Rapport annuel.",
            ),
            ScoredChunk(
                chunk_id="doc2_0003", doc_id="doc2",
                text="La méthodologie utilisée repose sur l'analyse quantitative.",
                page_number=12, section_title="Méthodologie",
                cosine_score=0.72, rerank_score=0.78,
                doc_title="Étude de marché",
            ),
        ]

        context = build_context(scored)

        # Vérifie la structure du contexte
        assert "SOURCE 1" in context
        assert "SOURCE 2" in context
        assert "FIN SOURCE 1" in context
        assert "FIN SOURCE 2" in context

        # Vérifie la référence APA utilisée en priorité
        assert "Dupont, J. (2024). Rapport annuel." in context
        # Vérifie le fallback vers doc_title
        assert "Étude de marché" in context

        # Vérifie les métadonnées
        assert "Page : 5" in context
        assert "Section : Résultats financiers" in context
        assert "augmentation de 15%" in context


# ── Test 3 : NEEDS_SOURCE marker detection ──

class TestNeedsSourceIntegration:
    def test_detect_markers_in_generated_content(self):
        """Simule du contenu généré avec des marqueurs."""
        generated = {
            "1": "L'introduction présente le contexte du rapport.",
            "2": ("Les résultats montrent une croissance. "
                  "{{NEEDS_SOURCE: données de vente Q4 2024}} "
                  "Cette tendance est confirmée."),
            "3": ("En conclusion, {{NEEDS_SOURCE: projection budgétaire 2025}} "
                  "les perspectives sont positives. "
                  "{{NEEDS_SOURCE: recommandations du comité}}"),
        }

        results = scan_all_sections_for_markers(generated)

        # Section 1 : pas de marqueur
        assert "1" not in results

        # Section 2 : 1 marqueur
        assert "2" in results
        assert len(results["2"]) == 1
        assert results["2"][0]["description"] == "données de vente Q4 2024"

        # Section 3 : 2 marqueurs
        assert "3" in results
        assert len(results["3"]) == 2
        descriptions = {m["description"] for m in results["3"]}
        assert "projection budgétaire 2025" in descriptions
        assert "recommandations du comité" in descriptions

    def test_no_markers_in_clean_content(self):
        generated = {
            "1": "Contenu propre sans marqueurs.",
            "2": "Autre section propre.",
        }
        results = scan_all_sections_for_markers(generated)
        assert results == {}


# ── Test 4 : Anti-hallucination in prompt engine ──

class TestAntiHallucinationIntegration:
    def test_enabled_by_default(self):
        engine = PromptEngine()
        prompt = engine.build_system_prompt()
        assert "SOURCES EXCLUSIVES" in prompt
        assert "NEEDS_SOURCE" in prompt
        assert "ATTRIBUTION" in prompt
        assert "TRANSPARENCE" in prompt

    def test_disabled(self):
        engine = PromptEngine(anti_hallucination_enabled=False)
        prompt = engine.build_system_prompt()
        assert "NEEDS_SOURCE" not in prompt
        assert "SOURCES EXCLUSIVES" not in prompt

    def test_with_persistent_instructions(self):
        engine = PromptEngine(
            persistent_instructions="Utilise un ton académique.",
            anti_hallucination_enabled=True,
        )
        prompt = engine.build_system_prompt()
        assert "ton académique" in prompt
        assert "SOURCES EXCLUSIVES" in prompt


# ── Test 5 : Metadata filtering for pre-filtering ──

class TestMetadataFiltering:
    def test_filter_then_search(self, tmp_path):
        """Simule le pré-filtrage SQLite avant recherche ChromaDB."""
        store = MetadataStore(str(tmp_path))

        docs = [
            DocumentMetadata(doc_id="fr1", filepath="a.pdf", filename="a.pdf",
                             language="fr", doc_type="article", year=2024),
            DocumentMetadata(doc_id="fr2", filepath="b.pdf", filename="b.pdf",
                             language="fr", doc_type="report", year=2023),
            DocumentMetadata(doc_id="en1", filepath="c.pdf", filename="c.pdf",
                             language="en", doc_type="article", year=2024),
        ]
        for d in docs:
            store.add_document(d)

        # Filtrer : FR uniquement
        fr_ids = store.get_doc_ids_by_filter(language="fr")
        assert set(fr_ids) == {"fr1", "fr2"}

        # Filtrer : articles uniquement
        article_ids = store.get_doc_ids_by_filter(doc_type="article")
        assert set(article_ids) == {"fr1", "en1"}

        # Filtrer combiné : FR + article
        combined = store.search_documents(language="fr", doc_type="article")
        assert len(combined) == 1
        assert combined[0].doc_id == "fr1"

        store.close()


# ── Test 6 : End-to-end chunk → context ──

class TestEndToEndChunkToContext:
    def test_from_extraction_to_prompt_context(self, tmp_path):
        """Test complet : extraction → chunking → stockage → scored → contexte."""
        # 1. Extraction (simulée)
        structure = [
            {"text": "Analyse financière", "type": "title", "page": 1, "level": 1},
            {"text": "Les revenus ont augmenté de 20% en 2024. " * 10,
             "type": "paragraph", "page": 1, "level": 0},
        ]
        extraction = _make_extraction("test", structure=structure)
        chunks = chunk_document(extraction, doc_id="fin_report")

        # 2. Stockage metadata
        store = MetadataStore(str(tmp_path))
        store.add_document(DocumentMetadata(
            doc_id="fin_report", filepath="/test/fin.pdf",
            filename="fin.pdf", title="Rapport financier 2024",
            language="fr", doc_type="report",
        ))
        store.add_chunks(chunks)

        # 3. Simuler un résultat de recherche
        scored_chunks = []
        for chunk in chunks:
            scored_chunks.append(ScoredChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                cosine_score=0.8,
                rerank_score=0.85,
                doc_title="Rapport financier 2024",
            ))

        # 4. Construire le contexte
        context = build_context(scored_chunks)

        assert "SOURCE 1" in context
        assert "revenus ont augmenté" in context
        assert "Rapport financier 2024" in context
        assert "Analyse financière" in context

        store.close()
