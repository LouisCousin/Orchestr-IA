"""Tests unitaires pour le chunking sémantique hiérarchique (Phase 2.5)."""

import pytest
from src.core.semantic_chunker import (
    Chunk, chunk_document, _chunk_by_sections, _chunk_by_tokens, _count_tokens,
)
from src.core.text_extractor import ExtractionResult


def _make_extraction(text: str, structure=None) -> ExtractionResult:
    """Crée un ExtractionResult de test."""
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


# ── Test Priorité 1 : document avec structure ──

def test_chunk_by_sections_basic():
    """Document avec 3 sections de taille raisonnable."""
    structure = [
        {"text": "Introduction", "type": "title", "page": 1, "level": 1},
        {"text": "Ceci est le paragraphe d'introduction. " * 10, "type": "paragraph", "page": 1, "level": 0},
        {"text": "Méthodologie", "type": "title", "page": 2, "level": 1},
        {"text": "Voici la méthodologie utilisée. " * 10, "type": "paragraph", "page": 2, "level": 0},
        {"text": "Résultats", "type": "title", "page": 3, "level": 1},
        {"text": "Les résultats sont présentés ici. " * 10, "type": "paragraph", "page": 3, "level": 0},
    ]
    extraction = _make_extraction("test", structure=structure)
    chunks = chunk_document(extraction, doc_id="doc1")

    assert len(chunks) == 3
    assert chunks[0].section_title == "Introduction"
    assert chunks[0].page_number == 1
    assert chunks[1].section_title == "Méthodologie"
    assert chunks[1].page_number == 2
    assert chunks[2].section_title == "Résultats"
    assert chunks[2].page_number == 3


def test_chunk_preserves_doc_id():
    """Chaque chunk a le bon doc_id."""
    structure = [
        {"text": "Titre", "type": "title", "page": 1, "level": 1},
        {"text": "Contenu. " * 20, "type": "paragraph", "page": 1, "level": 0},
    ]
    extraction = _make_extraction("test", structure=structure)
    chunks = chunk_document(extraction, doc_id="ABC123")

    for chunk in chunks:
        assert chunk.doc_id == "ABC123"
        assert chunk.chunk_id.startswith("ABC123_")


# ── Test Priorité 2 : section longue ──

def test_long_section_split():
    """Section longue (>800 tokens) découpée en plusieurs chunks."""
    long_text = "Ceci est une phrase de test avec suffisamment de mots pour dépasser la limite. " * 100
    structure = [
        {"text": "Analyse approfondie", "type": "title", "page": 1, "level": 1},
        {"text": long_text, "type": "paragraph", "page": 1, "level": 0},
    ]
    extraction = _make_extraction("test", structure=structure)
    chunks = chunk_document(extraction, doc_id="doc2", max_chunk_tokens=200)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.section_title == "Analyse approfondie"


# ── Test Priorité 3 : fallback par tokens ──

def test_chunk_by_tokens_no_structure():
    """Texte brut sans structure → fallback par tokens."""
    text = "Premier paragraphe de texte. " * 50 + "\n\n" + "Deuxième paragraphe. " * 50
    extraction = _make_extraction(text, structure=None)
    chunks = chunk_document(extraction, doc_id="doc3")

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.doc_id == "doc3"
        assert chunk.section_title == ""


def test_chunk_by_tokens_short_text():
    """Texte court → un seul chunk."""
    text = "Un court texte de test."
    extraction = _make_extraction(text, structure=None)
    chunks = chunk_document(extraction, doc_id="doc4")

    assert len(chunks) == 1
    assert chunks[0].text == text


# ── Test fusion ──

def test_short_chunks_merged():
    """Chunks < min_chunk_tokens fusionnés avec le précédent."""
    structure = [
        {"text": "Titre", "type": "title", "page": 1, "level": 1},
        {"text": "Contenu substantiel pour le premier chunk de la section. " * 10, "type": "paragraph", "page": 1, "level": 0},
        {"text": "Court.", "type": "paragraph", "page": 1, "level": 0},
    ]
    extraction = _make_extraction("test", structure=structure)
    chunks = chunk_document(extraction, doc_id="doc5", min_chunk_tokens=50)

    # Le "Court." devrait être fusionné avec le chunk précédent
    assert len(chunks) <= 2


# ── Test du chunk_id ──

def test_chunk_id_format():
    """Vérifier le format du chunk_id."""
    chunk = Chunk(doc_id="doc1", text="test", page_number=1,
                  section_title="Intro", chunk_index=5)
    assert chunk.chunk_id == "doc1_0005"


def test_token_count_estimation():
    """Vérifier l'estimation du nombre de tokens."""
    text = "word " * 100  # 100 mots
    count = _count_tokens(text.strip())
    assert count > 50  # ~100 * 4/3 ≈ 133
    assert count < 200


# ── Test texte vide ──

def test_empty_text():
    """Texte vide retourne une liste vide."""
    extraction = _make_extraction("", structure=None)
    chunks = chunk_document(extraction, doc_id="empty")
    assert chunks == []


def test_empty_structure():
    """Structure vide retourne une liste vide."""
    extraction = _make_extraction("", structure=[])
    chunks = chunk_document(extraction, doc_id="empty")
    assert chunks == []
