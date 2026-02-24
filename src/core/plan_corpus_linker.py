"""Liaison plan-corpus pour le planificateur.

Phase 2.5 : analyse le corpus avant la génération du plan pour produire
un plan basé sur le contenu réel des documents, pas sur les connaissances
générales du LLM.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("orchestria")


@dataclass
class PlanContext:
    """Contexte de planification issu de l'analyse du corpus."""
    corpus_summary: dict = field(default_factory=dict)
    themes: list[str] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)  # theme → {"avg_score", "nb_chunks"}

    def to_dict(self) -> dict:
        return {
            "corpus_summary": self.corpus_summary,
            "themes": self.themes,
            "coverage": self.coverage,
        }


def link_plan_to_corpus(
    objective: str,
    metadata_store,
    collection,
    config: dict,
    provider=None,
) -> PlanContext:
    """Analyse le corpus et produit un contexte de planification.

    Args:
        objective: Objectif du document décrit par l'utilisateur.
        metadata_store: Instance de MetadataStore (SQLite).
        collection: Collection ChromaDB indexée.
        config: Configuration du projet.
        provider: Fournisseur IA pour l'extraction de thèmes (optionnel).

    Returns:
        PlanContext avec résumé du corpus, thèmes et scores de couverture.
    """
    # ═══ ÉTAPE 1 : Inventaire du corpus ═══
    docs = metadata_store.get_all_documents()
    corpus_summary = {
        "total_documents": len(docs),
        "total_tokens": sum(d.token_count for d in docs),
        "languages": list(set(d.language for d in docs if d.language)),
        "types": list(set(d.doc_type for d in docs if d.doc_type)),
        "documents": [
            {
                "title": d.title or d.filename,
                "pages": d.page_count,
                "tokens": d.token_count,
                "type": d.doc_type,
            }
            for d in docs
        ],
    }

    # ═══ ÉTAPE 2 : Extraction des thèmes par échantillonnage ═══
    theme_chunks = []
    max_intro_chunks = config.get("plan_corpus_linking", {}).get("max_intro_chunks_per_doc", 3)
    max_docs = config.get("plan_corpus_linking", {}).get("max_documents_for_theme", 30)

    for doc in docs[:max_docs]:
        chunks = metadata_store.get_chunks_by_doc(doc.doc_id)
        theme_chunks.extend(chunks[:max_intro_chunks])

    # Extraire les thèmes
    themes = _extract_themes_simple(theme_chunks, objective)

    if provider and theme_chunks:
        try:
            themes = _extract_themes_llm(theme_chunks, objective, provider, config)
        except Exception as e:
            logger.warning(f"Extraction LLM des thèmes échouée, utilisation du fallback : {e}")

    # ═══ ÉTAPE 3 : Scores de couverture prévisionnels ═══
    coverage = {}
    try:
        from src.core.local_embedder import LocalEmbedder
        embedder = LocalEmbedder.get_instance()
        for theme in themes:
            try:
                query_embedding = embedder.embed_query(theme)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=10,
                )
                if results and results["distances"] and results["distances"][0]:
                    scores = [1 - d for d in results["distances"][0]]
                    coverage[theme] = {
                        "avg_score": sum(scores) / len(scores) if scores else 0,
                        "nb_chunks": len([s for s in scores if s > 0.3]),
                    }
                else:
                    coverage[theme] = {"avg_score": 0, "nb_chunks": 0}
            except Exception:
                coverage[theme] = {"avg_score": 0, "nb_chunks": 0}
    except ImportError:
        # Fallback si embeddings locaux non disponibles
        for theme in themes:
            try:
                results = collection.query(
                    query_texts=[theme],
                    n_results=10,
                )
                if results and results["distances"] and results["distances"][0]:
                    scores = [1 - d for d in results["distances"][0]]
                    coverage[theme] = {
                        "avg_score": sum(scores) / len(scores) if scores else 0,
                        "nb_chunks": len([s for s in scores if s > 0.3]),
                    }
                else:
                    coverage[theme] = {"avg_score": 0, "nb_chunks": 0}
            except Exception:
                coverage[theme] = {"avg_score": 0, "nb_chunks": 0}

    return PlanContext(
        corpus_summary=corpus_summary,
        themes=themes,
        coverage=coverage,
    )


def _extract_themes_simple(chunks: list[dict], objective: str) -> list[str]:
    """Extraction de thèmes par analyse textuelle simple (sans LLM).

    Utilise les titres de sections et les mots-clés fréquents.
    """
    import re
    from collections import Counter

    themes = []
    section_titles = set()

    for chunk in chunks:
        title = chunk.get("section_title", "") if isinstance(chunk, dict) else ""
        if title and title not in section_titles:
            section_titles.add(title)
            themes.append(title)

    # Si pas assez de thèmes depuis les sections, extraire des mots-clés
    if len(themes) < 3:
        all_text = " ".join(
            c.get("text", "")[:500] if isinstance(c, dict) else ""
            for c in chunks
        )
        # Mots significatifs (4+ caractères, pas de stop words)
        words = re.findall(r"\b[a-zà-ÿ]{4,}\b", all_text.lower())
        stop_words = frozenset(
            "dans avec pour cette cette comme plus être faire tout aussi entre"
            " mais très bien même peut sont leur leurs nous vous".split()
        )
        filtered = [w for w in words if w not in stop_words]
        most_common = Counter(filtered).most_common(10)
        for word, _ in most_common:
            if word not in themes:
                themes.append(word.capitalize())

    return themes[:15]


def _extract_themes_llm(
    chunks: list[dict],
    objective: str,
    provider,
    config: dict,
) -> list[str]:
    """Extraction de thèmes via un appel LLM dédié."""
    max_chunks = 30
    theme_text = "\n---\n".join(
        (c.get("text", "")[:500] if isinstance(c, dict) else "")
        for c in chunks[:max_chunks]
    )

    prompt = f"""Analyse les extraits de corpus suivants et identifie les thèmes principaux
couverts par ces documents, en relation avec l'objectif suivant :

OBJECTIF : {objective}

EXTRAITS DU CORPUS :
{theme_text}

Retourne une liste de 5 à 15 thèmes, un par ligne, du plus important au moins important.
Chaque thème doit être une phrase courte (3-8 mots) décrivant un sujet couvert par le corpus.
Retourne UNIQUEMENT la liste, sans numérotation, sans commentaires."""

    model = config.get("model")
    if not model:
        logger.warning("Clé 'model' absente de la config, fallback sur le modèle par défaut du provider")
        model = provider.get_default_model()
    response = provider.generate(
        prompt=prompt,
        system_prompt="Tu es un analyste documentaire expert.",
        model=model,
        temperature=0.3,
        max_tokens=500,
    )

    themes = []
    for line in response.content.strip().split("\n"):
        line = line.strip().lstrip("0123456789.-•) ")
        if line and len(line) > 2:
            themes.append(line)

    return themes[:15] if themes else _extract_themes_simple(chunks, objective)


def format_plan_context_for_prompt(context: PlanContext) -> str:
    """Formate le PlanContext pour injection dans le prompt de planification.

    Returns:
        Texte structuré avec résumé du corpus et thèmes avec couverture.
    """
    lines = []

    # Résumé du corpus
    summary = context.corpus_summary
    lines.append(f"Nombre de documents : {summary.get('total_documents', 0)}")
    lines.append(f"Tokens totaux : {summary.get('total_tokens', 0)}")
    langs = summary.get("languages", [])
    if langs:
        lines.append(f"Langues : {', '.join(langs)}")
    types = summary.get("types", [])
    if types:
        lines.append(f"Types : {', '.join(types)}")

    lines.append("")
    lines.append("Documents :")
    for doc in summary.get("documents", []):
        lines.append(f"  - {doc['title']} ({doc['pages']} pages, {doc['tokens']} tokens)")

    # Thèmes avec couverture
    if context.themes:
        lines.append("")
        lines.append("Thèmes identifiés dans le corpus :")
        for theme in context.themes:
            cov = context.coverage.get(theme, {})
            avg_score = cov.get("avg_score", 0)
            nb_chunks = cov.get("nb_chunks", 0)

            if avg_score >= 0.5:
                icon = "FORT"
            elif avg_score >= 0.3:
                icon = "PARTIEL"
            else:
                icon = "FAIBLE"

            lines.append(f"  - [{icon}] {theme} (score: {avg_score:.2f}, {nb_chunks} blocs)")

    return "\n".join(lines)


# Prompt de planification enrichi (Phase 2.5)
PLAN_PROMPT_WITH_CORPUS = """Tu es un planificateur de documents professionnels.

OBJECTIF DE L'UTILISATEUR :
{objective}

CORPUS DISPONIBLE :
{corpus_context}

RÈGLES IMPÉRATIVES :
1. Le plan DOIT être basé sur le contenu réel du corpus ci-dessus.
2. Ne propose PAS de sections sur des sujets absents du corpus.
3. Pour chaque section, indique le niveau de couverture du corpus :
   - Couverture forte (plusieurs documents pertinents)
   - Couverture partielle (1-2 documents)
   - Couverture insuffisante (pas de matière dans le corpus)
4. Favorise les sections avec couverture forte.
5. Si un thème important est absent du corpus, signale-le comme
   "thème non couvert" en fin de plan.
6. Génère un plan hiérarchique avec numérotation (1. / 1.1 / 1.1.1).
7. Taille cible : {target_pages} pages.

Génère un plan structuré avec numérotation hiérarchique.
Retourne uniquement le plan, sans commentaires ni explications.
"""
