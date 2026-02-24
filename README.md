# Orchestr'IA

Pipeline intelligent de génération de documents professionnels assistée par IA, intégrant RAG (ChromaDB + embeddings locaux + reranking), génération conditionnelle et export DOCX.

## Architecture des données

Chaque projet est stocké dans un répertoire dédié :

```
projects/
└── {project_id}/
    ├── state.json          # État complet du projet (config, plan, sections)
    ├── corpus/             # Fichiers sources bruts
    │   ├── 001_document.pdf
    │   ├── 002_rapport.docx
    │   └── 003_page_web.txt
    ├── chromadb/           # Base vectorielle ChromaDB (RAG)
    │   └── (fichiers internes SQLite + index HNSW)
    └── metadata.db         # Métadonnées SQLite (documents + chunks)
```

- **state.json** : sérialisé par `ProjectState.to_dict()`, contient la config, le plan, les sections générées, les coûts.
- **corpus/** : fichiers acquis (PDF, DOCX, TXT, etc.), nommés avec un préfixe séquentiel.
- **chromadb/** : persistance ChromaDB pour la recherche vectorielle RAG.
- **metadata.db** : base SQLite avec tables `documents` et `chunks`, contenant les métadonnées riches (titre, auteurs, année, APA).

## Pipeline

Le pipeline se décompose en 5 étapes :

1. **Configuration** — Fournisseur IA, modèle, paramètres de génération
2. **Acquisition du corpus** — Upload de fichiers ou URLs, avec détection anti-bot
3. **Plan du document** — Import ou génération IA du plan structuré
4. **Génération** — Génération séquentielle section par section (multi-pass, RAG)
5. **Export** — Export DOCX avec charte graphique configurable

## Structure du code

```
src/
├── app.py                       # Point d'entrée Streamlit
├── pages/
│   ├── page_accueil.py          # Gestion des projets
│   ├── page_configuration.py    # Configuration fournisseur/modèle
│   ├── page_acquisition.py      # Acquisition du corpus
│   ├── page_plan.py             # Plan du document
│   ├── page_generation.py       # Génération du contenu
│   └── page_export.py           # Export DOCX
├── core/
│   ├── orchestrator.py          # Orchestrateur principal + ProjectState
│   ├── prompt_engine.py         # Génération des prompts
│   ├── rag_engine.py            # Moteur RAG (ChromaDB)
│   ├── corpus_acquirer.py       # Acquisition fichiers/URLs
│   ├── corpus_extractor.py      # Structuration du corpus
│   ├── export_engine.py         # Export DOCX
│   ├── metadata_store.py        # Base SQLite métadonnées
│   ├── text_extractor.py        # Extraction texte PDF/DOCX/HTML
│   ├── semantic_chunker.py      # Chunking sémantique
│   ├── conditional_generator.py # Génération conditionnelle
│   └── plan_parser.py           # Parsing et normalisation de plans
├── providers/
│   ├── base.py                  # Interface abstraite des providers
│   ├── openai_provider.py       # Provider OpenAI
│   ├── anthropic_provider.py    # Provider Anthropic
│   └── gemini_provider.py       # Provider Google Gemini
└── utils/
    ├── providers_registry.py    # Registre centralisé des fournisseurs
    ├── content_validator.py     # Validation contenu web (anti-bot)
    ├── reference_cleaner.py     # Nettoyage des références [Source N]
    ├── config.py                # Chargement configuration
    ├── file_utils.py            # Utilitaires fichiers
    └── token_counter.py         # Comptage de tokens
```
