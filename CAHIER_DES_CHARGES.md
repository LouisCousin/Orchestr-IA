# Orchestr'IA — Cahier des Charges Technique Complet

> **Version** : 6.0
> **Date** : 26 février 2026
> **Auteur** : Équipe Orchestr'IA
> **Statut** : Document de référence

---

## Table des matières

1. [Vision du projet](#1-vision-du-projet)
2. [Architecture globale](#2-architecture-globale)
3. [Phase 1 — Pipeline fondamental](#3-phase-1--pipeline-fondamental) ✅
4. [Phase 2 — Multi-fournisseurs et RAG de base](#4-phase-2--multi-fournisseurs-et-rag-de-base) ✅
5. [Phase 2.5 — RAG avancé et garde-fous](#5-phase-25--rag-avancé-et-garde-fous) ✅
6. [Phase 3 — Intelligence du pipeline](#6-phase-3--intelligence-du-pipeline) ✅
7. [Phase 4 — Performance et optimisation](#7-phase-4--performance-et-optimisation) ✅
8. [Phase 5 — Intégration Gemini 3.1 et Context Caching](#8-phase-5--intégration-gemini-31-et-context-caching) 🔧
9. [Phase 6 — Acquisition GitHub (Clone de dépôts)](#9-phase-6--acquisition-github-clone-de-dépôts) 🔧
10. [Phase 7 — Orchestration multi-agents](#10-phase-7--orchestration-multi-agents) 📋
11. [Matrice des dépendances](#11-matrice-des-dépendances)
12. [Stack technique](#12-stack-technique)
13. [Annexes](#13-annexes)

**Légende** : ✅ Implémenté | 🔧 En cours | 📋 Planifié

---

## 1. Vision du projet

**Orchestr'IA** est un pipeline intelligent de génération documentaire assistée par IA. Il transforme un corpus de documents sources (PDF, DOCX, TXT, HTML, Excel, dépôts GitHub) en documents professionnels structurés via un processus en 5 étapes : Configuration → Acquisition → Plan → Génération → Export.

### 1.1 Principes directeurs

| Principe | Description |
|---|---|
| **Fiabilité factuelle** | Zéro hallucination : chaque affirmation est sourcée ou marquée `{{NEEDS_SOURCE}}` |
| **Human-in-the-Loop** | Checkpoints de validation humaine à chaque étape critique |
| **Multi-fournisseurs** | Support OpenAI, Anthropic et Google Gemini avec fallback automatique |
| **Optimisation des coûts** | Context caching, batch processing, modèles économiques pour les tâches secondaires |
| **Scalabilité** | Corpus de 500k+ chunks, documents de 100+ pages, traitement asynchrone |

### 1.2 Cas d'usage cibles

- Rapports d'analyse (20-80 pages) à partir de corpus documentaire
- Documents de formation à partir de supports techniques
- Synthèses de veille à partir d'articles et études
- Propositions de services à partir de spécifications client
- Documentation technique à partir de dépôts de code source

---

## 2. Architecture globale

### 2.1 Structure des modules

```
src/
├── app.py                          # Point d'entrée Streamlit
├── core/                           # 30 modules — Moteur du pipeline
│   ├── orchestrator.py             # Chef d'orchestre + ProjectState
│   ├── prompt_engine.py            # Génération dynamique de prompts
│   ├── rag_engine.py               # Pipeline RAG hybride (ChromaDB)
│   ├── semantic_chunker.py         # Chunking sémantique hiérarchique
│   ├── local_embedder.py           # Embeddings locaux (multilingual-e5-large)
│   ├── reranker.py                 # Cross-encoder reranking
│   ├── text_extractor.py           # Extraction multi-format (Docling/PyMuPDF)
│   ├── corpus_extractor.py         # Structuration corpus + TF-IDF
│   ├── corpus_acquirer.py          # Acquisition asynchrone (fichiers + URLs)
│   ├── corpus_deduplicator.py      # Dédoublonnage par hash
│   ├── plan_parser.py              # Parsing et normalisation de plans
│   ├── plan_corpus_linker.py       # Pré-analyse plan↔corpus
│   ├── conditional_generator.py    # Génération conditionnelle par couverture
│   ├── quality_evaluator.py        # Évaluation qualité (6 critères)
│   ├── factcheck_engine.py         # Vérification factuelle
│   ├── feedback_engine.py          # Apprentissage des corrections humaines
│   ├── glossary_engine.py          # Gestion terminologique
│   ├── citation_engine.py          # Citations APA 7e édition
│   ├── persona_engine.py           # Modélisation persona/audience
│   ├── export_engine.py            # Export DOCX avec styling
│   ├── cost_tracker.py             # Suivi des coûts API
│   ├── checkpoint_manager.py       # Checkpoints HITL
│   ├── metadata_store.py           # SQLite (documents + chunks)
│   ├── profile_manager.py          # Profils de projet pré-configurés
│   ├── template_library.py         # Bibliothèque de templates
│   ├── hitl_journal.py             # Journal des décisions HITL
│   ├── persistent_instructions.py  # Instructions persistantes hiérarchiques
│   ├── metadata_overrides.py       # Corrections manuelles de métadonnées
│   └── grobid_client.py            # Extraction bibliographique (Docker)
├── pages/                          # 8 pages Streamlit
│   ├── page_accueil.py             # Accueil et gestion de projets
│   ├── page_configuration.py       # Configuration fournisseur IA
│   ├── page_acquisition.py         # Upload/URL/GitHub acquisition
│   ├── page_plan.py                # Import/génération/édition du plan
│   ├── page_generation.py          # Génération avec barre de progression
│   ├── page_export.py              # Export DOCX et téléchargement
│   ├── page_dashboard.py           # Métriques et logs en temps réel
│   └── page_bibliotheque.py        # Gestion et recherche dans le corpus
├── providers/                      # 4 fournisseurs IA
│   ├── base.py                     # Interface abstraite + types Batch
│   ├── openai_provider.py          # GPT-4.1/4o/3.5 + Batch API
│   ├── anthropic_provider.py       # Claude Opus/Sonnet/Haiku + Batch
│   └── gemini_provider.py          # Gemini 3.1 Pro/Flash + Context Cache
└── utils/                          # 7 modules utilitaires
    ├── config.py                   # Chargement YAML + pricing
    ├── file_utils.py               # I/O fichiers + JSON
    ├── logger.py                   # ActivityLog structuré
    ├── token_counter.py            # Comptage tokens (tiktoken)
    ├── providers_registry.py       # Registre dynamique de providers
    └── content_validator.py        # Validation anti-bot pour scraping
```

### 2.2 Flux de données principal

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ ACQUISITION │───▶│   PLAN      │───▶│ GÉNÉRATION  │───▶│  ÉVALUATION │───▶│   EXPORT    │
│             │    │             │    │             │    │             │    │             │
│ • Fichiers  │    │ • Import    │    │ • RAG search│    │ • Qualité   │    │ • DOCX      │
│ • URLs      │    │ • Auto-gen  │    │ • Prompt    │    │ • Factcheck │    │ • Styling   │
│ • GitHub    │    │ • Édition   │    │ • LLM call  │    │ • Feedback  │    │ • Branding  │
│ • Extraction│    │ • Linking   │    │ • Multi-pass│    │ • HITL      │    │ • Download  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                                     │                 │
       ▼                                     ▼                 ▼
  ┌─────────┐                         ┌───────────┐    ┌────────────┐
  │ChromaDB │◀────────────────────────│ Providers │    │Cost Tracker│
  │ + SQLite│                         │ (3 APIs)  │    │            │
  └─────────┘                         └───────────┘    └────────────┘
```

### 2.3 Modèle de données (ProjectState)

```python
@dataclass
class ProjectState:
    name: str
    user_id: str = "user_default"
    plan: Optional[NormalizedPlan] = None
    corpus: Optional[StructuredCorpus] = None
    generated_sections: dict = {}          # section_id → contenu
    section_summaries: list[str] = []      # Résumés pour contexte inter-sections
    quality_reports: dict = {}             # section_id → QualityReport
    factcheck_reports: dict = {}           # section_id → FactcheckReport
    citations: dict = {}                   # section_id → liste de citations
    glossary: dict = {}                    # terme → définition
    personas: dict = {}                    # persona_id → PersonaConfig
    feedback_history: list = []            # Historique des corrections humaines
    cost_report: dict = {}                 # Rapport de coûts cumulés
    current_step: str = "init"             # init→plan→corpus→generation→review→export→done
    generation_config: dict = {}           # Paramètres de génération actifs
    cache_id: Optional[str] = None         # ID du cache Gemini actif (Phase 5)
```

### 2.4 Persistance par projet

```
projects/{project_id}/
├── state.json           # ProjectState sérialisé
├── corpus/              # Documents sources (001_doc.pdf, ...)
├── chromadb/            # Base vectorielle (HNSW + SQLite)
├── metadata.db          # Métadonnées documents + chunks (SQLite)
└── cache/               # Cache d'extraction (hash MD5 → JSON)
```

---

## 3. Phase 1 — Pipeline fondamental ✅

> **Statut** : Implémenté et opérationnel
> **Objectif** : Pipeline de base fonctionnel de bout en bout

### 3.1 Modules implémentés

| Module | Fichier | Fonctionnalités |
|---|---|---|
| **Orchestrateur** | `orchestrator.py` | Pipeline séquentiel, gestion d'état, sauvegarde JSON |
| **Prompt Engine** | `prompt_engine.py` | Templates système/section/raffinement/résumé/plan |
| **Plan Parser** | `plan_parser.py` | Parsing numéroté (1. / 1.1 / 1.1.1), normalisation hiérarchique |
| **Corpus Extractor** | `corpus_extractor.py` | Structuration, extraction mots-clés TF-IDF, digest multi-paliers |
| **Text Extractor** | `text_extractor.py` | Chaîne de fallback : Docling → PyMuPDF → pdfplumber → PyPDF2 |
| **Export Engine** | `export_engine.py` | DOCX avec styles (titres, corps, marges, logo, couleurs) |
| **Cost Tracker** | `cost_tracker.py` | Estimation pré-génération, suivi temps réel, rapport cumulé |
| **Config** | `config/default.yaml` | 204 paramètres configurables |

### 3.2 Interface utilisateur (Streamlit)

8 pages fonctionnelles :
- **Accueil** : Création/chargement de projets
- **Configuration** : Sélection provider/modèle, paramètres de génération
- **Acquisition** : Upload multi-fichier, saisie d'URLs
- **Plan** : Import texte/markdown, génération auto, édition inline
- **Génération** : Lancement section par section avec barre de progression
- **Export** : Génération DOCX, personnalisation charte, téléchargement
- **Dashboard** : Métriques temps réel, logs d'activité, graphiques
- **Bibliothèque** : Recherche sémantique dans le corpus indexé

### 3.3 Profils pré-configurés

5 profils YAML dans `profiles/default/` :

| Profil | Cible | Pages | Ton |
|---|---|---|---|
| `rapport_analyse.yaml` | Rapport d'analyse | ~20 | Professionnel, analytique |
| `document_formation.yaml` | Support de formation | ~15 | Pédagogique |
| `synthese_veille.yaml` | Synthèse de veille | ~10 | Informatif, concis |
| `proposition_services.yaml` | Proposition commerciale | ~12 | Persuasif |
| `compte_rendu.yaml` | Compte-rendu de réunion | ~5 | Factuel, structuré |

---

## 4. Phase 2 — Multi-fournisseurs et RAG de base ✅

> **Statut** : Implémenté et opérationnel
> **Objectif** : Support multi-providers avec recherche vectorielle

### 4.1 Fournisseurs IA

#### 4.1.1 Architecture Provider

```python
# Interface commune (base.py)
class BaseProvider(ABC):
    def generate(prompt, system_prompt, model, temperature, max_tokens) → AIResponse
    def is_available() → bool
    def get_default_model() → str
    def list_models() → list[str]
    # Batch (optionnel)
    def supports_batch() → bool
    def submit_batch(requests) → batch_id
    def poll_batch(batch_id) → BatchStatus
    def retrieve_batch_results(batch_id) → dict[custom_id, content]
```

#### 4.1.2 Providers implémentés

| Provider | Modèles | Batch | Retry |
|---|---|---|---|
| **OpenAI** | GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo | ✅ Batch API | ✅ Exponentiel |
| **Anthropic** | Claude Opus 4.6, Sonnet 4.5, Haiku 3.5 | ✅ Messages Batch | ✅ Exponentiel |
| **Google Gemini** | Gemini 3.0 Pro, 3.0 Flash (→ **3.1 Pro/Flash en Phase 5**) | ❌ Non | ✅ Exponentiel |

#### 4.1.3 Pricing actuel (`model_pricing.yaml`)

| Provider | Modèle | Input $/1M | Output $/1M | Contexte |
|---|---|---|---|---|
| OpenAI | gpt-4.1 | $2.00 | $8.00 | 1M |
| OpenAI | gpt-4.1-mini | $0.40 | $1.60 | 1M |
| OpenAI | gpt-4.1-nano | $0.10 | $0.40 | 1M |
| OpenAI | gpt-4o | $2.50 | $10.00 | 128K |
| Anthropic | claude-opus-4-6 | $15.00 | $75.00 | 200K |
| Anthropic | claude-sonnet-4-5 | $3.00 | $15.00 | 200K |
| Anthropic | claude-haiku-3-5 | $0.80 | $4.00 | 200K |
| Google | gemini-3.0-pro | $1.25 | $10.00 | 1M |
| Google | gemini-3.0-flash | $0.10 | $0.40 | 1M |

### 4.2 RAG de base

- Vectorisation via ChromaDB (stockage HNSW + cosine distance)
- Recherche top-k avec seuil de pertinence configurable
- Génération conditionnelle selon la couverture du corpus
- Mode batch pour OpenAI et Anthropic avec fallback temps réel

---

## 5. Phase 2.5 — RAG avancé et garde-fous ✅

> **Statut** : Implémenté et opérationnel
> **Objectif** : Pipeline RAG hybride de qualité production

### 5.1 Pipeline RAG en 5 étapes

```
Document → Extraction → Chunking sémantique → Embeddings → ChromaDB → Reranking → Prompt
            (Docling)    (hierarchique)        (local)      (HNSW)     (cross-enc.)
```

#### 5.1.1 Extraction PDF avancée

- **Moteur primaire** : Docling 2.0+ (extraction structurée avec tables, figures)
- **Fallback chain** : Docling → PyMuPDF → pdfplumber → PyPDF2
- **Batch Docling** : Traitement par lots de 30 pages pour les gros documents (>50 pages)
- **Coverage check** : Seuil de 80% des pages couvertes avant rattrapage PyMuPDF

#### 5.1.2 Chunking sémantique

```yaml
# Configuration (default.yaml)
rag.chunking:
  strategy: "semantic"      # "semantic" | "fixed"
  max_chunk_tokens: 800
  min_chunk_tokens: 100
  overlap_sentences: 2
```

Le `SemanticChunker` :
- Détecte les frontières de section (titres, sauts de paragraphe)
- Respecte la hiérarchie du document source
- Produit des chunks cohérents de 100-800 tokens avec overlap de 2 phrases

#### 5.1.3 Embeddings locaux

- **Modèle** : `intfloat/multilingual-e5-large` via FastEmbed (ONNX quantisé)
- **Performance** : ~300 Mo RAM, pas de GPU requis
- **Batch** : Vectorisation de masse (batch_size=512)
- **Fallback** : Embeddings ChromaDB natifs si FastEmbed indisponible
- **Providers alternatifs** : OpenAI `text-embedding-3-small` ou Gemini (via config)

#### 5.1.4 Reranking cross-encoder

- **Modèle** : `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Flow** : ChromaDB retourne 20 candidats → reranker sélectionne les top 10
- **Désactivable** : `rag.reranking_enabled: false`

### 5.2 Anti-hallucination

Bloc injecté systématiquement dans chaque prompt de génération :

```
═══ RÈGLES DE FIABILITÉ (NON NÉGOCIABLES) ═══
1. SOURCES EXCLUSIVES : seuls les blocs corpus fournis sont autorisés
2. MARQUEUR D'INSUFFISANCE : {{NEEDS_SOURCE: [description]}} pour les lacunes
3. ATTRIBUTION : citation par référence APA ou nom de fichier
4. TRANSPARENCE : section plus courte plutôt qu'information inventée
═══ FIN DES RÈGLES ═══
```

- Détection des marqueurs `{{NEEDS_SOURCE}}` résiduels avant export DOCX
- Alerte utilisateur si des points non sourcés subsistent

### 5.3 Liaison plan-corpus

Le `PlanCorpusLinker` effectue une pré-analyse avant la génération :
- Analyse thématique du corpus (jusqu'à 30 documents)
- Extraction des 3 premiers chunks par document
- Mapping plan↔corpus pour estimer la couverture par section

### 5.4 Métadonnées SQLite

Deux tables dans `metadata.db` :

```sql
-- Table documents
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    filepath TEXT, title TEXT, authors TEXT,
    year INTEGER, language TEXT, doc_type TEXT,
    hash TEXT, extraction_method TEXT, apa_reference TEXT
);

-- Table chunks
CREATE TABLE chunks (
    doc_id TEXT, chunk_id TEXT PRIMARY KEY,
    text TEXT, page_number INTEGER,
    section_title TEXT, token_count INTEGER,
    language TEXT, doc_type TEXT
);
```

---

## 6. Phase 3 — Intelligence du pipeline ✅

> **Statut** : Implémenté et opérationnel
> **Objectif** : Évaluation qualité, vérification factuelle, apprentissage

### 6.1 Évaluation qualité (`quality_evaluator.py`)

6 critères pondérés, évalués par le LLM après chaque section :

| Critère | Poids | Description |
|---|---|---|
| Conformité au plan | 1.0 | Respect du titre, niveau et consignes |
| Couverture corpus | 1.5 | Utilisation effective des sources |
| Cohérence narrative | 0.8 | Fluidité et enchaînement avec les sections précédentes |
| Taille cible | 0.5 | Respect du budget de pages |
| Fiabilité factuelle | 1.5 | Absence d'hallucinations |
| Traçabilité sources | 1.2 | Attribution correcte des citations |

- **Seuil auto-raffinement** : score global < 3.0/5.0 → raffinement automatique
- **Modèle d'évaluation** : le plus économique disponible (configurable)

### 6.2 Vérification factuelle (`factcheck_engine.py`)

- Extraction automatique des affirmations factuelles (max 30/section)
- Vérification croisée avec le corpus source
- Score de fiabilité par affirmation (%)
- **Seuil auto-correction** : score < 80% → correction automatique
- Rapport détaillé avec les affirmations non vérifiées

### 6.3 Boucle de feedback (`feedback_engine.py`)

- Analyse des corrections humaines (diff Levenshtein, seuil > 15%)
- Extraction des patterns de correction (style, terminologie, structure)
- Injection dans les prompts des sections suivantes
- Apprentissage cumulatif au fil du projet

### 6.4 Modules optionnels (désactivés par défaut)

| Module | Fichier | Description |
|---|---|---|
| **Glossaire** | `glossary_engine.py` | Extraction et harmonisation terminologique, injection dans les prompts |
| **Citations APA** | `citation_engine.py` | Références APA 7e édition, bibliographie automatique |
| **Personas** | `persona_engine.py` | Modélisation de l'audience cible, adaptation du ton |
| **GROBID** | `grobid_client.py` | Extraction bibliographique via Docker (articles scientifiques) |

---

## 7. Phase 4 — Performance et optimisation ✅

> **Statut** : Implémenté et opérationnel
> **Objectif** : Scalabilité et rapidité de traitement

### 7.1 Acquisition asynchrone (`corpus_acquirer.py`)

- Téléchargement parallèle via `aiohttp` + `aiofiles`
- Throttling configurable (1s entre les requêtes)
- User-agent rotation, timeout adaptatif (mode normal/lent)
- Validation anti-bot (`content_validator.py`)

### 7.2 Extraction parallèle (`text_extractor.py`)

- `ProcessPoolExecutor` avec scaling dynamique basé sur `psutil`
- Nombre de workers adapté à la RAM/CPU disponibles
- Cache d'extraction MD5 : pas de re-processing des fichiers déjà vus

### 7.3 Pipeline de génération asynchrone (`orchestrator.py`)

- `ThreadPoolExecutor` : l'évaluation post-génération de la section N tourne en parallèle avec la génération de la section N+1
- Verrou (`Lock`) pour protéger `save_state` et les mutations d'état
- Sauvegarde incrémentale après chaque section

### 7.4 Pipeline d'embedding asynchrone (`rag_engine.py`)

- **Phase 4.2** : Les embeddings du lot N+1 sont calculés pendant l'écriture ChromaDB du lot N
- Batch size configurable (défaut: 512)
- **Phase 5 (sécurité mémoire)** : Segmentation en lots de 10 000 chunks max pour éviter les OOM

### 7.5 Cache LRU RAG

- Cache en mémoire pour `search_for_section` (évite les recherches redondantes)
- Invalidation automatique à chaque `index_corpus()` ou `reset()`
- Thread-safe via `threading.Lock`

---

## 8. Phase 5 — Intégration Gemini 3.1 et Context Caching 🔧

> **Statut** : À implémenter
> **Objectif** : Exploiter Gemini 3.1 Pro comme "cerveau" principal avec context caching pour réduire les coûts de 90%
> **Priorité** : Haute

### 8.1 Contexte et motivations

**Problème actuel** : Le provider Gemini utilise les modèles `gemini-3.0-pro` et `gemini-3.0-flash`, qui sont **dépréciés** (shutdown le 9 mars 2026). De plus, le module d'embeddings Gemini dans `rag_engine.py` utilise l'ancien SDK `google.generativeai` (obsolète depuis novembre 2025).

**Opportunité** : Gemini 3.1 Pro offre une fenêtre de contexte de 1M tokens à $2/1M input, avec context caching à **$0.20/1M** (réduction de 90%). Pour un corpus typique de 200K tokens réutilisé sur 20 sections, cela représente une économie de ~$7.20 par document.

### 8.2 Mise à jour du provider Gemini

#### 8.2.1 Modèles cibles

| Modèle | Model ID | Usage | Input $/1M | Output $/1M |
|---|---|---|---|---|
| **Gemini 3.1 Pro** | `gemini-3.1-pro-preview` | Cerveau principal (raisonnement, génération) | $2.00 | $12.00 |
| **Gemini 3.1 Pro Custom Tools** | `gemini-3.1-pro-preview-customtools` | Workflows agentic multi-outils | $2.00 | $12.00 |
| **Gemini 3 Flash** | `gemini-3-flash-preview` | Tâches secondaires (résumés, évaluation) | $0.50 | $3.00 |

**Tokens cachés** :
- Gemini 3.1 Pro : $0.20/1M (90% de réduction sur l'input)
- Gemini 3 Flash : $0.05/1M (90% de réduction sur l'input)
- Stockage cache : ~$0.50/h/1M tokens (Pro) — ~$1.00/h/1M tokens (Flash)

#### 8.2.2 Modifications de `gemini_provider.py`

```python
class GeminiProvider(BaseProvider):
    """Fournisseur Google Gemini 3.1."""

    MODELS = [
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
        "gemini-3-flash-preview",
    ]

    def get_default_model(self) -> str:
        return "gemini-3-flash-preview"

    def generate(self, prompt, system_prompt=None, model=None,
                 temperature=0.7, max_tokens=4096,
                 cached_content=None, thinking_level=None) -> AIResponse:
        """Génère avec support du context caching et du thinking level."""
        ...
```

**Paramètres ajoutés** :
- `cached_content: Optional[str]` — Nom du cache à utiliser
- `thinking_level: Optional[str]` — `"minimal"`, `"low"`, `"medium"`, `"high"` (défaut: `"high"`)
- `max_output_tokens` — Doit être explicitement défini (défaut API = 8 192, max = 65 536)

#### 8.2.3 Système de pensée à 3 niveaux

Gemini 3.1 Pro introduit un paramètre `thinking_level` qui module la profondeur de raisonnement :

| Niveau | Usage recommandé | Latence |
|---|---|---|
| `minimal` | Résumés simples, extraction de métadonnées | ~2s |
| `low` | Reformulation, nettoyage de texte | ~5s |
| `medium` | Génération de contenu standard | ~15s |
| `high` | Analyse complexe, raisonnement long, factcheck | ~36s |

**Mapping par tâche Orchestr'IA** :

| Tâche | thinking_level |
|---|---|
| Résumé de section | `low` |
| Génération de plan | `medium` |
| Génération de section | `high` |
| Raffinement multi-pass | `high` |
| Évaluation qualité | `medium` |
| Vérification factuelle | `high` |
| Feedback analysis | `low` |

### 8.3 Context Caching

#### 8.3.1 Principe

Le context caching permet de stocker le corpus une seule fois côté Google et de le réutiliser pour chaque appel de génération. Le coût de lecture des tokens cachés est réduit de 90%.

```
Sans cache :  20 sections × 200K tokens input = 4M tokens × $2.00/1M = $8.00
Avec cache :  1 cache × 200K tokens + 20 lectures × $0.20/1M = $0.04 + stockage
Économie :    ~$7.56 (~95%)
```

#### 8.3.2 API cible (SDK `google-genai`)

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=API_KEY)

# 1. Créer le cache (une seule fois après indexation du corpus)
cache = client.caches.create(
    model='models/gemini-3.1-pro-preview',
    config=types.CreateCachedContentConfig(
        display_name=f'orchestria-{project_id}',
        system_instruction=system_prompt,      # Inclus dans le cache
        contents=[corpus_xml_content],          # Corpus complet en XML
        ttl='7200s',                            # 2 heures
    )
)

# 2. Utiliser le cache pour chaque section
response = client.models.generate_content(
    model='models/gemini-3.1-pro-preview',
    contents=section_prompt,                    # Seul le prompt section varie
    config=types.GenerateContentConfig(
        cached_content=cache.name,
        temperature=0.7,
        max_output_tokens=4096,
        # PAS de system_instruction ici (déjà dans le cache)
        # PAS de tools ici (incompatible avec cached_content)
    )
)

# 3. Mettre à jour le TTL si nécessaire
client.caches.update(
    name=cache.name,
    config=types.UpdateCachedContentConfig(ttl='3600s')
)
```

#### 8.3.3 Contraintes techniques

| Contrainte | Impact | Mitigation |
|---|---|---|
| `system_instruction` doit être dans le cache | Le prompt système ne peut pas varier entre sections | Construire un system prompt générique incluant anti-hallucination et persona |
| Pas de `tools` avec `cached_content` | Incompatible avec le mode agentic | Utiliser le mode agentic sans cache, ou désactiver les tools pour la génération documentaire |
| Contenu immutable après création | Pas de modification du corpus caché | Recréer un nouveau cache si le corpus change |
| Minimum 2 048 tokens pour le cache | Les petits corpus ne bénéficient pas du caching | Fallback en mode standard si corpus < 2 048 tokens |
| Caching implicite instable sur 3.x | Ne pas compter sur le cache automatique | Toujours utiliser le caching explicite |

#### 8.3.4 Stratégie de caching

```
corpus_tokens < 2048     → Mode standard (pas de cache)
2048 ≤ corpus_tokens < 200K → Cache explicite, TTL = 2h, thinking_level adaptatif
corpus_tokens ≥ 200K     → Cache explicite, TTL = 2h, ATTENTION repricing long-context
```

**Piège du repricing** : Au-delà de 200K tokens en input, Google facture TOUT le request (y compris l'output) au tarif long-context ($4/1M input au lieu de $2, $18/1M output au lieu de $12). Le cost_tracker doit en tenir compte.

#### 8.3.5 Nouveau module : `gemini_cache_manager.py`

```python
class GeminiCacheManager:
    """Gère le cycle de vie des caches Gemini pour un projet."""

    def create_corpus_cache(
        self, project_id: str, corpus_xml: str,
        system_prompt: str, model: str, ttl: int = 7200
    ) -> str:
        """Crée un cache contenant le corpus et le system prompt."""

    def get_or_create_cache(self, project_id: str, ...) -> str:
        """Récupère le cache existant ou en crée un nouveau."""

    def extend_cache_ttl(self, cache_name: str, ttl: int) -> None:
        """Prolonge le TTL d'un cache existant."""

    def delete_cache(self, cache_name: str) -> None:
        """Supprime un cache explicitement."""

    def estimate_cache_cost(
        self, corpus_tokens: int, num_sections: int, ttl_hours: float
    ) -> dict:
        """Estime le coût avec vs sans cache."""

    def should_use_cache(self, corpus_tokens: int, num_sections: int) -> bool:
        """Détermine si le caching est rentable pour ce projet."""
```

### 8.4 Mise à jour des embeddings Gemini

Dans `rag_engine.py`, la méthode `_get_embeddings_gemini()` utilise l'ancien SDK :

```python
# AVANT (obsolète)
import google.generativeai as genai
result = genai.embed_content(model=..., content=batch, task_type="retrieval_document")

# APRÈS (nouveau SDK)
from google import genai
client = genai.Client(api_key=API_KEY)
result = client.models.embed_content(
    model='models/text-embedding-004',
    contents=batch,
    config=types.EmbedContentConfig(task_type='RETRIEVAL_DOCUMENT')
)
```

### 8.5 Mise à jour du pricing

Ajouter dans `model_pricing.yaml` :

```yaml
google:
  gemini-3.1-pro-preview:
    input: 2.00
    input_cached: 0.20          # Nouveau
    input_long_context: 4.00    # >200K tokens — Nouveau
    output: 12.00
    output_long_context: 18.00  # >200K tokens — Nouveau
    cache_storage_per_hour: 0.50  # Nouveau
    context_window: 1000000
    max_output_tokens: 65536    # Nouveau
  gemini-3.1-pro-preview-customtools:
    input: 2.00
    input_cached: 0.20
    output: 12.00
    context_window: 1000000
    max_output_tokens: 65536
  gemini-3-flash-preview:
    input: 0.50
    input_cached: 0.05          # Nouveau
    output: 3.00
    cache_storage_per_hour: 1.00  # Nouveau
    context_window: 1000000
    max_output_tokens: 65536
```

### 8.6 Mise à jour du cost_tracker

Le `CostTracker` doit supporter :
- Le calcul des tokens cachés vs non-cachés
- Le repricing long-context (>200K tokens)
- L'estimation du coût de stockage du cache
- Le seuil de rentabilité du cache (break-even ~4 requêtes/heure pour 1M tokens)

### 8.7 Livrables Phase 5

| # | Livrable | Fichier(s) |
|---|---|---|
| 5.1 | Mise à jour provider Gemini 3.1 | `providers/gemini_provider.py` |
| 5.2 | Module de gestion du cache | `core/gemini_cache_manager.py` (nouveau) |
| 5.3 | Migration embeddings Gemini | `core/rag_engine.py` |
| 5.4 | Mise à jour pricing | `config/model_pricing.yaml` |
| 5.5 | Mise à jour cost tracker | `core/cost_tracker.py` |
| 5.6 | Intégration thinking_level | `core/orchestrator.py`, `providers/gemini_provider.py` |
| 5.7 | UI config caching | `pages/page_configuration.py` |
| 5.8 | Tests unitaires | `tests/unit/test_gemini_provider.py`, `tests/unit/test_gemini_cache.py` |
| 5.9 | Tests d'intégration | `tests/integration/test_gemini_caching_pipeline.py` |

---

## 9. Phase 6 — Acquisition GitHub (Clone de dépôts) 🔧

> **Statut** : À implémenter
> **Objectif** : Permettre l'acquisition de dépôts GitHub comme source de corpus pour la génération de documentation technique
> **Priorité** : Haute

### 9.1 Contexte et motivations

L'acquisition actuelle supporte les fichiers locaux (upload) et les URLs (scraping). Pour les projets de documentation technique, les développeurs ont besoin d'intégrer directement le code source et la documentation existante d'un dépôt GitHub.

**Cas d'usage** :
- Générer une documentation technique à partir du code source
- Créer un guide d'architecture à partir de la structure d'un dépôt
- Produire un rapport d'audit de code
- Résumer les README, CHANGELOG et issues d'un projet

### 9.2 Fonctionnalités

#### 9.2.1 Clone et filtrage

```python
class GitHubAcquirer:
    """Acquisition de dépôts GitHub comme corpus."""

    def clone_repo(
        self,
        repo_url: str,               # https://github.com/owner/repo
        branch: str = "main",         # Branche cible
        target_dir: Path = None,      # Répertoire de clone
        depth: int = 1,               # Shallow clone (défaut: dernier commit)
    ) -> Path:
        """Clone un dépôt GitHub (shallow par défaut)."""

    def filter_files(
        self,
        repo_path: Path,
        include_patterns: list[str],   # ["*.py", "*.md", "*.ts", "docs/**"]
        exclude_patterns: list[str],   # ["node_modules/**", "*.lock", ".git/**"]
        max_file_size_kb: int = 500,   # Ignorer les fichiers > 500 Ko
    ) -> list[Path]:
        """Filtre les fichiers pertinents du dépôt."""

    def extract_repo_structure(
        self,
        repo_path: Path,
    ) -> str:
        """Génère un arbre de la structure du dépôt (format tree)."""

    def extract_repo_metadata(
        self,
        repo_url: str,
    ) -> dict:
        """Extrait les métadonnées du dépôt (description, langages, stars, topics)."""
```

#### 9.2.2 Patterns de filtrage par défaut

```yaml
# Profil "Code source" (défaut)
github_acquisition:
  include_patterns:
    - "*.py"
    - "*.js"
    - "*.ts"
    - "*.tsx"
    - "*.jsx"
    - "*.java"
    - "*.go"
    - "*.rs"
    - "*.c"
    - "*.cpp"
    - "*.h"
    - "*.rb"
    - "*.php"
    - "*.swift"
    - "*.kt"
    - "*.md"
    - "*.rst"
    - "*.txt"
    - "*.yaml"
    - "*.yml"
    - "*.json"
    - "*.toml"
    - "Dockerfile"
    - "Makefile"
    - "*.sh"
  exclude_patterns:
    - ".git/**"
    - "node_modules/**"
    - "vendor/**"
    - "__pycache__/**"
    - "*.pyc"
    - "*.min.js"
    - "*.min.css"
    - "*.lock"
    - "*.sum"
    - "dist/**"
    - "build/**"
    - ".next/**"
    - "*.map"
    - "*.wasm"
    - "*.bin"
    - "*.png"
    - "*.jpg"
    - "*.gif"
    - "*.svg"
    - "*.ico"
    - "*.woff"
    - "*.woff2"
    - "*.ttf"
    - "*.eot"
  max_file_size_kb: 500
  shallow_clone: true
  depth: 1
```

#### 9.2.3 Transformation en corpus

Chaque fichier du dépôt est transformé en document corpus avec :

```python
@dataclass
class RepoDocument:
    """Document extrait d'un dépôt GitHub."""
    filepath: str              # Chemin relatif dans le repo
    content: str               # Contenu du fichier
    language: str              # Langage détecté
    file_type: str             # "code" | "documentation" | "config" | "test"
    line_count: int
    token_count: int
    repo_url: str
    branch: str
    last_modified: str         # Date du dernier commit sur ce fichier

    def to_corpus_entry(self) -> dict:
        """Convertit en entrée de corpus standard."""
        return {
            "text": self._format_for_corpus(),
            "source_file": f"github:{self.repo_url}#{self.filepath}",
            "metadata": {
                "doc_type": self.file_type,
                "language": self.language,
                "filepath": self.filepath,
                "line_count": self.line_count,
            }
        }

    def _format_for_corpus(self) -> str:
        """Formate le fichier pour l'injection dans le pipeline RAG."""
        header = f"# Fichier : {self.filepath}\n"
        header += f"# Langage : {self.language}\n"
        header += f"# Type : {self.file_type}\n\n"
        return header + self.content
```

#### 9.2.4 Chunking spécifique au code

Le `SemanticChunker` doit être étendu pour le code source :

| Stratégie | Application | Description |
|---|---|---|
| **Par classe/fonction** | Python, JavaScript, Java, Go | Une classe ou fonction = un chunk |
| **Par bloc logique** | Fichiers de config (YAML, JSON) | Un bloc de config = un chunk |
| **Par section** | Markdown, RST | Un titre = un chunk |
| **Par taille** | Fichiers longs | Fallback au chunking fixe |

```python
class CodeChunker:
    """Chunking sémantique spécifique au code source."""

    def chunk_python(self, content: str, filepath: str) -> list[CodeChunk]:
        """Découpe un fichier Python par classe/fonction."""

    def chunk_javascript(self, content: str, filepath: str) -> list[CodeChunk]:
        """Découpe un fichier JS/TS par export/function/class."""

    def chunk_generic(self, content: str, filepath: str) -> list[CodeChunk]:
        """Fallback : découpage par blocs de lignes."""
```

### 9.3 Interface utilisateur

#### 9.3.1 Modifications de `page_acquisition.py`

Ajout d'un onglet "GitHub" dans la page d'acquisition :

```
┌──────────────────────────────────────────────────────────────┐
│  📁 Fichiers  |  🌐 URLs  |  🐙 GitHub                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  URL du dépôt : [https://github.com/owner/repo         ]    │
│  Branche :      [main                                   ]    │
│                                                              │
│  ☑ Code source (*.py, *.js, *.ts, ...)                      │
│  ☑ Documentation (*.md, *.rst, README)                      │
│  ☐ Configuration (*.yaml, *.json, Dockerfile)               │
│  ☐ Tests (test_*, *_test.*)                                 │
│                                                              │
│  Taille max par fichier : [500] Ko                          │
│                                                              │
│  [🔍 Analyser le dépôt]  [📥 Cloner et indexer]             │
│                                                              │
│  ┌─ Aperçu du dépôt ─────────────────────────────────────┐  │
│  │ 📊 142 fichiers, 38 500 lignes, ~96K tokens           │  │
│  │ 🗂️ Langages : Python (65%), TypeScript (25%), MD (10%)│  │
│  │ 📝 README.md détecté                                  │  │
│  │ 📋 CHANGELOG.md détecté                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Fichiers sélectionnés : 87/142 (~64K tokens)               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ☑ src/main.py (245 lignes, Python)                    │  │
│  │ ☑ src/utils/helpers.py (120 lignes, Python)           │  │
│  │ ☑ README.md (180 lignes, Markdown)                    │  │
│  │ ☐ tests/test_main.py (90 lignes, Python)              │  │
│  │ ...                                                    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

#### 9.3.2 Workflow utilisateur

1. L'utilisateur colle l'URL du dépôt GitHub
2. **Analyser** : clone shallow, affiche la structure et les stats
3. L'utilisateur sélectionne les catégories de fichiers à inclure
4. **Cloner et indexer** : extrait le contenu, transforme en corpus, indexe dans ChromaDB
5. Le corpus GitHub est fusionné avec les autres sources (fichiers, URLs) dans le pipeline RAG

### 9.4 Intégration avec le pipeline existant

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│ GitHub   │────▶│ GitHubAcq.   │────▶│ CodeChunker  │────▶│ RAGEngine│
│ URL      │     │ clone+filter │     │ par langage  │     │ ChromaDB │
└──────────┘     └──────────────┘     └──────────────┘     └──────────┘
                       │                     │
                       ▼                     ▼
                 ┌──────────┐         ┌──────────────┐
                 │ Metadata │         │ SemanticChunk │
                 │ Store    │         │ (fallback)    │
                 └──────────┘         └──────────────┘
```

### 9.5 Dépendances

```
# Aucune dépendance supplémentaire requise
# git est utilisé via subprocess (présent sur tous les systèmes)
# L'API GitHub (optionnelle) est accessible via requests (déjà en dépendance)
```

### 9.6 Configuration

Ajout dans `config/default.yaml` :

```yaml
# Phase 6 — Acquisition GitHub
github_acquisition:
  enabled: true
  shallow_clone: true
  depth: 1
  max_file_size_kb: 500
  max_total_files: 500           # Limite de fichiers par dépôt
  max_total_tokens: 500000       # Limite de tokens par dépôt
  cleanup_after_indexing: true   # Supprimer le clone après indexation
  include_repo_structure: true   # Inclure l'arbre du dépôt comme document
  include_patterns:
    - "*.py"
    - "*.js"
    - "*.ts"
    - "*.md"
    # ... (liste complète dans le profil)
  exclude_patterns:
    - ".git/**"
    - "node_modules/**"
    # ... (liste complète dans le profil)
```

### 9.7 Gestion des erreurs

| Erreur | Handling |
|---|---|
| Dépôt privé sans token | Message clair : "Dépôt privé — configurez `GITHUB_TOKEN` dans `.env`" |
| Dépôt trop volumineux (>1 Go) | Shallow clone obligatoire, avertissement sur le temps de clone |
| Timeout de clone | Timeout configurable (60s), retry 1x |
| Fichier binaire dans les patterns | Détection automatique (magic bytes), skip avec log |
| Encodage non-UTF8 | Tentative de détection d'encodage, fallback latin-1 |

### 9.8 Livrables Phase 6

| # | Livrable | Fichier(s) |
|---|---|---|
| 6.1 | Module d'acquisition GitHub | `core/github_acquirer.py` (nouveau) |
| 6.2 | Chunking spécifique au code | `core/code_chunker.py` (nouveau) |
| 6.3 | Configuration GitHub | `config/default.yaml` (mise à jour) |
| 6.4 | UI onglet GitHub | `pages/page_acquisition.py` (mise à jour) |
| 6.5 | Métadonnées dépôt | `core/metadata_store.py` (mise à jour) |
| 6.6 | Profil "Documentation technique" | `profiles/default/documentation_technique.yaml` (nouveau) |
| 6.7 | Tests unitaires | `tests/unit/test_github_acquirer.py` |
| 6.8 | Tests d'intégration | `tests/integration/test_github_pipeline.py` |

---

## 10. Phase 7 — Orchestration multi-agents 📋

> **Statut** : Planifié
> **Objectif** : Pipeline agentic où plusieurs agents IA collaborent de manière autonome
> **Priorité** : Moyenne (dépend de la Phase 5)

### 10.1 Contexte

Actuellement, le pipeline est séquentiel : chaque section est générée une par une, avec évaluation post-génération. L'orchestration multi-agents permettrait :
- Analyse parallèle du corpus par plusieurs agents spécialisés
- Génération collaborative de sections interdépendantes
- Vérification factuelle en temps réel pendant la génération
- Auto-correction itérative sans intervention humaine

### 10.2 Agents planifiés

| Agent | Modèle | Rôle |
|---|---|---|
| **Architecte** | Gemini 3.1 Pro (`high`) | Planification, structure, cohérence globale |
| **Rédacteur** | Gemini 3.1 Pro (`medium`) | Génération du contenu section par section |
| **Vérificateur** | Gemini 3.1 Pro (`high`) | Factcheck, cohérence interne, sources |
| **Évaluateur** | Gemini 3 Flash | Scoring qualité rapide, métriques |
| **Correcteur** | Gemini 3.1 Pro Custom Tools | Raffinement, intégration du feedback |

### 10.3 Flux multi-agents

```
                    ┌───────────────┐
                    │  ARCHITECTE   │
                    │ (plan global) │
                    └──────┬────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │ RÉDACTEUR  │ │ RÉDACTEUR  │ │ RÉDACTEUR  │
       │ Section 1  │ │ Section 2  │ │ Section 3  │
       └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
             │              │              │
             ▼              ▼              ▼
       ┌────────────────────────────────────────┐
       │         VÉRIFICATEUR (parallèle)       │
       │  factcheck + cohérence inter-sections  │
       └──────────────────┬─────────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ ÉVALUATEUR  │
                   │ score final │
                   └──────┬──────┘
                          │
                    score < 3.0 ?
                     ▼         ▼
                   OUI        NON → Export
                     │
              ┌──────▼──────┐
              │ CORRECTEUR  │
              │ raffinement │
              └─────────────┘
```

### 10.4 Utilisation du `gemini-3.1-pro-preview-customtools`

La variante Custom Tools est conçue pour les workflows où le modèle doit utiliser des outils personnalisés plutôt que de tomber sur des commandes bash :

```python
tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_corpus",
            description="Recherche dans le corpus indexé",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(type="STRING"),
                    "top_k": types.Schema(type="INTEGER"),
                }
            )
        ),
        types.FunctionDeclaration(
            name="get_section_content",
            description="Récupère le contenu d'une section déjà générée",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "section_id": types.Schema(type="STRING"),
                }
            )
        ),
        types.FunctionDeclaration(
            name="evaluate_quality",
            description="Évalue la qualité d'un contenu généré",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "content": types.Schema(type="STRING"),
                    "section_title": types.Schema(type="STRING"),
                }
            )
        ),
    ])
]
```

**Contrainte** : Les tools ne sont PAS compatibles avec `cached_content`. Le mode agentic et le mode caché sont mutuellement exclusifs. L'orchestrateur doit choisir :
- **Mode documentaire** : cache + génération rapide (Phase 5)
- **Mode agentic** : tools + raisonnement autonome (Phase 7)

### 10.5 Livrables Phase 7

| # | Livrable | Fichier(s) |
|---|---|---|
| 7.1 | Framework d'agents | `core/agent_framework.py` (nouveau) |
| 7.2 | Agent Architecte | `core/agents/architect_agent.py` (nouveau) |
| 7.3 | Agent Rédacteur | `core/agents/writer_agent.py` (nouveau) |
| 7.4 | Agent Vérificateur | `core/agents/verifier_agent.py` (nouveau) |
| 7.5 | Agent Évaluateur | `core/agents/evaluator_agent.py` (nouveau) |
| 7.6 | Agent Correcteur | `core/agents/corrector_agent.py` (nouveau) |
| 7.7 | Orchestrateur multi-agents | `core/multi_agent_orchestrator.py` (nouveau) |
| 7.8 | Configuration mode agentic | `config/default.yaml` (mise à jour) |
| 7.9 | UI mode agentic | `pages/page_generation.py` (mise à jour) |
| 7.10 | Tests | `tests/integration/test_multi_agent.py` |

---

## 11. Matrice des dépendances

```
Phase 1 ──────────────────────────────────────────── ✅ Base
  │
  ├── Phase 2 ────────────────────────────────────── ✅ Multi-providers
  │     │
  │     ├── Phase 2.5 ────────────────────────────── ✅ RAG avancé
  │     │     │
  │     │     ├── Phase 3 ────────────────────────── ✅ Intelligence
  │     │     │     │
  │     │     │     └── Phase 4 ──────────────────── ✅ Performance
  │     │     │           │
  │     │     │           ├── Phase 5 ────────────── 🔧 Gemini 3.1 + Cache
  │     │     │           │     │
  │     │     │           │     └── Phase 7 ──────── 📋 Multi-agents
  │     │     │           │
  │     │     │           └── Phase 6 ────────────── 🔧 GitHub Acquisition
  │     │     │
  │     │     └── Phase 6 (aussi) ────────────────── 🔧 (dépend de RAG)
```

| Phase | Dépend de | Bloquant pour |
|---|---|---|
| Phase 5 | Phase 4 | Phase 7 |
| Phase 6 | Phase 2.5, Phase 4 | — |
| Phase 7 | Phase 5 | — |

**Phases 5 et 6 sont indépendantes** et peuvent être développées en parallèle.

---

## 12. Stack technique

### 12.1 Dépendances (`requirements.txt`)

| Catégorie | Package | Version | Usage |
|---|---|---|---|
| **Interface** | streamlit | ≥1.30.0 | UI web |
| **AI Providers** | openai | ≥1.12.0 | API OpenAI |
| | anthropic | ≥0.39.0 | API Anthropic |
| | google-genai | ≥1.51.0 | API Gemini 3.1 (**minimum pour 3.x**) |
| **RAG** | chromadb | ≥0.5.0 | Base vectorielle |
| | fastembed | ≥0.2.0 | Embeddings ONNX |
| | sentence-transformers | ≥3.0 | Cross-encoder reranking |
| **PDF** | docling | ≥2.0 | Extraction structurée |
| | pymupdf | ≥1.23.0 | Fallback PDF |
| | pdfplumber | ≥0.10.0 | Fallback PDF |
| | PyPDF2 | ≥3.0.0 | Fallback PDF |
| **Documents** | python-docx | ≥1.1.0 | Import/export DOCX |
| | beautifulsoup4 | ≥4.12.0 | Parsing HTML |
| | openpyxl | ≥3.1.0 | Import Excel |
| | pandas | ≥2.1.0 | Traitement données |
| **Async** | aiohttp | ≥3.9.0 | Téléchargement async |
| | aiofiles | ≥23.0.0 | I/O disque async |
| | psutil | ≥5.9.0 | Monitoring ressources |
| **Config** | pyyaml | ≥6.0.0 | Fichiers YAML |
| | python-dotenv | ≥1.0.0 | Variables d'environnement |
| **Tests** | pytest | ≥8.0.0 | Framework de test |
| | pytest-cov | ≥4.1.0 | Couverture de code |

### 12.2 Variables d'environnement (`.env`)

```bash
# Fournisseurs IA (au moins un requis)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...

# GitHub (optionnel, pour dépôts privés — Phase 6)
GITHUB_TOKEN=ghp_...
```

### 12.3 Infrastructure

| Composant | Technologie | Hébergement |
|---|---|---|
| Application | Streamlit | Local / VM |
| Base vectorielle | ChromaDB (embedded) | Local (SQLite + HNSW) |
| Métadonnées | SQLite | Local |
| Embeddings | FastEmbed ONNX | Local (CPU, ~300 Mo) |
| Reranker | Cross-encoder | Local (CPU, ~100 Mo) |
| Cache AI | Gemini Context Cache | Google Cloud |
| GROBID | Docker (optionnel) | Local |

---

## 13. Annexes

### 13.1 Glossaire technique

| Terme | Définition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — enrichit les prompts avec des données du corpus |
| **HITL** | Human-In-The-Loop — validation humaine à des points de contrôle |
| **Chunking** | Découpage du texte en blocs sémantiques pour la vectorisation |
| **Reranking** | Reclassement des résultats de recherche par un modèle cross-encoder |
| **Context Caching** | Stockage côté serveur d'un prefix de prompt pour réutilisation |
| **Thinking Level** | Paramètre Gemini 3.1 qui contrôle la profondeur de raisonnement |
| **TTL** | Time-To-Live — durée de vie d'un cache avant expiration |
| **Shallow Clone** | Clone git limité au dernier commit (économise bande passante) |

### 13.2 Patterns d'architecture utilisés

| Pattern | Implémentation | Localisation |
|---|---|---|
| **Registry** | Enregistrement dynamique des providers | `utils/providers_registry.py` |
| **Factory** | Instanciation des providers par nom | `pages/page_configuration.py` |
| **Strategy** | Chaîne de fallback pour l'extraction PDF | `core/text_extractor.py` |
| **Observer** | Notifications de checkpoint HITL | `core/checkpoint_manager.py` |
| **Pipeline** | Flux asynchrone embedding // écriture | `core/rag_engine.py` |
| **Cache** | LRU + disque pour extractions et recherches | `core/rag_engine.py`, `core/text_extractor.py` |
| **Chain of Resp.** | Fallback PDF (Docling → PyMuPDF → ...) | `core/text_extractor.py` |
| **Dataclass** | Modèles de données immuables | `core/orchestrator.py`, `providers/base.py` |

### 13.3 Métriques de qualité

| Critère | Poids | Seuil min | Auto-action |
|---|---|---|---|
| Conformité au plan | 1.0 | 2.0 | Raffinement |
| Couverture corpus | 1.5 | 2.0 | Raffinement |
| Cohérence narrative | 0.8 | 2.0 | Raffinement |
| Taille cible | 0.5 | 1.5 | Raffinement |
| Fiabilité factuelle | 1.5 | 2.5 | Factcheck + correction |
| Traçabilité sources | 1.2 | 2.0 | Injection citations |

### 13.4 Estimation des coûts par scénario

#### Scénario A : Rapport 20 pages, corpus 50K tokens, 15 sections

| Provider | Sans cache | Avec cache (Gemini) | Économie |
|---|---|---|---|
| GPT-4.1 | ~$3.50 | N/A | — |
| Claude Sonnet 4.5 | ~$4.20 | N/A | — |
| Gemini 3.1 Pro | ~$3.00 | ~$0.85 | **72%** |
| Gemini 3 Flash | ~$0.45 | ~$0.12 | **73%** |

#### Scénario B : Documentation technique 50 pages, dépôt GitHub 200K tokens, 30 sections

| Provider | Sans cache | Avec cache (Gemini) | Économie |
|---|---|---|---|
| GPT-4.1 | ~$14.00 | N/A | — |
| Gemini 3.1 Pro | ~$13.20 | ~$2.10 | **84%** |
| Gemini 3 Flash | ~$1.80 | ~$0.30 | **83%** |

*Note : les estimations incluent input, output, résumés et évaluation qualité. Le coût de stockage du cache (~$0.50/h pour 200K tokens) est inclus pour une session de 2h.*

### 13.5 Roadmap de livraison

| Phase | Estimation | Dépendance |
|---|---|---|
| Phase 5 — Gemini 3.1 + Cache | — | Phase 4 ✅ |
| Phase 6 — GitHub Acquisition | — | Phase 2.5 ✅ |
| Phase 7 — Multi-agents | — | Phase 5 |

**Phases 5 et 6 peuvent être développées en parallèle.**

---

> *Document généré pour le projet Orchestr'IA — Février 2026*
