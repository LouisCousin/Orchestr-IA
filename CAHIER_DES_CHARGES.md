# Orchestr'IA — Cahier des Charges Technique

> **Version** : 6.0
> **Date** : 27 février 2026
> **Statut** : Document de référence

---

## Table des matières

1. [État actuel du projet](#1-état-actuel-du-projet)
2. [Phase 5 — Intégration Gemini 3.1 et Context Caching](#2-phase-5--intégration-gemini-31-et-context-caching)
3. [Phase 6 — Acquisition GitHub (Clone de dépôts)](#3-phase-6--acquisition-github-clone-de-dépôts)
4. [Phase 7 — Orchestration multi-agents](#4-phase-7--orchestration-multi-agents)
5. [Matrice des dépendances](#5-matrice-des-dépendances)
6. [Estimation des coûts](#6-estimation-des-coûts)

---

## 1. État actuel du projet

Orchestr'IA est un pipeline de génération documentaire assistée par IA. Il transforme un corpus de documents sources (PDF, DOCX, TXT, HTML, Excel) en documents professionnels via 5 étapes : Configuration → Acquisition → Plan → Génération → Export.

### 1.1 Ce qui est opérationnel

| Composant | Détail |
|---|---|
| **Pipeline complet** | 30 modules core, 8 pages Streamlit, 5 profils pré-configurés |
| **3 providers IA** | OpenAI (GPT-4.1/4o), Anthropic (Claude Opus/Sonnet/Haiku), Google Gemini (3.0 Pro/Flash) |
| **RAG hybride** | ChromaDB + embeddings locaux (`multilingual-e5-large`) + reranking cross-encoder |
| **Intelligence** | Évaluation qualité (6 critères), factcheck, feedback loop, glossaire, citations APA, personas |
| **Performance** | Acquisition async, extraction parallèle, pipeline embedding asynchrone, cache LRU, batch API |
| **Anti-hallucination** | Marqueurs `{{NEEDS_SOURCE}}`, attribution par nom de fichier/APA, détection pré-export |

### 1.2 Points de friction identifiés

| Problème | Impact | Phase corrective |
|---|---|---|
| Les modèles Gemini 3.0 sont **dépréciés** (shutdown 9 mars 2026) | Provider Google inutilisable à court terme | Phase 5 |
| Le SDK `google.generativeai` (embeddings Gemini) est **obsolète** depuis nov. 2025 | Méthode `_get_embeddings_gemini()` dans `rag_engine.py` cassée | Phase 5 |
| Pas d'acquisition depuis un dépôt de code source | Impossible de documenter un projet à partir de son code | Phase 6 |
| Pipeline purement séquentiel en mode agentic | Pas de parallélisme entre agents spécialisés | Phase 7 |
| Pas de context caching | Chaque section repaie le corpus complet en tokens input | Phase 5 |

### 1.3 Architecture actuelle

```
src/
├── app.py                          # Point d'entrée Streamlit
├── core/                           # 30 modules (orchestrator, RAG, export, qualité...)
├── pages/                          # 8 pages UI
├── providers/                      # openai, anthropic, gemini, base
└── utils/                          # config, logger, token_counter, registry, validator

config/
├── default.yaml                    # 204 paramètres
└── model_pricing.yaml              # Tarifs des 3 providers

projects/{id}/
├── state.json                      # ProjectState sérialisé
├── corpus/                         # Documents sources
├── chromadb/                       # Base vectorielle
├── metadata.db                     # SQLite (documents + chunks)
└── cache/                          # Cache d'extraction MD5
```

---

## 2. Phase 5 — Intégration Gemini 3.1 et Context Caching

> **Priorité** : Haute (deadline : avant le 9 mars 2026)
> **Objectif** : Migrer vers Gemini 3.1, activer le context caching pour réduire les coûts de 75-90%

### 2.1 Motivations

Le provider Gemini actuel (`gemini_provider.py`) utilise les modèles `gemini-3.0-pro` et `gemini-3.0-flash`, dépréciés avec shutdown le 9 mars 2026. Gemini 3.1 Pro apporte :

- **Context caching** : stocker le corpus côté Google, réutiliser pour chaque section → -90% sur l'input
- **Thinking levels** : moduler la profondeur de raisonnement par tâche (minimal → high)
- **Custom Tools** : workflows agentic avec function calling amélioré
- **Fenêtre 1M tokens** maintenue, output jusqu'à 65 536 tokens

### 2.2 Mise à jour du provider Gemini

#### 2.2.1 Modèles cibles

| Modèle | Model ID | Usage | Input $/1M | Output $/1M |
|---|---|---|---|---|
| **Gemini 3.1 Pro** | `gemini-3.1-pro-preview` | Génération, raisonnement | $2.00 | $12.00 |
| **Gemini 3.1 Pro Custom Tools** | `gemini-3.1-pro-preview-customtools` | Mode agentic multi-outils | $2.00 | $12.00 |
| **Gemini 3 Flash** | `gemini-3-flash-preview` | Tâches secondaires (résumés, évaluation) | $0.50 | $3.00 |

**Tokens cachés** :
- Pro : $0.20/1M input (90% de réduction), stockage ~$0.50/h/1M tokens
- Flash : $0.05/1M input (90% de réduction), stockage ~$1.00/h/1M tokens

#### 2.2.2 Modifications de `gemini_provider.py`

**État actuel** :
```python
class GeminiProvider(BaseProvider):
    MODELS = ["gemini-3.0-pro", "gemini-3.0-flash"]
    def get_default_model(self): return "gemini-3.0-flash"
    def generate(self, prompt, system_prompt, model, temperature, max_tokens) -> AIResponse: ...
```

**Cible** :
```python
class GeminiProvider(BaseProvider):
    MODELS = [
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
        "gemini-3-flash-preview",
    ]

    def get_default_model(self) -> str:
        return "gemini-3-flash-preview"

    def generate(
        self, prompt, system_prompt=None, model=None,
        temperature=0.7, max_tokens=4096,
        cached_content=None,       # Nom du cache Gemini à utiliser
        thinking_level=None,       # "minimal" | "low" | "medium" | "high"
    ) -> AIResponse:
        """Génère avec support du context caching et du thinking level."""
        ...
```

**Changements clés dans `generate()`** :
- Si `cached_content` est fourni : passer `cached_content=cache_name` dans `GenerateContentConfig`, ne PAS inclure `system_instruction` (déjà dans le cache)
- Si `thinking_level` est fourni : l'ajouter dans `GenerateContentConfig` via `thinking_config`
- `max_output_tokens` explicite (le défaut API est 8 192, max 65 536)
- Extraire `usage_metadata.cached_content_token_count` pour le cost tracking

#### 2.2.3 Thinking levels

Gemini 3.1 Pro expose un paramètre `thinking_level` qui module la profondeur de raisonnement :

| Niveau | Latence type | Usage recommandé dans Orchestr'IA |
|---|---|---|
| `minimal` | ~2s | Extraction de métadonnées |
| `low` | ~5s | Résumés de sections, feedback analysis |
| `medium` | ~15s | Génération de plan, évaluation qualité |
| `high` | ~36s | Génération de sections, factcheck, raffinement |

**Mapping par tâche** (à configurer dans `default.yaml`) :

```yaml
gemini:
  thinking_levels:
    summary: "low"
    plan_generation: "medium"
    section_generation: "high"
    refinement: "high"
    quality_evaluation: "medium"
    factcheck: "high"
    feedback_analysis: "low"
```

L'`orchestrator.py` doit passer le bon `thinking_level` au provider selon le type de tâche en cours.

### 2.3 Context Caching

#### 2.3.1 Principe

Le context caching stocke le corpus une seule fois côté Google. Chaque appel de génération réutilise ce cache : les tokens cachés sont facturés à 10% du prix normal.

```
Sans cache :  20 sections × 200K tokens input = 4M tokens × $2.00/1M = $8.00
Avec cache :  1 création + 20 lectures × 200K × $0.20/1M = $0.80 + stockage
Économie :    ~$7.00 (~88%)
```

#### 2.3.2 API cible (SDK `google-genai`)

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
        contents=[corpus_content],              # Corpus complet
        ttl='7200s',                            # 2 heures
    )
)
# cache.name → "cachedContents/abc123" (à stocker dans ProjectState)

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

# 3. Prolonger le TTL si la génération prend du temps
client.caches.update(
    name=cache.name,
    config=types.UpdateCachedContentConfig(ttl='3600s')
)

# 4. Supprimer le cache à la fin du projet
client.caches.delete(name=cache.name)
```

#### 2.3.3 Contraintes techniques

| Contrainte | Impact | Mitigation |
|---|---|---|
| `system_instruction` dans le cache → immutable | Le prompt système ne peut pas varier entre sections | Construire un system prompt générique englobant anti-hallucination + persona |
| Pas de `tools` avec `cached_content` | Mode agentic (Phase 7) incompatible avec le cache | Deux modes exclusifs : documentaire (cache) vs agentic (tools) |
| Contenu immutable après création | Pas de modification du corpus caché | Recréer un cache si le corpus change |
| Minimum **2 048 tokens** dans le cache | Les petits corpus ne bénéficient pas | Fallback en mode standard si corpus < 2 048 tokens |
| Repricing >200K tokens input | Tout le request (input + output) facturé au tarif long-context | Le cost_tracker doit appliquer les tarifs long-context |

#### 2.3.4 Nouveau module : `core/gemini_cache_manager.py`

```python
class GeminiCacheManager:
    """Gère le cycle de vie des caches Gemini pour un projet."""

    def create_corpus_cache(
        self, project_id: str, corpus_content: str,
        system_prompt: str, model: str, ttl: int = 7200
    ) -> str:
        """Crée un cache contenant le corpus et le system prompt.
        Retourne le cache name (ex: 'cachedContents/abc123')."""

    def get_or_create_cache(self, project_id: str, ...) -> str:
        """Récupère le cache existant ou en crée un nouveau.
        Vérifie que le cache est toujours valide (TTL non expiré)."""

    def extend_cache_ttl(self, cache_name: str, ttl: int) -> None:
        """Prolonge le TTL d'un cache existant."""

    def delete_cache(self, cache_name: str) -> None:
        """Supprime un cache explicitement."""

    def estimate_cache_cost(
        self, corpus_tokens: int, num_sections: int, ttl_hours: float
    ) -> dict:
        """Retourne {'with_cache': float, 'without_cache': float, 'savings_pct': float}."""

    def should_use_cache(self, corpus_tokens: int, num_sections: int) -> bool:
        """True si le caching est rentable (corpus ≥ 2048 tokens, ≥ 3 sections)."""
```

**Stockage du cache_name** : ajouté dans `ProjectState.cache_id` (déjà prévu dans le dataclass).

#### 2.3.5 Stratégie de caching

```
corpus_tokens < 2048       → Mode standard (pas de cache)
2048 ≤ corpus_tokens < 200K → Cache explicite, TTL = 2h, tarif normal
corpus_tokens ≥ 200K       → Cache explicite, TTL = 2h, tarif long-context ($4/1M input, $18/1M output)
```

### 2.4 Migration des embeddings Gemini

Dans `rag_engine.py`, la méthode `_get_embeddings_gemini()` utilise l'ancien SDK obsolète :

```python
# AVANT (cassé — google.generativeai est obsolète)
import google.generativeai as genai
result = genai.embed_content(model=..., content=batch, task_type="retrieval_document")

# APRÈS (nouveau SDK google-genai)
from google import genai
from google.genai import types
client = genai.Client(api_key=API_KEY)
result = client.models.embed_content(
    model='models/text-embedding-004',
    contents=batch,
    config=types.EmbedContentConfig(task_type='RETRIEVAL_DOCUMENT')
)
embeddings = [e.values for e in result.embeddings]
```

### 2.5 Mise à jour du pricing

Remplacement complet de la section `google:` dans `config/model_pricing.yaml` :

```yaml
google:
  gemini-3.1-pro-preview:
    input: 2.00
    input_cached: 0.20
    input_long_context: 4.00        # >200K tokens
    output: 12.00
    output_long_context: 18.00      # >200K tokens
    cache_storage_per_hour: 0.50
    context_window: 1000000
    max_output_tokens: 65536
  gemini-3.1-pro-preview-customtools:
    input: 2.00
    input_cached: 0.20
    output: 12.00
    context_window: 1000000
    max_output_tokens: 65536
  gemini-3-flash-preview:
    input: 0.50
    input_cached: 0.05
    output: 3.00
    cache_storage_per_hour: 1.00
    context_window: 1000000
    max_output_tokens: 65536
```

### 2.6 Mise à jour du cost_tracker

Le `CostTracker` doit supporter les nouveaux champs :
- Calcul tokens cachés vs non-cachés (via `usage_metadata.cached_content_token_count`)
- Repricing long-context quand input total > 200K tokens
- Estimation du coût de stockage du cache (durée × taille × tarif/heure)
- Affichage de l'économie réalisée grâce au cache dans le dashboard

### 2.7 Livrables

| # | Livrable | Fichier(s) |
|---|---|---|
| 5.1 | Mise à jour provider Gemini 3.1 | `providers/gemini_provider.py` |
| 5.2 | Module de gestion du cache | `core/gemini_cache_manager.py` (nouveau) |
| 5.3 | Migration embeddings Gemini | `core/rag_engine.py` |
| 5.4 | Mise à jour pricing | `config/model_pricing.yaml` |
| 5.5 | Mise à jour cost tracker (tokens cachés, long-context) | `core/cost_tracker.py` |
| 5.6 | Config thinking levels | `config/default.yaml` |
| 5.7 | Intégration thinking_level dans l'orchestrateur | `core/orchestrator.py` |
| 5.8 | UI config caching | `pages/page_configuration.py` |
| 5.9 | Tests unitaires | `tests/unit/test_gemini_provider.py`, `tests/unit/test_gemini_cache.py` |
| 5.10 | Test d'intégration | `tests/integration/test_gemini_caching_pipeline.py` |

---

## 3. Phase 6 — Acquisition GitHub (Clone de dépôts)

> **Priorité** : Haute
> **Objectif** : Permettre l'acquisition de dépôts GitHub comme source de corpus pour la génération de documentation technique

### 3.1 Motivations

L'acquisition actuelle supporte les fichiers locaux (upload) et les URLs (scraping). Pour les projets de documentation technique, il manque l'intégration directe du code source d'un dépôt GitHub.

**Cas d'usage** :
- Générer une documentation technique à partir du code source
- Créer un guide d'architecture à partir de la structure d'un dépôt
- Produire un rapport d'audit de code
- Résumer les README, CHANGELOG et issues d'un projet open-source

### 3.2 Module d'acquisition : `core/github_acquirer.py`

```python
class GitHubAcquirer:
    """Acquisition de dépôts GitHub comme corpus."""

    def clone_repo(
        self,
        repo_url: str,                # https://github.com/owner/repo
        branch: str = "main",
        target_dir: Path = None,
        depth: int = 1,               # Shallow clone (défaut)
    ) -> Path:
        """Clone un dépôt GitHub via git subprocess.
        Retourne le chemin du clone local."""

    def filter_files(
        self,
        repo_path: Path,
        include_patterns: list[str],   # ["*.py", "*.md", "docs/**"]
        exclude_patterns: list[str],   # ["node_modules/**", ".git/**"]
        max_file_size_kb: int = 500,
    ) -> list[Path]:
        """Filtre les fichiers pertinents du dépôt cloné."""

    def extract_repo_structure(self, repo_path: Path) -> str:
        """Génère un arbre textuel de la structure du dépôt (format tree)."""

    def extract_repo_metadata(self, repo_url: str) -> dict:
        """Extrait les métadonnées via l'API GitHub (description, langages, topics).
        Nécessite GITHUB_TOKEN pour les dépôts privés."""

    def convert_to_corpus(
        self,
        repo_path: Path,
        filtered_files: list[Path],
    ) -> list[dict]:
        """Convertit les fichiers filtrés en entrées de corpus standard
        compatibles avec CorpusExtractor et RAGEngine."""
```

### 3.3 Patterns de filtrage

```yaml
# config/default.yaml — Section ajoutée
github_acquisition:
  enabled: true
  shallow_clone: true
  depth: 1
  max_file_size_kb: 500
  max_total_files: 500
  max_total_tokens: 500000
  cleanup_after_indexing: true     # Supprimer le clone après indexation
  include_repo_structure: true    # Inclure l'arbre du dépôt comme document corpus
  include_patterns:
    # Code source
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
    # Documentation
    - "*.md"
    - "*.rst"
    - "*.txt"
    # Configuration
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
    - "*.woff*"
    - "*.ttf"
    - "*.eot"
```

### 3.4 Transformation en corpus

Chaque fichier du dépôt est transformé en entrée de corpus avec un header de contexte :

```python
@dataclass
class RepoDocument:
    filepath: str              # Chemin relatif dans le repo
    content: str               # Contenu du fichier
    language: str              # Langage détecté (extension)
    file_type: str             # "code" | "documentation" | "config" | "test"
    line_count: int
    token_count: int
    repo_url: str
    branch: str

    def to_corpus_entry(self) -> dict:
        """Convertit en entrée de corpus standard pour CorpusExtractor."""
        header = f"# Fichier : {self.filepath}\n"
        header += f"# Langage : {self.language}\n"
        header += f"# Type : {self.file_type}\n\n"
        return {
            "text": header + self.content,
            "source_file": f"github:{self.repo_url}#{self.filepath}",
            "metadata": {
                "doc_type": self.file_type,
                "language": self.language,
                "filepath": self.filepath,
                "line_count": self.line_count,
            }
        }
```

### 3.5 Chunking spécifique au code : `core/code_chunker.py`

Le `SemanticChunker` existant est conçu pour du texte en prose. Le code source nécessite un chunking adapté :

| Stratégie | Application | Logique |
|---|---|---|
| **Par classe/fonction** | Python, JS/TS, Java, Go, Rust | Chaque classe ou fonction top-level = un chunk |
| **Par bloc logique** | YAML, JSON, TOML | Chaque clé de premier niveau = un chunk |
| **Par section** | Markdown, RST | Chaque titre (# / ##) = un chunk |
| **Par taille** | Tout fichier long | Fallback : découpage par blocs de N lignes |

```python
class CodeChunker:
    """Chunking sémantique adapté au code source."""

    def chunk_file(self, content: str, filepath: str) -> list[CodeChunk]:
        """Dispatch vers la stratégie de chunking selon l'extension."""

    def chunk_python(self, content: str, filepath: str) -> list[CodeChunk]:
        """Découpe par class/def de niveau 0 (via ast.parse ou regex)."""

    def chunk_javascript(self, content: str, filepath: str) -> list[CodeChunk]:
        """Découpe par export/function/class (regex-based)."""

    def chunk_markdown(self, content: str, filepath: str) -> list[CodeChunk]:
        """Découpe par titre (#, ##, ###)."""

    def chunk_generic(self, content: str, filepath: str, lines_per_chunk: int = 80) -> list[CodeChunk]:
        """Fallback : blocs de N lignes avec overlap."""
```

Les chunks produits par `CodeChunker` sont ensuite vectorisés et indexés dans ChromaDB via le pipeline RAG existant.

### 3.6 Interface utilisateur

Ajout d'un 3e onglet "GitHub" dans `pages/page_acquisition.py` :

```
┌──────────────────────────────────────────────────────────────┐
│  Fichiers  |  URLs  |  GitHub                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  URL du dépôt : [https://github.com/owner/repo         ]    │
│  Branche :      [main                                   ]    │
│                                                              │
│  Types de fichiers à inclure :                               │
│  [x] Code source (*.py, *.js, *.ts, ...)                    │
│  [x] Documentation (*.md, *.rst, README)                    │
│  [ ] Configuration (*.yaml, *.json, Dockerfile)             │
│  [ ] Tests (test_*, *_test.*)                               │
│                                                              │
│  Taille max par fichier : [500] Ko                          │
│                                                              │
│  [Analyser le dépôt]  [Cloner et indexer]                   │
│                                                              │
│  -- Aperçu du dépôt --------------------------------------- │
│  142 fichiers, 38 500 lignes, ~96K tokens                   │
│  Langages : Python (65%), TypeScript (25%), Markdown (10%)  │
│  README.md détecté | CHANGELOG.md détecté                   │
│  ---------------------------------------------------------- │
│                                                              │
│  Fichiers sélectionnés : 87/142 (~64K tokens)               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ [x] src/main.py (245 lignes, Python)                  │  │
│  │ [x] src/utils/helpers.py (120 lignes, Python)         │  │
│  │ [x] README.md (180 lignes, Markdown)                  │  │
│  │ [ ] tests/test_main.py (90 lignes, Python)            │  │
│  │ ...                                                    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Workflow** :
1. L'utilisateur colle l'URL du dépôt et choisit la branche
2. **Analyser** : shallow clone, affiche structure + stats (fichiers, lignes, tokens, langages)
3. L'utilisateur affine la sélection (catégories de fichiers, fichiers individuels)
4. **Cloner et indexer** : extraction, chunking par langage, vectorisation dans ChromaDB
5. Le corpus GitHub est fusionné avec les autres sources dans le pipeline RAG

### 3.7 Intégration avec le pipeline existant

```
GitHub URL ──▶ GitHubAcquirer ──▶ CodeChunker ──▶ RAGEngine (ChromaDB)
                 (clone+filter)    (par langage)      │
                      │                                │
                      ▼                                ▼
                MetadataStore                   Pipeline existant
                (doc_type="code")             (prompt, generate, export)
```

Les fichiers du dépôt deviennent des documents corpus standard. Le reste du pipeline (prompt engine, génération, évaluation) fonctionne sans modification.

### 3.8 Gestion des erreurs

| Erreur | Handling |
|---|---|
| Dépôt privé sans token | Message : "Dépôt privé — configurez `GITHUB_TOKEN` dans `.env`" |
| Dépôt trop volumineux (>1 Go) | Shallow clone obligatoire, avertissement temps de clone |
| Timeout de clone | Timeout 60s, 1 retry |
| Fichier binaire dans les patterns | Détection magic bytes, skip avec log |
| Encodage non-UTF8 | Détection chardet/latin-1, fallback avec remplacement |
| Branche inexistante | Message clair, proposition de branches disponibles |

### 3.9 Dépendances

Aucune dépendance supplémentaire : `git` est utilisé via `subprocess` (présent sur tous les systèmes) et l'API GitHub est accessible via `requests` (déjà dans les dépendances).

Variable d'environnement optionnelle dans `.env` :
```bash
GITHUB_TOKEN=ghp_...   # Pour les dépôts privés
```

### 3.10 Livrables

| # | Livrable | Fichier(s) |
|---|---|---|
| 6.1 | Module d'acquisition GitHub | `core/github_acquirer.py` (nouveau) |
| 6.2 | Chunking spécifique au code | `core/code_chunker.py` (nouveau) |
| 6.3 | Configuration GitHub | `config/default.yaml` (ajout section) |
| 6.4 | UI onglet GitHub | `pages/page_acquisition.py` (modification) |
| 6.5 | Métadonnées dépôt dans le store | `core/metadata_store.py` (modification) |
| 6.6 | Profil "Documentation technique" | `profiles/default/documentation_technique.yaml` (nouveau) |
| 6.7 | Tests unitaires | `tests/unit/test_github_acquirer.py`, `tests/unit/test_code_chunker.py` |
| 6.8 | Test d'intégration | `tests/integration/test_github_pipeline.py` |

---

## 4. Phase 7 — Orchestration multi-agents

> **Priorité** : Moyenne (dépend de la Phase 5)
> **Objectif** : Pipeline agentic avec agents spécialisés collaborant en parallèle

### 4.1 Motivations

Le pipeline actuel est séquentiel : chaque section est générée puis évaluée l'une après l'autre. L'orchestration multi-agents permettrait :
- Génération parallèle de sections indépendantes
- Vérification factuelle en temps réel pendant la génération
- Auto-correction itérative sans intervention humaine
- Allocation intelligente des modèles par tâche

### 4.2 Agents planifiés

| Agent | Modèle | Thinking | Rôle |
|---|---|---|---|
| **Architecte** | Gemini 3.1 Pro | `high` | Planification, structure globale, cohérence |
| **Rédacteur** | Gemini 3.1 Pro | `medium` | Génération du contenu section par section |
| **Vérificateur** | Gemini 3.1 Pro | `high` | Factcheck, cohérence inter-sections, sources |
| **Évaluateur** | Gemini 3 Flash | — | Scoring qualité rapide, métriques |
| **Correcteur** | Gemini 3.1 Pro Custom Tools | `high` | Raffinement avec accès aux outils (recherche corpus, lecture sections) |

### 4.3 Flux multi-agents

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
              │ (+ tools)   │
              └─────────────┘
```

### 4.4 Custom Tools pour l'agent Correcteur

La variante `gemini-3.1-pro-preview-customtools` permet au correcteur d'utiliser des outils pour accéder au contexte du projet :

```python
tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_corpus",
            description="Recherche sémantique dans le corpus indexé",
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
            description="Évalue la qualité d'un contenu",
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

**Contrainte critique** : les `tools` sont incompatibles avec `cached_content`. Le mode agentic (Phase 7) et le mode documentaire avec cache (Phase 5) sont **mutuellement exclusifs**. L'orchestrateur doit exposer les deux modes :
- **Mode documentaire** : cache activé, génération séquentielle rapide et économique
- **Mode agentic** : tools activés, agents parallèles, pas de cache

### 4.5 Livrables

| # | Livrable | Fichier(s) |
|---|---|---|
| 7.1 | Framework d'agents | `core/agent_framework.py` (nouveau) |
| 7.2 | Agents spécialisés | `core/agents/` (nouveau répertoire, 5 fichiers) |
| 7.3 | Orchestrateur multi-agents | `core/multi_agent_orchestrator.py` (nouveau) |
| 7.4 | Configuration mode agentic | `config/default.yaml` (ajout section) |
| 7.5 | UI sélection de mode | `pages/page_generation.py` (modification) |
| 7.6 | Tests d'intégration | `tests/integration/test_multi_agent.py` |

---

## 5. Matrice des dépendances

```
Phase 5 (Gemini 3.1 + Cache)  ◀── aucune dépendance
       │
       └──▶ Phase 7 (Multi-agents)  ◀── dépend de Phase 5

Phase 6 (GitHub Acquisition)   ◀── aucune dépendance
```

| Phase | Dépend de | Bloque |
|---|---|---|
| Phase 5 | — | Phase 7 |
| Phase 6 | — | — |
| Phase 7 | Phase 5 | — |

**Les phases 5 et 6 sont indépendantes et peuvent être développées en parallèle.**

---

## 6. Estimation des coûts

### 6.1 Scénario A : Rapport 20 pages, corpus 50K tokens, 15 sections

| Provider | Mode | Coût estimé |
|---|---|---|
| GPT-4.1 | Standard | ~$3.50 |
| Claude Sonnet 4.5 | Standard | ~$4.20 |
| Gemini 3.1 Pro | Standard | ~$3.00 |
| Gemini 3.1 Pro | **Avec cache** | **~$0.85** |
| Gemini 3 Flash | **Avec cache** | **~$0.12** |

### 6.2 Scénario B : Doc technique 50 pages, dépôt GitHub 200K tokens, 30 sections

| Provider | Mode | Coût estimé |
|---|---|---|
| GPT-4.1 | Standard | ~$14.00 |
| Gemini 3.1 Pro | Standard | ~$13.20 |
| Gemini 3.1 Pro | **Avec cache** | **~$2.10** |
| Gemini 3 Flash | **Avec cache** | **~$0.30** |

*Estimations incluant input, output, résumés et évaluation qualité. Le coût de stockage du cache (~$0.50/h pour 200K tokens) est inclus pour une session de 2h.*

---

> *Orchestr'IA — Cahier des charges v6.0 — Février 2026*
