"""Calcul d'embeddings locaux multilingues pour le pipeline RAG.

Phase 2.5 : utilise intfloat/multilingual-e5-large via sentence-transformers.
Produit des vecteurs de dimension 1024, normalisés L2.
Le modèle supporte 100+ langues dont FR, EN, ES, DE, PT.

IMPORTANT : Le modèle E5 exige des préfixes différents :
  - Documents : 'passage: '
  - Requêtes  : 'query: '
L'omission de ces préfixes dégrade la qualité de 15-20%.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("orchestria")

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_CACHE_DIR = "./models"


class LocalEmbedder:
    """Calcul d'embeddings en local avec multilingual-e5-large."""

    _instance: Optional["LocalEmbedder"] = None

    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        self._model_name = model_name or DEFAULT_MODEL
        self._cache_dir = cache_dir or os.environ.get("ORCHESTRIA_MODELS_DIR", DEFAULT_CACHE_DIR)
        self._model = None

    def _load_model(self):
        """Charge le modèle de manière paresseuse."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Chargement du modèle d'embeddings : {self._model_name}")
                self._model = SentenceTransformer(
                    self._model_name,
                    cache_folder=self._cache_dir,
                )
                logger.info("Modèle d'embeddings chargé.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers est requis pour les embeddings locaux. "
                    "Installez-le avec : pip install sentence-transformers"
                )
        return self._model

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> "LocalEmbedder":
        """Retourne l'instance singleton (le modèle est lourd à charger)."""
        if cls._instance is None:
            cls._instance = cls(model_name, cache_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Réinitialise le singleton (utile pour les tests)."""
        cls._instance = None

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode une liste de textes de corpus (avec préfixe 'passage:').

        Args:
            texts: Liste de textes à encoder.
            batch_size: Taille des lots pour l'inférence.

        Returns:
            Liste de vecteurs normalisés de dimension 1024.
        """
        model = self._load_model()
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode une requête de recherche (avec préfixe 'query:').

        Args:
            query: Texte de la requête.

        Returns:
            Vecteur normalisé de dimension 1024.
        """
        model = self._load_model()
        embedding = model.encode(f"query: {query}", normalize_embeddings=True)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Dimension des vecteurs produits."""
        return 1024
