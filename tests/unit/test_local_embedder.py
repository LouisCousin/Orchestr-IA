"""Tests unitaires pour le module local_embedder (Phase 2.5 — FastEmbed)."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.core.local_embedder import LocalEmbedder, DEFAULT_MODEL, DEFAULT_CACHE_DIR


@pytest.fixture(autouse=True)
def reset_singleton():
    """Réinitialise le singleton avant chaque test."""
    LocalEmbedder.reset_instance()
    yield
    LocalEmbedder.reset_instance()


class TestSingleton:
    def test_get_instance_returns_same_object(self):
        inst1 = LocalEmbedder.get_instance()
        inst2 = LocalEmbedder.get_instance()
        assert inst1 is inst2

    def test_reset_instance(self):
        inst1 = LocalEmbedder.get_instance()
        LocalEmbedder.reset_instance()
        inst2 = LocalEmbedder.get_instance()
        assert inst1 is not inst2

    def test_default_model_name(self):
        embedder = LocalEmbedder()
        assert embedder._model_name == DEFAULT_MODEL

    def test_custom_model_name(self):
        embedder = LocalEmbedder(model_name="custom/model")
        assert embedder._model_name == "custom/model"

    def test_default_cache_dir(self):
        embedder = LocalEmbedder()
        assert embedder._cache_dir == DEFAULT_CACHE_DIR

    def test_custom_cache_dir(self):
        embedder = LocalEmbedder(cache_dir="/custom/cache")
        assert embedder._cache_dir == "/custom/cache"


class TestDimension:
    def test_dimension_is_1024(self):
        embedder = LocalEmbedder()
        assert embedder.dimension == 1024


class TestEmbedDocuments:
    @patch("src.core.local_embedder.LocalEmbedder._load_model")
    def test_embed_documents_adds_passage_prefix(self, mock_load):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.zeros(1024), np.zeros(1024)])
        mock_load.return_value = mock_model

        embedder = LocalEmbedder()
        embedder.embed_documents(["texte 1", "texte 2"])

        call_args = mock_model.embed.call_args
        texts = call_args[0][0]
        assert texts[0] == "passage: texte 1"
        assert texts[1] == "passage: texte 2"

    @patch("src.core.local_embedder.LocalEmbedder._load_model")
    def test_embed_documents_returns_list_of_lists(self, mock_load):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([
            np.random.randn(1024) for _ in range(3)
        ])
        mock_load.return_value = mock_model

        embedder = LocalEmbedder()
        result = embedder.embed_documents(["a", "b", "c"])

        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], list)
        assert len(result[0]) == 1024

    @patch("src.core.local_embedder.LocalEmbedder._load_model")
    def test_embed_documents_returns_native_floats(self, mock_load):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.ones(1024)])
        mock_load.return_value = mock_model

        embedder = LocalEmbedder()
        result = embedder.embed_documents(["texte"])

        # Vérifie que les valeurs sont des float Python natifs, pas np.float32
        assert isinstance(result[0][0], float)

    @patch("src.core.local_embedder.LocalEmbedder._load_model")
    def test_embed_documents_custom_batch_size(self, mock_load):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.zeros(1024)])
        mock_load.return_value = mock_model

        embedder = LocalEmbedder()
        embedder.embed_documents(["texte"], batch_size=32)

        call_kwargs = mock_model.embed.call_args[1]
        assert call_kwargs["batch_size"] == 32

    @patch("src.core.local_embedder.LocalEmbedder._load_model")
    def test_embed_documents_default_batch_size(self, mock_load):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.zeros(1024)])
        mock_load.return_value = mock_model

        embedder = LocalEmbedder()
        embedder.embed_documents(["texte"])

        call_kwargs = mock_model.embed.call_args[1]
        assert call_kwargs["batch_size"] == 16


class TestEmbedQuery:
    @patch("src.core.local_embedder.LocalEmbedder._load_model")
    def test_embed_query_adds_query_prefix(self, mock_load):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.zeros(1024)])
        mock_load.return_value = mock_model

        embedder = LocalEmbedder()
        embedder.embed_query("ma requête")

        call_args = mock_model.embed.call_args
        texts = call_args[0][0]
        assert texts[0] == "query: ma requête"

    @patch("src.core.local_embedder.LocalEmbedder._load_model")
    def test_embed_query_returns_list(self, mock_load):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.random.randn(1024)])
        mock_load.return_value = mock_model

        embedder = LocalEmbedder()
        result = embedder.embed_query("test")

        assert isinstance(result, list)
        assert len(result) == 1024

    @patch("src.core.local_embedder.LocalEmbedder._load_model")
    def test_embed_query_returns_native_floats(self, mock_load):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.ones(1024)])
        mock_load.return_value = mock_model

        embedder = LocalEmbedder()
        result = embedder.embed_query("test")

        # Vérifie que les valeurs sont des float Python natifs, pas np.float32
        assert isinstance(result[0], float)


class TestLazyLoading:
    def test_model_not_loaded_on_init(self):
        embedder = LocalEmbedder()
        assert embedder._model is None

    @patch("src.core.local_embedder.LocalEmbedder._load_model")
    def test_model_loaded_on_embed(self, mock_load):
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([np.zeros(1024)])
        mock_load.return_value = mock_model

        embedder = LocalEmbedder()
        embedder.embed_query("test")

        mock_load.assert_called_once()
