"""Interface commune pour les fournisseurs d'IA.

Phase 2.5 : ajout du support batch (soumission, polling, récupération).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class AIResponse:
    """Réponse d'un fournisseur d'IA."""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    raw_response: Optional[dict] = field(default=None, repr=False)


@dataclass
class BatchRequest:
    """Requête individuelle dans un batch."""
    custom_id: str       # Identifiant unique (ex: "section_1.1")
    prompt: str          # Prompt utilisateur
    system_prompt: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


class BatchStatusEnum(str, Enum):
    """Statuts possibles d'un batch."""
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class BatchStatus:
    """Statut d'un batch en cours."""
    batch_id: str
    status: BatchStatusEnum
    total: int = 0
    completed: int = 0
    failed: int = 0
    error_message: str = ""


class BatchError(Exception):
    """Erreur lors du traitement d'un batch."""
    pass


class BaseProvider(ABC):
    """Interface commune pour tous les fournisseurs d'IA.

    Phase 2.5 : inclut les méthodes de batch (optionnelles).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du fournisseur."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AIResponse:
        """Génère du contenu à partir d'un prompt."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Vérifie si le fournisseur est configuré et disponible."""
        ...

    @abstractmethod
    def get_default_model(self) -> str:
        """Retourne le modèle par défaut du fournisseur."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """Liste les modèles disponibles pour ce fournisseur."""
        ...

    # ── Phase 2.5 : Méthodes Batch (implémentation par défaut) ──

    def supports_batch(self) -> bool:
        """Indique si le fournisseur supporte le mode batch."""
        return False

    def submit_batch(self, requests: list[BatchRequest]) -> str:
        """Soumet un batch et retourne un batch_id."""
        raise NotImplementedError(f"{self.name} ne supporte pas le mode batch")

    def poll_batch(self, batch_id: str) -> BatchStatus:
        """Vérifie le statut d'un batch."""
        raise NotImplementedError(f"{self.name} ne supporte pas le mode batch")

    def retrieve_batch_results(self, batch_id: str) -> dict[str, str]:
        """Récupère les résultats d'un batch terminé.

        Returns:
            Dict {custom_id: contenu_généré}.
        """
        raise NotImplementedError(f"{self.name} ne supporte pas le mode batch")
