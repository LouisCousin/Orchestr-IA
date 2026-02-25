"""Utilitaires de manipulation de chaînes."""

import re


def clean_json_string(text: str) -> str:
    """Nettoie les balises Markdown d'une chaîne JSON.

    Retire les blocs ```json ... ``` ou ``` ... ``` qui entourent
    souvent les réponses JSON des LLM.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()
