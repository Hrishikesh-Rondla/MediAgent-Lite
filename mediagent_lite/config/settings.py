"""Settings loader: merges config.yaml with .env overrides.

Why a separate settings module instead of reading config.yaml everywhere?
Single source of truth. Tests can monkeypatch get_settings() to inject
test config without touching the file system.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env from the repo root (two levels up from this file)
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)  # don't override already-set env vars

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class Settings:
    """Typed wrapper around the config.yaml dictionary.

    Provides attribute access for common paths, but also exposes the raw
    dict for less-common nested lookups.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    # ── LLM ──────────────────────────────────────────────────────────────
    def get_llm_config(self, role: str) -> dict[str, Any]:
        """Return the LLM config for a role, falling back to 'default'."""
        roles = self._config.get("llm_roles", {})
        return roles.get(role) or roles["default"]

    # ── RAG ──────────────────────────────────────────────────────────────
    @property
    def rag(self) -> dict[str, Any]:
        return self._config["rag"]

    @property
    def top_k(self) -> int:
        return self.rag["top_k"]

    @property
    def similarity_threshold(self) -> float:
        return self.rag["similarity_threshold"]

    @property
    def collection_name(self) -> str:
        return self.rag["collection_name"]

    @property
    def qdrant_path(self) -> str:
        return self.rag["qdrant_path"]

    @property
    def embedding_model(self) -> str:
        return self.rag["embedding_model"]

    @property
    def chunk_size(self) -> int:
        return self.rag["chunk_size"]

    @property
    def chunk_overlap(self) -> int:
        return self.rag["chunk_overlap"]

    # ── Confidence ───────────────────────────────────────────────────────
    @property
    def confidence(self) -> dict[str, Any]:
        return self._config["confidence"]

    @property
    def confidence_weights(self) -> tuple[float, float, float]:
        c = self.confidence
        return c["w1"], c["w2"], c["w3"]

    @property
    def tau(self) -> float:
        return self.confidence["tau"]

    # ── Optimizer ────────────────────────────────────────────────────────
    @property
    def max_iter(self) -> int:
        return self._config["optimizer"]["max_iter"]

    # ── PubMed ───────────────────────────────────────────────────────────
    @property
    def pubmed(self) -> dict[str, Any]:
        return self._config["pubmed"]

    # ── Evidence weights ─────────────────────────────────────────────────
    @property
    def evidence_weights(self) -> dict[str, float]:
        return self._config["evidence_weights"]

    # ── Eval ─────────────────────────────────────────────────────────────
    @property
    def eval(self) -> dict[str, Any]:
        return self._config["eval"]

    # ── Ingestion ────────────────────────────────────────────────────────
    @property
    def ingestion(self) -> dict[str, Any]:
        return self._config["ingestion"]

    # ── Raw access ───────────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings. Cached so config.yaml is read once per process."""
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return Settings(config)
