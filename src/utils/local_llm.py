"""Utilities for running local LLaMA inference pipelines."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import pathlib

import logging

try:  # pragma: no cover - optional dependency
    import requests
except ImportError:  # pragma: no cover - handled gracefully
    requests = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - handled gracefully
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration describing a locally hosted LLaMA model."""

    name: str
    llm_provider: str = "local_llama"
    model_path: Optional[str] = None
    inference_engine: str = "ollama"
    temperature: float = 0.3
    max_tokens: int = 1024
    seed: Optional[int] = None
    base_url: Optional[str] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Represents a cached LLM response."""

    label: str
    score: float
    reasoning: str
    raw_response: str


class PromptCache:
    """SQLite-backed cache for deterministic LLM responses."""

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        self.path = path or pathlib.Path("data/cache/llm_cache.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._connection:  # pragma: no cover - minimal schema creation
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (model, prompt_hash)
                )
                """
            )

    def _hash(self, payload: str) -> str:
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return digest

    def get(self, model: str, payload: str) -> Optional[CacheEntry]:
        """Retrieve a cached entry for a given model and payload."""

        key = self._hash(payload)
        with self._lock:
            cursor = self._connection.execute(
                "SELECT payload FROM cache WHERE model=? AND prompt_hash=?",
                (model, key),
            )
            row = cursor.fetchone()

        if not row:
            return None

        data = json.loads(row[0])
        return CacheEntry(**data)

    def set(self, model: str, payload: str, entry: CacheEntry) -> None:
        """Persist an entry in the cache."""

        key = self._hash(payload)
        with self._lock:
            with self._connection:  # pragma: no cover - sqlite writes are atomic
                self._connection.execute(
                    "REPLACE INTO cache (model, prompt_hash, payload) VALUES (?, ?, ?)",
                    (model, key, json.dumps(asdict(entry))),
                )


class LocalLlamaEngine:
    """Wrapper around supported local inference providers."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device: Optional[str] = None
        self._lock = threading.Lock()

    # Public API -----------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text for a given prompt using the configured backend."""

        engine = (self.config.inference_engine or "ollama").lower()
        if engine == "ollama":
            return self._generate_with_ollama(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
        if engine == "transformers":
            return self._generate_with_transformers(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )

        raise ValueError(f"Unsupported inference engine: {self.config.inference_engine}")

    # Internal helpers -----------------------------------------------------------
    def _generate_with_ollama(
        self,
        prompt: str,
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        seed: Optional[int],
    ) -> str:
        if requests is None:
            raise RuntimeError("requests not installed; Ollama inference unavailable")

        endpoint = (self.config.base_url or "http://localhost:11434").rstrip("/")
        url = f"{endpoint}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.config.model_path or self.config.name,
            "prompt": prompt,
            "stream": False,
        }

        options: Dict[str, Any] = dict(self.config.generation_kwargs)
        if temperature is not None:
            options["temperature"] = temperature
        else:
            options["temperature"] = self.config.temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        else:
            options["num_predict"] = self.config.max_tokens
        if seed is not None:
            options["seed"] = seed
        elif self.config.seed is not None:
            options["seed"] = self.config.seed
        payload["options"] = options

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:  # type: ignore[attr-defined]
            raise RuntimeError(
                f"Unable to reach Ollama at {endpoint}. "
                "Ensure `ollama serve` is running and accessible."
            ) from exc
        except requests.exceptions.Timeout as exc:  # type: ignore[attr-defined]
            raise RuntimeError(
                f"Ollama request timed out for model '{payload['model']}'."
            ) from exc
        except requests.exceptions.RequestException as exc:  # type: ignore[attr-defined]
            raise RuntimeError(
                f"Ollama request failed for model '{payload['model']}': {exc}"
            ) from exc

        data = response.json()
        return data.get("response", "")

    def _lazy_load_transformers(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers not installed; install transformers to use this backend"
            )

        model_path = self.config.model_path or self.config.name
        LOGGER.info("Loading transformers model from %s", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = "cpu"
        dtype = None

        if torch is not None and torch.cuda.is_available():  # pragma: no cover - GPU
            device = "cuda"
            dtype = torch.float16
        elif torch is not None:
            dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu" and torch is not None:
            model.to(device)

        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _generate_with_transformers(
        self,
        prompt: str,
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        seed: Optional[int],
    ) -> str:
        with self._lock:
            if self._model is None or self._tokenizer is None:
                self._lazy_load_transformers()

        assert self._model is not None
        assert self._tokenizer is not None

        if torch is not None:
            if seed is not None:
                torch.manual_seed(seed)
            elif self.config.seed is not None:
                torch.manual_seed(self.config.seed)

        device = self._device or "cpu"
        tokenizer_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
        inputs = self._tokenizer(prompt, **tokenizer_kwargs)

        if torch is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = dict(self.config.generation_kwargs)
        gen_kwargs.setdefault("max_new_tokens", max_tokens or self.config.max_tokens)
        gen_kwargs.setdefault("temperature", temperature or self.config.temperature)
        gen_kwargs.setdefault("do_sample", (temperature or self.config.temperature) > 0)
        gen_kwargs.setdefault("pad_token_id", self._tokenizer.eos_token_id)

        with torch.no_grad() if torch is not None else _nullcontext():
            output = self._model.generate(**inputs, **gen_kwargs)

        sequence = output[0]
        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = sequence[prompt_length:]
        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text


class _nullcontext:
    """Fallback context manager when torch is unavailable."""

    def __enter__(self) -> None:  # pragma: no cover - trivial
        return None

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None


__all__ = ["ModelConfig", "CacheEntry", "PromptCache", "LocalLlamaEngine"]
