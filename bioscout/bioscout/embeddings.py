"""Turns text into vectors.

GeminiEmbedder calls the Gemini embedding model. FakeEmbedder is a tiny offline stand-in used by the
tests (and when BIOSCOUT_FAKE_EMBED=1), so the pipeline can be checked without an API key.
"""
import hashlib
import math
import os
import re
from typing import Protocol

from .config import settings


class Embedder(Protocol):
    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]: ...


class GeminiEmbedder:
    BATCH = 50  # the API accepts a limited number of texts per call

    def __init__(self, model: str = settings.embed_model, dim: int = settings.embed_dim):
        self.model, self.dim = model, dim
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai  # reads GOOGLE_API_KEY or Vertex AI settings from the environment
            self._client = genai.Client()
        return self._client

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        from google.genai import types

        vectors = []
        for i in range(0, len(texts), self.BATCH):
            resp = self.client.models.embed_content(
                model=self.model,
                contents=texts[i : i + self.BATCH],
                # Documents and queries are embedded slightly differently for better retrieval.
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
            )
            vectors.extend(e.values for e in resp.embeddings)
        return vectors


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-size vector. Good enough to test that retrieval is wired up."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def get_embedder() -> Embedder:
    return FakeEmbedder() if os.getenv("BIOSCOUT_FAKE_EMBED") == "1" else GeminiEmbedder()
