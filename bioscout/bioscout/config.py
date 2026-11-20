"""All settings in one place. Values come from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("BIOSCOUT_LLM_MODEL", "gemini-2.5-flash")
    embed_model: str = os.getenv("BIOSCOUT_EMBED_MODEL", "gemini-embedding-001")
    embed_dim: int = 768

    data_dir: Path = ROOT / "data"
    store_dir: Path = field(default_factory=lambda: Path(os.getenv("BIOSCOUT_STORE_DIR", ROOT / ".store")))

    # Chunking: split long documents into overlapping windows of words.
    chunk_words: int = 180
    chunk_overlap: int = 40
    top_k: int = 4

    # USD per 1M tokens. Used to compute cost-per-request. Update from the pricing page.
    price_in_per_m: float = float(os.getenv("BIOSCOUT_PRICE_IN", "0.30"))
    price_out_per_m: float = float(os.getenv("BIOSCOUT_PRICE_OUT", "2.50"))

    @property
    def sqlite_path(self) -> Path:
        return self.store_dir / "proteins.db"

    @property
    def chroma_path(self) -> Path:
        return self.store_dir / "chroma"

    @property
    def traces_path(self) -> Path:
        return self.store_dir / "traces.jsonl"


settings = Settings()
