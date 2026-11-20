import os
import sys
from pathlib import Path

# Offline: hashed embeddings instead of the Gemini API, and a temp store directory.
os.environ["BIOSCOUT_FAKE_EMBED"] = "1"
os.environ.setdefault("BIOSCOUT_STORE_DIR", str(Path(__file__).parent / ".teststore"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
