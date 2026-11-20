"""Ask one question from the terminal and print the answer + metrics.
Usage:  python scripts/ask.py "Why is TP53 called the guardian of the genome?" [--trace]"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bioscout.service import BioScout  # noqa: E402


async def main():
    args = [a for a in sys.argv[1:] if a != "--trace"]
    result = await BioScout().ask(" ".join(args) or "What does p53 do after DNA damage?")
    print("\n" + result["answer"] + "\n")
    print(json.dumps(result["metrics"], indent=2))
    if "--trace" in sys.argv:
        for step in result["trace"]:
            print(step)


asyncio.run(main())
