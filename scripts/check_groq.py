"""Quick connectivity check for your Groq API key.  Run: python scripts/check_groq.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config  # noqa: E402  (loads .env)
from rag.llm import GroqLLM, LLMError  # noqa: E402

print(f"Sending test message to {config.LLM_MODEL}...")
try:
    text, tokens = GroqLLM().chat(
        [
            {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely."},
            {"role": "user", "content": "What is generative AI in 2 sentences?"},
        ]
    )
except LLMError as exc:
    sys.exit(f"✗ {exc}")
print("-" * 40)
print(text)
print("-" * 40)
print(f"✓ Groq connection working ({tokens} tokens)")
