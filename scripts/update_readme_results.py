"""Paste evaluation/results.md into README.md between the EVAL-RESULTS markers.

python -m evaluation.run_eval          # produces evaluation/results.md
python scripts/update_readme_results.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

START, END = "<!-- EVAL-RESULTS-START -->", "<!-- EVAL-RESULTS-END -->"


def inject(readme: str, table: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(readme):
        raise ValueError(f"README is missing the {START} / {END} markers")
    return pattern.sub(lambda _: f"{START}\n{table.strip()}\n{END}", readme)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    results, readme = root / "evaluation" / "results.md", root / "README.md"
    if not results.exists():
        print("evaluation/results.md not found. Run: python -m evaluation.run_eval", file=sys.stderr)
        return 1
    readme.write_text(
        inject(readme.read_text(encoding="utf-8"), results.read_text(encoding="utf-8")), encoding="utf-8"
    )
    print("README.md updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
