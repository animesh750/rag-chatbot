import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "update_readme_results", Path(__file__).resolve().parent.parent / "scripts" / "update_readme_results.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_inject_replaces_only_the_marked_block():
    readme = f"before\n{tool.START}\nold table\n{tool.END}\nafter"
    out = tool.inject(readme, "| new |")
    assert "old table" not in out and "| new |" in out
    assert out.startswith("before\n") and out.endswith("\nafter")


def test_inject_requires_markers():
    with pytest.raises(ValueError):
        tool.inject("no markers here", "x")


def test_inject_handles_backslashes_in_table():
    assert r"a\b" in tool.inject(f"{tool.START}{tool.END}", r"a\b")
