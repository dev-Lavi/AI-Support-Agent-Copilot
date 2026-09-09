"""Initial structural integrity tests for the AI Support Agent repository."""

import sys
from pathlib import Path

# Ensure src package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_documentation_files_exist():
    """Verify that all 13 required documentation files exist."""
    required_docs = [
        "01-problem-framing.md",
        "02-system-design.md",
        "03-data-strategy.md",
        "04-intent-taxonomy.md",
        "05-reply-generation.md",
        "06-escalation-policy.md",
        "07-evaluation-strategy.md",
        "08-baselines.md",
        "09-golden-set.md",
        "10-failure-analysis.md",
        "11-decision-log.md",
        "12-reproducibility.md",
        "13-roadmap.md",
    ]
    docs_dir = PROJECT_ROOT / "docs"
    assert docs_dir.exists(), "docs/ directory must exist"
    for doc in required_docs:
        doc_path = docs_dir / doc
        assert doc_path.exists(), f"Missing required documentation: {doc}"
        assert doc_path.stat().st_size > 500, f"Documentation file {doc} appears empty or too brief"


def test_package_structure_exists():
    """Verify that all src modules have __init__.py files."""
    required_modules = [
        "src",
        "src/data",
        "src/intents",
        "src/retrieval",
        "src/generation",
        "src/escalation",
        "src/evaluation",
        "src/pipeline",
    ]
    for mod in required_modules:
        init_file = PROJECT_ROOT / mod / "__init__.py"
        assert init_file.exists(), f"Missing package init: {init_file}"


def test_readme_exists():
    """Verify root README exists with non-trivial content."""
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists(), "Root README.md must exist"
    assert readme.stat().st_size > 1000, "README.md appears too brief"
