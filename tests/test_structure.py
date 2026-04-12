"""Tests de structure du projet (inspiré de loan-default-mlops)."""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_dockerfile_exists():
    """Vérifie que le Dockerfile existe."""
    assert (ROOT / "Dockerfile").exists()


def test_dockerfile_contenu():
    """Vérifie que le Dockerfile contient les éléments essentiels."""
    content = (ROOT / "Dockerfile").read_text()
    assert "FROM python" in content
    assert "requirements.txt" in content
    assert "--no-cache-dir" in content
    assert "chainlit" in content


def test_requirements_exists():
    """Vérifie que requirements.txt existe."""
    assert (ROOT / "requirements.txt").exists()


def test_requirements_contenu():
    """Vérifie les dépendances essentielles."""
    content = (ROOT / "requirements.txt").read_text()
    assert "langchain-anthropic" in content
    assert "langgraph" in content
    assert "chainlit" in content
    assert "faiss-cpu" in content
    assert "rank-bm25" in content
    assert "langchain-groq" not in content  # on a migré vers Anthropic


def test_app_py_exists():
    """Vérifie que app.py existe."""
    assert (ROOT / "app.py").exists()


def test_claude_md_exists():
    """Vérifie que CLAUDE.md existe."""
    assert (ROOT / "CLAUDE.md").exists()


def test_env_example_exists():
    """Vérifie que .env.example existe."""
    assert (ROOT / ".env.example").exists()


def test_env_example_anthropic():
    """Vérifie que .env.example utilise Anthropic, pas Groq."""
    content = (ROOT / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY" in content
    assert "GROQ_API_KEY" not in content


def test_gitignore_env():
    """Vérifie que .env est dans le .gitignore."""
    content = (ROOT / ".gitignore").read_text()
    assert ".env" in content


def test_src_structure():
    """Vérifie la structure src/."""
    assert (ROOT / "src" / "__init__.py").exists()
    assert (ROOT / "src" / "config.py").exists()
    assert (ROOT / "src" / "rag" / "loader.py").exists()
    assert (ROOT / "src" / "rag" / "embeddings.py").exists()
    assert (ROOT / "src" / "rag" / "retriever.py").exists()
    assert (ROOT / "src" / "rag" / "hybrid_retriever.py").exists()
    assert (ROOT / "src" / "agents" / "tools.py").exists()
    assert (ROOT / "src" / "agents" / "agent.py").exists()
    assert (ROOT / "src" / "memory" / "memory.py").exists()


def test_github_workflows():
    """Vérifie que les workflows CI/CD existent."""
    workflows = ROOT / ".github" / "workflows"
    assert workflows.exists()
    files = [f.name for f in workflows.iterdir()]
    assert "github-docker-cicd.yaml" in files
    assert "azure.yml" in files


def test_tests_directory():
    """Vérifie que le dossier tests/ contient des tests."""
    tests_dir = ROOT / "tests"
    assert tests_dir.exists()
    test_files = [f.name for f in tests_dir.glob("test_*.py")]
    assert len(test_files) >= 3
