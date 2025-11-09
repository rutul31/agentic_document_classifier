"""Tests for prompt tree loading and dynamic prompt generation."""

from src.prompt_tree import PROMPT_TRACE_PATH, PromptTree, build_prompt_tree


PROMPT_LOG = PROMPT_TRACE_PATH


def test_prompt_tree_from_yaml(tmp_path):
    yaml_path = tmp_path / "tree.yaml"
    yaml_path.write_text(
        """
name: root
prompt: root prompt
children:
  - name: child
    prompt: child prompt
"""
    )

    tree = PromptTree.from_yaml(str(yaml_path))
    assert tree.root.name == "root"
    assert tree.root.children[0].name == "child"


def test_build_prompt_tree_text_includes_tc2(tmp_path):
    if PROMPT_LOG.exists():
        PROMPT_LOG.unlink()

    prompt = build_prompt_tree({
        "title": "Payroll Memo",
        "has_text": True,
        "text_length": 120,
    })

    assert "TC2" in prompt
    assert "Payroll Memo" in prompt
    assert PROMPT_LOG.exists()
    log_text = PROMPT_LOG.read_text(encoding="utf-8")
    assert "tc2_pii_review" in log_text


def test_build_prompt_tree_image_includes_tc4(tmp_path):
    if PROMPT_LOG.exists():
        PROMPT_LOG.unlink()

    prompt = build_prompt_tree({
        "content_type": "image",
        "image_count": 3,
    })

    assert "TC4" in prompt
    log_text = PROMPT_LOG.read_text(encoding="utf-8")
    assert "tc4_image_classification" in log_text
