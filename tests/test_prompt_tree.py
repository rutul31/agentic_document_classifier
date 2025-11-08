"""Tests for prompt tree loading."""

from src.prompt_tree import PromptTree


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
