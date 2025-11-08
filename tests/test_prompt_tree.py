"""Prompt tree construction and traversal tests."""

from src.prompt_tree import PromptTree


def test_prompt_tree_iteration(sample_prompt_tree):
    """Iterating over the prompt tree should cover every node in order."""

    names = [node.name for node in sample_prompt_tree.root.iter_prompts()]
    assert names == ["root", "policy", "sensitivity", "safety"]


def test_prompt_tree_round_trip(sample_prompt_tree):
    """Trees serialized to dictionaries can be restored losslessly."""

    as_dict = sample_prompt_tree.to_dict()
    restored = PromptTree.from_dict(as_dict)

    assert restored.root.name == "root"
    assert restored.root.find("safety").prompt == "Identify safety issues."
    assert len(list(restored.root.iter_prompts())) == 4
