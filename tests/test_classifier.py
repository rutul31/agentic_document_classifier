"""Tests for the dual LLM classifier."""

import pathlib

from src.classifier import DualLLMClassifier
from src.hitl_feedback import FeedbackRepository
from src.preprocess import DocumentBundle
from src.prompt_tree import PromptNode, PromptTree


def build_prompt_tree() -> PromptTree:
    root = PromptNode(name="root", prompt="root")
    return PromptTree(root=root)


def test_classifier_detects_confidential(tmp_path):
    repository = FeedbackRepository(database_url="sqlite:///:memory:")
    classifier = DualLLMClassifier("gpt-4", "claude", repository)
    bundle = DocumentBundle(text="This memo is confidential and contains salary data.")
    result = classifier.classify(bundle, build_prompt_tree(), pathlib.Path("memo.txt"))

    assert result.label in {"Highly Sensitive", "Confidential"}
    assert result.citations
    assert "salary" in result.citations[0].snippet.lower()
