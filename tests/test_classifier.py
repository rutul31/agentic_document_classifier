"""Tests for the dual LLM classifier."""

import pathlib

from src.classifier import ClassificationEngine, DualLLMClassifier
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


def test_classification_engine_single_model_confidential():
    engine = ClassificationEngine(primary_model="gpt-4")
    bundle = DocumentBundle(text="TC3 internal memo: for internal use only.")
    result = engine.classify(bundle, prompt="Classify document", document_path=pathlib.Path("tc3.txt"))

    assert result.category == "Confidential"
    payload = result.to_dict()
    assert payload["category"] == "Confidential"
    assert payload["citations"]
    assert payload["confidence"] >= 0.7


def test_classification_engine_dual_model_agreement():
    engine = ClassificationEngine(primary_model="gpt-4", secondary_model="claude")
    bundle = DocumentBundle(text="Employee salary records shared under NDA.")
    result = engine.classify(bundle, prompt="Classify document", document_path=pathlib.Path("salary.txt"))

    assert result.category in {"Highly Sensitive", "Confidential"}
    assert set(result.model_outputs.keys()) == {"gpt-4", "claude"}
    assert isinstance(result.agreement, bool)
