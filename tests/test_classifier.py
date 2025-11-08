"""Tests for the dual LLM classifier and HITL feedback flow."""

from typing import Dict

import pytest
from src.classifier import ClassificationEngine, DualLLMClassifier
from src.hitl_feedback import FeedbackRepository
from src.preprocess import DocumentBundle
from src.prompt_tree import PromptNode, PromptTree

from src.classifier import DualLLMClassifier
from src.hitl_feedback import Feedback, FeedbackRepository


@pytest.mark.parametrize(
    "case_id",
    ["TC1", "TC2", "TC3", "TC4", "TC5"],
)
def test_classifier_assigns_expected_labels(case_id: str, tc_documents: Dict[str, Dict], sample_prompt_tree) -> None:
    """Classification results should match the heuristic expectations."""

    repository = FeedbackRepository(database_url="sqlite:///:memory:")
    classifier = DualLLMClassifier("gpt-4", "claude", repository)

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
