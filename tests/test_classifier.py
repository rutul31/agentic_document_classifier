"""Tests for the dual LLM classifier and HITL feedback flow."""

from typing import Dict

import pytest

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

    case = tc_documents[case_id]
    result = classifier.classify(case["bundle"], sample_prompt_tree, case["path"])

    assert result.label == case["config"].expected_label
    assert isinstance(result.score, float)
    assert result.citations and result.citations[0].source == str(case["path"])

    if case["config"].critical:
        assert set(result.safety_flags) >= {"malware", "breach"}
    else:
        assert result.safety_flags == []


def test_feedback_repository_records_and_lists_feedback() -> None:
    """HITL feedback repository should persist reviewer updates."""

    repository = FeedbackRepository(database_url="sqlite:///:memory:")
    classification_id = repository.record_classification(
        document_path="/tmp/tc1.txt", classification="Public", score=0.5
    )

    feedback = Feedback(reviewer="analyst", notes="Looks good", quality_score=0.95)
    repository.add_feedback(classification_id, feedback)

    stored = repository.list_feedback(classification_id)
    assert len(stored) == 1
    assert stored[0].reviewer == "analyst"
    assert pytest.approx(stored[0].quality_score, rel=1e-3) == 0.95
