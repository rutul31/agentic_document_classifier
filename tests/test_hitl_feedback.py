"""Tests covering the human-in-the-loop feedback utilities."""

from src.hitl_feedback import (
    AdaptivePromptRefiner,
    Feedback,
    FeedbackRepository,
)
from src.prompt_tree import PromptNode, PromptTree


def build_tree() -> PromptTree:
    return PromptTree(PromptNode(name="root", prompt="Classify"))


def test_feedback_repository_tracks_misclassifications(tmp_path):
    repository = FeedbackRepository(database_url="sqlite:///:memory:")
    classification_id = repository.record_classification(
        "doc.txt", "Public", score=0.9
    )

    repository.add_feedback(
        classification_id,
        Feedback(
            reviewer="alice",
            decision="override",
            comments="Contains payroll data",
            quality_score=0.8,
            suggested_label="Confidential",
        ),
    )

    misclassified = repository.list_misclassifications()
    assert misclassified

    history = repository.get_feedback_history()
    assert history
    assert history[0]["decision"] == "override"

    export_path = tmp_path / "feedback_history.json"
    repository.export_feedback_history(export_path)
    assert export_path.exists()

    recalibrated = repository.recalibrate_confidence("Public", 0.9)
    assert recalibrated < 0.9


def test_adaptive_prompt_refiner_updates_thresholds(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
thresholds:
  public: 0.5
  confidential: 0.7
  highly_sensitive: 0.85
  unsafe: 0.95
""".strip()
        + "\n",
        encoding="utf-8",
    )

    repository = FeedbackRepository(database_url="sqlite:///:memory:")
    classification_id = repository.record_classification(
        "doc.txt", "Public", score=0.9
    )
    repository.add_feedback(
        classification_id,
        Feedback(
            reviewer="bob",
            decision="override",
            comments="Should be confidential",
            quality_score=0.9,
            suggested_label="Confidential",
        ),
    )

    prompt_tree = build_tree()
    refiner = AdaptivePromptRefiner(repository, prompt_tree, config_path)
    updated = refiner.refine()

    assert updated["public"] != 0.5
    assert "# HITL Adjustments" in prompt_tree.root.prompt
