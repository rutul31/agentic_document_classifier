"""Benchmark suite tests covering accuracy, latency, and safety metrics."""

from pathlib import Path

from src.classifier import DualLLMClassifier
from src.evaluation import EvaluationCase, run_benchmark
from src.hitl_feedback import FeedbackRepository


def test_run_benchmark_generates_report(tmp_path, evaluation_cases, sample_prompt_tree):
    """Running the benchmark should create a JSON report with metrics."""

    repository = FeedbackRepository(database_url="sqlite:///:memory:")
    classifier = DualLLMClassifier("gpt-4", "claude", repository)

    cases = [
        EvaluationCase(
            case_id=data["id"],
            expected_label=data["expected_label"],
            bundle=data["bundle"],
            document_path=Path(data["path"]),
            critical=data["critical"],
            expected_flags=data["expected_flags"],
        )
        for data in evaluation_cases
    ]

    output_path = tmp_path / "evaluation_report.json"
    report = run_benchmark(classifier, sample_prompt_tree, cases, output_path)

    assert output_path.exists()
    assert report["metrics"]["accuracy"] == 1.0
    assert report["metrics"]["safety"]["critical_detection_rate"] == 1.0
    assert report["metrics"]["safety"]["expected_flag_match_rate"] == 1.0
    assert report["metrics"]["latency_ms"]["average"] >= 0.0
    assert len(report["cases"]) == len(cases)
    assert all(entry["predicted_label"] for entry in report["cases"])
