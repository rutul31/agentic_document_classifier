"""Benchmark utilities for evaluating the document classifier pipeline."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Sequence

from .classifier import DualLLMClassifier
from .preprocess import DocumentBundle
from .prompt_tree import PromptTree


@dataclass
class EvaluationCase:
    """Definition of a benchmark test case."""

    case_id: str
    expected_label: str
    bundle: DocumentBundle
    document_path: Path
    critical: bool = False
    expected_flags: Sequence[str] = ()


def run_benchmark(
    classifier: DualLLMClassifier,
    prompt_tree: PromptTree,
    cases: Iterable[EvaluationCase],
    output_path: Path,
) -> dict:
    """Execute the benchmark suite and persist an evaluation report."""

    latencies: List[float] = []
    case_reports: List[dict] = []
    correct = 0
    flagged_cases = 0
    critical_total = 0
    critical_detected = 0
    expected_flag_cases = 0
    expected_flag_matches = 0

    for case in cases:
        start = perf_counter()
        result = classifier.classify(case.bundle, prompt_tree, case.document_path)
        latency_ms = (perf_counter() - start) * 1000
        latencies.append(latency_ms)

        is_correct = result.label == case.expected_label
        correct += int(is_correct)

        has_flags = bool(result.safety_flags)
        flagged_cases += int(has_flags)

        if case.critical:
            critical_total += 1
            if has_flags:
                critical_detected += 1

        expected_flag_set = set(case.expected_flags)
        if expected_flag_set:
            expected_flag_cases += 1
            if expected_flag_set.issubset(set(result.safety_flags)):
                expected_flag_matches += 1

        case_reports.append(
            {
                "id": case.case_id,
                "expected_label": case.expected_label,
                "predicted_label": result.label,
                "score": result.score,
                "latency_ms": latency_ms,
                "verifier_agreement": result.verifier_agreement,
                "safety_flags": result.safety_flags,
            }
        )

    total_cases = len(case_reports) or 1
    accuracy = correct / total_cases
    average_latency = statistics.mean(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0

    flagged_rate = flagged_cases / total_cases
    critical_detection_rate = (
        critical_detected / critical_total if critical_total else 1.0
    )
    expected_flag_match_rate = (
        expected_flag_matches / expected_flag_cases if expected_flag_cases else 1.0
    )

    report = {
        "total_cases": total_cases,
        "metrics": {
            "accuracy": accuracy,
            "latency_ms": {
                "average": average_latency,
                "max": max_latency,
                "min": min_latency,
            },
            "safety": {
                "flagged_cases": flagged_cases,
                "flagged_rate": flagged_rate,
                "critical_cases": critical_total,
                "critical_detection_rate": critical_detection_rate,
                "expected_flag_match_rate": expected_flag_match_rate,
            },
        },
        "cases": case_reports,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["EvaluationCase", "run_benchmark"]
