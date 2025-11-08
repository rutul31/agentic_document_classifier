"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.preprocess import DocumentPreprocessor
from src.prompt_tree import PromptNode, PromptTree


@dataclass(frozen=True)
class TestCaseConfig:
    """Configuration for a synthetic evaluation document."""

    case_id: str
    text: str
    expected_label: str
    expected_flags: Iterable[str]
    critical: bool


@pytest.fixture(scope="session")
def tc_configs() -> Dict[str, TestCaseConfig]:
    """Return the synthetic TC1–TC5 configurations used in tests."""

    return {
        "TC1": TestCaseConfig(
            case_id="TC1",
            text="This policy is public and contains general guidelines.",
            expected_label="Public",
            expected_flags=(),
            critical=False,
        ),
        "TC2": TestCaseConfig(
            case_id="TC2",
            text="The document is confidential and references an NDA.",
            expected_label="Confidential",
            expected_flags=(),
            critical=False,
        ),
        "TC3": TestCaseConfig(
            case_id="TC3",
            text="This report includes salary data and employee SSN numbers.",
            expected_label="Highly Sensitive",
            expected_flags=(),
            critical=False,
        ),
        "TC4": TestCaseConfig(
            case_id="TC4",
            text="Incident response log detailing a suspected malware breach.",
            expected_label="Unsafe",
            expected_flags=("malware", "breach"),
            critical=True,
        ),
        "TC5": TestCaseConfig(
            case_id="TC5",
            text="Marketing flyer with no sensitive information.",
            expected_label="Public",
            expected_flags=(),
            critical=False,
        ),
    }


@pytest.fixture(scope="session")
def tc_documents(tc_configs: Dict[str, TestCaseConfig], tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Dict]:
    """Materialize synthetic documents on disk and return metadata."""

    base_dir = tmp_path_factory.mktemp("tc_docs")
    preprocessor = DocumentPreprocessor(enable_ocr=False)
    documents: Dict[str, Dict] = {}

    for case in tc_configs.values():
        doc_path = base_dir / f"{case.case_id}.txt"
        doc_path.write_text(case.text, encoding="utf-8")
        bundle = preprocessor.process_document(doc_path)
        documents[case.case_id] = {
            "config": case,
            "path": doc_path,
            "bundle": bundle,
        }

    return documents


@pytest.fixture(scope="session")
def sample_prompt_tree() -> PromptTree:
    """Construct a prompt tree with multiple evaluation steps."""

    safety_node = PromptNode(name="safety", prompt="Identify safety issues.")
    sensitivity_node = PromptNode(
        name="sensitivity", prompt="Determine sensitivity level.", children=[safety_node]
    )
    policy_node = PromptNode(name="policy", prompt="Check for policy terms.")
    root = PromptNode(
        name="root",
        prompt="Classify the document for access level.",
        children=[policy_node, sensitivity_node],
    )
    return PromptTree(root=root)


@pytest.fixture
    
def evaluation_cases(tc_documents: Dict[str, Dict]) -> List[Dict]:
    """Return evaluation case dictionaries for benchmarking tests."""

    cases: List[Dict] = []
    for data in tc_documents.values():
        case_config = data["config"]
        cases.append(
            {
                "id": case_config.case_id,
                "expected_label": case_config.expected_label,
                "expected_flags": tuple(case_config.expected_flags),
                "critical": case_config.critical,
                "bundle": data["bundle"],
                "path": data["path"],
            }
        )
    return cases
