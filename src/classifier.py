"""Dual-LLM document classification orchestration."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:  # pragma: no cover - handled gracefully
    letter = None  # type: ignore[assignment]
    canvas = None  # type: ignore[assignment]

from .citations import Citation, CitationManager
from .hitl_feedback import FeedbackRepository
from .preprocess import DocumentBundle
from .prompt_tree import PromptNode, PromptTree
from .utils.logger import get_logger

LOGGER = get_logger(__name__)

CLASSIFICATION_LABELS = ["Public", "Confidential", "Highly Sensitive", "Unsafe"]


@dataclass
class ClassificationResult:
    """Represents the outcome of a classification."""

    label: str
    score: float
    citations: List[Citation] = field(default_factory=list)
    safety_flags: List[str] = field(default_factory=list)
    verifier_agreement: bool = True
    classification_id: Optional[int] = None

    def to_json(self) -> str:
        """Serialize the classification result to JSON."""

        payload = {
            "label": self.label,
            "score": self.score,
            "citations": [citation.__dict__ for citation in self.citations],
            "safety_flags": self.safety_flags,
            "verifier_agreement": self.verifier_agreement,
            "classification_id": self.classification_id,
        }
        return json.dumps(payload, ensure_ascii=False)


class LLMClient:
    """A lightweight LLM client abstraction for testing."""

    def __init__(self, name: str) -> None:
        self.name = name

    def classify(self, prompt: str, document: DocumentBundle) -> Tuple[str, float]:
        """Return a pseudo classification based on heuristic scoring."""

        text = document.text.lower()
        if "breach" in text or "malware" in text:
            return "Unsafe", 0.99
        if "salary" in text or "ssn" in text:
            return "Highly Sensitive", 0.85
        if "confidential" in text or "nda" in text:
            return "Confidential", 0.75
        return "Public", 0.6


class DualLLMClassifier:
    """Coordinates classification using two LLMs and prompt trees."""

    def __init__(
        self,
        primary_model: str,
        secondary_model: str,
        feedback_repository: FeedbackRepository,
        safety_keywords: Optional[Sequence[str]] = None,
    ) -> None:
        self.primary = LLMClient(primary_model)
        self.secondary = LLMClient(secondary_model)
        self.feedback_repository = feedback_repository
        self.safety_keywords = set(safety_keywords or ["breach", "malware", "leak"])

    def classify(
        self,
        bundle: DocumentBundle,
        prompt_tree: PromptTree,
        document_path: pathlib.Path,
    ) -> ClassificationResult:
        """Classify a document using dual-LLM verification."""

        primary_label, primary_score = self._run_chain(
            self.primary, prompt_tree, bundle
        )
        secondary_label, secondary_score = self._run_chain(
            self.secondary, prompt_tree, bundle
        )
        agreement = primary_label == secondary_label

        label = (
            primary_label
            if agreement
            else self._resolve_disagreement(primary_label, secondary_label)
        )
        score = (primary_score + secondary_score) / 2.0

        citations = self._build_citations(bundle, document_path, label)
        safety_flags = self._run_safety_checks(bundle)

        classification_id = self.feedback_repository.record_classification(
            document_path=str(document_path), classification=label, score=score
        )
        LOGGER.debug("Classification stored under ID %s", classification_id)

        recalibrated_score = self.feedback_repository.recalibrate_confidence(
            label, score
        )

        return ClassificationResult(
            label=label,
            score=recalibrated_score,
            citations=citations.citations,
            safety_flags=safety_flags,
            verifier_agreement=agreement,
            classification_id=classification_id,
        )

    # Internal helpers -----------------------------------------------------------
    def _run_chain(
        self, client: LLMClient, prompt_tree: PromptTree, bundle: DocumentBundle
    ) -> Tuple[str, float]:
        """Simulate traversal of the prompt tree."""

        label, score = "Public", 0.0
        for node in prompt_tree.root.iter_prompts():
            label, score = client.classify(node.prompt, bundle)
        LOGGER.debug("%s classified as %s (score %.2f)", client.name, label, score)
        return label, score

    def _resolve_disagreement(self, primary: str, secondary: str) -> str:
        """Resolve disagreements by selecting the more sensitive label."""

        priority = {label: index for index, label in enumerate(CLASSIFICATION_LABELS)}
        resolved = primary if priority[primary] > priority[secondary] else secondary
        LOGGER.info(
            "Resolved disagreement between %s and %s -> %s",
            primary,
            secondary,
            resolved,
        )
        return resolved

    def _build_citations(
        self, bundle: DocumentBundle, path: pathlib.Path, label: str
    ) -> CitationManager:
        """Create citation evidence based on document text snippets."""

        manager = CitationManager()
        snippet = bundle.text[:280] if bundle.text else ""
        manager.add(
            Citation(
                source=str(path),
                page=1,
                snippet=snippet,
                confidence=0.6 if label == "Public" else 0.8,
                metadata={"label": label},
            )
        )
        return manager

    def _run_safety_checks(self, bundle: DocumentBundle) -> List[str]:
        """Perform basic keyword-based safety checks."""

        text = bundle.text.lower()
        flags = [keyword for keyword in self.safety_keywords if keyword in text]
        LOGGER.debug("Safety flags detected: %s", flags)
        return flags

    # Report generation ----------------------------------------------------------
    @staticmethod
    def generate_reports(
        result: ClassificationResult, output_dir: pathlib.Path
    ) -> Dict[str, pathlib.Path]:
        """Generate JSON and PDF reports for a classification result."""

        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "report.json"
        pdf_path = output_dir / "report.pdf"

        json_path.write_text(result.to_json(), encoding="utf-8")
        DualLLMClassifier._write_pdf_report(result, pdf_path)

        return {"json": json_path, "pdf": pdf_path}

    @staticmethod
    def _write_pdf_report(result: ClassificationResult, pdf_path: pathlib.Path) -> None:
        """Create a lightweight PDF report summarizing the classification."""

        if canvas is None or letter is None:
            pdf_path.write_text(result.to_json(), encoding="utf-8")
            return

        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        text_object = c.beginText(40, 750)
        text_object.textLine(f"Classification: {result.label}")
        text_object.textLine(f"Score: {result.score:.2f}")
        text_object.textLine(f"Agreement: {result.verifier_agreement}")
        text_object.textLine("Safety Flags:")
        for flag in result.safety_flags or ["None"]:
            text_object.textLine(f" - {flag}")
        text_object.textLine("Citations:")
        for citation in result.citations:
            text_object.textLine(f" - {citation.source} (p{citation.page})")
        c.drawText(text_object)
        c.showPage()
        c.save()


__all__ = ["DualLLMClassifier", "ClassificationResult", "LLMClient"]
