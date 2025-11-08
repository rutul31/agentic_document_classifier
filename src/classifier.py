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

KEYWORD_RULES: List[Tuple[str, float, Tuple[str, ...]]] = [
    ("Unsafe", 0.99, ("breach", "malware", "ransomware", "exploit")),
    (
        "Highly Sensitive",
        0.9,
        ("salary", "ssn", "social security", "passport", "medical"),
    ),
    (
        "Confidential",
        0.78,
        (
            "confidential",
            "nda",
            "internal memo",
            "internal use only",
            "for internal use",
            "proprietary",
        ),
    ),
]


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


@dataclass
class _ModelInference:
    """Internal representation of a single model inference."""

    model: str
    category: str
    confidence: float
    reasoning: str


@dataclass
class ClassificationEngineResult:
    """Structured payload produced by :class:`ClassificationEngine`."""

    category: str
    confidence: float
    citations: List[Dict[str, object]]
    model_outputs: Dict[str, Dict[str, object]]
    agreement: bool

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable dictionary."""

        return {
            "category": self.category,
            "confidence": self.confidence,
            "citations": self.citations,
            "model_outputs": self.model_outputs,
            "agreement": self.agreement,
        }

    def to_json(self) -> str:
        """Serialize the engine result to JSON."""

        return json.dumps(self.to_dict(), ensure_ascii=False)


class LLMClient:
    """A lightweight LLM client abstraction for testing."""

    def __init__(self, name: str) -> None:
        self.name = name

    def classify(self, prompt: str, document: DocumentBundle) -> Tuple[str, float]:
        """Return a pseudo classification based on heuristic scoring."""

        text = document.text.lower()
        for label, score, keywords in KEYWORD_RULES:
            if any(keyword in text for keyword in keywords):
                return label, score
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


class ClassificationEngine:
    """Execute document classification with optional dual-LLM validation."""

    def __init__(
        self,
        primary_model: str,
        *,
        secondary_model: Optional[str] = None,
    ) -> None:
        self.primary = LLMClient(primary_model)
        self.secondary = LLMClient(secondary_model) if secondary_model else None

    # Public API -----------------------------------------------------------------
    def classify(
        self,
        bundle: DocumentBundle,
        prompt: str,
        *,
        document_path: Optional[pathlib.Path] = None,
    ) -> ClassificationEngineResult:
        """Classify a document bundle using one or two models."""

        LOGGER.info("Starting classification with primary model %s", self.primary.name)
        primary_output = self._invoke_model(self.primary, prompt, bundle)
        model_outputs: Dict[str, Dict[str, object]] = {
            primary_output.model: {
                "category": primary_output.category,
                "confidence": primary_output.confidence,
                "reasoning": primary_output.reasoning,
            }
        }

        if self.secondary is not None:
            LOGGER.info("Running secondary verification with %s", self.secondary.name)
            secondary_output = self._invoke_model(self.secondary, prompt, bundle)
            model_outputs[secondary_output.model] = {
                "category": secondary_output.category,
                "confidence": secondary_output.confidence,
                "reasoning": secondary_output.reasoning,
            }
            category, confidence, agreement = self._merge_outputs(
                primary_output, secondary_output
            )
        else:
            category, confidence, agreement = (
                primary_output.category,
                primary_output.confidence,
                True,
            )

        citations = self._build_citations(bundle, document_path, category)
        LOGGER.info("Final classification -> %s (%.2f)", category, confidence)

        return ClassificationEngineResult(
            category=category,
            confidence=confidence,
            citations=citations.to_payload(),
            model_outputs=model_outputs,
            agreement=agreement,
        )

    # Internal helpers -----------------------------------------------------------
    def _invoke_model(
        self, client: LLMClient, prompt: str, bundle: DocumentBundle
    ) -> _ModelInference:
        label, score = client.classify(prompt, bundle)
        reasoning = self._derive_reasoning(bundle.text.lower(), label)
        LOGGER.info(
            "Model %s classified document as %s (%.2f)", client.name, label, score
        )
        LOGGER.debug("Reasoning from %s: %s", client.name, reasoning)
        return _ModelInference(
            model=client.name, category=label, confidence=score, reasoning=reasoning
        )

    def _derive_reasoning(self, text: str, label: str) -> str:
        matched_keywords: List[str] = []
        for rule_label, _, keywords in KEYWORD_RULES:
            if rule_label != label:
                continue
            matched_keywords.extend([kw for kw in keywords if kw in text])

        if matched_keywords:
            return (
                "Detected keywords "
                + ", ".join(sorted(set(matched_keywords)))
                + f" leading to {label} classification."
            )

        if label == "Public":
            return "No sensitive keywords detected; defaulting to Public risk level."
        return (
            "Model heuristics inferred a "
            f"{label} classification without explicit keyword matches."
        )

    def _merge_outputs(
        self, primary: _ModelInference, secondary: _ModelInference
    ) -> Tuple[str, float, bool]:
        if primary.category == secondary.category:
            confidence = (primary.confidence + secondary.confidence) / 2.0
            LOGGER.info(
                "Models agree on %s with averaged confidence %.2f",
                primary.category,
                confidence,
            )
            return primary.category, confidence, True

        resolved = self._resolve_disagreement(primary.category, secondary.category)
        confidence = min(primary.confidence, secondary.confidence)
        LOGGER.warning(
            "Disagreement detected (%s vs %s); resolved to %s with confidence %.2f",
            primary.category,
            secondary.category,
            resolved,
            confidence,
        )
        return resolved, confidence, False

    def _resolve_disagreement(self, primary: str, secondary: str) -> str:
        priority = {label: index for index, label in enumerate(CLASSIFICATION_LABELS)}
        return primary if priority[primary] > priority[secondary] else secondary

    def _build_citations(
        self,
        bundle: DocumentBundle,
        path: Optional[pathlib.Path],
        label: str,
    ) -> CitationManager:
        manager = CitationManager()
        snippet = (bundle.text or "")[:280]
        manager.add(
            Citation(
                source=str(path) if path else "memory",
                page=1,
                snippet=snippet,
                confidence=0.6 if label == "Public" else 0.85,
                metadata={"label": label},
            )
        )
        return manager


__all__ = [
    "DualLLMClassifier",
    "ClassificationResult",
    "LLMClient",
    "ClassificationEngine",
    "ClassificationEngineResult",
]
