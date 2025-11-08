"""FastAPI entry point for the document classification assistant."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .classifier import CLASSIFICATION_LABELS, ClassificationResult, DualLLMClassifier
from .hitl_feedback import Feedback, FeedbackRepository
from .preprocess import DocumentBundle
from .prompt_tree import PromptTree
from .utils.logger import get_logger
from .utils.simple_yaml import load as load_yaml

LOGGER = get_logger(__name__)

CONFIG_PATH = pathlib.Path("config/config.yaml")
PROMPT_LIBRARY_PATH = pathlib.Path("config/prompts_library.yaml")


@dataclass
class Settings:
    """Runtime configuration."""

    primary_model: str
    secondary_model: str
    safety_keywords: List[str]


class ClassificationRequest(BaseModel):
    """Request payload for single document classification."""

    text: str
    document_path: Optional[str] = None


class BatchRequest(BaseModel):
    """Request payload for batch classification."""

    documents: List[ClassificationRequest]


class FeedbackRequest(BaseModel):
    """Request payload for reviewer feedback."""

    classification_id: int
    reviewer: str
    notes: str
    quality_score: float


def load_settings(path: pathlib.Path = CONFIG_PATH) -> Settings:
    """Load configuration settings from disk."""

    payload = load_yaml(str(path))

    models = payload.get("models", {})
    thresholds = payload.get("thresholds", {})
    keywords = payload.get("safety_keywords", ["breach", "malware"])

    LOGGER.debug("Loaded settings: %s", payload)

    return Settings(
        primary_model=models.get("primary", "gpt-4"),
        secondary_model=models.get("secondary", "claude-3-opus"),
        safety_keywords=list(keywords),
    )


def create_classifier(
    settings: Settings, repository: FeedbackRepository
) -> DualLLMClassifier:
    """Factory helper for classifier construction."""

    return DualLLMClassifier(
        primary_model=settings.primary_model,
        secondary_model=settings.secondary_model,
        feedback_repository=repository,
        safety_keywords=settings.safety_keywords,
    )


def _bundle_from_request(request: ClassificationRequest) -> DocumentBundle:
    """Create a document bundle from request payload."""

    metadata = {}
    if request.document_path:
        metadata["source"] = request.document_path
    return DocumentBundle(text=request.text, images=[], metadata=metadata)


def get_prompt_tree(path: pathlib.Path = PROMPT_LIBRARY_PATH) -> PromptTree:
    """Load the default prompt tree."""

    return PromptTree.from_yaml(str(path))


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    app = FastAPI(title="Document Classification Assistant", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    settings = load_settings()
    repository = FeedbackRepository()
    classifier = create_classifier(settings, repository)
    prompt_tree = get_prompt_tree()

    @app.post("/classify")
    async def classify(request: ClassificationRequest) -> Dict[str, object]:
        """Classify a single document."""

        bundle = _bundle_from_request(request)
        result = classifier.classify(
            bundle, prompt_tree, pathlib.Path(request.document_path or "inline")
        )
        reports = DualLLMClassifier.generate_reports(result, pathlib.Path("reports"))
        response = {
            "result": result.to_json(),
            "reports": {key: str(value) for key, value in reports.items()},
        }
        LOGGER.info("Interactive classification complete for %s", request.document_path)
        return response

    @app.post("/batch")
    async def batch(
        request: BatchRequest, background_tasks: BackgroundTasks
    ) -> Dict[str, object]:
        """Classify multiple documents asynchronously."""

        job_id = id(request)
        background_tasks.add_task(
            _process_batch, request, classifier, prompt_tree, job_id
        )
        return {"job_id": job_id, "status": "queued"}

    @app.post("/feedback")
    async def submit_feedback(request: FeedbackRequest) -> Dict[str, object]:
        """Persist reviewer feedback for a classification."""

        repository.add_feedback(
            classification_id=request.classification_id,
            feedback=Feedback(
                reviewer=request.reviewer,
                notes=request.notes,
                quality_score=request.quality_score,
            ),
        )
        return {"status": "recorded"}

    return app


def _process_batch(
    request: BatchRequest,
    classifier: DualLLMClassifier,
    prompt_tree: PromptTree,
    job_id: int,
) -> None:
    """Process a batch request sequentially while logging progress."""

    LOGGER.info("Starting batch job %s", job_id)
    for index, document in enumerate(request.documents, start=1):
        bundle = _bundle_from_request(document)
        classifier.classify(
            bundle,
            prompt_tree,
            pathlib.Path(document.document_path or f"batch-{index}"),
        )
        LOGGER.info(
            "Batch job %s progress: %s/%s", job_id, index, len(request.documents)
        )
    LOGGER.info("Batch job %s complete", job_id)


app = create_app()


__all__ = [
    "CLASSIFICATION_LABELS",
    "ClassificationResult",
    "DualLLMClassifier",
    "create_app",
    "load_settings",
]
