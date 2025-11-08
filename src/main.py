"""FastAPI application exposing the document classification backend."""

from __future__ import annotations

import asyncio
import pathlib
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .classifier import CLASSIFICATION_LABELS, ClassificationResult, DualLLMClassifier
from .hitl_feedback import AdaptivePromptRefiner, Feedback, FeedbackRepository
from .preprocess import DocumentBundle, DocumentPreprocessor
from .prompt_tree import PromptTree
from .utils.local_llm import ModelConfig
from .utils.logger import get_logger
from .utils.simple_yaml import load as load_yaml

LOGGER = get_logger(__name__)

CONFIG_PATH = pathlib.Path("config/config.yaml")
PROMPT_LIBRARY_PATH = pathlib.Path("config/prompts_library.yaml")
UPLOAD_DIR = pathlib.Path("data/uploads")
REPORTS_DIR = pathlib.Path("reports")


# ---------------------------------------------------------------------------
# Configuration helpers


@dataclass
class Settings:
    """Runtime configuration pulled from disk."""

    primary_model_id: str
    secondary_model_id: Optional[str]
    safety_keywords: List[str]
    model_configs: Dict[str, ModelConfig]


def load_settings(path: pathlib.Path = CONFIG_PATH) -> Settings:
    """Load application settings from a YAML file."""

    payload = load_yaml(str(path))
    models = payload.get("models", {})
    keywords = payload.get("safety_keywords", ["breach", "malware"])
    registry: Dict[str, ModelConfig] = {}

    def _register(alias: str, spec) -> Optional[str]:
        if isinstance(spec, dict):
            name = str(spec.get("name") or spec.get("model") or alias)
            config = ModelConfig(
                name=name,
                llm_provider=str(spec.get("llm_provider", "local_llama")),
                model_path=spec.get("model_path") or name,
                inference_engine=str(spec.get("inference_engine", "ollama")),
                temperature=float(spec.get("temperature", 0.3)),
                max_tokens=int(spec.get("max_tokens", 1024)),
                seed=spec.get("seed"),
                base_url=spec.get("base_url"),
                generation_kwargs=dict(spec.get("generation_kwargs", {})),
            )
            registry[name] = config
            registry[alias] = config
            return name
        if isinstance(spec, str):
            config = ModelConfig(name=spec)
            registry[spec] = config
            registry[alias] = config
            return spec
        return None

    primary_spec = models.get("primary")
    secondary_spec = models.get("secondary")
    primary_model_id = _register("primary", primary_spec) or "local-llama"
    secondary_model_id = _register("secondary", secondary_spec) if secondary_spec else None

    LOGGER.debug("Loaded settings: %s", payload)

    return Settings(
        primary_model_id=primary_model_id,
        secondary_model_id=secondary_model_id,
        safety_keywords=list(keywords),
        model_configs=registry,
    )


def create_classifier(settings: Settings, repository: FeedbackRepository) -> DualLLMClassifier:
    """Instantiate the dual LLM classifier with configured models."""

    return DualLLMClassifier(
        primary_model=settings.model_configs.get(
            settings.primary_model_id, settings.primary_model_id
        ),
        secondary_model=(
            settings.model_configs.get(settings.secondary_model_id, settings.secondary_model_id)
            if settings.secondary_model_id
            else None
        ),
        feedback_repository=repository,
        safety_keywords=settings.safety_keywords,
        model_configs=settings.model_configs,
    )


def get_prompt_tree(path: pathlib.Path = PROMPT_LIBRARY_PATH) -> PromptTree:
    """Load the configured prompt tree from disk."""

    return PromptTree.from_yaml(str(path))


# ---------------------------------------------------------------------------
# Data stores and job registry


@dataclass
class StoredDocument:
    """Persisted document artefacts waiting to be classified."""

    document_id: str
    path: pathlib.Path
    bundle: DocumentBundle


class DocumentStore:
    """In-memory store for uploaded documents."""

    def __init__(self) -> None:
        self._documents: Dict[str, StoredDocument] = {}
        self._lock = asyncio.Lock()

    async def add(self, record: StoredDocument) -> None:
        async with self._lock:
            self._documents[record.document_id] = record
            LOGGER.debug("Stored document %s", record.document_id)

    async def get(self, document_id: str) -> StoredDocument:
        async with self._lock:
            record = self._documents.get(document_id)
        if record is None:
            LOGGER.error("Document %s not found", document_id)
            raise KeyError(document_id)
        return record

    async def ensure_exists(self, document_ids: Sequence[str]) -> None:
        async with self._lock:
            missing = [doc_id for doc_id in document_ids if doc_id not in self._documents]
        if missing:
            LOGGER.error("Missing documents for classification: %s", missing)
            raise KeyError(
                ", ".join(missing)
            )


TaskStatus = Literal["queued", "processing", "completed", "failed"]


@dataclass
class TaskState:
    """Represents the status of an asynchronous classification job."""

    status: TaskStatus
    progress: float = 0.0
    results: List[Dict[str, object]] = field(default_factory=list)
    error: Optional[str] = None

    def clone(self) -> "TaskState":
        return TaskState(
            status=self.status,
            progress=self.progress,
            results=list(self.results),
            error=self.error,
        )


class TaskRegistry:
    """Tracks asynchronous classification tasks and their progress."""

    def __init__(self) -> None:
        self._states: Dict[str, TaskState] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> str:
        task_id = uuid.uuid4().hex
        async with self._lock:
            self._states[task_id] = TaskState(status="queued", progress=0.0)
        LOGGER.debug("Created task %s", task_id)
        return task_id

    async def get(self, task_id: str) -> TaskState:
        async with self._lock:
            state = self._states.get(task_id)
        if state is None:
            LOGGER.error("Task %s not found", task_id)
            raise KeyError(task_id)
        return state.clone()

    async def set_status(self, task_id: str, status: TaskStatus) -> None:
        async with self._lock:
            state = self._states.get(task_id)
            if state is None:
                raise KeyError(task_id)
            state.status = status
        LOGGER.debug("Task %s status -> %s", task_id, status)

    async def set_progress(self, task_id: str, progress: float) -> None:
        async with self._lock:
            state = self._states.get(task_id)
            if state is None:
                raise KeyError(task_id)
            state.progress = progress
        LOGGER.debug("Task %s progress -> %.2f", task_id, progress)

    async def add_result(self, task_id: str, payload: Dict[str, object]) -> None:
        async with self._lock:
            state = self._states.get(task_id)
            if state is None:
                raise KeyError(task_id)
            state.results.append(payload)
        LOGGER.debug("Task %s appended result", task_id)

    async def fail(self, task_id: str, error: str) -> None:
        async with self._lock:
            state = self._states.get(task_id)
            if state is None:
                raise KeyError(task_id)
            state.status = "failed"
            state.error = error
            state.progress = 0.0
        LOGGER.error("Task %s failed: %s", task_id, error)


@dataclass
class ClassificationJob:
    """Job submitted to the asynchronous queue."""

    task_id: str
    document_ids: List[str]


class ClassificationQueue:
    """Simple asyncio-based worker queue for document classification."""

    def __init__(
        self,
        document_store: DocumentStore,
        classifier: DualLLMClassifier,
        prompt_tree: PromptTree,
        task_registry: TaskRegistry,
        report_dir: pathlib.Path,
        workers: int = 1,
    ) -> None:
        self._document_store = document_store
        self._classifier = classifier
        self._prompt_tree = prompt_tree
        self._task_registry = task_registry
        self._report_dir = report_dir
        self._queue: "asyncio.Queue[Optional[ClassificationJob]]" = asyncio.Queue()
        self._workers = workers
        self._worker_handles: List[asyncio.Task[None]] = []

    async def start(self) -> None:
        for index in range(self._workers):
            handle = asyncio.create_task(self._worker_loop(index))
            self._worker_handles.append(handle)
        LOGGER.info("Started %s classification worker(s)", self._workers)

    async def shutdown(self) -> None:
        for _ in range(self._workers):
            await self._queue.put(None)
        await asyncio.gather(*self._worker_handles, return_exceptions=True)
        self._worker_handles.clear()
        LOGGER.info("Classification workers stopped")

    async def enqueue(self, job: ClassificationJob) -> None:
        LOGGER.info("Queueing classification job %s", job.task_id)
        await self._queue.put(job)

    async def _worker_loop(self, index: int) -> None:
        LOGGER.debug("Worker %s waiting for jobs", index)
        while True:
            job = await self._queue.get()
            if job is None:
                LOGGER.debug("Worker %s received shutdown signal", index)
                break
            try:
                await self._process_job(job)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Worker %s failed job %s: %s", index, job.task_id, exc)
                await self._task_registry.fail(job.task_id, str(exc))
            finally:
                self._queue.task_done()

    async def _process_job(self, job: ClassificationJob) -> None:
        await self._task_registry.set_status(job.task_id, "processing")
        total = len(job.document_ids)
        if total == 0:
            await self._task_registry.fail(job.task_id, "No documents submitted")
            return

        for index, document_id in enumerate(job.document_ids, start=1):
            record = await self._document_store.get(document_id)
            try:
                result = await asyncio.to_thread(
                    self._classifier.classify,
                    record.bundle,
                    self._prompt_tree,
                    record.path,
                )
                reports = await asyncio.to_thread(
                    DualLLMClassifier.generate_reports,
                    result,
                    self._report_dir / job.task_id / document_id,
                )
            except Exception as exc:  # pragma: no cover - propagation handled above
                await self._task_registry.fail(job.task_id, str(exc))
                raise

            payload = _serialize_classification(document_id, result, reports)
            await self._task_registry.add_result(job.task_id, payload)
            await self._task_registry.set_progress(job.task_id, index / total)

        await self._task_registry.set_status(job.task_id, "completed")
        await self._task_registry.set_progress(job.task_id, 1.0)


# ---------------------------------------------------------------------------
# Pydantic schemas


class UploadResponse(BaseModel):
    """Response returned once documents are uploaded and processed."""

    document_ids: List[str]


class ClassificationJobRequest(BaseModel):
    """Request body for submitting a classification job."""

    document_ids: Optional[List[str]] = None
    text: Optional[str] = None
    document_name: Optional[str] = None


class ClassificationJobResponse(BaseModel):
    """Acknowledgement response once a job is accepted."""

    task_id: str
    status: TaskStatus


class CitationPayload(BaseModel):
    """Serializable citation payload returned to API clients."""

    source: str
    page: Optional[int]
    snippet: str
    confidence: float
    metadata: Dict[str, str]


class DocumentClassificationResult(BaseModel):
    """Represents classification output for an individual document."""

    document_id: str
    label: str
    score: float
    safety_flags: List[str]
    verifier_agreement: bool
    citations: List[CitationPayload]
    reports: Dict[str, str]


class TaskStatusResponse(BaseModel):
    """Status response for asynchronous jobs."""

    task_id: str
    status: TaskStatus
    progress: float
    results: Optional[List[DocumentClassificationResult]] = None
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Incoming payload for human-in-the-loop feedback."""

    classification_id: int
    reviewer: str
    notes: str
    quality_score: float


class FeedbackResponse(BaseModel):
    """Simple acknowledgement payload after recording feedback."""

    status: str


# ---------------------------------------------------------------------------
# Serialization helpers


def _serialize_classification(
    document_id: str,
    result: ClassificationResult,
    reports: Dict[str, pathlib.Path],
) -> Dict[str, object]:
    """Convert a classification result into an API-friendly dictionary."""

    citation_payloads = [
        {
            "source": citation.source,
            "page": citation.page,
            "snippet": citation.snippet,
            "confidence": citation.confidence,
            "metadata": citation.metadata,
        }
        for citation in result.citations
    ]

    return {
        "document_id": document_id,
        "label": result.label,
        "score": result.score,
        "safety_flags": list(result.safety_flags),
        "verifier_agreement": result.verifier_agreement,
        "classification_id": result.classification_id,
        "citations": citation_payloads,
        "reports": {key: str(path) for key, path in reports.items()},
    }


# ---------------------------------------------------------------------------
# FastAPI application factory


def create_app() -> FastAPI:
    """Instantiate the FastAPI application and wire dependencies."""

    app = FastAPI(
        title="AI Document Classifier",
        version="1.0.0",
        description="Backend services powering the AI-powered document classifier.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    repository = FeedbackRepository()
    classifier = create_classifier(settings, repository)
    prompt_tree = get_prompt_tree()
    preprocessor = DocumentPreprocessor()
    document_store = DocumentStore()
    task_registry = TaskRegistry()
    queue = ClassificationQueue(
        document_store=document_store,
        classifier=classifier,
        prompt_tree=prompt_tree,
        task_registry=task_registry,
        report_dir=REPORTS_DIR,
        workers=2,
    )

    async def _store_inline_document(
        text: str, document_name: Optional[str]
    ) -> str:
        """Persist ad-hoc text input as a document in the store."""

        document_id = uuid.uuid4().hex
        filename = document_name or f"inline-{document_id}.txt"
        suffix = pathlib.Path(filename).suffix or ".txt"
        target_path = UPLOAD_DIR / f"{document_id}{suffix}"

        def _write_text(path: pathlib.Path, data: str) -> None:
            path.write_text(data, encoding="utf-8")

        await asyncio.to_thread(_write_text, target_path, text)
        bundle = await asyncio.to_thread(preprocessor.process_document, target_path)
        bundle.metadata.setdefault("source", str(target_path))
        bundle.metadata.setdefault("filename", filename)

        await document_store.add(
            StoredDocument(document_id=document_id, path=target_path, bundle=bundle)
        )
        LOGGER.info("Prepared inline document %s", document_id)
        return document_id

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - lifecycle wiring
        await queue.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - lifecycle wiring
        await queue.shutdown()

    @app.post("/upload", response_model=UploadResponse, tags=["documents"])
    async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
        """Upload multi-modal documents and run preprocessing."""

        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        document_ids: List[str] = []
        for upload in files:
            filename = upload.filename or "document"
            suffix = pathlib.Path(filename).suffix or ".bin"
            document_id = uuid.uuid4().hex
            target_path = UPLOAD_DIR / f"{document_id}{suffix}"
            data = await upload.read()
            target_path.write_bytes(data)

            bundle = await asyncio.to_thread(
                preprocessor.process_document, target_path
            )
            bundle.metadata.setdefault("source", str(target_path))
            bundle.metadata.setdefault("filename", filename)

            await document_store.add(
                StoredDocument(document_id=document_id, path=target_path, bundle=bundle)
            )
            document_ids.append(document_id)
            LOGGER.info("Uploaded %s as %s", filename, document_id)

        return UploadResponse(document_ids=document_ids)

    @app.post(
        "/classify",
        response_model=ClassificationJobResponse,
        tags=["classification"],
    )
    async def classify(request: ClassificationJobRequest) -> ClassificationJobResponse:
        """Submit documents for asynchronous classification."""

        document_ids = list(request.document_ids or [])

        if request.text:
            if document_ids:
                raise HTTPException(
                    status_code=400,
                    detail="Provide either document IDs or raw text, not both.",
                )
            try:
                new_document_id = await _store_inline_document(
                    request.text, request.document_name
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Failed to prepare inline document: %s", exc)
                raise HTTPException(
                    status_code=500,
                    detail="Failed to prepare document for classification.",
                ) from exc
            document_ids.append(new_document_id)

        if not document_ids:
            raise HTTPException(
                status_code=400,
                detail="No document IDs or text supplied",
            )

        try:
            await document_store.ensure_exists(document_ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown documents: {exc}") from exc

        task_id = await task_registry.create()
        await queue.enqueue(
            ClassificationJob(task_id=task_id, document_ids=list(document_ids))
        )

        return ClassificationJobResponse(task_id=task_id, status="queued")

    @app.get(
        "/status/{task_id}",
        response_model=TaskStatusResponse,
        tags=["classification"],
    )
    async def status(task_id: str) -> TaskStatusResponse:
        """Retrieve status information for a submitted classification job."""

        try:
            state = await task_registry.get(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Task not found") from exc

        results = None
        if state.results:
            results = [
                DocumentClassificationResult(**payload) for payload in state.results
            ]

        return TaskStatusResponse(
            task_id=task_id,
            status=state.status,
            progress=state.progress,
            results=results,
            error=state.error,
        )

    @app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
    async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
        """Store human review feedback for a prior classification."""

        try:
            repository.add_feedback(
                classification_id=request.classification_id,
                feedback=Feedback(
                    reviewer=request.reviewer,
                    notes=request.notes,
                    quality_score=request.quality_score,
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return FeedbackResponse(status="recorded")

    return app


app = create_app()


__all__ = [
    "CLASSIFICATION_LABELS",
    "ClassificationResult",
    "DualLLMClassifier",
    "create_app",
    "load_settings",
]

