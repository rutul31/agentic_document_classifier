"""Human-in-the-loop feedback storage and retrieval."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, List, Optional

import yaml

from .prompt_tree import PromptTree
from .utils.simple_yaml import load as load_yaml

from .utils.logger import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        ForeignKey,
        Integer,
        String,
        create_engine,
        text,
    )
    from sqlalchemy.orm import (
        DeclarativeBase,
        Mapped,
        Session,
        mapped_column,
        relationship,
    )
    from sqlalchemy.pool import StaticPool

    SQLALCHEMY_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully
    SQLALCHEMY_AVAILABLE = False


if SQLALCHEMY_AVAILABLE:

    class Base(DeclarativeBase):
        """Base class for SQLAlchemy models."""

    class ClassificationRecord(Base):
        """Persisted record of a classification run."""

        __tablename__ = "classification_records"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        document_path: Mapped[str] = mapped_column(String, nullable=False)
        classification: Mapped[str] = mapped_column(String, nullable=False)
        score: Mapped[float] = mapped_column(Float, default=0.0)
        created_at: Mapped[dt.datetime] = mapped_column(
            DateTime, default=dt.datetime.utcnow
        )

        feedback: Mapped[List["FeedbackRecord"]] = relationship(
            "FeedbackRecord",
            back_populates="classification",
            cascade="all, delete-orphan",
        )
        misclassifications: Mapped[List["MisclassificationRecord"]] = relationship(
            "MisclassificationRecord",
            back_populates="classification",
            cascade="all, delete-orphan",
        )

    class FeedbackRecord(Base):
        """Feedback attached to a classification."""

        __tablename__ = "feedback_records"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        classification_id: Mapped[int] = mapped_column(
            ForeignKey("classification_records.id")
        )
        reviewer: Mapped[str] = mapped_column(String, nullable=False)
        decision: Mapped[str] = mapped_column(String, nullable=False)
        comments: Mapped[str] = mapped_column(String, nullable=False)
        suggested_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
        quality_score: Mapped[float] = mapped_column(Float, default=0.0)
        created_at: Mapped[dt.datetime] = mapped_column(
            DateTime, default=dt.datetime.utcnow
        )

        classification: Mapped[ClassificationRecord] = relationship(
            "ClassificationRecord", back_populates="feedback"
        )

    class MisclassificationRecord(Base):
        """Human validated misclassification instances."""

        __tablename__ = "misclassification_records"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        classification_id: Mapped[int] = mapped_column(
            ForeignKey("classification_records.id"), nullable=False
        )
        expected_label: Mapped[str] = mapped_column(String, nullable=False)
        reviewer_notes: Mapped[str] = mapped_column(String, nullable=False)
        created_at: Mapped[dt.datetime] = mapped_column(
            DateTime, default=dt.datetime.utcnow
        )

        classification: Mapped[ClassificationRecord] = relationship(
            "ClassificationRecord", back_populates="misclassifications"
        )

else:

    @dataclass
    class FeedbackRecord:  # type: ignore[no-redef]
        """Fallback feedback record when SQLAlchemy is unavailable."""

        id: int
        classification_id: int
        reviewer: str
        decision: str
        comments: str
        quality_score: float
        suggested_label: Optional[str] = None
        created_at: dt.datetime = field(default_factory=dt.datetime.utcnow)

    @dataclass
    class ClassificationRecord:  # type: ignore[no-redef]
        """Fallback classification record when SQLAlchemy is unavailable."""

        id: int
        document_path: str
        classification: str
        score: float
        created_at: dt.datetime = field(default_factory=dt.datetime.utcnow)
        feedback: List[FeedbackRecord] = field(default_factory=list)
        misclassifications: List["MisclassificationRecord"] = field(
            default_factory=list
        )

    @dataclass
    class MisclassificationRecord:  # type: ignore[no-redef]
        """Fallback misclassification record."""

        id: int
        classification_id: int
        expected_label: str
        reviewer_notes: str
        created_at: dt.datetime = field(default_factory=dt.datetime.utcnow)


@dataclass
class Feedback:
    """DTO for feedback information."""

    reviewer: str
    decision: str
    comments: str
    quality_score: float
    suggested_label: Optional[str] = None


class FeedbackRepository:
    """Repository for managing classification results and feedback."""

    def __init__(self, database_url: str = "sqlite:///./assistant.db") -> None:
        if SQLALCHEMY_AVAILABLE:
            engine_kwargs = {"future": True}
            if database_url.endswith(":memory:"):
                engine_kwargs.update(
                    {
                        "connect_args": {"check_same_thread": False},
                        "poolclass": StaticPool,
                    }
                )
            self.engine = create_engine(database_url, **engine_kwargs)
            # Create any missing tables defined by the ORM models.
            Base.metadata.create_all(self.engine)
            # Run lightweight runtime migrations for missing columns that
            # can occur when the DB was created by an older release.
            try:
                self._ensure_feedback_columns()
            except Exception:
                LOGGER.exception("Failed to ensure feedback table schema")
    
    def _ensure_feedback_columns(self) -> None:
        """Ensure that expected feedback-related columns exist in the DB.

        Some users may have an older sqlite database created prior to new
        columns being added to the model (for example `decision`). SQLAlchemy's
        create_all won't alter existing tables, so we apply lightweight ALTER
        TABLE statements at runtime for small compatible changes.
        """
        if not getattr(self, "engine", None):
            return

        try:
            with self.engine.connect() as conn:
                # Query existing columns for the feedback_records table.
                result = conn.execute(text("PRAGMA table_info('feedback_records')"))
                rows = result.fetchall()
                # PRAGMA table_info rows: (cid, name, type, notnull, dflt_value, pk)
                existing = [row[1] for row in rows]

                if "decision" not in existing:
                    LOGGER.info("Adding missing 'decision' column to feedback_records")
                    with conn.begin():
                        # Add a nullable text column so existing rows continue to work.
                        conn.exec_driver_sql(
                            "ALTER TABLE feedback_records ADD COLUMN decision VARCHAR"
                        )
        except Exception:
            LOGGER.exception("Error while ensuring feedback table columns")
        else:
            self.engine = None
            self._classifications: Dict[int, ClassificationRecord] = {}
            self._feedback: Dict[int, List[FeedbackRecord]] = {}
            self._misclassifications: Dict[int, List[MisclassificationRecord]] = {}
            self._counter = 0

    # Classification persistence -------------------------------------------------
    def record_classification(
        self, document_path: str, classification: str, score: float
    ) -> int:
        """Persist a classification result and return its identifier."""

        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as session:  # type: ignore[arg-type]
                record = ClassificationRecord(
                    document_path=document_path,
                    classification=classification,
                    score=score,
                )
                session.add(record)
                session.commit()
                session.refresh(record)
                LOGGER.debug(
                    "Stored classification %s for %s", record.id, document_path
                )
                return record.id

        self._counter += 1
        record = ClassificationRecord(
            id=self._counter,
            document_path=document_path,
            classification=classification,
            score=score,
        )
        self._classifications[record.id] = record
        LOGGER.debug(
            "Stored classification %s for %s (fallback)", record.id, document_path
        )
        return record.id

    def add_feedback(self, classification_id: int, feedback: Feedback) -> None:
        """Attach feedback to an existing classification."""

        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as session:  # type: ignore[arg-type]
                record = session.get(ClassificationRecord, classification_id)
                if record is None:
                    msg = f"Classification {classification_id} not found"
                    LOGGER.error(msg)
                    raise ValueError(msg)
                session.add(
                    FeedbackRecord(
                        classification=record,
                        reviewer=feedback.reviewer,
                        decision=feedback.decision,
                        comments=feedback.comments,
                        suggested_label=feedback.suggested_label,
                        quality_score=feedback.quality_score,
                    )
                )
                if self._is_misclassification(feedback):
                    session.add(
                        MisclassificationRecord(
                            classification=record,
                            expected_label=feedback.suggested_label
                            or record.classification,
                            reviewer_notes=feedback.comments,
                        )
                    )
                session.commit()
                LOGGER.debug("Stored feedback for classification %s", classification_id)
                return

        record = self._classifications.get(classification_id)
        if record is None:
            msg = f"Classification {classification_id} not found"
            LOGGER.error(msg)
            raise ValueError(msg)
        feedback_record = FeedbackRecord(
            id=len(self._feedback.get(classification_id, [])) + 1,
            classification_id=classification_id,
            reviewer=feedback.reviewer,
            decision=feedback.decision,
            comments=feedback.comments,
            suggested_label=feedback.suggested_label,
            quality_score=feedback.quality_score,
        )
        record.feedback.append(feedback_record)
        self._feedback.setdefault(classification_id, []).append(feedback_record)
        if self._is_misclassification(feedback):
            self._log_misclassification_fallback(record, feedback)
        LOGGER.debug(
            "Stored feedback for classification %s (fallback)", classification_id
        )

    def list_feedback(
        self, classification_id: Optional[int] = None
    ) -> List[FeedbackRecord]:
        """Return stored feedback records."""

        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as session:  # type: ignore[arg-type]
                query = session.query(FeedbackRecord)
                if classification_id is not None:
                    query = query.filter(
                        FeedbackRecord.classification_id == classification_id
                    )
                return query.all()

        if classification_id is not None:
            return list(self._feedback.get(classification_id, []))
        aggregated: List[FeedbackRecord] = []
        for records in self._feedback.values():
            aggregated.extend(records)
        return aggregated

    # Misclassification logging --------------------------------------------------
    def _is_misclassification(self, feedback: Feedback) -> bool:
        """Return True if the feedback indicates a misclassification."""

        decision = feedback.decision.lower().strip()
        return decision in {"override", "reclassify"} or bool(
            feedback.suggested_label
        )

    def _log_misclassification_fallback(
        self, record: ClassificationRecord, feedback: Feedback
    ) -> None:
        """Persist misclassification info for the fallback storage."""

        misclassification = MisclassificationRecord(
            id=len(self._misclassifications.get(record.id, [])) + 1,
            classification_id=record.id,
            expected_label=feedback.suggested_label or record.classification,
            reviewer_notes=feedback.comments,
        )
        record.misclassifications.append(misclassification)
        self._misclassifications.setdefault(record.id, []).append(misclassification)

    def list_misclassifications(self) -> List[MisclassificationRecord]:
        """Return all misclassification records."""

        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as session:  # type: ignore[arg-type]
                return session.query(MisclassificationRecord).all()

        aggregated: List[MisclassificationRecord] = []
        for records in self._misclassifications.values():
            aggregated.extend(records)
        return aggregated

    # Analytics -----------------------------------------------------------------
    def get_label_statistics(self) -> Dict[str, Dict[str, float]]:
        """Aggregate statistics per label from captured feedback."""

        stats: Dict[str, Dict[str, float]] = {}

        def _register(label: str, decision: str, quality: float) -> None:
            bucket = stats.setdefault(
                label,
                {"total": 0.0, "overrides": 0.0, "avg_quality": 0.0},
            )
            bucket["total"] += 1
            if decision.lower() in {"override", "reclassify"}:
                bucket["overrides"] += 1
            # incremental average
            bucket["avg_quality"] = (
                ((bucket["total"] - 1) * bucket["avg_quality"]) + quality
            ) / bucket["total"]

        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as session:  # type: ignore[arg-type]
                records = session.query(FeedbackRecord).all()
                for entry in records:
                    label = entry.classification.classification  # type: ignore[attr-defined]
                    _register(label, entry.decision, entry.quality_score)
            return stats

        for classification in self._classifications.values():
            for entry in classification.feedback:
                _register(classification.classification, entry.decision, entry.quality_score)
        return stats

    def recalibrate_confidence(self, label: str, score: float) -> float:
        """Adjust confidence scores based on reviewer overrides."""

        stats = self.get_label_statistics()
        bucket = stats.get(label)
        if not bucket or bucket["total"] == 0:
            return score

        override_ratio = bucket["overrides"] / bucket["total"]
        quality_delta = bucket["avg_quality"] - 0.5
        penalty = min(0.3, override_ratio * 0.3)
        bonus = max(-0.1, min(0.1, quality_delta * 0.2))
        recalibrated = score * (1 - penalty) + bonus
        return max(0.0, min(1.0, recalibrated))

    # History export -------------------------------------------------------------
    def get_feedback_history(self) -> List[Dict[str, object]]:
        """Return a serialisable history of classifications and feedback."""

        history: List[Dict[str, object]] = []

        if SQLALCHEMY_AVAILABLE:
            with Session(self.engine) as session:  # type: ignore[arg-type]
                records = (
                    session.query(FeedbackRecord)
                    .join(FeedbackRecord.classification)
                    .order_by(FeedbackRecord.created_at.desc())
                    .all()
                )
                for entry in records:
                    history.append(
                        {
                            "classification_id": entry.classification_id,
                            "document_path": entry.classification.document_path,
                            "predicted_label": entry.classification.classification,
                            "score": entry.classification.score,
                            "decision": entry.decision,
                            "suggested_label": entry.suggested_label,
                            "comments": entry.comments,
                            "quality_score": entry.quality_score,
                            "created_at": entry.created_at.isoformat(),
                        }
                    )
        else:
            for record in self._classifications.values():
                for entry in record.feedback:
                    history.append(
                        {
                            "classification_id": entry.classification_id,
                            "document_path": record.document_path,
                            "predicted_label": record.classification,
                            "score": record.score,
                            "decision": entry.decision,
                            "suggested_label": entry.suggested_label,
                            "comments": entry.comments,
                            "quality_score": entry.quality_score,
                            "created_at": entry.created_at.isoformat(),
                        }
                    )

        return history

    def export_feedback_history(self, path: pathlib.Path) -> pathlib.Path:
        """Write the feedback history to disk for auditing."""

        history = self.get_feedback_history()
        payload = json.dumps(history, indent=2)
        path.write_text(payload, encoding="utf-8")
        LOGGER.info("Exported feedback history to %s", path)
        return path


class AdaptivePromptRefiner:
    """Applies adaptive refinement rules based on reviewer feedback."""

    def __init__(
        self,
        repository: FeedbackRepository,
        prompt_tree: PromptTree,
        config_path: pathlib.Path,
    ) -> None:
        self.repository = repository
        self.prompt_tree = prompt_tree
        self.config_path = config_path

    def refine(self) -> Dict[str, float]:
        """Update thresholds and prompt annotations based on feedback."""

        stats = self.repository.get_label_statistics()
        thresholds = self._update_thresholds(stats)
        self._annotate_prompt_tree(stats)
        return thresholds

    # Internal helpers ---------------------------------------------------------
    def _update_thresholds(self, stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Adjust threshold values in the configuration file."""

        config = load_yaml(str(self.config_path))
        thresholds = config.get("thresholds", {})
        updated: Dict[str, float] = {}
        for key, value in thresholds.items():
            label = key.replace("_", " ").title()
            bucket = stats.get(label)
            numeric_value = float(value)
            if not bucket or bucket["total"] == 0:
                updated[key] = numeric_value
                continue
            override_ratio = bucket["overrides"] / bucket["total"]
            adjustment = (0.2 - override_ratio) * 0.02
            new_value = max(0.05, min(0.99, numeric_value + adjustment))
            updated[key] = round(new_value, 4)

        config["thresholds"] = updated
        with open(self.config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        LOGGER.info("Updated thresholds based on feedback: %s", updated)
        return updated

    def _annotate_prompt_tree(self, stats: Dict[str, Dict[str, float]]) -> None:
        """Append feedback insights to the root prompt."""

        if not self.prompt_tree or not self.prompt_tree.root:
            return

        marker = "\n\n# HITL Adjustments\n"
        base_prompt = self.prompt_tree.root.prompt
        if marker in base_prompt:
            base_prompt = base_prompt.split(marker)[0]

        lines = ["Recent reviewer insights:"]
        for label, bucket in stats.items():
            if bucket["total"] == 0:
                continue
            override_pct = bucket["overrides"] / bucket["total"]
            lines.append(
                f"- {label}: overrides {override_pct:.0%}, avg quality {bucket['avg_quality']:.2f}"
            )

        summary = "\n".join(lines)
        self.prompt_tree.root.prompt = base_prompt + marker + summary
        LOGGER.debug("Prompt tree annotated with HITL insights")


__all__ = [
    "FeedbackRepository",
    "Feedback",
    "ClassificationRecord",
    "FeedbackRecord",
    "MisclassificationRecord",
    "AdaptivePromptRefiner",
]
