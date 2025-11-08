"""Human-in-the-loop feedback storage and retrieval."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

    class FeedbackRecord(Base):
        """Feedback attached to a classification."""

        __tablename__ = "feedback_records"

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        classification_id: Mapped[int] = mapped_column(
            ForeignKey("classification_records.id")
        )
        reviewer: Mapped[str] = mapped_column(String, nullable=False)
        notes: Mapped[str] = mapped_column(String, nullable=False)
        quality_score: Mapped[float] = mapped_column(Float, default=0.0)
        created_at: Mapped[dt.datetime] = mapped_column(
            DateTime, default=dt.datetime.utcnow
        )

        classification: Mapped[ClassificationRecord] = relationship(
            "ClassificationRecord", back_populates="feedback"
        )

else:

    @dataclass
    class FeedbackRecord:  # type: ignore[no-redef]
        """Fallback feedback record when SQLAlchemy is unavailable."""

        id: int
        classification_id: int
        reviewer: str
        notes: str
        quality_score: float
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


@dataclass
class Feedback:
    """DTO for feedback information."""

    reviewer: str
    notes: str
    quality_score: float


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
            Base.metadata.create_all(self.engine)
        else:
            self.engine = None
            self._classifications: Dict[int, ClassificationRecord] = {}
            self._feedback: Dict[int, List[FeedbackRecord]] = {}
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
                        notes=feedback.notes,
                        quality_score=feedback.quality_score,
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
            notes=feedback.notes,
            quality_score=feedback.quality_score,
        )
        record.feedback.append(feedback_record)
        self._feedback.setdefault(classification_id, []).append(feedback_record)
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


__all__ = [
    "FeedbackRepository",
    "Feedback",
    "ClassificationRecord",
    "FeedbackRecord",
]
