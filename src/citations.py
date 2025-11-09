"""Citation generation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Citation:
    """Represents evidence supporting a classification decision."""

    source: str
    page: Optional[int]
    snippet: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CitationManager:
    """Manage citations for generated classifications."""

    def __init__(self) -> None:
        self._citations: List[Citation] = []

    def add(self, citation: Citation) -> None:
        """Add a citation to the manager."""

        self._citations.append(citation)

    def extend(self, citations: Iterable[Citation]) -> None:
        """Extend the citation list."""

        for citation in citations:
            self.add(citation)

    def to_payload(self) -> List[Dict[str, object]]:
        """Serialize the citations to a JSON-compatible payload."""

        return [
            {
                "source": citation.source,
                "page": citation.page,
                "snippet": citation.snippet,
                "confidence": citation.confidence,
                "metadata": citation.metadata,
            }
            for citation in self._citations
        ]

    @property
    def citations(self) -> List[Citation]:
        """Return the collected citations."""

        return list(self._citations)


__all__ = ["Citation", "CitationManager"]
