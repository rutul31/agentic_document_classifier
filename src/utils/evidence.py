"""Heuristic evidence extraction utilities for classification support."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from ..preprocess import DocumentBundle


@dataclass(frozen=True)
class RiskSignal:
    """Represents a heuristic indicator of sensitivity or unsafe content."""

    label: str  # One of Public, Confidential, Highly Sensitive, Unsafe
    snippet: str
    description: str
    page: Optional[int]
    confidence: float
    source_type: str = "text"


@dataclass(frozen=True)
class _PatternRule:
    name: str
    label: str
    pattern: re.Pattern[str]
    confidence: float = 0.85
    description: Optional[str] = None


_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PASSPORT = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")
_CREDIT_CARD = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")
_BANK_ACCOUNT = re.compile(r"\b\d{9,12}\b")
_EMAIL = re.compile(r"[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z0-9-.]+", re.I)

_PATTERN_RULES: Sequence[_PatternRule] = (
    _PatternRule(
        name="ssn",
        label="Highly Sensitive",
        pattern=_SSN,
        confidence=0.95,
        description="Possible social security number detected",
    ),
    _PatternRule(
        name="passport",
        label="Highly Sensitive",
        pattern=_PASSPORT,
        confidence=0.8,
        description="Passport or government ID pattern",
    ),
    _PatternRule(
        name="credit_card",
        label="Highly Sensitive",
        pattern=_CREDIT_CARD,
        confidence=0.9,
        description="Credit or debit card number pattern",
    ),
    _PatternRule(
        name="bank_account",
        label="Confidential",
        pattern=_BANK_ACCOUNT,
        confidence=0.75,
        description="Potential bank or routing number",
    ),
    _PatternRule(
        name="email",
        label="Confidential",
        pattern=_EMAIL,
        confidence=0.6,
        description="Email address detected",
    ),
)

_CONFIDENTIAL_KEYWORDS: Sequence[str] = (
    "confidential",
    "internal use only",
    "proprietary",
    "nda",
    "do not distribute",
    "restricted",
    "customer list",
    "employee roster",
    "pricing sheet",
)

_UNSAFE_KEYWORDS: Sequence[str] = (
    "malware",
    "exploit",
    "payload",
    "weapon",
    "kill chain",
    "attack vector",
    "child sexual",
    "csam",
    "extremist",
    "hate speech",
)


def extract_risk_signals(bundle: DocumentBundle, limit: int = 8) -> List[RiskSignal]:
    """Pull high-signal snippets from text to support citations and overrides."""

    text = (bundle.text or "").strip()
    if not text:
        return []

    metadata = bundle.metadata or {}
    spans = metadata.get("page_spans") or []

    signals: List[RiskSignal] = []

    for rule in _PATTERN_RULES:
        for match in rule.pattern.finditer(text):
            snippet = _extract_snippet(text, match.start(), match.end())
            if not snippet:
                continue
            page = _lookup_page(spans, match.start())
            signals.append(
                RiskSignal(
                    label=rule.label,
                    snippet=snippet,
                    description=rule.description or rule.name,
                    page=page,
                    confidence=rule.confidence,
                )
            )
            if len(signals) >= limit:
                return signals

    lowered = text.lower()
    for keyword in _CONFIDENTIAL_KEYWORDS:
        if keyword in lowered:
            index = lowered.index(keyword)
            snippet = _extract_snippet(text, index, index + len(keyword))
            page = _lookup_page(spans, index)
            signals.append(
                RiskSignal(
                    label="Confidential",
                    snippet=snippet,
                    description=f"Keyword '{keyword}'",
                    page=page,
                    confidence=0.55,
                )
            )
            if len(signals) >= limit:
                return signals

    for keyword in _UNSAFE_KEYWORDS:
        if keyword in lowered:
            index = lowered.index(keyword)
            snippet = _extract_snippet(text, index, index + len(keyword))
            page = _lookup_page(spans, index)
            signals.append(
                RiskSignal(
                    label="Unsafe",
                    snippet=snippet,
                    description=f"Unsafe keyword '{keyword}'",
                    page=page,
                    confidence=0.9,
                )
            )
            if len(signals) >= limit:
                return signals

    return signals


def _extract_snippet(text: str, start: int, end: int, radius: int = 140) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].strip()
    return re.sub(r"\s+", " ", snippet)


def _lookup_page(spans: Iterable[Dict[str, int]], index: int) -> Optional[int]:
    for span in spans:
        start = span.get("start")
        end = span.get("end")
        if start is None or end is None:
            continue
        if start <= index < end:
            return span.get("page")
    return None


__all__ = ["RiskSignal", "extract_risk_signals"]
