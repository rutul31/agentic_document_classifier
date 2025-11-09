"""Document preprocessing utilities for multi-modal content."""

from __future__ import annotations

import io
import json
import mimetypes
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover - handled gracefully
    Image = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import pdfplumber
except ImportError:  # pragma: no cover - handled gracefully
    pdfplumber = None

try:  # pragma: no cover - optional dependency
    import pytesseract
except ImportError:  # pragma: no cover - handled gracefully
    pytesseract = None

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
    cv2 = None

from .utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass
class DocumentImage:
    """Representation of an extracted image."""

    image: object
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentBundle:
    """Container for all extracted artefacts from a document."""

    text: str
    images: List[DocumentImage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the bundle into JSON for persistence."""

        payload = {
            "text": self.text,
            "images": [
                {
                    "page": img.page,
                    "metadata": img.metadata,
                }
                for img in self.images
            ],
            "metadata": self.metadata,
        }
        return json.dumps(payload, ensure_ascii=False)


class DocumentPreprocessor:
    """High-level interface for document preprocessing."""

    def __init__(self, enable_ocr: bool = True) -> None:
        self.enable_ocr = enable_ocr

    # Public API -----------------------------------------------------------------
    def process_document(self, source: pathlib.Path) -> DocumentBundle:
        """Process a single document and return structured artefacts."""

        text, page_count, page_spans = self._extract_text(source)
        images = list(self._extract_images(source))
        metadata = self._build_metadata(source, text, images, page_count, page_spans)
        LOGGER.debug("Processed document %s", source)
        return DocumentBundle(text=text, images=images, metadata=metadata)

    def _build_metadata(
        self,
        source: pathlib.Path,
        text: str,
        images: List[DocumentImage],
        page_count: int,
        page_spans: List[Dict[str, int]],
    ) -> Dict[str, Any]:
        """Derive structured metadata for downstream prompt templating."""

        text_content = text or ""
        stripped = text_content.strip()
        has_text = bool(stripped)
        image_count = len(images)
        pages_with_images = sorted(
            {img.page for img in images if img.page is not None}
        )
        word_count = len(stripped.split()) if stripped else 0

        if has_text and image_count:
            content_type = "mixed"
        elif image_count:
            content_type = "image"
        elif has_text:
            content_type = "text"
        else:
            content_type = "unknown"

        metadata: Dict[str, Any] = {
            "source": str(source),
            "has_text": has_text,
            "text_length": len(text_content),
            "character_count": len(text_content),
            "word_count": word_count,
            "has_images": image_count > 0,
            "image_count": image_count,
            "content_type": content_type,
            "page_count": max(int(page_count or 0), 0),
        }

        if pages_with_images:
            metadata["pages_with_images"] = pages_with_images
        if page_spans:
            metadata["page_spans"] = page_spans

        return metadata

    # Internal helpers -----------------------------------------------------------
    def _extract_text(self, source: pathlib.Path) -> Tuple[str, int, List[Dict[str, int]]]:
        """Extract text plus basic pagination metadata."""

        mime, _ = mimetypes.guess_type(source)
        if not mime:
            mime = "application/octet-stream"

        LOGGER.debug("Extracting text from %s with mime %s", source, mime)

        if mime.startswith("text"):
            text = source.read_text(encoding="utf-8")
            if not text:
                return "", 0, []
            sections = text.split("\f")
            spans: List[Dict[str, int]] = []
            cursor = 0
            for index, section in enumerate(sections, start=1):
                start = cursor
                cursor += len(section)
                spans.append({"page": index, "start": start, "end": cursor})
                cursor += 1  # account for removed delimiter
            return text, max(len(sections), 1), spans

        if mime == "application/pdf":
            return self._extract_pdf_text(source)

        if mime.startswith("image"):
            text = self._extract_image_text(source)
            span = {"page": 1, "start": 0, "end": len(text)} if text else {}
            return text, 1 if text else 0, [span] if span else []

        LOGGER.warning("Unsupported mime type %s. Returning empty text.", mime)
        return "", 0, []

    def _extract_pdf_text(self, source: pathlib.Path) -> Tuple[str, int, List[Dict[str, int]]]:
        """Extract textual content and page count from a PDF document."""

        if pdfplumber is None:  # pragma: no cover - optional dependency
            LOGGER.warning("pdfplumber not installed; returning empty text.")
            return "", 0, []

        texts: List[str] = []
        page_spans: List[Dict[str, int]] = []
        separator = "\n\n"
        current = 0
        with pdfplumber.open(str(source)) as pdf:  # type: ignore[arg-type]
            for index, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                texts.append(page_text)
                start = current
                current += len(page_text)
                page_spans.append({"page": index, "start": start, "end": current})
                if index < len(pdf.pages):
                    current += len(separator)
        combined = separator.join(texts)
        return combined, len(texts), page_spans

    def _extract_image_text(self, source: pathlib.Path) -> str:
        """Extract text from an image using OCR when available."""

        if not self.enable_ocr or pytesseract is None or Image is None:
            LOGGER.info("OCR disabled or pytesseract missing; returning empty text.")
            return ""

        image = Image.open(source)  # type: ignore[call-arg]
        return pytesseract.image_to_string(image)

    def _extract_images(self, source: pathlib.Path) -> Iterable[DocumentImage]:
        """Extract images from supported document types."""

        mime, _ = mimetypes.guess_type(source)
        if mime == "application/pdf":
            yield from self._extract_pdf_images(source)
        elif mime and mime.startswith("image") and Image is not None:
            image = Image.open(source)  # type: ignore[call-arg]
            yield DocumentImage(image=image, page=None)

    def _extract_pdf_images(self, source: pathlib.Path) -> Iterable[DocumentImage]:
        """Extract images from a PDF file."""

        if (
            pdfplumber is None or cv2 is None or Image is None
        ):  # pragma: no cover - optional dependency
            LOGGER.info("Skipping PDF image extraction due to missing dependencies.")
            return []

        with pdfplumber.open(str(source)) as pdf:  # type: ignore[arg-type]
            for page_number, page in enumerate(pdf.pages, start=1):
                pil_images = page.images
                for img_meta in pil_images:
                    if "stream" not in img_meta:
                        continue
                    try:
                        stream = io.BytesIO(img_meta["stream"].get_data())
                        image = Image.open(stream)
                        yield DocumentImage(
                            image=image, page=page_number, metadata=img_meta
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        LOGGER.warning(
                            "Failed to extract image on page %s: %s", page_number, exc
                        )
                        continue


__all__ = ["DocumentPreprocessor", "DocumentBundle", "DocumentImage"]
