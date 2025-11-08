"""Document preprocessing utilities for multi-modal content."""

from __future__ import annotations

import io
import json
import mimetypes
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

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
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class DocumentBundle:
    """Container for all extracted artefacts from a document."""

    text: str
    images: List[DocumentImage] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

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

        text = self._extract_text(source)
        images = list(self._extract_images(source))
        metadata = {"source": str(source)}
        LOGGER.debug("Processed document %s", source)
        return DocumentBundle(text=text, images=images, metadata=metadata)

    # Internal helpers -----------------------------------------------------------
    def _extract_text(self, source: pathlib.Path) -> str:
        """Extract text from supported document types."""

        mime, _ = mimetypes.guess_type(source)
        if not mime:
            mime = "application/octet-stream"

        LOGGER.debug("Extracting text from %s with mime %s", source, mime)

        if mime.startswith("text"):
            return source.read_text(encoding="utf-8")

        if mime == "application/pdf":
            return self._extract_pdf_text(source)

        if mime.startswith("image"):
            return self._extract_image_text(source)

        LOGGER.warning("Unsupported mime type %s. Returning empty text.", mime)
        return ""

    def _extract_pdf_text(self, source: pathlib.Path) -> str:
        """Extract textual content from a PDF document."""

        if pdfplumber is None:  # pragma: no cover - optional dependency
            LOGGER.warning("pdfplumber not installed; returning empty text.")
            return ""

        texts: List[str] = []
        with pdfplumber.open(str(source)) as pdf:  # type: ignore[arg-type]
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        return "\n".join(texts)

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
