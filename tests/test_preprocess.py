"""Tests for preprocess module."""

import pathlib

from src.preprocess import DocumentPreprocessor


def test_process_plain_text(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "doc.txt"
    source.write_text("This is a confidential memo.", encoding="utf-8")

    preprocessor = DocumentPreprocessor(enable_ocr=False)
    bundle = preprocessor.process_document(source)

    assert "confidential" in bundle.text.lower()
    assert bundle.metadata["source"] == str(source)
