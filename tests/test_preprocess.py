"""Integration-focused tests for the preprocessing module."""

import json

from src.preprocess import DocumentBundle, DocumentPreprocessor


def test_process_tc_documents(tc_documents):
    """Ensure the preprocessor extracts text and metadata for TC1–TC5."""

    for case_id, data in tc_documents.items():
        bundle = data["bundle"]
        config = data["config"]

        assert isinstance(bundle, DocumentBundle)
        assert bundle.text == config.text
        assert bundle.metadata["source"] == str(data["path"])

        payload = json.loads(bundle.to_json())
        assert payload["text"] == config.text
        assert payload["metadata"]["source"] == str(data["path"])
        assert payload["images"] == []


def test_preprocessor_handles_unknown_mime(tmp_path):
    """Unknown file types should not raise and should return empty text."""

    binary_path = tmp_path / "payload.bin"
    binary_path.write_bytes(b"\x00\x01\x02")

    preprocessor = DocumentPreprocessor(enable_ocr=False)
    bundle = preprocessor.process_document(binary_path)

    assert bundle.text == ""
    assert bundle.metadata["source"] == str(binary_path)
