"""Streamlit dashboard for the document classification assistant."""

from __future__ import annotations

import json
from typing import Any, Dict

import requests
import streamlit as st

API_BASE = "http://localhost:8000"


def _render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("Classification Assistant")
    mode = st.sidebar.selectbox("Mode", ["Interactive", "Batch"])
    return {"mode": mode}


def interactive_mode() -> None:
    st.header("Interactive Classification")
    text = st.text_area("Document Text")
    document_path = st.text_input("Document Path (optional)")
    if st.button("Classify"):
        response = requests.post(
            f"{API_BASE}/classify",
            json={"text": text, "document_path": document_path or None},
            timeout=60,
        )
        if response.status_code != 200:
            st.error(f"Error: {response.text}")
            return
        payload = response.json()
        st.json(json.loads(payload["result"]))
        st.write("Reports:")
        for name, path in payload["reports"].items():
            st.write(f"{name}: {path}")


def batch_mode() -> None:
    st.header("Batch Classification")
    documents_raw = st.text_area("Documents JSON", value="[]")
    if st.button("Submit Batch"):
        documents = json.loads(documents_raw)
        response = requests.post(f"{API_BASE}/batch", json={"documents": documents}, timeout=60)
        if response.status_code != 200:
            st.error(f"Error: {response.text}")
            return
        st.success(f"Batch job queued: {response.json()['job_id']}")


def main() -> None:
    st.set_page_config(page_title="Document Classification Assistant", layout="wide")
    context = _render_sidebar()
    if context["mode"] == "Interactive":
        interactive_mode()
    else:
        batch_mode()


if __name__ == "__main__":
    main()
