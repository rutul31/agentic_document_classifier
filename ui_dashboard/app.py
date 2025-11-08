"""Streamlit dashboard for the document classification assistant."""

from __future__ import annotations

import json
from typing import Any, Dict, List

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
        st.session_state["last_classification_id"] = payload.get("classification_id")

    classification_id = st.session_state.get("last_classification_id")
    if classification_id:
        st.subheader("Reviewer Feedback")
        with st.form("feedback_form", clear_on_submit=True):
            reviewer = st.text_input("Reviewer")
            decision = st.selectbox(
                "Decision", ["accept", "override", "reclassify", "flag"]
            )
            suggested_label = st.text_input("Suggested Label (optional)")
            comments = st.text_area("Comments")
            quality_score = st.slider(
                "Quality Score", min_value=0.0, max_value=1.0, value=0.75, step=0.05
            )
            submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            response = requests.post(
                f"{API_BASE}/feedback",
                json={
                    "classification_id": classification_id,
                    "reviewer": reviewer,
                    "decision": decision,
                    "comments": comments,
                    "quality_score": quality_score,
                    "suggested_label": suggested_label or None,
                },
                timeout=30,
            )
            if response.status_code != 200:
                st.error(f"Failed to submit feedback: {response.text}")
            else:
                result = response.json()
                st.success("Feedback recorded")
                st.write(f"History exported to {result['history_path']}")
                st.json(result.get("thresholds", {}))

    st.subheader("Feedback History")
    if st.button("Refresh History"):
        _render_feedback_history()
    if st.button("Export Feedback History"):
        response = requests.post(f"{API_BASE}/feedback/export", timeout=30)
        if response.status_code != 200:
            st.error(f"Export failed: {response.text}")
        else:
            st.success(f"Exported to {response.json()['path']}")


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


def _render_feedback_history() -> None:
    """Fetch and display feedback records."""

    response = requests.get(f"{API_BASE}/feedback/history", timeout=30)
    if response.status_code != 200:
        st.error(f"Unable to fetch history: {response.text}")
        return
    data = response.json()
    items: List[Dict[str, Any]] = data.get("items", [])
    if not items:
        st.info("No feedback captured yet")
        return
    st.dataframe(items)


if __name__ == "__main__":
    main()
