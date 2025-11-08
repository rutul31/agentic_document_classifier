"""Interactive Streamlit dashboard for the document classification assistant."""

from __future__ import annotations

import html
import io
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import Any, Dict, List

import requests
import streamlit as st

try:  # pragma: no cover - optional dependency
    import pdfplumber
except Exception:  # pragma: no cover - gracefully handled at runtime
    pdfplumber = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from plotly import graph_objects as go
except Exception as exc:  # pragma: no cover - Streamlit will surface the error
    raise RuntimeError(
        "Plotly is required for the dashboard. Install via `pip install plotly`."
    ) from exc

DEFAULT_API_BASE = "http://localhost:8000"
CITATION_COLORS = ["#7F5AF0", "#FF8E3C", "#2CB1BC", "#EF4565", "#16BDCA"]


@dataclass
class ClassificationState:
    """Persisted state for the most recent classification run."""

    label: str
    score: float
    citations: List[Dict[str, Any]]
    safety_flags: List[str]
    verifier_agreement: bool
    classification_id: Optional[int]
    reports: Dict[str, str]
    document_text: str
    document_name: str
    timestamp: datetime


def _apply_theme(dark_mode: bool) -> None:
    """Inject custom theming and responsive layout tweaks."""

    background = "#0E1117" if dark_mode else "#F7F9FC"
    panel_background = "#161B26" if dark_mode else "#FFFFFF"
    text_color = "#E4ECFA" if dark_mode else "#1F2933"
    accent = "#7F5AF0"

    st.markdown(
        f"""
        <style>
        html, body, [class*="css"], .stApp {{
            background-color: {background};
            color: {text_color};
        }}
        :root {{
            --panel-bg: {panel_background};
            --text-color: {text_color};
            --accent-color: {accent};
            --viewer-bg: {panel_background};
            --viewer-text: {text_color};
        }}
        .stSidebar {{
            background: rgba(0, 0, 0, 0.05);
        }}
        .document-viewer {{
            background-color: var(--viewer-bg);
            color: var(--viewer-text);
            padding: 1.2rem;
            border-radius: 0.75rem;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(127, 90, 240, 0.2);
        }}
        .citation-legend {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.6rem;
            border-radius: 999px;
            margin-bottom: 0.5rem;
            background-color: rgba(127, 90, 240, 0.12);
            color: var(--viewer-text);
            width: fit-content;
        }}
        .citation-bullet {{
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 50%;
        }}
        @media (max-width: 768px) {{
            .stTabs [role="tablist"] {{
                flex-wrap: wrap;
            }}
            .document-viewer {{
                max-height: 320px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> Dict[str, Any]:
    """Render sidebar controls and return the UI context."""

    st.sidebar.title("Classification Controls")
    stored_api = st.session_state.get("api_base", DEFAULT_API_BASE)
    stored_dark_mode = st.session_state.get("dark_mode", True)

    api_base = st.sidebar.text_input("API base URL", value=stored_api)
    dark_mode = st.sidebar.toggle("Dark mode", value=stored_dark_mode)

    st.session_state["api_base"] = api_base
    st.session_state["dark_mode"] = dark_mode

    history = st.session_state.get("history", [])
    if history:
        st.sidebar.metric("Sessions completed", len(history))
    st.sidebar.caption(
        "Upload a document or paste text to classify with real-time verification."
    )
    return {"api_base": api_base.rstrip("/"), "dark_mode": dark_mode}


def _extract_text(uploaded_file: Optional[Any], text_fallback: str) -> str:
    """Extract textual content from an uploaded asset or text input."""

    if uploaded_file is None:
        return text_fallback.strip()

    file_bytes = uploaded_file.getvalue()
    if uploaded_file.type == "text/plain":
        return file_bytes.decode("utf-8")

    if uploaded_file.type in {"application/pdf", "application/x-pdf"} and pdfplumber:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return text_fallback.strip()


def _simulate_progress(status: Any, progress: Any) -> None:
    """Animate progress updates while waiting for the API response."""

    progress.progress(10)
    status.write("Preparing document bundle…")
    time.sleep(0.1)
    progress.progress(35)
    status.write("Sending request to classification service…")
    time.sleep(0.1)
    progress.progress(60)
    status.write("Awaiting verifier agreement…")


def _run_classification(api_base: str, text: str, document_name: str) -> Optional[ClassificationState]:
    """Send the document to the backend and return structured state."""

    if not text:
        st.warning("Provide document text or upload a file before running classification.")
        return None

    endpoint = f"{api_base}/classify"
    payload = {"text": text, "document_path": document_name or None}

    with st.status("Running classification", expanded=True) as status:
        progress_placeholder = st.empty()
        progress = progress_placeholder.progress(0)
        _simulate_progress(status, progress)

        try:
            response = requests.post(endpoint, json=payload, timeout=120)
        except requests.RequestException as exc:
            status.update(label="Network error", state="error")
            st.error(f"Failed to contact the classification service: {exc}")
            return None

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
            status.update(label="Classification failed", state="error")
            st.error(f"API error {response.status_code}: {response.text}")
            return None

        try:
            raw_payload = response.json()
            result_payload = json.loads(raw_payload.get("result", "{}"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            status.update(label="Malformed response", state="error")
            st.error(f"Unable to parse API response: {exc}")
            return None

        progress.progress(100)
        status.write("Generating downloadable reports…")
        status.update(label="Classification complete", state="complete")
        progress_placeholder.empty()

    classification_state = ClassificationState(
        label=result_payload.get("label", "Unknown"),
        score=float(result_payload.get("score", 0.0)),
        citations=result_payload.get("citations", []),
        safety_flags=result_payload.get("safety_flags", []),
        verifier_agreement=bool(result_payload.get("verifier_agreement", False)),
        classification_id=result_payload.get("classification_id"),
        reports=raw_payload.get("reports", {}),
        document_text=text,
        document_name=document_name or "uploaded-document",
        timestamp=datetime.utcnow(),
    )

    history = st.session_state.setdefault("history", [])
    history.append(
        {
            "timestamp": classification_state.timestamp.isoformat(),
            "label": classification_state.label,
            "score": classification_state.score,
        }
    )
    st.session_state["latest_result"] = classification_state
    return classification_state


def _highlight_document(text: str, citations: List[Dict[str, Any]]) -> str:
    """Return HTML with highlighted citation snippets."""

    if not text:
        return "<div class='document-viewer'>No document text available.</div>"

    html_text = html.escape(text)
    for index, citation in enumerate(citations, start=1):
        snippet = citation.get("snippet") or ""
        if not snippet.strip():
            continue
        escaped_snippet = html.escape(snippet)
        color = CITATION_COLORS[(index - 1) % len(CITATION_COLORS)]
        highlight = (
            f"<mark style=\"background-color:{color}; color: var(--viewer-text);\">"
            f"{escaped_snippet}</mark>"
        )
        if escaped_snippet in html_text:
            html_text = html_text.replace(escaped_snippet, highlight, 1)

    html_text = html_text.replace("\n", "<br>")
    return f"<div class='document-viewer'>{html_text}</div>"


def _render_citation_legend(citations: List[Dict[str, Any]]) -> None:
    """Display a legend mapping highlight colors to citation metadata."""

    if not citations:
        st.info("No citations were generated for this document.")
        return

    for index, citation in enumerate(citations, start=1):
        color = CITATION_COLORS[(index - 1) % len(CITATION_COLORS)]
        st.markdown(
            f"""
            <div class="citation-legend">
                <span class="citation-bullet" style="background:{color};"></span>
                <strong>Citation {index}</strong>
                <span>{html.escape(citation.get('source', 'N/A'))}</span>
                <span>p.{citation.get('page') or '—'}</span>
                <span>confidence: {citation.get('confidence', 0):.2f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_downloads(state: ClassificationState) -> None:
    """Expose JSON and PDF downloads for the classification."""

    st.subheader("Download reports")
    json_payload = {
        "label": state.label,
        "score": state.score,
        "citations": state.citations,
        "safety_flags": state.safety_flags,
        "verifier_agreement": state.verifier_agreement,
        "classification_id": state.classification_id,
        "generated_at": state.timestamp.isoformat(),
        "document_name": state.document_name,
    }
    st.download_button(
        "Download JSON report",
        data=json.dumps(json_payload, indent=2).encode("utf-8"),
        file_name=f"{state.document_name}-classification.json",
        mime="application/json",
    )

    pdf_path = state.reports.get("pdf")
    pdf_bytes: Optional[bytes] = None
    if pdf_path:
        try:
            pdf_bytes = Path(pdf_path).read_bytes()
        except OSError:
            pdf_bytes = None
    if pdf_bytes:
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=f"{state.document_name}-classification.pdf",
            mime="application/pdf",
        )
    else:
        st.info("PDF report will be available once generated by the backend service.")


def _render_history_chart(history: List[Dict[str, Any]]) -> None:
    """Visualise category distribution using Plotly."""

    if not history:
        st.info("Run a classification to see category analytics.")
        return

    counts = Counter(entry["label"] for entry in history)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    colors = [CITATION_COLORS[i % len(CITATION_COLORS)] for i in range(len(labels))]

    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=values,
                textposition="outside",
            )
        ]
    )
    figure.update_layout(
        title="Classification label distribution",
        yaxis_title="Count",
        xaxis_title="Label",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "var(--text-color)"},
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_feedback(state: Optional[ClassificationState], api_base: str) -> None:
    """Render a feedback pane for human-in-the-loop review."""

    st.subheader("Reviewer feedback")
    if state is None or state.classification_id is None:
        st.info(
            "Submit a classification to unlock feedback capture. Ensure the backend"
            " returns a classification identifier."
        )
        return

    with st.form("feedback_form", clear_on_submit=True):
        reviewer = st.text_input("Reviewer name")
        quality_score = st.slider("Quality score", min_value=0.0, max_value=1.0, value=0.8)
        notes = st.text_area("Feedback notes", help="Highlight corrections or additional context.")
        submitted = st.form_submit_button("Send feedback")

    if submitted:
        payload = {
            "classification_id": state.classification_id,
            "reviewer": reviewer or "Analyst",
            "notes": notes,
            "quality_score": quality_score,
        }
        try:
            response = requests.post(f"{api_base}/feedback", json=payload, timeout=30)
        except requests.RequestException as exc:
            st.error(f"Failed to send feedback: {exc}")
            return

        if response.status_code == 200:
            st.success("Feedback recorded for auditing.")
        else:  # pragma: no cover - defensive
            st.error(f"Backend rejected feedback: {response.status_code} {response.text}")


def main() -> None:
    """Entry point for running the Streamlit dashboard."""

    st.set_page_config(
        page_title="Document Classification Dashboard",
        page_icon="🗂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    context = _render_sidebar()
    _apply_theme(context["dark_mode"])

    st.title("Interactive document classification")
    st.caption("Upload documents, monitor progress, review evidence, and capture feedback in one place.")

    latest_state: Optional[ClassificationState] = st.session_state.get("latest_result")

    with st.form("classification_form"):
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["txt", "pdf"],
            help="Accepted formats: plain text (.txt) or PDF (.pdf).",
        )
        manual_text = st.text_area(
            "Or paste document text",
            height=180,
            placeholder="Paste raw content if no file is provided.",
        )
        document_name = st.text_input(
            "Document identifier",
            value=(uploaded_file.name if uploaded_file else "inline-document"),
        )
        submitted = st.form_submit_button("Run classification", type="primary")

    if submitted:
        document_text = _extract_text(uploaded_file, manual_text)
        latest_state = _run_classification(
            context["api_base"], document_text, document_name.strip()
        )

    if latest_state:
        summary_cols = st.columns(4)
        summary_cols[0].metric("Predicted label", latest_state.label)
        summary_cols[1].metric("Confidence", f"{latest_state.score * 100:.1f}%")
        summary_cols[2].metric(
            "Verifier agreement",
            "Yes" if latest_state.verifier_agreement else "No",
        )
        summary_cols[3].metric(
            "Safety flags",
            ", ".join(latest_state.safety_flags) if latest_state.safety_flags else "None",
        )

        viewer_tab, citations_tab, downloads_tab, feedback_tab, analytics_tab = st.tabs(
            [
                "Document viewer",
                "Citations",
                "Reports",
                "Feedback",
                "Analytics",
            ]
        )

        with viewer_tab:
            st.markdown(_highlight_document(latest_state.document_text, latest_state.citations), unsafe_allow_html=True)

        with citations_tab:
            _render_citation_legend(latest_state.citations)
            if latest_state.citations:
                st.json(latest_state.citations)

        with downloads_tab:
            _render_downloads(latest_state)

        with feedback_tab:
            _render_feedback(latest_state, context["api_base"])

        with analytics_tab:
            _render_history_chart(st.session_state.get("history", []))
    else:
        st.info("Submit a document to unlock document viewer, reports, and analytics.")
        _render_history_chart(st.session_state.get("history", []))


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
