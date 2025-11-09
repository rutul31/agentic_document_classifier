"""Interactive Streamlit dashboard for the document classification assistant."""

from __future__ import annotations

import html
import io
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

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

    identifier: str
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
    category: str
    page_count: int
    image_count: int
    evidence: List[Dict[str, Any]]
    content_safety: str
    llm_debug: Dict[str, Any] = field(default_factory=dict)
    single_prompt: bool = False


def _bucket_category(raw_label: str) -> str:
    lowered = (raw_label or "").lower()
    if "high" in lowered:
        return "Highly Sensitive"
    if "confidential" in lowered:
        return "Confidential"
    return "Public"


def _loading_indicator(label: str) -> str:
    safe_label = html.escape(label)
    return f"""
    <div class="loading-indicator">
        <div class="spinner-circle"></div>
        <span>{safe_label}</span>
    </div>
    """


def _info_card(title: str, value: str, subtitle: Optional[str] = None) -> str:
    subtitle_html = (
        f'<p class="info-card__subtitle">{html.escape(subtitle)}</p>'
        if subtitle
        else ""
    )
    return (
        f'<div class="info-card">'
        f'<p class="info-card__title">{html.escape(title)}</p>'
        f'<p class="info-card__value">{html.escape(value)}</p>'
        f"{subtitle_html}"
        f"</div>"
    )


def _normalise_evidence_items(evidence: List[Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for entry in evidence or []:
        if isinstance(entry, dict):
            items.append(entry)
        else:
            items.append({"type": "text", "content": str(entry)})
    return items


def _prepare_evidence_payload(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return _normalise_evidence_items(raw)
    if raw:
        return _normalise_evidence_items([raw])
    return []


def _render_evidence_block(
    evidence: List[Any],
    *,
    show_header: bool = True,
) -> None:
    if show_header:
        st.subheader("Evidence")
    normalised = _normalise_evidence_items(evidence)
    if not normalised:
        st.info("No evidence references were generated for this document.")
        return

    for index, item in enumerate(normalised[:10], start=1):
        item_type = (item.get("type") or "text").lower()
        if item_type == "image" and item.get("content"):
            bbox = item.get("bbox") or [0.02, 0.02, 0.96, 0.96]
            left = bbox[0] * 100
            top = bbox[1] * 100
            width = bbox[2] * 100
            height = bbox[3] * 100
            caption = item.get("description") or "Image evidence"
            page = item.get("page")
            caption_text = (
                f"{caption} • Page {page}" if page else caption
            )
            st.markdown(
                f"""
                <div class="image-evidence">
                    <div class="image-wrapper">
                        <img src="{item['content']}" alt="Image evidence {index}" />
                        <span class="bbox-overlay" style="left:{left}%; top:{top}%; width:{width}%; height:{height}%;"></span>
                    </div>
                    <p class="image-caption">{html.escape(caption_text)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            content = item.get("content") or ""
            description = item.get("description")
            page = item.get("page")
            subtitle = []
            if page:
                subtitle.append(f"Page {page}")
            if description:
                subtitle.append(description)
            subtitle_text = " · ".join(subtitle)
            subtitle_html = (
                f"<small>{html.escape(subtitle_text)}</small>" if subtitle_text else ""
            )
            st.markdown(
                f"""
                <div class="evidence-item text-evidence">
                    <span class="badge">{index}</span>
                    <div>
                        <p>{html.escape(str(content))}</p>
                        {subtitle_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _record_history(state: ClassificationState) -> None:
    entry = {
        "id": state.identifier,
        "document_name": state.document_name,
        "timestamp": state.timestamp,
        "label": state.label,
        "category": state.category,
        "category_bucket": _bucket_category(state.category or state.label),
        "score": state.score,
        "content_safety": state.content_safety,
        "page_count": state.page_count,
        "image_count": state.image_count,
        "reports": dict(state.reports),
        "evidence": list(state.evidence),
        "safety_flags": list(state.safety_flags),
        "llm_debug": dict(state.llm_debug or {}),
        "single_prompt": state.single_prompt,
    }
    history = st.session_state.setdefault("history", [])
    history.insert(0, entry)
    del history[25:]


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.utcnow()


def _render_history_panel(history: List[Dict[str, Any]]) -> None:
    st.subheader("Past documents")
    if not history:
        st.info("No past classifications yet. Run a document to build history.")
        return

    option_map: Dict[str, Dict[str, Any]] = {}
    option_labels: Dict[str, str] = {}
    for index, entry in enumerate(history):
        entry_id = entry.get("id") or f"legacy-{index}"
        option_map[entry_id] = entry
        timestamp = _coerce_timestamp(entry.get("timestamp"))
        bucket = entry.get("category_bucket") or _bucket_category(
            entry.get("category") or entry.get("label") or "Public"
        )
        name = entry.get("document_name") or "Document"
        option_labels[entry_id] = (
            f"{name} • {timestamp.strftime('%b %d %H:%M')} • {bucket}"
        )

    placeholder = "__none__"
    options = [placeholder] + list(option_map.keys())

    def _format_history_option(key: str) -> str:
        if key == placeholder:
            return "Choose a document…"
        return option_labels.get(key, key)

    selected_id = st.selectbox(
        "Select a document to preview",
        options,
        format_func=_format_history_option,
        key="history_select",
    )

    if selected_id == placeholder:
        st.info("Select a document above to view details.")
    else:
        selected = option_map[selected_id]
        timestamp = _coerce_timestamp(selected.get("timestamp"))
        bucket = selected.get("category_bucket") or _bucket_category(
            selected.get("category") or selected.get("label") or "Public"
        )

        with st.expander(
            f"Details for {option_labels.get(selected_id, 'document')}", expanded=False
        ):
            mode_label = "Single prompt" if selected.get("single_prompt") else "Prompt tree"
            st.caption(
                f"Processed on {timestamp.strftime('%Y-%m-%d %H:%M')} • {mode_label}"
            )
            cards = st.columns(4)
            cards[0].markdown(_info_card("Category", bucket), unsafe_allow_html=True)
            cards[1].markdown(
                _info_card("# of pages", str(selected.get("page_count", "—"))),
                unsafe_allow_html=True,
            )
            cards[2].markdown(
                _info_card("# of images", str(selected.get("image_count", "—"))),
                unsafe_allow_html=True,
            )
            cards[3].markdown(
                _info_card("Content safety", selected.get("content_safety", "Unknown")),
                unsafe_allow_html=True,
            )

            evidence = selected.get("evidence") or []
            _render_evidence_block(
                evidence if isinstance(evidence, list) else [], show_header=False
            )

            reports = selected.get("reports") or {}
            cols = st.columns(2)
            json_path = reports.get("json")
            pdf_path = reports.get("pdf")
            if json_path and Path(json_path).exists():
                try:
                    json_bytes = Path(json_path).read_bytes()
                    cols[0].download_button(
                        "Download JSON report",
                        data=json_bytes,
                        file_name=f"{selected.get('document_name','document')}-history.json",
                        mime="application/json",
                        key=f"history-json-{selected_id}",
                    )
                except OSError:
                    cols[0].info("JSON report unavailable.")
            else:
                cols[0].info("JSON report unavailable.")

            if pdf_path and Path(pdf_path).exists():
                try:
                    pdf_bytes = Path(pdf_path).read_bytes()
                    cols[1].download_button(
                        "Download PDF report",
                        data=pdf_bytes,
                        file_name=f"{selected.get('document_name','document')}-history.pdf",
                        mime="application/pdf",
                        key=f"history-pdf-{selected_id}",
                    )
                except OSError:
                    cols[1].info("PDF report unavailable.")
            else:
                cols[1].info("PDF report unavailable.")

    table_rows = []
    for entry in history[:10]:
        ts = _coerce_timestamp(entry.get("timestamp"))
        mode = "Single Prompt" if entry.get("single_prompt") else "Prompt Tree"
        table_rows.append(
            {
                "Document": entry.get("document_name", "Document"),
                "Category": entry.get("category_bucket")
                or _bucket_category(entry.get("label") or "Public"),
                "Score": f"{entry.get('score', 0.0):.2f}",
                "Safety": entry.get("content_safety", "Unknown"),
                "Mode": mode,
                "Run at": ts.strftime("%Y-%m-%d %H:%M"),
            }
        )
    st.dataframe(table_rows, use_container_width=True)

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
        .info-card {{
            background-color: var(--panel-bg);
            border: 1px solid rgba(127, 90, 240, 0.25);
            border-radius: 0.75rem;
            padding: 1rem 1.2rem;
            min-height: 120px;
        }}
        .info-card__title {{
            margin: 0;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: rgba(255, 255, 255, 0.7);
        }}
        .info-card__value {{
            margin: 0.1rem 0;
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--accent-color);
        }}
        .info-card__subtitle {{
            margin: 0;
            font-size: 0.9rem;
            color: var(--viewer-text);
            opacity: 0.85;
        }}
        .loading-indicator {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: 600;
            color: var(--accent-color);
        }}
        .spinner-circle {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 3px solid rgba(127, 90, 240, 0.25);
            border-top-color: var(--accent-color);
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        .evidence-list {{
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }}
        .evidence-item {{
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            padding: 0.75rem;
            border: 1px solid rgba(127, 90, 240, 0.2);
            border-radius: 0.6rem;
            background: var(--panel-bg);
        }}
        .text-evidence p {{
            margin-bottom: 0.2rem;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--accent-color);
            color: #fff;
            font-weight: 600;
        }}
        .image-evidence {{
            margin-bottom: 1rem;
        }}
        .image-wrapper {{
            position: relative;
            border-radius: 0.6rem;
            overflow: hidden;
            border: 1px solid rgba(127, 90, 240, 0.2);
        }}
        .image-wrapper img {{
            width: 100%;
            display: block;
        }}
        .bbox-overlay {{
            position: absolute;
            border: 2px solid var(--accent-color);
            border-radius: 0.25rem;
            pointer-events: none;
        }}
        .image-caption {{
            margin: 0.4rem 0 0;
            font-size: 0.9rem;
            color: var(--viewer-text);
        }}
        .hero-card {{
            background: linear-gradient(135deg, rgba(127,90,240,0.25), rgba(22,189,202,0.2));
            border-radius: 1rem;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .hero-card h2 {{
            margin-bottom: 0.5rem;
        }}
        .hero-metrics {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }}
        .hero-metrics div {{
            background: var(--panel-bg);
            border-radius: 0.75rem;
            padding: 0.75rem;
            border: 1px solid rgba(127, 90, 240, 0.2);
            text-align: center;
        }}
        .hero-metrics strong {{
            font-size: 1.4rem;
            color: var(--accent-color);
        }}
        .citation-card {{
            border: 1px solid rgba(127, 90, 240, 0.25);
            border-radius: 0.75rem;
            padding: 0.9rem 1rem;
            background: var(--panel-bg);
            margin-bottom: 0.8rem;
        }}
        .citation-card__header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.4rem;
            font-size: 0.9rem;
            color: rgba(255,255,255,0.7);
        }}
        .citation-card__chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.4rem;
        }}
        .chip {{
            display: inline-flex;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: rgba(127, 90, 240, 0.18);
            font-size: 0.8rem;
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

    dark_mode = st.sidebar.toggle("Dark mode", value=stored_dark_mode)
    single_prompt_mode = st.sidebar.toggle(
        "Single prompt mode", value=st.session_state.get("single_prompt_mode", False)
    )
    with st.sidebar.expander("Advanced settings", expanded=False):
        api_base = st.text_input("API base URL", value=stored_api)

    st.session_state["api_base"] = api_base
    st.session_state["dark_mode"] = dark_mode
    st.session_state["single_prompt_mode"] = single_prompt_mode

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


def _run_classification(
    api_base: str,
    text: str,
    document_name: str,
    uploaded_file: Optional[Any],
) -> Optional[ClassificationState]:
    """Send the document to the backend and return structured state."""

    if not text:
        st.warning("Provide document text or upload a file before running classification.")
        return None

    endpoint = f"{api_base}/classify"
    single_prompt_mode = bool(st.session_state.get("single_prompt_mode", False))

    with st.status("Running classification", expanded=True) as status:
        progress_placeholder = st.empty()
        progress = progress_placeholder.progress(0)
        job_status_placeholder = st.empty()
        spinner_placeholder = st.empty()
        progress.progress(5)
        status.write("Preparing document bundle…")

        document_ids: List[str] = []
        if uploaded_file is not None:
            status.write("Uploading document to the classification service…")
            file_bytes = uploaded_file.getvalue()
            files = [
                (
                    "files",
                    (
                        uploaded_file.name,
                        file_bytes,
                        uploaded_file.type or "application/octet-stream",
                    ),
                )
            ]
            try:
                upload_response = requests.post(
                    f"{api_base}/upload", files=files, timeout=120
                )
            except requests.RequestException as exc:
                status.update(label="Upload failed", state="error")
                st.error(f"Unable to upload document: {exc}")
                return None

            if upload_response.status_code != 200:
                status.update(label="Upload failed", state="error")
                st.error(
                    f"Upload failed with status {upload_response.status_code}: {upload_response.text}"
                )
                return None

            upload_payload = upload_response.json()
            document_ids = upload_payload.get("document_ids", [])
            if not document_ids:
                status.update(label="Upload failed", state="error")
                st.error("The upload endpoint did not return document identifiers.")
                return None
            progress.progress(20)

        if document_ids:
            payload = {"document_ids": document_ids}
        else:
            payload = {"text": text, "document_name": document_name or None}
        payload["single_prompt"] = single_prompt_mode

        status.write("Submitting classification job…")

        try:
            response = requests.post(endpoint, json=payload, timeout=120)
        except requests.RequestException as exc:
            status.update(label="Network error", state="error")
            st.error(f"Failed to contact the classification service: {exc}")
            return None

        if response.status_code != 200:
            status.update(label="Submission failed", state="error")
            st.error(f"Classification request failed: {response.text}")
            return None

        job_payload = response.json()
        task_id = job_payload.get("task_id")
        if not task_id:
            status.update(label="Submission failed", state="error")
            st.error("Classification service did not return a task identifier.")
            return None

        status.write("Awaiting classification results…")
        spinner_placeholder.markdown(
            _loading_indicator("Processing"), unsafe_allow_html=True
        )
        last_status: Optional[str] = None

        poll_url = f"{api_base}/status/{task_id}"
        poll_timeout = 300.0
        poll_interval = 1.0
        start_time = time.time()
        result_payload: Optional[Dict[str, Any]] = None

        while True:
            try:
                status_response = requests.get(poll_url, timeout=60)
            except requests.RequestException as exc:
                spinner_placeholder.empty()
                status.update(label="Status polling failed", state="error")
                st.error(f"Failed to fetch job status: {exc}")
                return None

            if status_response.status_code != 200:
                spinner_placeholder.empty()
                status.update(label="Status polling failed", state="error")
                st.error(
                    f"Status endpoint returned {status_response.status_code}: {status_response.text}"
                )
                return None

            status_payload = status_response.json()
            progress_value = status_payload.get("progress") or 0.0
            progress.progress(max(20, int(progress_value * 100)))
            current_status = (status_payload.get("status") or "queued").title()
            if current_status != last_status:
                job_status_placeholder.markdown(
                    f"**Job status:** {current_status}", unsafe_allow_html=True
                )
                last_status = current_status

            if status_payload.get("status") == "failed":
                spinner_placeholder.empty()
                status.update(label="Classification failed", state="error")
                st.error(status_payload.get("error") or "Classification job failed.")
                return None

            if status_payload.get("status") == "completed":
                results = status_payload.get("results") or []
                if not results:
                    spinner_placeholder.empty()
                    status.update(label="Classification failed", state="error")
                    st.error("Classification completed but no results were returned.")
                    return None
                result_payload = results[0]
                break

            if time.time() - start_time > poll_timeout:
                spinner_placeholder.empty()
                status.update(label="Classification timed out", state="error")
                st.error("Classification did not complete within the allotted time.")
                return None

            time.sleep(poll_interval)

        progress.progress(100)
        spinner_placeholder.empty()
        status.update(label="Classification complete", state="complete")
        progress_placeholder.empty()

    if result_payload is None:
        st.error("Classification job completed without returning a result.")
        return None

    evidence_payload = _prepare_evidence_payload(result_payload.get("evidence", []))
    llm_debug = result_payload.get("llm_debug") or {}

    classification_state = ClassificationState(
        identifier=uuid4().hex,
        label=result_payload.get("label", "Unknown"),
        score=float(result_payload.get("score", 0.0)),
        citations=result_payload.get("citations", []),
        safety_flags=result_payload.get("safety_flags", []),
        verifier_agreement=bool(result_payload.get("verifier_agreement", False)),
        classification_id=result_payload.get("classification_id"),
        reports=result_payload.get("reports", {}),
        document_text=text,
        document_name=document_name or (uploaded_file.name if uploaded_file else "document"),
        timestamp=datetime.utcnow(),
        category=result_payload.get("category", result_payload.get("label", "Unknown")),
        page_count=int(result_payload.get("page_count", 0) or 0),
        image_count=int(result_payload.get("image_count", 0) or 0),
        evidence=evidence_payload,
        content_safety=result_payload.get("content_safety", "Unknown"),
        llm_debug=llm_debug,
        single_prompt=bool(result_payload.get("single_prompt", False)),
    )

    _record_history(classification_state)
    st.session_state["latest_result"] = classification_state
    return classification_state

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

    evidence_payload = _prepare_evidence_payload(result_payload.get("evidence", []))
    llm_debug = result_payload.get("llm_debug") or {}

    classification_state = ClassificationState(
        identifier=uuid4().hex,
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
        category=result_payload.get("category", result_payload.get("label", "Unknown")),
        page_count=int(result_payload.get("page_count", 0) or 0),
        image_count=int(result_payload.get("image_count", 0) or 0),
        evidence=evidence_payload,
        content_safety=result_payload.get("content_safety", "Unknown"),
        llm_debug=llm_debug,
        single_prompt=bool(result_payload.get("single_prompt", False)),
    )

    _record_history(classification_state)
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


def _render_citation_cards(citations: List[Dict[str, Any]]) -> None:
    if not citations:
        st.info("No citations were generated for this document.")
        return

    for index, citation in enumerate(citations, start=1):
        snippet = citation.get("snippet") or "No snippet provided."
        page = citation.get("page") or "—"
        metadata = citation.get("metadata") or {}
        chips = []
        if metadata.get("label"):
            chips.append(metadata["label"])
        if metadata.get("signal"):
            chips.append(metadata["signal"])
        chips_html = "".join(
            f"<span class='chip'>{html.escape(str(chip))}</span>" for chip in chips
        )
        st.markdown(
            f"""
            <div class="citation-card">
                <div class="citation-card__header">
                    <strong>Citation {index}</strong>
                    <span>Page {page}</span>
                </div>
                <p>{html.escape(snippet)}</p>
                <div class="citation-card__chips">{chips_html}</div>
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
        "category": state.category,
        "page_count": state.page_count,
        "image_count": state.image_count,
        "content_safety": state.content_safety,
        "evidence": state.evidence,
        "llm_debug": state.llm_debug,
        "single_prompt_mode": state.single_prompt,
    }
    st.download_button(
        "Download JSON report",
        data=json.dumps(json_payload, indent=2, default=str).encode("utf-8"),
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

    counts = Counter(
        _bucket_category(entry.get("category") or entry.get("label") or "Public")
        for entry in history
    )
    ordered_labels = ["Highly Sensitive", "Confidential", "Public"]
    labels = [label for label in ordered_labels if label in counts] or list(counts.keys())
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
        decision_display = st.selectbox(
            "Decision",
            ["Accept", "Override", "Reclassify", "Flag"],
            index=0,
        )
        suggested_label = st.text_input(
            "Suggested label (optional)",
            help="Provide the correct label if this classification should change.",
        )
        quality_score = st.slider(
            "Quality score", min_value=0.0, max_value=1.0, value=0.8
        )
        notes = st.text_area(
            "Feedback notes", help="Highlight corrections or additional context."
        )
        submitted = st.form_submit_button("Send feedback")

    if submitted:
        decision_value = decision_display.lower()
        payload = {
            "classification_id": state.classification_id,
            "reviewer": reviewer or "Analyst",
            "decision": decision_value,
            "notes": notes,
            "quality_score": quality_score,
            "suggested_label": suggested_label or None,
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

    history: List[Dict[str, Any]] = st.session_state.get("history", [])
    total_docs = len(history)
    last_bucket = (
        history[0].get("category_bucket")
        if history
        else "—"
    )
    unsafe_count = sum(
        1
        for entry in history
        if (entry.get("category_bucket") or "").lower().startswith("high")
    )

    hero_cols = st.columns([2, 1])
    hero_cols[0].markdown(
        """
        <div class="hero-card">
            <h2>Document Risk Command Center</h2>
            <p>Monitor classification outcomes and trace the exact evidence—text or imagery—that triggered each decision.</p>
            <ul>
                <li>Real-time dual-LLM verification</li>
                <li>Visual redaction cues for sensitive screenshots</li>
                <li>Persistent audit trail of past submissions</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    hero_cols[1].markdown(
        f"""
        <div class="hero-metrics">
            <div>
                <p>Total documents</p>
                <strong>{total_docs}</strong>
            </div>
            <div>
                <p>Last category</p>
                <strong>{last_bucket}</strong>
            </div>
            <div>
                <p>High-risk cases</p>
                <strong>{unsafe_count}</strong>
            </div>
            <div>
                <p>Session mode</p>
                <strong>{"Dark" if context["dark_mode"] else "Light"}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            context["api_base"], document_text, document_name.strip(), uploaded_file
        )

    if latest_state:
        bucketed = _bucket_category(latest_state.category or latest_state.label)
        primary_cols = st.columns(4)
        primary_cols[0].markdown(
            _info_card("Category", bucketed), unsafe_allow_html=True
        )
        primary_cols[1].markdown(
            _info_card("# of pages", str(latest_state.page_count)), unsafe_allow_html=True
        )
        primary_cols[2].markdown(
            _info_card("# of images", str(latest_state.image_count)),
            unsafe_allow_html=True,
        )
        primary_cols[3].markdown(
            _info_card("Content safety", latest_state.content_safety),
            unsafe_allow_html=True,
        )

        secondary_cols = st.columns(4)
        secondary_cols[0].markdown(
            _info_card("Confidence", f"{latest_state.score * 100:.1f}%"),
            unsafe_allow_html=True,
        )
        secondary_cols[1].markdown(
            _info_card(
                "Verifier agreement",
                "Yes" if latest_state.verifier_agreement else "No",
            ),
            unsafe_allow_html=True,
        )
        safety_text = (
            ", ".join(latest_state.safety_flags) if latest_state.safety_flags else "None"
        )
        secondary_cols[2].markdown(
            _info_card("Safety flags", safety_text), unsafe_allow_html=True
        )
        mode_label = "Single prompt" if latest_state.single_prompt else "Prompt tree"
        secondary_cols[3].markdown(
            _info_card("Mode", mode_label, subtitle=latest_state.document_name),
            unsafe_allow_html=True,
        )

        _render_evidence_block(latest_state.evidence)

        viewer_tab, citations_tab, downloads_tab, feedback_tab = st.tabs(
            [
                "Document viewer",
                "Citations",
                "Reports",
                "Feedback",
            ]
        )

        with viewer_tab:
            st.markdown(_highlight_document(latest_state.document_text, latest_state.citations), unsafe_allow_html=True)

        with citations_tab:
            _render_citation_legend(latest_state.citations)
            _render_citation_cards(latest_state.citations)

        with downloads_tab:
            _render_downloads(latest_state)

        with feedback_tab:
            _render_feedback(latest_state, context["api_base"])

    else:
        st.info("Submit a document to unlock document viewer, reports, and analytics.")

    st.markdown("---")
    _render_history_panel(history)
    _render_history_chart(history)


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
