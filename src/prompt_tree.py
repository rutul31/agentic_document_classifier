"""Prompt tree abstractions for orchestrating LLM interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from jinja2 import Environment, StrictUndefined

from .utils.logger import get_logger
from .utils.simple_yaml import load as load_yaml

LOGGER = get_logger(__name__)


@dataclass
class PromptNode:
    """A node in the prompt tree representing a classification step."""

    name: str
    prompt: str
    children: List["PromptNode"] = field(default_factory=list)

    def find(self, name: str) -> Optional["PromptNode"]:
        """Find a node by name within the subtree rooted at this node."""

        if self.name == name:
            return self
        for child in self.children:
            result = child.find(name)
            if result:
                return result
        return None

    def iter_prompts(self) -> Iterable["PromptNode"]:
        """Iterate over the node and all of its descendants."""

        yield self
        for child in self.children:
            yield from child.iter_prompts()

    def add_child(self, child: "PromptNode") -> None:
        """Attach a new child node to the current node."""

        self.children.append(child)


class PromptTree:
    """Encapsulates a tree of prompts used to query LLMs."""

    def __init__(self, root: PromptNode):
        self.root = root

    @classmethod
    def from_dict(cls, payload: Dict) -> "PromptTree":
        """Build a prompt tree from a dictionary representation."""

        def _build(node: Dict) -> PromptNode:
            children = [_build(child) for child in node.get("children", [])]
            return PromptNode(
                name=node["name"], prompt=node["prompt"], children=children
            )

        return cls(root=_build(payload))

    @classmethod
    def from_yaml(cls, path: str) -> "PromptTree":
        """Load a prompt tree from a YAML file."""

        LOGGER.debug("Loading prompt tree from %s", path)
        data = load_yaml(path)
        return cls.from_dict(data)

    def to_dict(self) -> Dict:
        """Serialize the tree into a dictionary."""

        def _serialize(node: PromptNode) -> Dict:
            return {
                "name": node.name,
                "prompt": node.prompt,
                "children": [_serialize(child) for child in node.children],
            }

        return _serialize(self.root)


PROMPT_LIBRARY_PATH = Path("config/prompts_library.yaml")
PROMPT_TRACE_PATH = Path("/logs/prompt_trace.log")

_JINJA_ENV = Environment(undefined=StrictUndefined, autoescape=False)


def _render_template(template: str, metadata: Dict) -> str:
    """Render a prompt template using document metadata."""

    context = {"doc": metadata or {}}
    if metadata:
        context.update(metadata)
    return _JINJA_ENV.from_string(template).render(context)


def _detect_content_type(metadata: Dict) -> str:
    """Infer the document content type from metadata."""

    declared = (metadata or {}).get("content_type")
    if declared in {"text", "image", "mixed"}:
        return declared

    has_text = False
    has_images = False
    if metadata:
        has_text = any(
            bool(metadata.get(key))
            for key in ("has_text", "text_length", "word_count", "character_count")
        )
        has_images = any(
            bool(metadata.get(key))
            for key in ("has_images", "image_count", "pages_with_images")
        )

    if has_text and has_images:
        return "mixed"
    if has_images:
        return "image"
    return "text"


def _append_dynamic_branches(root: PromptNode, metadata: Dict) -> None:
    """Attach content-type specific nodes to the prompt tree."""

    content_type = _detect_content_type(metadata)
    LOGGER.debug("Detected content type: %s", content_type)

    text_prompt = (
        "TC2 – PII Exposure Review:\n"
        "Document title: {{ doc.title | default('Unknown document') }}.\n"
        "Summarise detected personal identifiers (names, emails, government IDs) and\n"
        "rate severity on a 1-5 scale."
    )
    image_prompt = (
        "TC4 – Image Classification:\n"
        "Evaluate all provided imagery (count: {{ doc.image_count | default(0) }}).\n"
        "Identify scene type, presence of people, and any compliance risks."
    )

    if content_type in {"text", "mixed"}:
        rendered = _render_template(text_prompt, metadata)
        root.add_child(PromptNode(name="tc2_pii_review", prompt=rendered))
    if content_type in {"image", "mixed"}:
        rendered = _render_template(image_prompt, metadata)
        root.add_child(PromptNode(name="tc4_image_classification", prompt=rendered))


def _ensure_log_destination() -> None:
    """Make sure the prompt trace log exists."""

    PROMPT_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PROMPT_TRACE_PATH.exists():
        PROMPT_TRACE_PATH.touch()


def _log_prompt(node: PromptNode) -> None:
    """Append the prompt content to the trace log."""

    _ensure_log_destination()
    with PROMPT_TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{node.name}]\n{node.prompt}\n\n")


def _apply_templates(node: PromptNode, metadata: Dict) -> None:
    """Apply templating to the entire tree."""

    node.prompt = _render_template(node.prompt, metadata)
    for child in node.children:
        _apply_templates(child, metadata)


def _serialise_tree(node: PromptNode, depth: int = 0) -> List[str]:
    """Create a text representation of the tree."""

    indent = "  " * depth
    lines = [f"{indent}- ({node.name}) {node.prompt}"]
    for child in node.children:
        lines.extend(_serialise_tree(child, depth + 1))
    return lines


def build_prompt_tree(doc_metadata: Dict) -> str:
    """Build and serialise a prompt tree tailored to a document."""

    LOGGER.debug("Building prompt tree with metadata: %s", doc_metadata)
    tree = PromptTree.from_yaml(str(PROMPT_LIBRARY_PATH))
    _apply_templates(tree.root, doc_metadata or {})
    _append_dynamic_branches(tree.root, doc_metadata or {})

    for node in tree.root.iter_prompts():
        _log_prompt(node)

    return "\n".join(_serialise_tree(tree.root))


__all__ = ["PromptTree", "PromptNode", "build_prompt_tree"]
