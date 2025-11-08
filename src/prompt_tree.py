"""Prompt tree abstractions for orchestrating LLM interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

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


__all__ = ["PromptTree", "PromptNode"]
