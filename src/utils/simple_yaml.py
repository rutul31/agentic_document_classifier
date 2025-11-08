"""Minimal YAML loader supporting a subset of YAML features."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def parse_block_scalar(
    lines: List[str], start: int, indent: int, style: str
) -> Tuple[str, int]:
    """Parse a YAML block scalar (``|`` or ``>``)."""

    collected: List[str] = []
    index = start

    while index < len(lines):
        raw_line = lines[index]
        current_indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        if not stripped:
            if current_indent < indent:
                break
            collected.append("")
            index += 1
            continue

        if current_indent < indent:
            break

        collected.append(raw_line[indent:])
        index += 1

    if style == ">":
        folded: List[str] = []
        buffer: List[str] = []
        for line in collected:
            if line == "":
                if buffer:
                    folded.append(" ".join(buffer))
                    buffer = []
                folded.append("")
            else:
                buffer.append(line)
        if buffer:
            folded.append(" ".join(buffer))
        value = "\n".join(folded)
    else:
        value = "\n".join(collected)

    return value, index


def parse_scalar(value: str) -> Any:
    """Parse a scalar YAML value into a Python object."""

    if value in {"null", "~", "None"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_block(lines: List[str], start: int, indent: int) -> Tuple[Any, int]:
    """Parse a YAML block recursively."""

    items: List[Any] = []
    mapping: Dict[str, Any] = {}
    is_list = False

    index = start
    while index < len(lines):
        raw_line = lines[index]
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            index += 1
            continue

        current_indent = len(raw_line) - len(raw_line.lstrip(" "))
        if current_indent < indent:
            break

        line = raw_line.strip()
        if line.startswith("- "):
            if not is_list:
                is_list = True
                items = []
            value = line[2:].strip()
            if not value:
                nested, index = parse_block(lines, index + 1, indent + 2)
                items.append(nested)
                continue
            if value in {"|", ">"}:
                scalar, index = parse_block_scalar(
                    lines, index + 1, indent + 2, value
                )
                items.append(scalar)
                continue
            if ":" in value:
                key, remainder = value.split(":", 1)
                item: Dict[str, Any] = {}
                normalized = remainder.strip()
                if normalized and normalized not in {"|", ">"}:
                    item[key.strip()] = parse_scalar(normalized)
                next_index = index + 1
                if next_index < len(lines):
                    next_line = lines[next_index]
                    next_indent = len(next_line) - len(next_line.lstrip(" "))
                    if next_indent >= indent + 2:
                        if normalized in {"|", ">"}:
                            scalar, index = parse_block_scalar(
                                lines, next_index, indent + 2, normalized
                            )
                            item[key.strip()] = scalar
                            items.append(item)
                            continue
                        nested, index = parse_block(lines, next_index, indent + 2)
                        if isinstance(nested, dict):
                            item.update(nested)
                        else:
                            item[key.strip()] = nested
                        items.append(item)
                        continue
                if not normalized:
                    nested, index = parse_block(lines, index + 1, indent + 2)
                    item[key.strip()] = nested
                    items.append(item)
                    continue
                if normalized in {"|", ">"}:
                    scalar, index = parse_block_scalar(
                        lines, index + 1, indent + 2, normalized
                    )
                    item[key.strip()] = scalar
                    items.append(item)
                    continue
                items.append(item)
                index += 1
                continue
            items.append(parse_scalar(value))
            index += 1
        else:
            if is_list:
                break
            if ":" not in line:
                raise ValueError(f"Invalid line: {line}")
            key, remainder = line.split(":", 1)
            key = key.strip()
            remainder = remainder.strip()
            if remainder in {"|", ">"}:
                scalar, index = parse_block_scalar(
                    lines, index + 1, indent + 2, remainder
                )
                mapping[key] = scalar
            elif remainder:
                mapping[key] = parse_scalar(remainder)
                index += 1
            else:
                nested, index = parse_block(lines, index + 1, indent + 2)
                mapping[key] = nested

    return (items if is_list else mapping), index


def load(path: str) -> Any:
    """Load YAML content from a file path using a minimal parser."""

    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read().splitlines()
    data, _ = parse_block(content, 0, 0)
    return data


__all__ = ["load"]
