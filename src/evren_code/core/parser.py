"""Hybrid Tool Calling and Reasoning (<think>) Parser (`parser.py`).

Models on EVREN AI (GLM-5.3, DeepSeek-V4.1, Qwen 3.8, Gemma) may return standard OpenAI
`tool_calls` arrays, or emit `<tool_call>...</tool_call>` tags / JSON markdown blocks directly in text.
This module transparently captures both, auto-repairs truncated or malformed JSON payloads,
and extracts `<think>...</think>` reasoning chains.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolCallItem:
    """Normalized data structure representing a single tool invocation."""

    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""
    from_xml: bool = False
    is_truncated: bool = False


@dataclass
class ParsedAssistantResponse:
    """Parsed model response containing clean content, reasoning traces, and tool calls."""

    content: str
    thinking: str = ""
    tool_calls: List[ToolCallItem] = field(default_factory=list)
    repetition_detected: bool = False
    overthinking_detected: bool = False
    overthinking_reason: str = ""
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    is_truncated: bool = False


_RUMINATION_MARKERS_RE = re.compile(
    r"\b(?:hmm+|wait\s*[—\-:,]|actually\s*,|alternative\s+approach|let\s+me\s+reconsider|"
    r"let\s+me\s+think\s+about|better\s+approach|simplest\s+of\s+all|options\s*:\s*1\.)\b",
    re.IGNORECASE,
)
_CODE_FENCE_IN_THINK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n.*?(?:```|$)", re.DOTALL)


def analyze_reasoning_overthinking(
    thinking: str,
    max_clean_chars: int = 1200,
) -> Tuple[bool, str]:
    """Analyze reasoning overthinking (overthinking)."""
    if not thinking or not thinking.strip():
        return False, ""

    stripped = thinking.strip()
    reasons: List[str] = []

    if "```" in stripped:
        reasons.append("code_block_in_thinking")

    rumination_hits = _RUMINATION_MARKERS_RE.findall(stripped)
    if len(rumination_hits) >= 3:
        reasons.append(f"rumination_loop({len(rumination_hits)})")

    if len(stripped) > max_clean_chars:
        reasons.append(f"excessive_length({len(stripped)}>{max_clean_chars})")

    if reasons:
        return True, ", ".join(reasons)
    return False, ""


def sanitize_thinking_for_context(thinking: str, max_chars: int = 200) -> str:
    """Sanitize thinking for context."""
    if not thinking or not thinking.strip():
        return ""

    cleaned = _CODE_FENCE_IN_THINK_RE.sub(" ", thinking)

    kept_lines: List[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith((";", "{", "}")) and any(tok in line for tok in ("=", "(", ")", "public ", "private ", "var ", "const ", "def ", "return ")):
            continue
        if _RUMINATION_MARKERS_RE.search(line) and kept_lines:
            continue
        kept_lines.append(line)

    compact = " ".join(kept_lines).strip()
    compact = re.sub(r"\s+", " ", compact)
    if not compact:
        compact = re.sub(r"\s+", " ", cleaned).strip()

    if len(compact) <= max_chars:
        return compact

    head_len = max(80, (max_chars * 2) // 3)
    tail_len = max(40, max_chars - head_len - 5)
    return f"{compact[:head_len].rstrip()} ... {compact[-tail_len:].lstrip()}"


_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TOOL_XML_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json|tool_call)\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def _clean_trailing_comma(out_chars: List[str]) -> None:
    """Clean trailing comma."""
    idx = len(out_chars) - 1
    while idx >= 0 and out_chars[idx].isspace():
        idx -= 1
    if idx >= 0 and out_chars[idx] == ",":
        out_chars.pop(idx)


class StatefulJsonRepair:
    """StatefulJsonRepair: stateful lexer."""

    @classmethod
    def repair_with_status(cls, raw: str) -> Tuple[str, bool]:
        """Repair with status (is_truncated)."""
        if not raw or not raw.strip():
            return "{}", False

        text = raw.strip()

        start_obj = text.find("{")
        start_arr = text.find("[")
        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            text = text[start_obj:]
        elif start_arr != -1:
            text = text[start_arr:]
        else:
            return raw, False

        out: List[str] = []
        stack: List[str] = []

        in_string = False
        escape_active = False
        quote_char = '"'
        was_string_truncated = False

        i = 0
        n = len(text)

        while i < n:
            ch = text[i]

            if in_string:
                if escape_active:
                    out.append(ch)
                    escape_active = False
                elif ch == "\\":
                    out.append(ch)
                    escape_active = True
                elif ch == quote_char:
                    is_structural_close = True
                    if quote_char == '"':
                        k = i + 1
                        while k < n and text[k] in (" ", "\t", "\r", "\n"):
                            k += 1
                        if k < n and text[k] not in (":", ",", "}", "]"):
                            is_structural_close = False
                    if is_structural_close:
                        out.append('"')
                        in_string = False
                        quote_char = '"'
                    else:
                        out.append('\\"')
                elif ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    pass
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(ch)
            else:
                if ch in ('"', "'"):
                    in_string = True
                    quote_char = ch
                    out.append('"')
                elif ch == "{":
                    stack.append("}")
                    out.append(ch)
                elif ch == "[":
                    stack.append("]")
                    out.append(ch)
                elif ch == "}":
                    if stack and stack[-1] == "}":
                        stack.pop()
                    _clean_trailing_comma(out)
                    out.append(ch)
                elif ch == "]":
                    if stack and stack[-1] == "]":
                        stack.pop()
                    _clean_trailing_comma(out)
                    out.append(ch)
                else:
                    out.append(ch)

            i += 1

        if in_string:
            was_string_truncated = True
            if escape_active:
                if out and out[-1] == "\\":
                    out.pop()
            out.append('"')
            in_string = False

        _clean_trailing_comma(out)

        idx = len(out) - 1
        while idx >= 0 and out[idx].isspace():
            idx -= 1
        if idx >= 0 and out[idx] == ":":
            out.append(" null")
            was_string_truncated = True

        _clean_trailing_comma(out)

        if stack:
            was_string_truncated = True
        while stack:
            closing = stack.pop()
            out.append(closing)

        return "".join(out), was_string_truncated

    @classmethod
    def repair(cls, raw: str) -> str:
        repaired, _ = cls.repair_with_status(raw)
        return repaired


def repair_and_parse_json(raw: str) -> Dict[str, Any]:
    """Repair and parse json (stateful lexer)."""
    if not raw or not raw.strip():
        return {}

    text = raw.strip()
    # 1. Standart json.loads
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        pass

    try:
        repaired, is_trunc = StatefulJsonRepair.repair_with_status(text)
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            if is_trunc:
                parsed["_is_truncated"] = True
            return parsed
        res_dict: Dict[str, Any] = {"value": parsed}
        if is_trunc:
            res_dict["_is_truncated"] = True
        return res_dict
    except Exception:
        pass

    try:
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        repaired, is_trunc = StatefulJsonRepair.repair_with_status(cleaned)
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            if is_trunc:
                parsed["_is_truncated"] = True
            return parsed
        res_dict = {"value": parsed}
        if is_trunc:
            res_dict["_is_truncated"] = True
        return res_dict
    except Exception:
        pass

    return {"_raw_unparsed": raw}


def extract_thinking_blocks(content: str, explicit_reasoning: str = "") -> Tuple[str, str]:
    """Extract thinking blocks."""
    thinking_parts: List[str] = []
    if explicit_reasoning and explicit_reasoning.strip():
        thinking_parts.append(explicit_reasoning.strip())

    def _replacer(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if inner:
            thinking_parts.append(inner)
        return ""

    cleaned_content = _THINK_PATTERN.sub(_replacer, content or "").strip()
    return cleaned_content, "\n\n".join(thinking_parts)


def parse_assistant_message(
    message: Dict[str, Any],
    repetition_detected: bool = False,
    usage: Optional[Dict[str, int]] = None,
    finish_reason: Optional[str] = None,
) -> ParsedAssistantResponse:
    """Parse assistant message."""
    raw_content = message.get("content") or ""
    explicit_reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or ""
    )

    cleaned_content, thinking = extract_thinking_blocks(raw_content, explicit_reasoning)
    tool_calls: List[ToolCallItem] = []

    native_calls = message.get("tool_calls") or []
    for tc in native_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name")
        if not name:
            continue
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, dict):
            args = raw_args
            raw_args_str = json.dumps(raw_args, ensure_ascii=False)
            is_trunc = False
        else:
            raw_args_str = str(raw_args)
            args = repair_and_parse_json(raw_args_str)
            is_trunc = bool(args.pop("_is_truncated", False)) if isinstance(args, dict) else False
        tool_calls.append(
            ToolCallItem(
                id=tc.get("id") or f"call_{uuid.uuid4().hex[:10]}",
                name=name,
                arguments=args,
                raw_arguments=raw_args_str,
                is_truncated=is_trunc,
            )
        )

    if not tool_calls and cleaned_content:
        xml_matches = list(_TOOL_XML_PATTERN.finditer(cleaned_content))
        if xml_matches:
            for match in xml_matches:
                payload = repair_and_parse_json(match.group(1))
                name = payload.get("name") or payload.get("tool")
                if name:
                    args = payload.get("arguments") or payload.get("parameters")
                    if args is None:
                        args = {
                            k: v for k, v in payload.items()
                            if k not in ("name", "tool", "id", "type", "_is_truncated", "action")
                        }
                    is_trunc = False
                    if isinstance(args, str):
                        args = repair_and_parse_json(args)
                    if isinstance(args, dict):
                        is_trunc = bool(args.pop("_is_truncated", False)) or bool(payload.get("_is_truncated", False))
                    tool_calls.append(
                        ToolCallItem(
                            id=f"call_{uuid.uuid4().hex[:10]}",
                            name=str(name),
                            arguments=args if isinstance(args, dict) else {},
                            raw_arguments=json.dumps(args, ensure_ascii=False),
                            from_xml=True,
                            is_truncated=is_trunc,
                        )
                    )
            cleaned_content = _TOOL_XML_PATTERN.sub("", cleaned_content).strip()
        else:
            fence_matches = list(_JSON_FENCE_PATTERN.finditer(cleaned_content))
            for match in fence_matches:
                payload = repair_and_parse_json(match.group(1))
                name = payload.get("name") or payload.get("tool")
                if name:
                    args = payload.get("arguments") or payload.get("parameters")
                    if args is None:
                        args = {
                            k: v for k, v in payload.items()
                            if k not in ("name", "tool", "id", "type", "_is_truncated", "action")
                        }
                    is_trunc = False
                    if isinstance(args, str):
                        args = repair_and_parse_json(args)
                    if isinstance(args, dict):
                        is_trunc = bool(args.pop("_is_truncated", False)) or bool(payload.get("_is_truncated", False))
                    tool_calls.append(
                        ToolCallItem(
                            id=f"call_{uuid.uuid4().hex[:10]}",
                            name=str(name),
                            arguments=args if isinstance(args, dict) else {},
                            raw_arguments=json.dumps(args, ensure_ascii=False),
                            from_xml=True,
                            is_truncated=is_trunc,
                        )
                    )
            if tool_calls:
                cleaned_content = _JSON_FENCE_PATTERN.sub("", cleaned_content).strip()

        if not tool_calls and "<tool_call>" in cleaned_content:
            unclosed_match = re.search(r"<tool_call>\s*(.*)$", cleaned_content, re.DOTALL | re.IGNORECASE)
            if unclosed_match:
                payload = repair_and_parse_json(unclosed_match.group(1))
                name = payload.get("name") or payload.get("tool")
                if name:
                    args = payload.get("arguments") or payload.get("parameters")
                    if args is None:
                        args = {
                            k: v for k, v in payload.items()
                            if k not in ("name", "tool", "id", "type", "_is_truncated", "action")
                        }
                    is_trunc = False
                    if isinstance(args, str):
                        args = repair_and_parse_json(args)
                    if isinstance(args, dict):
                        is_trunc = bool(args.pop("_is_truncated", False)) or bool(payload.get("_is_truncated", False))
                    tool_calls.append(
                        ToolCallItem(
                            id=f"call_{uuid.uuid4().hex[:10]}",
                            name=str(name),
                            arguments=args if isinstance(args, dict) else {},
                            raw_arguments=json.dumps(args, ensure_ascii=False),
                            from_xml=True,
                            is_truncated=is_trunc,
                        )
                    )
                    cleaned_content = re.sub(r"<tool_call>.*$", "", cleaned_content, flags=re.DOTALL | re.IGNORECASE).strip()

    has_truncated_tc = any(getattr(tc, "is_truncated", False) for tc in tool_calls)
    is_truncated = (finish_reason == "length") or has_truncated_tc
    overthinking_detected, overthinking_reason = analyze_reasoning_overthinking(thinking)

    return ParsedAssistantResponse(
        content=cleaned_content,
        thinking=thinking,
        tool_calls=tool_calls,
        repetition_detected=repetition_detected,
        overthinking_detected=overthinking_detected,
        overthinking_reason=overthinking_reason,
        usage=usage,
        finish_reason=finish_reason,
        is_truncated=is_truncated,
    )
