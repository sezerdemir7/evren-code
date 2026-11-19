"""Context and Memory Management Module.

- Consolidates system prompt and project rules (`EVREN.md`, `AGENTS.md`, `.evrenrules`).
- Manages conversation history (`user`, `assistant`, `tool`).
- Ensures strict message invariants for all LLM providers (preventing HTTP 400 errors).
- Truncates and compacts history when approaching `context_length` limits.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional
import uuid

from evren_code.config import AppConfig, ModelSpec
from evren_code.core.parser import ToolCallItem, sanitize_thinking_for_context

logger = logging.getLogger(__name__)


def ensure_tool_call_invariants(messages: List[Dict[str, Any]], task_prompt: str = "") -> List[Dict[str, Any]]:
    """Guarantees tool/assistant message pairings to prevent HTTP 400 errors across all LLM providers.

    Rules:
    1. Message history must never start with an orphan `role == 'tool'` message.
       Any tool messages preceding their corresponding assistant message are discarded.
    2. If an `assistant` message contains `tool_calls`:
       - Subsequent `tool` messages matching these IDs must be present.
       - If a tool response is missing (e.g. truncated or pruned),
         a synthetic tool response is injected:
         `{"role": "tool", "tool_call_id": id, "content": "[Tool call output archived during context optimization]"}`
    3. Orphan `tool` messages following messages without `tool_calls` are safely cleaned.
    4. Ensures at least one `role == 'user'` message exists in the conversation history.
    """
    if not messages:
        return []

    sanitized: List[Dict[str, Any]] = []
    i = 0
    n = len(messages)

    while i < n:
        msg = messages[i]
        role = msg.get("role")

        # Rule 1: Skip leading or detached orphan tool messages
        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            is_valid_tool = False
            if sanitized:
                for prev in reversed(sanitized):
                    if prev.get("role") == "assistant":
                        expected_ids = {
                            tc.get("id")
                            for tc in prev.get("tool_calls", [])
                            if isinstance(tc, dict) and tc.get("id")
                        }
                        if tool_call_id in expected_ids:
                            is_valid_tool = True
                        break
                    elif prev.get("role") != "tool":
                        break
            if is_valid_tool:
                sanitized.append(msg)
            else:
                logger.debug("Orphan tool message removed from context (tool_call_id: %s)", tool_call_id)
            i += 1
            continue

        if role == "assistant" and msg.get("tool_calls"):
            tcalls = msg.get("tool_calls", [])
            expected_ids = [
                tc.get("id")
                for tc in tcalls
                if isinstance(tc, dict) and tc.get("id")
            ]
            sanitized.append(msg)
            i += 1

            # Collect tool messages following this assistant
            actual_tool_msgs: Dict[str, Dict[str, Any]] = {}
            while i < n and messages[i].get("role") == "tool":
                t_msg = messages[i]
                t_id = t_msg.get("tool_call_id")
                if t_id and t_id in expected_ids:
                    actual_tool_msgs[t_id] = t_msg
                i += 1

            # Append for each expected ID; inject synthetic closer if missing
            for exp_id in expected_ids:
                if exp_id in actual_tool_msgs:
                    sanitized.append(actual_tool_msgs[exp_id])
                else:
                    logger.debug("Injected synthetic closer for missing tool_call_id '%s'", exp_id)
                    sanitized.append({
                        "role": "tool",
                        "tool_call_id": exp_id,
                        "content": "[Tool call output archived during context optimization]",
                    })
            continue

        sanitized.append(msg)
        i += 1

    # Rule 4: Guarantee at least one user message
    has_user = any(m.get("role") == "user" for m in sanitized)
    if not has_user and sanitized:
        fallback_prompt = (task_prompt or "").strip() or "Execute user instructions."
        logger.debug("Injected anchor user message as none was found in sanitized history.")
        sanitized.insert(0, {
            "role": "user",
            "content": f"[User Goal]: {fallback_prompt}",
        })

    return sanitized


_TIKTOKEN_ENC: Any = None


def _get_tiktoken_encoding():
    """Caches tiktoken encoder at module level to prevent redundant loads."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is False:
        return None
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENC = False
            return None
    return _TIKTOKEN_ENC


def estimate_text_tokens(text: str) -> int:
    """Safe conservative token estimator accounting for symbol density and non-ASCII characters."""
    if not text:
        return 0
    enc = _get_tiktoken_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass

    length = len(text)
    if length == 0:
        return 0

    symbol_chars = sum(1 for ch in text if ch in "{}[]();:<>=/\\\"'_-&*|!+~^%#@$\n\t")
    non_ascii_chars = sum(1 for ch in text if ord(ch) > 127)
    symbol_ratio = (symbol_chars + non_ascii_chars) / length

    if symbol_ratio > 0.25:
        chars_per_tok = 2.2
    elif symbol_ratio > 0.12:
        chars_per_tok = 2.5
    else:
        chars_per_tok = 2.8

    return int(length / chars_per_tok) + 1


estimate_tokens = estimate_text_tokens


class ContextManager:
    """Conversation history, token estimation, and context compaction manager."""

    def __init__(
        self,
        config: AppConfig,
        base_system_prompt: str,
    ) -> None:
        self.config = config
        self.base_system_prompt = base_system_prompt
        self._messages: List[Dict[str, Any]] = []
        self._summary_memory: str = ""
        self._current_task_prompt: str = ""
        self._workspace_fingerprint: Optional[str] = None
        self._project_rules_cache: Optional[str] = None
        self._project_memory_cache: Optional[str] = None
        self._real_token_count: int = 0
        self._real_token_msg_idx: int = 0

    @property
    def raw_messages(self) -> List[Dict[str, Any]]:
        return list(self._messages)

    @property
    def summary_memory(self) -> str:
        return self._summary_memory

    def restore_state(
        self,
        messages: List[Dict[str, Any]],
        summary_memory: str = "",
    ) -> None:
        """Restores conversation history and summary memory from a saved session."""
        deduped = self._deduplicate_repetitive_tool_messages(messages)
        normalized = self._normalize_legacy_xml_messages(deduped)
        self._messages = normalized
        self._summary_memory = summary_memory

    @staticmethod
    def _normalize_legacy_xml_messages(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize legacy xml messages."""
        import re
        import uuid

        normalized: List[Dict[str, Any]] = []
        last_tool_id_map: Dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # 1. Convert XML tool_call in assistant message to native tool_calls
            if role == "assistant" and isinstance(content, str) and "<tool_call>" in content:
                xml_matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL)
                if xml_matches:
                    converted_calls = []
                    for raw_json in xml_matches:
                        try:
                            parsed = json.loads(raw_json.strip())
                            call_name = parsed.get("name", "unknown_tool")
                            call_args = parsed.get("arguments", {})
                            call_id = f"call_{uuid.uuid4().hex[:8]}"
                            last_tool_id_map[call_name] = call_id
                            converted_calls.append({
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": call_name,
                                    "arguments": json.dumps(call_args, ensure_ascii=False) if isinstance(call_args, dict) else str(call_args),
                                },
                            })
                        except Exception:
                            pass

                    if converted_calls:
                        clean_text = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
                        normalized.append({
                            "role": "assistant",
                            "content": clean_text or None,
                            "tool_calls": converted_calls,
                        })
                        continue

            if role == "user" and isinstance(content, str) and (content.startswith("[Araç Sonucu: ") or content.startswith("[Tool Result: ")):
                tool_match = re.match(r"\[(?:Araç Sonucu|Tool Result):\s*([a-zA-Z0-9_\-]+)\]", content)
                tool_name = tool_match.group(1) if tool_match else "unknown"
                tool_id = last_tool_id_map.get(tool_name, f"call_{uuid.uuid4().hex[:8]}")
                clean_result = content
                if tool_match:
                    clean_result = content[tool_match.end():].strip()
                clean_result = re.sub(r"\n\n(?:Görev tamamlanmadıysa bir sonraki aracı çağır|If the task is not complete, call next tool).*$", "", clean_result, flags=re.DOTALL).strip()

                normalized.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": clean_result,
                })
                continue

            normalized.append(msg)

        return normalized

    @staticmethod
    def _deduplicate_repetitive_tool_messages(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Removes consecutive repetitive tool calls and results from conversation history."""
        if len(messages) < 4:
            return list(messages)

        deduped: List[Dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if i + 1 < len(messages):
                next_msg = messages[i + 1]
                sig = f"{msg.get('role')}:{str(msg.get('content'))[:200]}"
                next_sig = f"{next_msg.get('role')}:{str(next_msg.get('content'))[:200]}"

                j = i + 2
                has_duplicates = False
                while j + 1 < len(messages):
                    cand_msg = messages[j]
                    cand_next = messages[j + 1]
                    cand_sig = f"{cand_msg.get('role')}:{str(cand_msg.get('content'))[:200]}"
                    cand_next_sig = f"{cand_next.get('role')}:{str(cand_next.get('content'))[:200]}"
                    if cand_sig == sig and cand_next_sig == next_sig:
                        has_duplicates = True
                        j += 2
                    else:
                        break

                if has_duplicates:
                    i = j - 2
                    deduped.append(messages[i])
                    deduped.append(messages[i + 1])
                    i += 2
                    continue

            deduped.append(msg)
            i += 1

        return deduped

    def load_project_rules(self, active_filepath: Optional[str] = None) -> str:
        """Reads project rules (`EVREN.md`, `AGENTS.md`, `.evrenrules`) from workspace and subdirectories."""
        rule_filenames = set(self.config.agent.project_rules_filenames) | {
            "EVREN.md", "AGENTS.md", ".cursorrules", ".windsurfrules"
        }
        rules_parts: List[str] = []
        seen_paths: Set[Path] = set()

        # 1. Root directory rules
        for filename in rule_filenames:
            candidate = self.config.workspace_root / filename
            if candidate.is_file() and candidate not in seen_paths:
                seen_paths.add(candidate)
                try:
                    content = candidate.read_text(encoding="utf-8", errors="replace").strip()
                    if content:
                        rules_parts.append(f"### Project Rule / Proje Kuralı (`{filename}`):\n{content}")
                except OSError:
                    pass

        # 2. Hierarchical subdirectory rules
        if active_filepath:
            try:
                p = (self.config.workspace_root / active_filepath).resolve()
                parent = p.parent if p.is_file() else p
                while parent and parent != self.config.workspace_root and parent.is_relative_to(self.config.workspace_root):
                    for filename in ("EVREN.md", "AGENTS.md"):
                        sub_candidate = parent / filename
                        if sub_candidate.is_file() and sub_candidate not in seen_paths:
                            seen_paths.add(sub_candidate)
                            content = sub_candidate.read_text(encoding="utf-8", errors="replace").strip()
                            if content:
                                rel_name = sub_candidate.relative_to(self.config.workspace_root)
                                rules_parts.append(f"### Directory Rule / Dizin Kuralı (`{rel_name}`):\n{content}")
                    parent = parent.parent
            except Exception:
                pass

        self._project_rules_cache = "\n\n".join(rules_parts)
        return self._project_rules_cache

    def auto_learn_user_preference(self, user_text: str) -> Optional[str]:
        """Detects strict user instructions or preferences and persists them to EVREN.md (Auto-Memory)."""
        triggers = [
            "bundan sonra her zaman",
            "asla unutma",
            "kural olarak ekle",
            "her zaman şöyle yap",
            "bu projede her zaman",
            "always use",
            "never use",
            "remember that",
            "from now on",
        ]
        lower_t = user_text.lower()
        matched = any(tr in lower_t for tr in triggers)
        if not matched:
            return None

        evren_md_path = self.config.workspace_root / "EVREN.md"
        rule_entry = f"\n- [Learned Rule / Öğrenilen Kural]: {user_text.strip()}\n"
        try:
            if evren_md_path.is_file():
                existing = evren_md_path.read_text(encoding="utf-8", errors="replace")
                evren_md_path.write_text(existing + rule_entry, encoding="utf-8")
            else:
                evren_md_path.write_text(f"# Project Rules (EVREN.md)\n{rule_entry}", encoding="utf-8")
            self.reload_project_rules()
            return f"Saved new rule to `EVREN.md`: {user_text.strip()}"
        except OSError:
            return None

    def load_project_memory(self) -> str:
        """Reads persistent memory from `.evren/memory.md` in workspace."""
        if self._project_memory_cache is None:
            try:
                from evren_code.core.memory import ProjectMemoryManager
                mem = ProjectMemoryManager(self.config.workspace_root).load_memory()
                self._project_memory_cache = f"## Persistent Project Memory / Persistent Project Memory (.evren/memory.md)\n{mem}" if mem else ""
            except Exception:
                self._project_memory_cache = ""
        return self._project_memory_cache

    def reload_project_rules(self) -> None:
        """Refreshes project rules (`EVREN.md`) and persistent project memory caches."""
        self._project_rules_cache = None
        self._project_memory_cache = None
        self.load_project_rules()
        self.load_project_memory()

    def detect_workspace_fingerprint(self) -> str:
        """Detects workspace architecture and technology stack."""
        if self._workspace_fingerprint is None:
            try:
                from evren_code.core.fingerprint import WorkspaceFingerprinter
                self._workspace_fingerprint = WorkspaceFingerprinter(self.config.workspace_root).detect()
            except Exception:
                self._workspace_fingerprint = ""
        return self._workspace_fingerprint

    def build_system_message(
        self,
        extra_instructions: str = "",
        active_plan_markdown: str = "",
    ) -> Dict[str, Any]:
        """Combines base system prompt, workspace information, rules, and memory."""
        import os as _os
        os_label = "Windows (nt)" if _os.name == "nt" else "Linux/Unix"
        shell_label = "PowerShell 5.1 (Windows)" if _os.name == "nt" else "bash/sh"
        ws_info = (
            f"## Workspace Environment / Çalışma Ortamı\n"
            f"- Root Directory / Kök Dizin: `{self.config.workspace_root}`\n"
            f"- Operating System: `{os_label}` | Default Shell: `{shell_label}`\n"
        )
        if _os.name == "nt":
            ws_info += (
                "- **PowerShell Guidelines**: Do not use Unix commands (`ls -la`, `wc -l`, `tail`, `touch`, `cat`) "
                "in terminal commands (`run_command`)! Use standard PowerShell cmdlets (`Get-ChildItem`, `Get-Content`, `Select-String`, `Measure-Object`)."
            )

        sections = [
            self.base_system_prompt.strip(),
            ws_info,
        ]
        fingerprint = self.detect_workspace_fingerprint()
        if fingerprint:
            sections.append(fingerprint)
        project_rules = self.load_project_rules()
        if project_rules:
            sections.append(project_rules)
        project_memory = self.load_project_memory()
        if project_memory:
            sections.append(project_memory)
        if self._summary_memory:
            sections.append(
                "## Previous Conversation & Discovery Summary (Compressed Context / Önceki Konuşma ve Keşif Özeti)\n"
                "CRITICAL RULE (ÖNEMLİ KURAL): The summary below reflects previous exploratory analysis, inspected files, and architecture notes. "
                "Do NOT redundantly re-invoke tools (`get_project_tree`, `get_project_memory`, or re-reading identical files) for answers already present here; "
                "build directly upon these findings!\n\n"
                f"{self._summary_memory}"
            )
        if active_plan_markdown:
            sections.append(active_plan_markdown.strip())
        if extra_instructions:
            sections.append(extra_instructions.strip())

        return {
            "role": "system",
            "content": "\n\n".join(sections),
        }

    @staticmethod
    def _clean_assistant_summary_text(text: str, max_chars: int = 4000) -> str:
        """Strips thinking blocks (<think>, [Muhakeme: ...]) and verification report headers from assistant text."""
        if not text:
            return ""
        cleaned = text.strip()
        # 1. Strip leading VerificationSummaryReport block
        if "\n\n---\n\n" in cleaned and ("Doğrulama" in cleaned[:400] or "Verification" in cleaned[:400] or "Test" in cleaned[:400]):
            cleaned = cleaned.split("\n\n---\n\n", 1)[1].strip()
        # 2. Clean thinking blocks
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        cleaned = re.sub(r"\[(?:Düşünce|Muhakeme|Reasoning|Thinking)\s*:.*?\]", "", cleaned, flags=re.DOTALL).strip()
        if len(cleaned) > max_chars:
            half = max_chars // 2
            cleaned = f"{cleaned[:half]}\n... [Summary Truncated / Özet Kırpıldı] ...\n{cleaned[-half:]}"
        return cleaned

    def _extract_turn_discovery_notes(self) -> str:
        """Summarizes explored files, skeletons, and discovery findings from turn messages."""
        explored_items: List[str] = []
        key_findings: List[str] = []

        for m in self._messages:
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                for tc in m.get("tool_calls", []):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}

                    if name in ("read_file", "skeletonize_code", "get_file_symbols", "edit_file", "write_file"):
                        p = args.get("path") or args.get("file_path")
                        if p:
                            tag = f"{name}({p})"
                            if tag not in explored_items:
                                explored_items.append(tag)
                    elif name == "read_multiple_files":
                        files = args.get("files") or args.get("paths") or []
                        if isinstance(files, list):
                            for f in files[:10]:
                                f_path = f.get("path") if isinstance(f, dict) else str(f)
                                if f_path:
                                    tag = f"read({f_path})"
                                    if tag not in explored_items:
                                        explored_items.append(tag)
                    elif name == "get_project_tree":
                        p = args.get("path", ".")
                        tag = f"tree({p})"
                        if tag not in explored_items:
                            explored_items.append(tag)
            elif role == "tool":
                t_name = m.get("name", "")
                content = str(m.get("content") or "").strip()
                if t_name in ("get_project_tree", "skeletonize_code", "read_multiple_files") and content:
                    first_lines = [ln.strip() for ln in content.splitlines()[:6] if ln.strip()]
                    snippet = " | ".join(first_lines)[:300]
                    if snippet and snippet not in key_findings:
                        key_findings.append(f"[{t_name}]: {snippet}")

        parts: List[str] = []
        if explored_items:
            parts.append(f"  - Explored Files & Tools / İncelenen Dosyalar ve Araçlar: {', '.join(explored_items[:25])}")
        if key_findings:
            parts.append("  - Discovery Findings / Keşif Bulguları: " + " ; ".join(key_findings[:5]))
        return "\n".join(parts)

    def start_new_task(self, prompt: Any) -> None:
        """Initializes a new user task while preventing cross-task context contamination and preserving summary."""
        if self._messages:
            first_user_msg = ""
            for m in self._messages:
                if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
                    u_txt = m["content"].strip()
                    if not u_txt.startswith("["):
                        first_user_msg = u_txt[:300]
                        break

            last_assistant_msg = ""
            for m in reversed(self._messages):
                if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
                    cleaned_asst = self._clean_assistant_summary_text(m["content"], max_chars=4000)
                    if cleaned_asst:
                        last_assistant_msg = cleaned_asst
                        break

            discovery_notes = self._extract_turn_discovery_notes()

            bullet_lines: List[str] = []
            if first_user_msg:
                bullet_lines.append(f"- Previous User Prompt / Önceki Kullanıcı İstemi: {first_user_msg}")
            if discovery_notes:
                bullet_lines.append(discovery_notes)
            if last_assistant_msg:
                bullet_lines.append(f"- Completed Task Summary / Tamamlanan Görev Özeti: {last_assistant_msg}")
            else:
                bullet_lines.append("- Completed Task Summary / Tamamlanan Görev Özeti: Previous task completed. (Önceki görev tamamlandı.)")

            summary_bullet = "\n".join(bullet_lines)
            if self._summary_memory:
                combined = f"{self._summary_memory}\n\n{summary_bullet}"
                if len(combined) > 12000:
                    combined = "... [Previous task summaries truncated / Önceki görev özetleri kırpıldı]\n" + combined[-12000:]
                self._summary_memory = combined
            else:
                self._summary_memory = summary_bullet

            self._messages.clear()
            self._real_token_count = 0
            self._real_token_msg_idx = 0

        self._current_task_prompt = str(prompt) if prompt else ""
        self.add_user_message(prompt)

    def clear_task_context(self) -> None:
        """Clears active task messages while preserving persistent summary memory."""
        self._messages.clear()
        self._real_token_count = 0
        self._real_token_msg_idx = 0

    def add_user_message(self, content: Any) -> None:
        """Appends a user message to conversation history."""
        if isinstance(content, str) and content.strip():
            self._current_task_prompt = content.strip()
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[ToolCallItem]] = None,
        native_tools_supported: bool = True,
        thinking: str = "",
    ) -> None:
        """Appends assistant response, thinking summary, and optional tool calls to history."""
        thinking_note = ""
        if thinking and thinking.strip():
            compact_thk = sanitize_thinking_for_context(thinking, max_chars=200)
            if compact_thk:
                thinking_note = f"[Muhakeme]: {compact_thk}"

        if tool_calls and native_tools_supported:
            serialized_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    c_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    fn = tc.get("function", {})
                    fn_name = fn.get("name") if isinstance(fn, dict) else tc.get("name", "tool")
                    raw_args = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments", {})
                    arg_str = json.dumps(raw_args, ensure_ascii=False) if isinstance(raw_args, dict) else str(raw_args)
                    serialized_calls.append({
                        "id": c_id,
                        "type": "function",
                        "function": {"name": fn_name, "arguments": arg_str},
                    })
                else:
                    arg_str = json.dumps(tc.arguments, ensure_ascii=False) if isinstance(tc.arguments, dict) else str(tc.arguments)
                    serialized_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": arg_str,
                        },
                    })
            clean_content = content or ""
            if any(getattr(tc, "from_xml", False) if not isinstance(tc, dict) else tc.get("from_xml", False) for tc in tool_calls):
                import re
                clean_content = re.sub(r"<tool_call>.*?</tool_call>", "", clean_content, flags=re.DOTALL).strip()

            if not clean_content and thinking_note:
                clean_content = thinking_note

            msg: Dict[str, Any] = {
                "role": "assistant",
                "content": clean_content or None,
                "tool_calls": serialized_calls,
            }
            self._messages.append(msg)
        elif tool_calls and not native_tools_supported:
            xml_blocks = [
                f"<tool_call>{json.dumps({'name': tc.name, 'arguments': tc.arguments}, ensure_ascii=False)}</tool_call>"
                for tc in tool_calls
            ]
            eff_content = content or thinking_note
            combined = (eff_content + "\n" if eff_content else "") + "\n".join(xml_blocks)
            self._messages.append({"role": "assistant", "content": combined})
        else:
            self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result_text: str,
        native_tools_supported: bool = True,
        from_xml: bool = False,
    ) -> None:
        """Appends tool execution result to message history."""
        if native_tools_supported:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result_text,
                }
            )
        else:
            self._messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[Tool Result / Araç Sonucu: {tool_name}]\n{result_text}\n\n"
                        "If the task is not yet complete, invoke the next tool or provide your final response to the user."
                    ),
                }
            )

    def get_messages(
        self,
        extra_system_instructions: str = "",
        active_plan_markdown: str = "",
    ) -> List[Dict[str, Any]]:
        """Returns the complete message payload for the LLM API, optimized for KV prefix caching."""
        if not self._messages:
            return [
                self.build_system_message(
                    extra_instructions=extra_system_instructions,
                    active_plan_markdown=active_plan_markdown,
                )
            ]

        sys_msg = self.build_system_message(
            extra_instructions=extra_system_instructions,
            active_plan_markdown="",
        )
        history = [{k: v for k, v in m.items() if not k.startswith("_")} for m in self._messages]

        if active_plan_markdown and history:
            last_msg = history[-1]
            last_content = last_msg.get("content")
            if isinstance(last_content, str):
                last_msg["content"] = f"{last_content}\n\n{active_plan_markdown.strip()}"
            elif last_content is None and last_msg.get("role") == "assistant":
                last_msg["content"] = active_plan_markdown.strip()

        sanitized_history = ensure_tool_call_invariants(history, task_prompt=self._current_task_prompt)
        return [sys_msg] + sanitized_history

    @property
    def history_length(self) -> int:
        return len(self._messages)

    def record_usage(self, usage: Optional[Dict[str, int]]) -> None:
        """Record usage."""
        if not usage:
            return
        self._last_usage = usage
        prompt_t = int(usage.get("prompt_tokens") or 0)
        total_t = int(usage.get("total_tokens") or 0)
        effective_t = max(prompt_t, total_t)
        if effective_t > 0:
            self._real_token_count = effective_t
            self._real_token_msg_idx = len(self._messages)

    @classmethod
    def _truncate_json_obj(cls, obj: Any, max_str_len: int) -> Any:
        """Truncate json obj."""
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in ("path", "file_path", "id", "name", "type"):
                    out[k] = v
                else:
                    out[k] = cls._truncate_json_obj(v, max_str_len)
            return out
        if isinstance(obj, list):
            if len(obj) > 10:
                trimmed_list = [cls._truncate_json_obj(item, max_str_len) for item in obj[:5]]
                trimmed_list.append(f"... [{len(obj) - 5} items truncated for context savings / bağlam tasarrufu için kırpıldı]")
                return trimmed_list
            return [cls._truncate_json_obj(item, max_str_len) for item in obj]
        if isinstance(obj, str) and len(obj) > max_str_len:
            half = max(100, max_str_len // 2)
            return (
                f"{obj[:half]}\n"
                f"... [Legacy tool arguments truncated ({len(obj)} -> {max_str_len} chars) / bağlam tasarrufu için kırpıldı] ...\n"
                f"{obj[-half:]}"
            )
        return obj

    def evict_large_tool_results(
        self,
        max_chars_per_result: int = 1500,
        keep_recent: Optional[int] = None,
        file_tracker: Optional[Any] = None,
    ) -> int:
        """Progressive eviction strategy: Truncates oversized past tool results and assistant arguments."""
        evicted_count = 0
        assistant_indices = [
            i for i, m in enumerate(self._messages)
            if m.get("role") == "assistant"
        ]
        if len(assistant_indices) >= 2:
            protected_turn_boundary = assistant_indices[-2]
        elif assistant_indices:
            protected_turn_boundary = assistant_indices[-1]
        else:
            protected_turn_boundary = len(self._messages)

        if keep_recent is not None:
            safe_boundary = min(max(0, len(self._messages) - keep_recent), protected_turn_boundary)
        else:
            safe_boundary = protected_turn_boundary

        for idx in range(safe_boundary):
            msg = self._messages[idx]
            role = msg.get("role")
            if role == "tool" or (role == "user" and ("tool_call_id" in str(msg.get("content", "")) or str(msg.get("content", "")).startswith("[Araç Sonucu:") or str(msg.get("content", "")).startswith("[Tool Result:"))):
                content = str(msg.get("content") or "")
                if len(content) > max_chars_per_result:
                    if file_tracker is not None:
                        file_matches = re.findall(r"(?:Dosya|File):\s*([^\s\(\)]+)", content)
                        for fpath in file_matches:
                            try:
                                file_tracker.invalidate(fpath.strip())
                            except Exception:
                                pass

                    head = content[: max_chars_per_result // 2]
                    tail = content[-max_chars_per_result // 2 :]
                    msg["content"] = (
                        f"{head}\n\n... [Past tool output truncated for context savings / bağlam tasarrufu için kırpıldı "
                        f"({len(content)} bytes -> {max_chars_per_result} bytes)] ...\n\n{tail}"
                    )
                    evicted_count += 1

            elif role == "assistant":
                tcalls = msg.get("tool_calls")
                if isinstance(tcalls, list):
                    for tc in tcalls:
                        fn = tc.get("function") if isinstance(tc, dict) else None
                        if isinstance(fn, dict):
                            raw_args = fn.get("arguments")
                            if isinstance(raw_args, str) and len(raw_args) > max_chars_per_result:
                                try:
                                    parsed_args = json.loads(raw_args)
                                    if file_tracker is not None and isinstance(parsed_args, dict):
                                        for pk in ("path", "file_path"):
                                            pval = parsed_args.get(pk)
                                            if isinstance(pval, str) and pval.strip():
                                                file_tracker.invalidate(pval.strip())
                                        f_list = parsed_args.get("files") or parsed_args.get("paths")
                                        if isinstance(f_list, list):
                                            for item in f_list:
                                                if isinstance(item, dict) and isinstance(item.get("path"), str):
                                                    file_tracker.invalidate(item["path"].strip())
                                                elif isinstance(item, str) and item.strip():
                                                    file_tracker.invalidate(item.strip())
                                    compacted_args = self._truncate_json_obj(parsed_args, max_chars_per_result // 2)
                                    fn["arguments"] = json.dumps(compacted_args, ensure_ascii=False)
                                except Exception:
                                    half = max_chars_per_result // 2
                                    fn["arguments"] = json.dumps(
                                        {"_truncated_arguments": f"{raw_args[:half]} ... {raw_args[-half:]}"},
                                        ensure_ascii=False,
                                    )
                                evicted_count += 1

                asst_content = msg.get("content")
                if isinstance(asst_content, str) and len(asst_content) > max_chars_per_result * 2:
                    limit = max_chars_per_result * 2
                    half = limit // 2
                    msg["content"] = (
                        f"{asst_content[:half]}\n\n"
                        f"... [Past assistant message truncated for context savings / bağlam tasarrufu için kırpıldı ({len(asst_content)} -> {limit} chars)] ...\n\n"
                        f"{asst_content[-half:]}"
                    )
                    evicted_count += 1

            elif role == "user":
                usr_content = msg.get("content")
                if isinstance(usr_content, str) and len(usr_content) > max_chars_per_result * 3:
                    limit = max_chars_per_result * 3
                    half = limit // 2
                    msg["content"] = (
                        f"{usr_content[:half]}\n\n"
                        f"... [Past message truncated for context savings / bağlam tasarrufu için kırpıldı ({len(usr_content)} -> {limit} chars)] ...\n\n"
                        f"{usr_content[-half:]}"
                    )
                    evicted_count += 1

        if evicted_count > 0:
            self._real_token_count = 0
            self._real_token_msg_idx = 0
        return evicted_count

    def compact_sliding_window_tools(
        self,
        keep_recent_turns: int = 3,
        max_chars_per_result: int = 400,
    ) -> int:
        """Compact sliding window tools (Sliding Window Historical Tool Eviction)."""
        assistant_indices = [
            i for i, m in enumerate(self._messages)
            if m.get("role") == "assistant"
        ]
        if len(assistant_indices) <= keep_recent_turns:
            return 0

        cutoff_idx = assistant_indices[-keep_recent_turns]
        evicted_count = 0

        for idx in range(cutoff_idx):
            msg = self._messages[idx]
            role = msg.get("role")
            if role == "tool" or (
                role == "user"
                and (
                    "tool_call_id" in str(msg.get("content", ""))
                    or str(msg.get("content", "")).startswith("[Araç Sonucu:")
                    or str(msg.get("content", "")).startswith("[Tool Result:")
                )
            ):
                content = str(msg.get("content") or "")
                if "... [Auto-Compacted" in content or "... [Past tool output truncated" in content:
                    continue
                if len(content) > max_chars_per_result:
                    first_line = content.strip().splitlines()[0] if content.strip().splitlines() else "Tool Output"
                    if len(first_line) > 120:
                        first_line = first_line[:120] + "..."
                    msg["content"] = (
                        f"{first_line}\n\n"
                        f"... [Auto-Compacted / Bağlam Tasarrufu İçin Özetlendi: {len(content)} karakter — İçerik hafızaya alındı] ..."
                    )
                    evicted_count += 1

        if evicted_count > 0:
            self._real_token_count = 0
            self._real_token_msg_idx = 0
        return evicted_count

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        return estimate_text_tokens(text)

    @classmethod
    def _get_message_tokens(cls, msg: Dict[str, Any]) -> int:
        """Calculates approximate tokens for a single message and caches with `_cached_tokens` ($O(1)$)."""
        cached = msg.get("_cached_tokens")
        if isinstance(cached, int):
            return cached
        clean_msg = {k: v for k, v in msg.items() if not k.startswith("_")}
        tok = cls._estimate_text_tokens(json.dumps(clean_msg, ensure_ascii=False))
        msg["_cached_tokens"] = tok
        return tok

    def estimate_tokens(self) -> int:
        """Estimates approximate tokens in current context with safe conservative margin."""
        if self._real_token_count > 0 and self._real_token_msg_idx <= len(self._messages):
            delta_tokens = sum(
                self._get_message_tokens(msg)
                for msg in self._messages[self._real_token_msg_idx :]
            )
            return self._real_token_count + delta_tokens

        base_tokens = (
            self._estimate_text_tokens(self.base_system_prompt)
            + self._estimate_text_tokens(self._summary_memory)
            + self._estimate_text_tokens(self.load_project_rules())
            + self._estimate_text_tokens(self.load_project_memory())
            + self._estimate_text_tokens(self.detect_workspace_fingerprint())
        )
        msg_tokens = sum(
            self._get_message_tokens(msg)
            for msg in self._messages
        )
        return base_tokens + msg_tokens

    def needs_compaction(self, model: ModelSpec) -> bool:
        """Checks if context compaction is needed based on model limits."""
        raw_window = max(16384, model.context_length - model.max_output_tokens)
        max_working_limit = getattr(self.config.agent, "max_working_context_tokens", 96000)
        usable_window = min(raw_window, max_working_limit) if max_working_limit > 0 else raw_window
        threshold = int(usable_window * self.config.agent.compaction_threshold_ratio)
        return (
            len(self._messages) > 2
            and self.estimate_tokens() >= threshold
        )

    def compact(
        self,
        summarizer_fn: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
        aggressive: bool = False,
        file_tracker: Optional[Any] = None,
    ) -> int:
        """Compacts older messages to reduce token usage. Returns number of compacted items."""
        # 1. Prune large tool results and assistant arguments first
        evict_keep = 2 if aggressive else None
        evict_max_chars = 600 if aggressive else 1500
        evicted = self.evict_large_tool_results(
            max_chars_per_result=evict_max_chars,
            keep_recent=evict_keep,
            file_tracker=file_tracker,
        )

        keep = 2 if aggressive else self.config.agent.keep_recent_messages
        if len(self._messages) <= keep:
            if len(self._messages) > 2:
                keep = max(2, len(self._messages) // 2)
            else:
                if aggressive:
                    evicted += self.evict_large_tool_results(max_chars_per_result=600, keep_recent=0)
                return evicted

        split_idx = len(self._messages) - keep
        while split_idx > 0 and self._messages[split_idx].get("role") == "tool":
            split_idx -= 1

        if split_idx <= 0:
            for candidate_idx in range(len(self._messages) - 1, 0, -1):
                if self._messages[candidate_idx].get("role") != "tool":
                    split_idx = candidate_idx
                    break

        if split_idx <= 0:
            if aggressive:
                evicted += self.evict_large_tool_results(max_chars_per_result=600, keep_recent=1)
            return evicted

        older_messages = self._messages[:split_idx]
        self._messages = ensure_tool_call_invariants(self._messages[split_idx:], task_prompt=self._current_task_prompt)
        self._real_token_count = 0
        self._real_token_msg_idx = 0

        if summarizer_fn is not None and not aggressive:
            try:
                new_summary = summarizer_fn(older_messages)
                if new_summary:
                    combined_summary = (
                        f"{self._summary_memory}\n{new_summary}".strip()
                        if self._summary_memory
                        else new_summary.strip()
                    )
                    if len(combined_summary) > 4000:
                        combined_summary = "... [Previous summary truncated / Önceki özet kırpıldı]\n" + combined_summary[-4000:]
                    self._summary_memory = combined_summary
                    return len(older_messages) + evicted
            except Exception as exc:
                logger.debug("Summarization call failed, falling back to rule-based summary: %s", exc)

        bullet_points: List[str] = []
        for m in older_messages:
            role = m.get("role", "")
            content = str(m.get("content") or "")[:200].replace("\n", " ").strip()
            if role in ("user", "assistant") and content:
                bullet_points.append(f"- [{role}]: {content}")
            elif role == "assistant" and m.get("tool_calls"):
                tc_names = [
                    tc.get("function", {}).get("name", "tool")
                    for tc in (m.get("tool_calls") or [])
                    if isinstance(tc, dict)
                ]
                if tc_names:
                    bullet_points.append(f"- [assistant tools]: {', '.join(tc_names)}")
        fallback_summary = "\n".join(bullet_points[-15:])
        combined_fallback = (
            f"{self._summary_memory}\n{fallback_summary}".strip()
            if self._summary_memory
            else fallback_summary
        )
        if len(combined_fallback) > 12000:
            combined_fallback = "... [Previous summary truncated / Önceki özet kırpıldı]\n" + combined_fallback[-12000:]
        self._summary_memory = combined_fallback
        return len(older_messages) + evicted

    def clear(self) -> None:
        """Resets conversation history and summary memory."""
        self._messages.clear()
        self._summary_memory = ""
        self._real_token_count = 0
        self._real_token_msg_idx = 0
