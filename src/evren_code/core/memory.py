"""Project Memory Manager (`ProjectMemoryManager`).

Persists critical learned project insights (build commands, architecture rules,
frequent bug resolutions, developer preferences) to `.evren/memory.md` and injects them
into context across future agent sessions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from evren_code.config import AppConfig
from evren_code.tools.registry import ToolRegistry


class ProjectMemoryManager:
    """Manages reading and updating project memory (`.evren/memory.md`)."""

    def __init__(self, workspace_root: Path, max_chars: int = 4000) -> None:
        self.workspace_root = workspace_root.resolve()
        self.memory_file = self.workspace_root / ".evren" / "memory.md"
        self.max_chars = max_chars

    def load_memory(self) -> str:
        """Reads persisted project memory context."""
        if not self.memory_file.is_file():
            return ""
        try:
            content = self.memory_file.read_text(encoding="utf-8", errors="replace").strip()
            if len(content) > self.max_chars:
                content = content[-self.max_chars :]
            return content
        except OSError:
            return ""

    def remember_fact(self, fact: str, category: str = "General") -> str:
        """Appends new verified project context to `.evren/memory.md` without duplicate lines."""
        cleaned_fact = fact.strip()
        if not cleaned_fact:
            return "Hata: Kaydedilecek bilgi (`fact`) boş olamaz."

        cleaned_cat = category.strip() or "Genel"
        existing = self.load_memory()

        if cleaned_fact.lower() in existing.lower():
            return f"Bilgi zaten proje hafızasında mevcut: '{cleaned_fact}'"

        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            entry = f"- **[{cleaned_cat}]** ({timestamp}): {cleaned_fact}\n"

            if not existing:
                header = "# Evren-Code Persistent Project Memory\n\n"
                self.memory_file.write_text(header + entry, encoding="utf-8")
            else:
                with open(self.memory_file, "a", encoding="utf-8") as f:
                    if not existing.endswith("\n"):
                        f.write("\n")
                    f.write(entry)
            return f"Başarılı: Proje hafızasına kaydedildi -> [{cleaned_cat}] {cleaned_fact}"
        except OSError as exc:
            return f"Hata: Proje hafızası yazılamadı: {exc}"

    def query_memory(self, query: str, max_results: int = 5) -> str:
        """Query memory."""
        content = self.load_memory()
        if not content:
            return "Henüz kayıtlı bir proje hafızası bulunmuyor."

        q_terms = [t.lower() for t in query.strip().split() if len(t) >= 2]
        if not q_terms:
            return content

        lines = [line.strip() for line in content.splitlines() if line.strip().startswith("- ")]
        if not lines:
            return content

        import difflib
        scored = []
        for line in lines:
            line_lower = line.lower()
            hits = sum(1 for t in q_terms if t in line_lower)
            sim = difflib.SequenceMatcher(None, " ".join(q_terms), line_lower).ratio()
            score = (hits * 2.0) + sim
            if score > 0.3:
                scored.append((score, line))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for _, item in scored[:max_results]]

        if not top:
            return f"'{query}' konusuyla ilgili proje hafızasında doğrudan eşleşen bir kayıt bulunamadı."

        return (
            f"🧠 **[Proje Hafızası Eşleşmeleri - '{query}']**:\n"
            + "\n".join(top)
        )

    def clear_memory(self) -> bool:
        """Clear memory."""
        try:
            if self.memory_file.is_file():
                self.memory_file.unlink()
            return True
        except OSError:
            return False


def register_memory_tools(
    registry: ToolRegistry,
    memory_manager: ProjectMemoryManager,
) -> None:
    """Register memory tools."""

    def remember_project_fact(fact: str, category: str = "Mimari") -> str:
        return memory_manager.remember_fact(fact=fact, category=category)

    def get_project_memory() -> str:
        mem = memory_manager.load_memory()
        if not mem:
            return "Henüz kayıtlı bir kalıcı proje hafızası bulunmuyor."
        return mem

    registry.register(
        name="remember_project_fact",
        description=(
            "Persistently saves a critical project fact, build/test command, architectural rule, "
            "or user preference to '.evren/memory.md' for future sessions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "Clear and concise technical rule (e.g. 'Backend tests run via pytest -m unit').",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category (e.g. 'Build', 'Architecture', 'Database', 'User Preference').",
                },
            },
            "required": ["fact"],
        },
        handler=remember_project_fact,
        requires_approval=False,
    )

    def query_project_memory(query: str, max_results: Optional[int] = 5) -> str:
        limit = int(max_results or 5)
        return memory_manager.query_memory(query=query, max_results=limit)

    registry.register(
        name="query_project_memory",
        description="Searches persistent project memory for relevant facts, configs, or rules by keyword or topic.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword or technical topic."},
                "max_results": {"type": "integer", "description": "Maximum entries to return (default: 5)."},
            },
            "required": ["query"],
        },
        handler=query_project_memory,
        requires_approval=False,
    )

    registry.register(
        name="get_project_memory",
        description="Reads the entire persistent project memory file ('.evren/memory.md').",
        parameters={"type": "object", "properties": {}},
        handler=get_project_memory,
        requires_approval=False,
    )
