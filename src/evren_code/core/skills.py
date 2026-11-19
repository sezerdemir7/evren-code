"""Evren-Code Custom Skills and Commands Engine (`skills.py`).

Generates dynamic slash commands (e.g. `/test-all`, `/deploy`, `/refactor`) from markdown
definitions stored in `.evren/commands/*.md` or `.evren/skills/*.md`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CustomSkill:
    """Individual custom skill or slash command definition."""
    name: str          # "deploy"
    command: str       # "/deploy"
    description: str   # "Builds and deploys project"
    template: str      # Prompt template
    file_path: Path


class SkillManager:
    """Discovers and manages project and global custom skills."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()
        self.skills: Dict[str, CustomSkill] = {}
        self.reload()

    def reload(self) -> None:
        """Reload."""
        self.skills.clear()

        search_dirs = [
            self.workspace_root / ".evren" / "commands",
            self.workspace_root / ".evren" / "skills",
            Path.home() / ".evren-code" / "commands",
        ]

        for sdir in search_dirs:
            if not sdir.is_dir():
                continue
            for file in sdir.glob("*.md"):
                skill = self._parse_skill_file(file)
                if skill:
                    self.skills[skill.command] = skill

    def _parse_skill_file(self, file_path: Path) -> Optional[CustomSkill]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            desc = "Özel kullanıcı komutu"
            body = content

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    body = parts[2].strip()
                    for line in frontmatter.splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            if k.strip().lower() == "description":
                                desc = v.strip().strip("\"'")

            cmd_name = file_path.stem.lower()
            return CustomSkill(
                name=cmd_name,
                command=f"/{cmd_name}",
                description=desc,
                template=body,
                file_path=file_path,
            )
        except Exception:
            return None

    def get_command_list(self) -> List[str]:
        """Get command list."""
        return list(self.skills.keys())

    def render(self, command_str: str, args: str = "") -> Optional[str]:
        """Render."""
        clean_cmd = command_str.strip().lower()
        skill = self.skills.get(clean_cmd)
        if not skill:
            return None
        return skill.template.replace("$ARGUMENTS", args.strip()).replace("$args", args.strip())
