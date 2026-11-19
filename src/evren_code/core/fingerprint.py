"""Workspace Fingerprinter.

Scans the project root and 1st-level directories to automatically detect the tech stack,
architecture type (Monorepo / Single Project), primary modules, and dependencies.
This context is injected into system prompts so the agent recognizes the environment from step zero.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


IGNORED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".evren",
    "node_modules",
    "target",
    "build",
    "dist",
    ".next",
    ".turbo",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    "bin",
    "obj",
}


def _analyze_node_package(pkg_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": "Node/JavaScript", "details": []}
    try:
        data = json.loads(pkg_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return info

    deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
    
    # Frameworks
    if "next" in deps:
        v = deps["next"].lstrip("^~")
        info["type"] = f"Next.js {v.split('.')[0] if '.' in v else v}"
    elif "nuxt" in deps or "nuxt3" in deps:
        info["type"] = "Nuxt"
    elif "vue" in deps:
        info["type"] = "Vue"
    elif "@angular/core" in deps:
        info["type"] = "Angular"
    elif "svelte" in deps:
        info["type"] = "Svelte"
    elif "express" in deps:
        info["type"] = "Express"
    elif "@nestjs/core" in deps:
        info["type"] = "NestJS"
    elif "react" in deps:
        info["type"] = "React"

    # Language
    is_ts = (
        "typescript" in deps
        or (pkg_path.parent / "tsconfig.json").is_file()
    )
    if is_ts:
        info["details"].append("TypeScript")

    if "react" in deps and not info["type"].startswith("Next"):
        info["details"].append("React")
    elif "react" in deps and info["type"].startswith("Next"):
        v_react = deps["react"].lstrip("^~").split(".")[0]
        info["details"].append(f"React {v_react}")

    if "tailwindcss" in deps:
        info["details"].append("TailwindCSS")
    if "axios" in deps:
        info["details"].append("Axios")
    if "lucide-react" in deps:
        info["details"].append("Lucide Icons")

    return info


def _analyze_maven_pom(pom_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": "Java (Maven)", "details": []}
    try:
        content = pom_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return info

    if "spring-boot" in content:
        info["type"] = "Java Spring Boot (Maven)"

    java_vers = re.findall(r"<java\.version>(.*?)</java\.version>", content)
    if java_vers:
        info["details"].append(f"Java {java_vers[0].strip()}")

    if "lombok" in content:
        info["details"].append("Lombok")
    if "postgresql" in content:
        info["details"].append("PostgreSQL")
    if "spring-boot-starter-security" in content or "spring-security" in content:
        info["details"].append("Spring Security")
    if "flyway" in content:
        info["details"].append("Flyway")

    return info


def _analyze_gradle(gradle_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": "Java/Kotlin (Gradle)", "details": []}
    try:
        content = gradle_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return info

    if "spring" in content:
        info["type"] = "Java/Kotlin Spring Boot (Gradle)"
    elif "com.android.application" in content:
        info["type"] = "Android (Gradle)"

    return info


def _analyze_python(dir_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": "Python", "details": []}
    content = ""
    for fname in ["pyproject.toml", "requirements.txt", "setup.py"]:
        fpath = dir_path / fname
        if fpath.is_file():
            try:
                content += "\n" + fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

    if "fastapi" in content.lower():
        info["type"] = "Python (FastAPI)"
    elif "django" in content.lower():
        info["type"] = "Python (Django)"
    elif "flask" in content.lower():
        info["type"] = "Python (Flask)"

    for lib in ["pydantic", "pytest", "torch", "sqlalchemy", "numpy"]:
        if lib in content.lower():
            info["details"].append(lib)

    return info


def _inspect_single_directory(d: Path) -> Optional[Dict[str, Any]]:
    """Inspect single directory."""
    if (d / "package.json").is_file():
        res = _analyze_node_package(d / "package.json")
        res["dir_name"] = d.name
        return res
    if (d / "pom.xml").is_file():
        res = _analyze_maven_pom(d / "pom.xml")
        res["dir_name"] = d.name
        return res
    if (d / "build.gradle").is_file() or (d / "build.gradle.kts").is_file():
        res = _analyze_gradle(d / "build.gradle" if (d / "build.gradle").is_file() else d / "build.gradle.kts")
        res["dir_name"] = d.name
        return res
    if (d / "pyproject.toml").is_file() or (d / "requirements.txt").is_file():
        res = _analyze_python(d)
        res["dir_name"] = d.name
        return res
    if (d / "pubspec.yaml").is_file():
        return {"type": "Flutter / Dart", "details": [], "dir_name": d.name}
    if (d / "go.mod").is_file():
        return {"type": "Go", "details": [], "dir_name": d.name}
    if (d / "Cargo.toml").is_file():
        return {"type": "Rust", "details": [], "dir_name": d.name}
    if (d / "composer.json").is_file():
        return {"type": "PHP (Composer)", "details": [], "dir_name": d.name}

    return None


class WorkspaceFingerprinter:
    """Workspace fingerprinter implementation."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root

    def detect(self) -> str:
        """Detect."""
        if not self.workspace_root.is_dir():
            return ""

        modules: List[Dict[str, Any]] = []

        root_info = _inspect_single_directory(self.workspace_root)

        subdirs: List[Path] = []
        try:
            for item in self.workspace_root.iterdir():
                if item.is_dir() and item.name not in IGNORED_DIRS and not item.name.startswith("."):
                    subdirs.append(item)
        except OSError:
            pass

        for sd in subdirs:
            sub_info = _inspect_single_directory(sd)
            if sub_info:
                sub_info["dir_name"] = sd.name
                modules.append(sub_info)

        if not root_info and not modules:
            return ""

        lines = ["## Proje Parmak İzi (Workspace Fingerprint)"]
        if modules:
            lines.append("- **Mimari:** Çoklu Modül / Monorepo")
            lines.append("- **Tespit Edilen Modüller ve Teknoloji Yığını:**")
            for m in modules:
                d_name = m["dir_name"]
                t_name = m["type"]
                details = f" ({', '.join(m['details'])})" if m.get("details") else ""
                lines.append(f"  - `{d_name}`: {t_name}{details}")
        elif root_info:
            lines.append("- **Mimari:** Tekil Proje")
            t_name = root_info["type"]
            details = f" ({', '.join(root_info['details'])})" if root_info.get("details") else ""
            lines.append(f"- **Teknoloji Yığını:** {t_name}{details}")

        key_dirs = []
        for sd in subdirs:
            dname = sd.name.lower()
            if any(k in dname for k in ["front", "web", "ui", "client"]):
                key_dirs.append(f"`{sd.name}/`: Kullanıcı arayüzü ve frontend kodları")
            elif any(k in dname for k in ["back", "api", "server"]):
                key_dirs.append(f"`{sd.name}/`: Backend API ve servis kodları")
            elif dname in ["src", "lib", "app"]:
                key_dirs.append(f"`{sd.name}/`: Ana uygulama kaynak kodları")

        if key_dirs:
            lines.append("- **Temel Çalışma Dizinleri:**")
            for kd in key_dirs:
                lines.append(f"  - {kd}")

        return "\n".join(lines)
