"""Detector module."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from evren_code.core.repo_intelligence.models import (
    DevCommand,
    ImportantFile,
    ModuleInfo,
    RepoMetadata,
)
from evren_code.core.repo_intelligence.scanner import ScanResult


class ProjectDetector:
    """Project detector implementation."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root.resolve()

    def _read_text_safe(self, rel_path: str, max_chars: int = 100_000) -> str:
        fpath = self.workspace_root / rel_path
        if not fpath.is_file():
            return ""
        try:
            return fpath.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except (OSError, PermissionError):
            return ""

    def _read_json_safe(self, rel_path: str) -> Optional[Dict[str, Any]]:
        content = self._read_text_safe(rel_path)
        if not content:
            return None
        try:
            return json.loads(content)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def analyze_directory(self, dir_rel_path: str, files_in_dir: List[str]) -> Optional[ModuleInfo]:
        """Analyze directory."""
        manifest_names = {Path(f).name: f for f in files_in_dir}
        base_name = Path(dir_rel_path).name if dir_rel_path != "." else self.workspace_root.name

        languages: Set[str] = set()
        frameworks: Set[str] = set()
        manifests: List[str] = []
        test_cmds: List[DevCommand] = []
        build_cmds: List[DevCommand] = []
        dev_cmds: List[DevCommand] = []
        entry_points: List[str] = []
        important_files: List[ImportantFile] = []
        build_system = "unknown"
        detected_type = "unknown"

        # 1. Node / JS / TS Ekosistemi
        if "package.json" in manifest_names:
            pkg_rel = manifest_names["package.json"]
            manifests.append(pkg_rel)
            pkg_data = self._read_json_safe(pkg_rel) or {}
            if pkg_data.get("name"):
                base_name = str(pkg_data["name"])
            build_system = "npm"
            if "pnpm-lock.yaml" in manifest_names:
                build_system = "pnpm"
            elif "yarn.lock" in manifest_names:
                build_system = "yarn"
            elif "bun.lock" in manifest_names or "bun.lockb" in manifest_names:
                build_system = "bun"

            languages.add("JavaScript")
            if "tsconfig.json" in manifest_names:
                languages.add("TypeScript")
                manifests.append(manifest_names["tsconfig.json"])

            deps = {**pkg_data.get("dependencies", {}), **pkg_data.get("devDependencies", {})}
            if "typescript" in deps:
                languages.add("TypeScript")

            # Frameworks
            if "next" in deps:
                ver = str(deps["next"]).lstrip("^~")
                v_major = ver.split(".")[0] if "." in ver else ver
                frameworks.add(f"Next.js {v_major}")
                detected_type = "frontend"
            elif "nuxt" in deps or "nuxt3" in deps:
                frameworks.add("Nuxt")
                detected_type = "frontend"
            elif "vue" in deps:
                frameworks.add("Vue")
                detected_type = "frontend"
            elif "@angular/core" in deps:
                frameworks.add("Angular")
                detected_type = "frontend"
            elif "svelte" in deps:
                frameworks.add("Svelte")
                detected_type = "frontend"
            elif "express" in deps:
                frameworks.add("Express")
                detected_type = "backend"
            elif "@nestjs/core" in deps:
                frameworks.add("NestJS")
                detected_type = "backend"

            if "react" in deps:
                frameworks.add("React")
                if detected_type == "unknown":
                    detected_type = "frontend"

            if not frameworks:
                frameworks.add("Node.js")
                if detected_type == "unknown":
                    detected_type = "backend" if "server" in base_name.lower() or "api" in base_name.lower() else "library"

            if "vite" in deps:
                frameworks.add("Vite")
            if "tailwindcss" in deps:
                frameworks.add("TailwindCSS")

            # Komutlar (scripts)
            scripts = pkg_data.get("scripts", {})
            pkg_mgr = build_system if build_system in ("pnpm", "yarn", "bun") else "npm"
            run_prefix = f"{pkg_mgr} run"
            for sname, scmd in scripts.items():
                cmd_obj = DevCommand(
                    name=sname,
                    command=f"{pkg_mgr} test" if sname == "test" else f"{run_prefix} {sname}",
                    source=f"{pkg_rel}:scripts.{sname}",
                    description=scmd,
                )
                if sname in ("test", "test:unit", "test:e2e", "jest"):
                    test_cmds.append(cmd_obj)
                elif sname in ("build", "compile", "bundle"):
                    build_cmds.append(cmd_obj)
                elif sname in ("dev", "start", "serve"):
                    dev_cmds.append(cmd_obj)

        # 2. Java / JVM Ekosistemi
        has_pom = "pom.xml" in manifest_names
        has_gradle = any(g in manifest_names for g in ("build.gradle", "build.gradle.kts"))

        if has_pom or has_gradle:
            languages.add("Java")
            if has_pom:
                pom_rel = manifest_names["pom.xml"]
                manifests.append(pom_rel)
                build_system = "maven"
                pom_content = self._read_text_safe(pom_rel)
                if "spring-boot" in pom_content:
                    frameworks.add("Spring Boot")
                    detected_type = "backend"
                java_v = re.findall(r"<java\.version>(.*?)</java\.version>", pom_content)
                if java_v:
                    frameworks.add(f"Java {java_v[0].strip()}")
                test_cmds.append(DevCommand(name="test", command="mvn test", source=f"{pom_rel} (Maven)"))
                build_cmds.append(DevCommand(name="build", command="mvn clean package", source=f"{pom_rel} (Maven)"))

            if has_gradle:
                gradle_file = manifest_names.get("build.gradle") or manifest_names.get("build.gradle.kts")
                if gradle_file:
                    manifests.append(gradle_file)
                    build_system = "gradle"
                    g_content = self._read_text_safe(gradle_file)
                    if "spring" in g_content:
                        frameworks.add("Spring Boot")
                        detected_type = "backend"
                    if "kotlin" in g_content or gradle_file.endswith(".kts"):
                        languages.add("Kotlin")
                    test_cmds.append(DevCommand(name="test", command="./gradlew test", source=f"{gradle_file} (Gradle)"))
                    build_cmds.append(DevCommand(name="build", command="./gradlew build", source=f"{gradle_file} (Gradle)"))

            if detected_type == "unknown":
                detected_type = "backend"

        # 3. Python Ekosistemi
        py_manifests = [
            m for m in ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile", "poetry.lock", "uv.lock")
            if m in manifest_names
        ]
        if py_manifests:
            languages.add("Python")
            detected_type = "backend" if detected_type == "unknown" else detected_type
            combined_content = ""
            for pm in py_manifests:
                pm_rel = manifest_names[pm]
                manifests.append(pm_rel)
                combined_content += "\n" + self._read_text_safe(pm_rel)

            if "poetry.lock" in manifest_names or "[tool.poetry]" in combined_content:
                build_system = "poetry"
            elif "uv.lock" in manifest_names or "[tool.uv]" in combined_content:
                build_system = "uv"
            elif "Pipfile" in manifest_names:
                build_system = "pipenv"
            elif "setup.py" in manifest_names or "setup.cfg" in manifest_names:
                build_system = "setuptools"
            else:
                build_system = "pip"

            # Frameworks
            c_low = combined_content.lower()
            if "fastapi" in c_low:
                frameworks.add("FastAPI")
                detected_type = "backend"
            if "django" in c_low or "manage.py" in manifest_names:
                frameworks.add("Django")
                detected_type = "backend"
            if "flask" in c_low:
                frameworks.add("Flask")
                detected_type = "backend"
            if "pytest" in c_low or "pytest.ini" in manifest_names:
                frameworks.add("Pytest")
                test_cmds.append(DevCommand(name="test", command="pytest", source="pyproject/pytest convention"))
            else:
                test_cmds.append(DevCommand(name="test", command="python -m unittest", source="python standard library"))

            if build_system == "poetry":
                build_cmds.append(DevCommand(name="build", command="poetry build", source="poetry"))
            elif build_system == "uv":
                build_cmds.append(DevCommand(name="build", command="uv build", source="uv"))
            else:
                build_cmds.append(DevCommand(name="build", command="python -m build", source="python build"))

        # 4. Go Ekosistemi
        if "go.mod" in manifest_names:
            go_rel = manifest_names["go.mod"]
            manifests.append(go_rel)
            languages.add("Go")
            build_system = "go_modules"
            detected_type = "backend" if detected_type == "unknown" else detected_type
            go_content = self._read_text_safe(go_rel)
            if "github.com/gin-gonic/gin" in go_content:
                frameworks.add("Gin")
            elif "github.com/gofiber/fiber" in go_content:
                frameworks.add("Fiber")
            elif "github.com/labstack/echo" in go_content:
                frameworks.add("Echo")
            test_cmds.append(DevCommand(name="test", command="go test ./...", source=f"{go_rel} (Go)"))
            build_cmds.append(DevCommand(name="build", command="go build ./...", source=f"{go_rel} (Go)"))

        # 5. Rust Ekosistemi
        if "Cargo.toml" in manifest_names:
            cargo_rel = manifest_names["Cargo.toml"]
            manifests.append(cargo_rel)
            languages.add("Rust")
            build_system = "cargo"
            detected_type = "backend" if detected_type == "unknown" else detected_type
            cargo_content = self._read_text_safe(cargo_rel)
            if "actix-web" in cargo_content:
                frameworks.add("Actix-web")
            elif "axum" in cargo_content:
                frameworks.add("Axum")
            elif "tokio" in cargo_content:
                frameworks.add("Tokio")
            test_cmds.append(DevCommand(name="test", command="cargo test", source=f"{cargo_rel} (Cargo)"))
            build_cmds.append(DevCommand(name="build", command="cargo build", source=f"{cargo_rel} (Cargo)"))

        # 6. .NET Ekosistemi
        dotnet_files = [f for f in manifest_names if f.endswith(".sln") or f.endswith(".csproj")]
        if dotnet_files:
            languages.add("C#")
            build_system = "dotnet"
            detected_type = "backend" if detected_type == "unknown" else detected_type
            frameworks.add(".NET")
            for df in dotnet_files:
                manifests.append(manifest_names[df])
            test_cmds.append(DevCommand(name="test", command="dotnet test", source=".NET CLI"))
            build_cmds.append(DevCommand(name="build", command="dotnet build", source=".NET CLI"))

        # 7. PHP Ekosistemi
        if "composer.json" in manifest_names:
            comp_rel = manifest_names["composer.json"]
            manifests.append(comp_rel)
            languages.add("PHP")
            build_system = "composer"
            detected_type = "backend" if detected_type == "unknown" else detected_type
            comp_data = self._read_json_safe(comp_rel) or {}
            deps = {**comp_data.get("require", {}), **comp_data.get("require-dev", {})}
            if "laravel/framework" in deps:
                frameworks.add("Laravel")
            elif "symfony/framework-bundle" in deps:
                frameworks.add("Symfony")
            test_cmds.append(DevCommand(name="test", command="composer test", source=f"{comp_rel} (Composer)"))

        # 8. Flutter / Dart Ekosistemi
        if "pubspec.yaml" in manifest_names:
            pub_rel = manifest_names["pubspec.yaml"]
            manifests.append(pub_rel)
            languages.add("Dart")
            frameworks.add("Flutter")
            build_system = "flutter"
            detected_type = "frontend"
            test_cmds.append(DevCommand(name="test", command="flutter test", source=f"{pub_rel} (Flutter)"))

        if not languages and not frameworks and not manifests:
            return None

        for f in files_in_dir:
            name = Path(f).name.lower()
            if name in (
                "main.py", "app.py", "manage.py", "server.py", "wsgi.py", "asgi.py",
                "index.ts", "index.tsx", "index.js", "main.ts", "main.tsx", "main.js",
                "server.ts", "server.js", "app.tsx", "app.ts", "app.jsx", "app.js",
                "main.go", "main.rs", "application.java", "main.java"
            ):
                entry_points.append(f)

        d_lower = base_name.lower()
        if detected_type == "unknown":
            if any(k in d_lower for k in ("front", "ui", "web", "client")):
                detected_type = "frontend"
            elif any(k in d_lower for k in ("back", "api", "server", "service")):
                detected_type = "backend"
            elif any(k in d_lower for k in ("lib", "core", "common", "shared")):
                detected_type = "library"
            elif any(k in d_lower for k in ("cli", "tool")):
                detected_type = "cli"
            else:
                detected_type = "root" if dir_rel_path == "." else "service"

        return ModuleInfo(
            name=base_name,
            path=dir_rel_path,
            module_type=detected_type,
            languages=sorted(list(languages)),
            frameworks=sorted(list(frameworks)),
            build_system=build_system,
            dependency_manifests=manifests,
            entry_points=entry_points,
            test_commands=test_cmds,
            build_commands=build_cmds,
            dev_commands=dev_cmds,
            important_files=important_files,
        )

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def classify_important_files(self, scan_result: ScanResult) -> List[ImportantFile]:
        """Classify important files."""
        important: List[ImportantFile] = []
        for rel_path, file_info in scan_result.files.items():
            name = Path(rel_path).name.lower()
            category: Optional[str] = None
            desc = ""

            # README
            if name.startswith("readme"):
                category = "readme"
                desc = "Proje dokümantasyonu ve genel bakış"

            # Dependency Manifest
            elif name in (
                "package.json", "pyproject.toml", "requirements.txt", "pom.xml",
                "build.gradle", "build.gradle.kts", "cargo.toml", "go.mod",
                "composer.json", "pubspec.yaml", "gemfile", "pipfile"
            ):
                category = "dependency_manifest"
                desc = "Paket ve bağımlılık yöneticisi tanımı"

            # Build Config
            elif name in (
                "makefile", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
                "webpack.config.js", "vite.config.ts", "vite.config.js", "tsconfig.json",
                "settings.gradle", "settings.gradle.kts"
            ):
                category = "build_config"
                desc = "Derleme veya konteyner yapılandırması"

            # Test Config
            elif name in (
                "pytest.ini", "jest.config.js", "jest.config.ts", "vitest.config.ts",
                "vitest.config.js", ".rspec", "setup.cfg"
            ):
                category = "test_config"
                desc = "Test koşucu ve yapılandırma dosyası"

            # Env Example
            elif name in (".env.example", ".env.sample", ".env.template"):
                category = "env_example"
                desc = "Çevre değişkenleri örnek şablonu"

            # CI/CD
            elif ".github/workflows" in rel_path or name in (".gitlab-ci.yml", "jenkinsfile"):
                category = "ci_cd_config"
                desc = "Sürekli entegrasyon ve dağıtım boru hattı (CI/CD)"

            # Main Config
            elif name in (
                "application.properties", "application.yml", "application.yaml",
                "config.yaml", "config.json", "settings.py"
            ):
                category = "main_config"
                desc = "Uygulama ana konfigürasyon dosyası"

            # Entry Points
            elif name in (
                "main.py", "app.py", "index.ts", "index.tsx", "index.js",
                "main.go", "main.rs", "application.java"
            ):
                category = "entry_point"
                desc = "Uygulama başlatma giriş noktası"

            if category:
                important.append(
                    ImportantFile(
                        path=rel_path,
                        category=category,
                        description=desc,
                        size_bytes=file_info.size_bytes,
                    )
                )

        return important

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def detect_project_structure(self, scan_result: ScanResult) -> Tuple[Optional[ModuleInfo], List[ModuleInfo], RepoMetadata]:
        """Detect project structure."""
        files_by_dir: Dict[str, List[str]] = {}
        for rel_path in scan_result.files.keys():
            parts = rel_path.split("/")
            if len(parts) == 1:
                dir_key = "."
            else:
                dir_key = "/".join(parts[:-1])

            files_by_dir.setdefault(dir_key, []).append(rel_path)

        root_files = files_by_dir.get(".", [])
        root_module = self.analyze_directory(".", root_files)

        candidate_module_dirs: Set[str] = set()
        for d in scan_result.directories:
            parts = d.split("/")
            if len(parts) <= 2:
                candidate_module_dirs.add(d)

        submodules: List[ModuleInfo] = []
        for md in sorted(list(candidate_module_dirs)):
            m_files = files_by_dir.get(md, [])
            sub_mod = self.analyze_directory(md, m_files)
            if sub_mod and sub_mod.dependency_manifests:
                if sub_mod.path != ".":
                    submodules.append(sub_mod)

        is_monorepo = False
        project_type = "single_project"

        root_manifest_names = [Path(f).name for f in root_files]
        if "pnpm-workspace.yaml" in root_manifest_names or "lerna.json" in root_manifest_names:
            is_monorepo = True
            project_type = "monorepo"
        elif "package.json" in root_manifest_names:
            pkg_data = self._read_json_safe("package.json") or {}
            if "workspaces" in pkg_data:
                is_monorepo = True
                project_type = "monorepo"

        if not is_monorepo:
            if "settings.gradle" in root_manifest_names or "settings.gradle.kts" in root_manifest_names:
                s_content = self._read_text_safe("settings.gradle") or self._read_text_safe("settings.gradle.kts")
                if "include" in s_content and len(submodules) > 0:
                    is_monorepo = True
                    project_type = "multi_module"
            elif "pom.xml" in root_manifest_names:
                pom_content = self._read_text_safe("pom.xml")
                if "<modules>" in pom_content and len(submodules) > 0:
                    is_monorepo = True
                    project_type = "multi_module"

        if not is_monorepo and len(submodules) >= 2:
            is_monorepo = True
            project_type = "monorepo"

        if not root_module and not submodules:
            project_type = "unknown"

        all_langs: Set[str] = set()
        all_frameworks: Set[str] = set()
        if root_module:
            all_langs.update(root_module.languages)
            all_frameworks.update(root_module.frameworks)
        for sm in submodules:
            all_langs.update(sm.languages)
            all_frameworks.update(sm.frameworks)

        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        metadata = RepoMetadata(
            workspace_root=str(self.workspace_root),
            project_name=self.workspace_root.name,
            detected_languages=sorted(list(all_langs)),
            detected_frameworks=sorted(list(all_frameworks)),
            project_type=project_type,
            is_monorepo=is_monorepo,
            scan_timestamp=timestamp,
            scan_version="2.1",
            is_partial=scan_result.is_partial,
            partial_reasons=list(scan_result.partial_reasons),
            total_files_scanned=scan_result.total_files_scanned,
            total_dirs_scanned=scan_result.total_dirs_scanned,
            total_bytes_scanned=scan_result.total_bytes_scanned,
            ignored_files_count=scan_result.ignored_files_count,
        )

        return root_module, submodules, metadata
