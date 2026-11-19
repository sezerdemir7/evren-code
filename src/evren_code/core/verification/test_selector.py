"""Test selector module."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from evren_code.core.repo_intelligence.engine import RepoIntelligenceEngine
from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine
from evren_code.core.verification.models import VerificationStep


class TestSelectionEngine:
    """Test selection engine implementation."""

    def __init__(
        self,
        workspace_root: Path,
        repo_intelligence: Optional[RepoIntelligenceEngine] = None,
        symbol_engine: Optional[CodebaseSymbolEngine] = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.repo_intelligence = repo_intelligence
        self.symbol_engine = symbol_engine

    def select_tests_for_changes(self, changed_files: List[str]) -> List[VerificationStep]:
        """Select tests for changes."""
        steps: List[VerificationStep] = []
        seen_commands: Set[str] = set()

        direct_test_files: Set[str] = set()
        impacted_test_files: Set[str] = set()
        selection_reasons: Dict[str, List[str]] = {}

        for f in changed_files:
            rel_f = self._normalize_path(f)
            if self._is_test_file(rel_f):
                direct_test_files.add(rel_f)
                selection_reasons.setdefault(rel_f, []).append(f"Değiştirilen test dosyası: {rel_f}")

        for f in changed_files:
            rel_f = self._normalize_path(f)
            if self._is_test_file(rel_f):
                continue

            if self.symbol_engine and hasattr(self.symbol_engine, "test_linker"):
                linked_tests = self.symbol_engine.test_linker.find_tests_for_file(rel_f)
                for lt in linked_tests:
                    tf = self._normalize_path(lt["test_file"])
                    direct_test_files.add(tf)
                    evidence = lt.get("evidence", lt.get("match_type", "bağlantılı"))
                    selection_reasons.setdefault(tf, []).append(
                        f"Doğrudan test bağlantısı ({evidence}): {tf} <- {rel_f}"
                    )

            if self.symbol_engine and hasattr(self.symbol_engine, "impact_analyzer"):
                symbols = self.symbol_engine.graph.get_file_symbols(rel_f)
                for sym in symbols[:5]:
                    try:
                        impact = self.symbol_engine.impact_analyzer.analyze_symbol(sym.name)
                        for rtf in impact.related_test_files:
                            tf = self._normalize_path(rtf["test_file"])
                            if tf not in direct_test_files:
                                impacted_test_files.add(tf)
                                selection_reasons.setdefault(tf, []).append(
                                    f"Sembol etki analizi ({sym.name}): {tf}"
                                )
                    except Exception:
                        pass

        all_target_test_files = list(direct_test_files) + [
            f for f in impacted_test_files if f not in direct_test_files
        ]

        if all_target_test_files:
            targeted_steps = self._create_targeted_test_steps(all_target_test_files, selection_reasons)
            for step in targeted_steps:
                cmd_key = f"{step.cwd}:{step.command}"
                if cmd_key not in seen_commands:
                    seen_commands.add(cmd_key)
                    steps.append(step)

        build_steps = self._create_build_or_lint_steps(changed_files)
        for b_step in build_steps:
            cmd_key = f"{b_step.cwd}:{b_step.command}"
            if cmd_key not in seen_commands:
                seen_commands.add(cmd_key)
                steps.insert(0, b_step)

        if not any(s.target_type == "test" for s in steps):
            fallback_step = self._create_fallback_test_step(changed_files)
            if fallback_step:
                cmd_key = f"{fallback_step.cwd}:{fallback_step.command}"
                if cmd_key not in seen_commands:
                    seen_commands.add(cmd_key)
                    steps.append(fallback_step)

        return steps

    def _normalize_path(self, path_str: str) -> str:
        """Normalize path."""
        try:
            p = Path(path_str)
            if p.is_absolute():
                return str(p.relative_to(self.workspace_root)).replace("\\", "/")
            return str(p).replace("\\", "/")
        except Exception:
            return path_str.replace("\\", "/")

    def _is_test_file(self, file_path: str) -> bool:
        """Is test file."""
        if self.symbol_engine and hasattr(self.symbol_engine, "test_linker"):
            return self.symbol_engine.test_linker.is_test_file(file_path)

        p_low = file_path.lower().replace("\\", "/")
        name = Path(file_path).name.lower()
        if any(seg in p_low.split("/") for seg in ("test", "tests", "__tests__", "spec", "specs")):
            return True
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        if name.endswith("test.java") or name.endswith("tests.java"):
            return True
        if name.endswith(".test.ts") or name.endswith(".spec.ts") or name.endswith(".test.js"):
            return True
        if name.endswith("_test.go") or name.endswith("_test.rs"):
            return True
        return False

    def _create_targeted_test_steps(
        self,
        test_files: List[str],
        reasons: Dict[str, List[str]],
    ) -> List[VerificationStep]:
        """Create targeted test steps."""
        steps: List[VerificationStep] = []

        py_tests = [f for f in test_files if f.endswith(".py")]
        java_tests = [f for f in test_files if f.endswith(".java")]
        ts_js_tests = [f for f in test_files if f.endswith((".ts", ".js", ".tsx", ".jsx"))]
        go_tests = [f for f in test_files if f.endswith(".go")]
        rs_tests = [f for f in test_files if f.endswith(".rs")]

        if py_tests:
            target_str = " ".join(py_tests[:10])
            why = " | ".join(sum([reasons.get(f, []) for f in py_tests[:5]], []))
            cmd = f"pytest {target_str}" if self._has_pytest() else f"python -m unittest {target_str}"
            steps.append(
                VerificationStep(
                    name="python:targeted_tests",
                    command=cmd,
                    cwd=".",
                    target_type="test",
                    description=f"{len(py_tests)} adet hedeflenmiş Python test dosyası yürütülüyor.",
                    why_selected=why or "Değişikliklerle ilişkili Python testleri seçildi.",
                )
            )

        if java_tests:
            class_names = [Path(f).stem for f in java_tests]
            target_str = ",".join(class_names[:8])
            why = " | ".join(sum([reasons.get(f, []) for f in java_tests[:3]], []))
            is_mvn = (self.workspace_root / "pom.xml").exists()
            cmd = f"mvn test -o -Dtest={target_str}" if is_mvn else f"./gradlew test --offline --tests '*{class_names[0]}*'"
            steps.append(
                VerificationStep(
                    name="java:targeted_tests",
                    command=cmd,
                    cwd=".",
                    target_type="test",
                    description=f"{len(java_tests)} adet hedeflenmiş Java test sınıfı yürütülüyor.",
                    why_selected=why or "Değişikliklerle ilişkili Java testleri seçildi.",
                )
            )

        if ts_js_tests:
            target_str = " ".join(ts_js_tests[:8])
            why = " | ".join(sum([reasons.get(f, []) for f in ts_js_tests[:3]], []))
            steps.append(
                VerificationStep(
                    name="node:targeted_tests",
                    command=f"npm test -- {target_str}",
                    cwd=".",
                    target_type="test",
                    description=f"{len(ts_js_tests)} adet hedeflenmiş JS/TS test dosyası yürütülüyor.",
                    why_selected=why or "Değişikliklerle ilişkili JS/TS testleri seçildi.",
                )
            )

        if go_tests:
            dirs = sorted(list({str(Path(f).parent).replace("\\", "/") for f in go_tests}))
            dir_str = " ".join([f"./{d}" if not d.startswith(".") else d for d in dirs[:5]])
            why = " | ".join(sum([reasons.get(f, []) for f in go_tests[:3]], []))
            steps.append(
                VerificationStep(
                    name="go:targeted_tests",
                    command=f"go test -mod=readonly -v {dir_str}",
                    cwd=".",
                    target_type="test",
                    description="Hedeflenmiş Go paket testleri yürütülüyor.",
                    why_selected=why or "Değişikliklerle ilişkili Go testleri seçildi.",
                )
            )

        if rs_tests:
            why = " | ".join(sum([reasons.get(f, []) for f in rs_tests[:3]], []))
            steps.append(
                VerificationStep(
                    name="rust:targeted_tests",
                    command="cargo test --offline",
                    cwd=".",
                    target_type="test",
                    description="Rust test paketi yürütülüyor.",
                    why_selected=why or "Değişikliklerle ilişkili Rust testleri seçildi.",
                )
            )

        return steps

    def _create_build_or_lint_steps(self, changed_files: List[str]) -> List[VerificationStep]:
        """Create build or lint steps."""
        steps: List[VerificationStep] = []

        if any(f.endswith((".ts", ".tsx")) for f in changed_files):
            if (self.workspace_root / "tsconfig.json").exists():
                steps.append(
                    VerificationStep(
                        name="tsc:typecheck",
                        command="npx --no-install tsc --noEmit",
                        cwd=".",
                        target_type="typecheck",
                        description="TypeScript statik tip denetimi.",
                        why_selected="TypeScript dosyaları değiştiği için statik tip doğrulaması yapılıyor.",
                    )
                )

        # Java Maven derleme
        if any(f.endswith(".java") for f in changed_files):
            if (self.workspace_root / "pom.xml").exists():
                steps.append(
                    VerificationStep(
                        name="mvn:compile",
                        command="mvn compile -o -DskipTests",
                        cwd=".",
                        target_type="build",
                        description="Maven derleme doğrulaması (çevrimdışı).",
                        why_selected="Java kaynak kodları değiştiği için derleme kontrolü yapılıyor.",
                    )
                )

        # Rust cargo check
        if any(f.endswith(".rs") for f in changed_files):
            if (self.workspace_root / "Cargo.toml").exists():
                steps.append(
                    VerificationStep(
                        name="cargo:check",
                        command="cargo check --offline",
                        cwd=".",
                        target_type="build",
                        description="Rust hızlı derleme kontrolü.",
                        why_selected="Rust kaynak dosyaları değiştiği için derlenebilirlik kontrol ediliyor.",
                    )
                )

        return steps

    def _create_fallback_test_step(self, changed_files: List[str]) -> Optional[VerificationStep]:
        """Create fallback test step."""
        if self.repo_intelligence:
            try:
                pmap = self.repo_intelligence.get_project_map()
                if pmap and pmap.dev_commands and pmap.dev_commands.get("test"):
                    return VerificationStep(
                        name="repo:test_suite",
                        command=pmap.dev_commands["test"],
                        cwd=".",
                        target_type="test",
                        description="Proje genel test paketi.",
                        why_selected="Özel hedef test izole edilemediği için proje test komutu yürütülüyor.",
                    )
            except Exception:
                pass

        if (self.workspace_root / "pytest.ini").exists() or (self.workspace_root / "tests").is_dir() or any(f.endswith(".py") for f in changed_files):
            cmd = "pytest" if self._has_pytest() else "python -m unittest discover"
            return VerificationStep(
                name="python:test_suite",
                command=cmd,
                cwd=".",
                target_type="test",
                description="Python test paketi yürütülüyor.",
                why_selected="Değişiklikler sonrası genel Python test paketi çalıştırılıyor.",
            )

        if (self.workspace_root / "pom.xml").exists():
            return VerificationStep(
                name="mvn:test",
                command="mvn test -o",
                cwd=".",
                target_type="test",
                description="Maven genel test paketi (çevrimdışı).",
                why_selected="Genel Maven testleri yürütülüyor.",
            )

        if (self.workspace_root / "build.gradle").exists() or (self.workspace_root / "build.gradle.kts").exists():
            return VerificationStep(
                name="gradle:test",
                command="./gradlew test --offline",
                cwd=".",
                target_type="test",
                description="Gradle genel test paketi (çevrimdışı).",
                why_selected="Genel Gradle testleri yürütülüyor.",
            )

        if (self.workspace_root / "package.json").exists():
            return VerificationStep(
                name="npm:test",
                command="npm test",
                cwd=".",
                target_type="test",
                description="NPM test paketi.",
                why_selected="Genel NPM testleri yürütülüyor.",
            )

        if (self.workspace_root / "go.mod").exists():
            return VerificationStep(
                name="go:test",
                command="go test -mod=readonly ./...",
                cwd=".",
                target_type="test",
                description="Go test paketi.",
                why_selected="Genel Go testleri yürütülüyor.",
            )

        if (self.workspace_root / "Cargo.toml").exists():
            return VerificationStep(
                name="cargo:test",
                command="cargo test --offline",
                cwd=".",
                target_type="test",
                description="Rust test paketi.",
                why_selected="Genel Cargo testleri yürütülüyor.",
            )

        return None

    def _has_pytest(self) -> bool:
        """Has pytest."""
        return (
            (self.workspace_root / "pytest.ini").exists()
            or (self.workspace_root / "pyproject.toml").exists()
            or (self.workspace_root / "setup.cfg").exists()
        )
