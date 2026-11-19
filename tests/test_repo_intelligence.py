"""Test repo intelligence module."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.repo_intelligence.detector import ProjectDetector
from evren_code.core.repo_intelligence.engine import RepoIntelligenceEngine
from evren_code.core.repo_intelligence.models import (
    DevCommand,
    ImportantFile,
    ModuleInfo,
    ProjectMap,
    RepoMetadata,
    ScanLimits,
)
from evren_code.core.repo_intelligence.persistence import ProjectMapStore
from evren_code.core.repo_intelligence.scanner import (
    DEFAULT_IGNORED_DIRS,
    GitIgnoreMatcher,
    RepoScanner,
    ScanResult,
)
from evren_code.core.repo_intelligence.tools import register_repo_intelligence_tools
from evren_code.core.security import PermissionManager, ToolExecutor
from evren_code.tools.registry import ToolRegistry


class TestRepoIntelligence(unittest.TestCase):
    """Test repo intelligence implementation."""

    # -------------------------------------------------------------------------
    # 1. Proje Tespiti (Project Detection Engine)
    # -------------------------------------------------------------------------

    def test_detect_python_fastapi_pytest_project(self):
        """Verify detect python fastapi pytest project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text(
                """
                [project]
                name = "my-service"
                dependencies = ["fastapi", "uvicorn", "pydantic"]
                [tool.pytest.ini_options]
                testpaths = ["tests"]
                """,
                encoding="utf-8",
            )
            (root / "main.py").write_text("import fastapi\napp = fastapi.FastAPI()", encoding="utf-8")

            detector = ProjectDetector(root)
            files = ["pyproject.toml", "main.py"]
            mod = detector.analyze_directory(".", files)

            self.assertIsNotNone(mod)
            self.assertIn("Python", mod.languages)
            self.assertIn("FastAPI", mod.frameworks)
            self.assertIn("Pytest", mod.frameworks)
            self.assertEqual(mod.build_system, "pip")
            self.assertIn("main.py", mod.entry_points)
            cmd_names = [c.command for c in mod.test_commands]
            self.assertIn("pytest", cmd_names)

    def test_detect_java_spring_boot_maven_project(self):
        """Verify detect java spring boot maven project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pom.xml").write_text(
                """
                <project>
                    <properties><java.version>17</java.version></properties>
                    <dependencies>
                        <dependency><artifactId>spring-boot-starter-web</artifactId></dependency>
                    </dependencies>
                </project>
                """,
                encoding="utf-8",
            )
            (root / "Application.java").write_text("class Application {}", encoding="utf-8")

            detector = ProjectDetector(root)
            mod = detector.analyze_directory(".", ["pom.xml", "Application.java"])

            self.assertIsNotNone(mod)
            self.assertIn("Java", mod.languages)
            self.assertIn("Spring Boot", mod.frameworks)
            self.assertIn("Java 17", mod.frameworks)
            self.assertEqual(mod.build_system, "maven")
            self.assertIn("Application.java", mod.entry_points)
            self.assertTrue(any(c.command == "mvn test" for c in mod.test_commands))
            self.assertTrue(any(c.command == "mvn clean package" for c in mod.build_commands))

    def test_detect_java_gradle_spring_boot(self):
        """Verify detect java gradle spring boot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "build.gradle.kts").write_text(
                """
                plugins {
                    id("org.springframework.boot") version "3.2.0"
                    kotlin("jvm") version "1.9.20"
                }
                """,
                encoding="utf-8",
            )

            detector = ProjectDetector(root)
            mod = detector.analyze_directory(".", ["build.gradle.kts"])

            self.assertIsNotNone(mod)
            self.assertIn("Java", mod.languages)
            self.assertIn("Kotlin", mod.languages)
            self.assertIn("Spring Boot", mod.frameworks)
            self.assertEqual(mod.build_system, "gradle")
            self.assertTrue(any("./gradlew test" in c.command for c in mod.test_commands))

    def test_detect_react_nextjs_typescript_project(self):
        """Verify detect react nextjs typescript project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "package.json").write_text(
                json.dumps({
                    "dependencies": {"next": "15.0.0", "react": "19.0.0"},
                    "devDependencies": {"typescript": "^5.0.0"},
                    "scripts": {"dev": "next dev", "build": "next build", "test": "jest"},
                }),
                encoding="utf-8",
            )
            (root / "tsconfig.json").write_text("{}", encoding="utf-8")
            (root / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'", encoding="utf-8")

            detector = ProjectDetector(root)
            mod = detector.analyze_directory(".", ["package.json", "tsconfig.json", "pnpm-lock.yaml"])

            self.assertIsNotNone(mod)
            self.assertIn("TypeScript", mod.languages)
            self.assertIn("JavaScript", mod.languages)
            self.assertIn("Next.js 15", mod.frameworks)
            self.assertIn("React", mod.frameworks)
            self.assertEqual(mod.build_system, "pnpm")
            self.assertTrue(any("pnpm test" in c.command for c in mod.test_commands))
            self.assertTrue(any("pnpm run build" in c.command for c in mod.build_commands))

    def test_detect_rust_and_go_projects(self):
        """Verify detect rust and go projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Rust
            (root / "Cargo.toml").write_text('[package]\nname = "rust_app"\n[dependencies]\naxum = "0.7"', encoding="utf-8")
            detector = ProjectDetector(root)
            mod_rust = detector.analyze_directory(".", ["Cargo.toml"])
            self.assertIn("Rust", mod_rust.languages)
            self.assertIn("Axum", mod_rust.frameworks)
            self.assertEqual(mod_rust.build_system, "cargo")

            # Go
            (root / "go.mod").write_text("module example.com/go_app\ngo 1.22\nrequire github.com/gin-gonic/gin v1.9.1", encoding="utf-8")
            mod_go = detector.analyze_directory(".", ["go.mod"])
            self.assertIn("Go", mod_go.languages)
            self.assertIn("Gin", mod_go.frameworks)
            self.assertEqual(mod_go.build_system, "go_modules")

    def test_detect_unknown_project(self):
        """Verify detect unknown project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.txt").write_text("sadece bir not", encoding="utf-8")
            detector = ProjectDetector(root)
            mod = detector.analyze_directory(".", ["notes.txt"])
            self.assertIsNone(mod)

    def test_detect_monorepo_multi_module(self):
        """Verify detect monorepo multi module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            be = root / "backend"
            be.mkdir()
            (be / "pom.xml").write_text("<project><dependencies><dependency><artifactId>spring-boot</artifactId></dependency></dependencies></project>", encoding="utf-8")

            fe = root / "frontend"
            fe.mkdir()
            (fe / "package.json").write_text(json.dumps({"dependencies": {"next": "14.2.0", "react": "18.3.0"}}), encoding="utf-8")

            scanner = RepoScanner(root)
            scan_res = scanner.scan()
            detector = ProjectDetector(root)
            root_mod, submodules, meta = detector.detect_project_structure(scan_res)

            self.assertTrue(meta.is_monorepo)
            self.assertEqual(meta.project_type, "monorepo")
            self.assertEqual(len(submodules), 2)
            sub_paths = [sm.path for sm in submodules]
            self.assertIn("backend", sub_paths)
            self.assertIn("frontend", sub_paths)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_scanner_respects_gitignore(self):
        """Verify scanner respects gitignore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".gitignore").write_text("*.log\nsecret_dir/\n!important.log\n", encoding="utf-8")
            (root / "app.py").write_text("print('hello')", encoding="utf-8")
            (root / "test.log").write_text("debug log", encoding="utf-8")
            (root / "important.log").write_text("keep me", encoding="utf-8")
            sec_dir = root / "secret_dir"
            sec_dir.mkdir()
            (sec_dir / "keys.txt").write_text("secret", encoding="utf-8")

            scanner = RepoScanner(root)
            res = scanner.scan()

            self.assertIn("app.py", res.files)
            self.assertIn("important.log", res.files)
            self.assertNotIn("test.log", res.files)
            self.assertNotIn("secret_dir/keys.txt", res.files)

    def test_scanner_excludes_default_generated_directories(self):
        """Verify scanner excludes default generated directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            (root / "src" / "index.js").write_text("console.log(1)", encoding="utf-8")

            (root / "node_modules").mkdir()
            (root / "node_modules" / "dep.js").write_text("var x = 1;", encoding="utf-8")

            (root / "venv").mkdir()
            (root / "venv" / "pip.py").write_text("pass", encoding="utf-8")

            (root / "target").mkdir()
            (root / "target" / "app.jar").write_bytes(b"PK\x03\x04")

            scanner = RepoScanner(root)
            res = scanner.scan()

            self.assertIn("src/index.js", res.files)
            self.assertNotIn("node_modules/dep.js", res.files)
            self.assertNotIn("venv/pip.py", res.files)
            self.assertNotIn("target/app.jar", res.files)

    def test_scanner_binary_file_detection(self):
        """Verify scanner binary file detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (root / "text.txt").write_text("düz metin", encoding="utf-8")
            (root / "nullbyte.dat").write_bytes(b"abc\x00def")

            scanner = RepoScanner(root)
            res = scanner.scan()

            self.assertTrue(res.files["image.png"].is_binary)
            self.assertFalse(res.files["text.txt"].is_binary)
            self.assertTrue(res.files["nullbyte.dat"].is_binary)

    def test_scanner_depth_and_file_limits(self):
        """Verify scanner depth and file limits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            d = root / "l1" / "l2" / "l3"
            d.mkdir(parents=True)
            (d / "deep.txt").write_text("deep", encoding="utf-8")

            # max_depth = 2 verelim
            limits = ScanLimits(max_depth=2, max_files=100)
            scanner = RepoScanner(root, limits=limits)
            res = scanner.scan()

            self.assertTrue(res.is_partial)
            self.assertTrue(any("derinliği" in r for r in res.partial_reasons))
            self.assertNotIn("l1/l2/l3/deep.txt", res.files)

    def test_scanner_workspace_boundary_enforcement(self):
        """Verify scanner workspace boundary enforcement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir) / "workspace"
            ws_root.mkdir()
            outside_dir = Path(tmpdir) / "outside"
            outside_dir.mkdir()
            (outside_dir / "secret.txt").write_text("secret outside", encoding="utf-8")

            (ws_root / "inside.txt").write_text("inside", encoding="utf-8")

            link_path = ws_root / "outside_link"
            try:
                os.symlink(str(outside_dir), str(link_path), target_is_directory=True)
            except (OSError, NotImplementedError):
                pass  # Symlink izni yoksa atla

            scanner = RepoScanner(ws_root, follow_symlinks=False)
            res = scanner.scan()

            self.assertIn("inside.txt", res.files)
            self.assertNotIn("outside_link/secret.txt", res.files)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_important_file_classification(self):
        """Verify important file classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# Demo Project", encoding="utf-8")
            (root / "Dockerfile").write_text("FROM python:3.11", encoding="utf-8")
            (root / ".env.example").write_text("PORT=8000", encoding="utf-8")
            (root / "pytest.ini").write_text("[pytest]", encoding="utf-8")
            (root / "application.yml").write_text("server: port 8080", encoding="utf-8")

            scanner = RepoScanner(root)
            scan_res = scanner.scan()
            detector = ProjectDetector(root)
            important = detector.classify_important_files(scan_res)

            cat_map = {f.path: f.category for f in important}
            self.assertEqual(cat_map.get("README.md"), "readme")
            self.assertEqual(cat_map.get("Dockerfile"), "build_config")
            self.assertEqual(cat_map.get(".env.example"), "env_example")
            self.assertEqual(cat_map.get("pytest.ini"), "test_config")
            self.assertEqual(cat_map.get("application.yml"), "main_config")

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_persistence_save_load_and_corrupted_recovery(self):
        """Verify persistence save load and corrupted recovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("print('hello')", encoding="utf-8")

            store = ProjectMapStore(root)
            self.assertIsNone(store.load())

            # Sahte harita kaydet
            meta = RepoMetadata(workspace_root=str(root), project_name="demo", detected_languages=["Python"])
            pmap = ProjectMap(metadata=meta, directory_tree=["."], file_hashes={"main.py": "123:456"})
            self.assertTrue(store.save(pmap))

            loaded = store.load()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.metadata.project_name, "demo")
            self.assertEqual(loaded.file_hashes.get("main.py"), "123:456")

            store.cache_file.write_text("{bozuk_json: true,", encoding="utf-8")
            self.assertIsNone(store.load())

    def test_incremental_refresh_detects_added_modified_deleted_files(self):
        """Verify incremental refresh detects added modified deleted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "file1.py").write_text("a = 1\n", encoding="utf-8")
            (root / "file2.py").write_text("b = 2\n", encoding="utf-8")

            engine = RepoIntelligenceEngine(root)
            pmap1, diff1 = engine.refresh(force=False)
            self.assertEqual(diff1.get("mode"), "initial_scan")

            time.sleep(0.05)
            (root / "file3.py").write_text("c = 3\n", encoding="utf-8")
            (root / "file1.py").write_text("a = 99999\n", encoding="utf-8")
            (root / "file2.py").unlink()

            pmap2, diff2 = engine.refresh(force=False)
            self.assertEqual(diff2.get("mode"), "incremental")
            self.assertEqual(diff2.get("added_count"), 1)
            self.assertEqual(diff2.get("modified_count"), 1)
            self.assertEqual(diff2.get("deleted_count"), 1)
            self.assertIn("file3.py", diff2["added"])
            self.assertIn("file1.py", diff2["modified"])
            self.assertIn("file2.py", diff2["deleted"])

    def test_incremental_refresh_invalidates_on_manifest_change(self):
        """Verify incremental refresh invalidates on manifest change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "package.json").write_text('{"dependencies": {"react": "18.0.0"}}', encoding="utf-8")

            engine = RepoIntelligenceEngine(root)
            pmap1, _ = engine.refresh(force=False)
            self.assertIn("React", pmap1.metadata.detected_frameworks)

            time.sleep(0.05)
            (root / "package.json").write_text('{"dependencies": {"next": "14.0.0", "react": "18.0.0"}}', encoding="utf-8")

            pmap2, diff2 = engine.refresh(force=False)
            self.assertTrue(diff2.get("reanalyzed"))
            self.assertTrue(any("Next.js" in f for f in pmap2.metadata.detected_frameworks))

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_levels_of_information_retrieval(self):
        """Verify levels of information retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text('[project]\nname = "core"\ndependencies = ["fastapi", "pytest"]', encoding="utf-8")
            (root / "main.py").write_text("import fastapi", encoding="utf-8")

            engine = RepoIntelligenceEngine(root)

            # Level 1
            overview = engine.get_overview()
            self.assertIn("## Proje Mimarisi ve Teknoloji Özeti", overview)
            self.assertIn("Python", overview)
            self.assertIn("FastAPI", overview)
            self.assertIn("pytest", overview)

            # Level 2
            mod_det = engine.get_module_details(".")
            self.assertIn("Modül Detayları", mod_det)
            self.assertIn("pyproject.toml", mod_det)

            # Level 3
            targeted = engine.query_targeted("fastapi")
            self.assertIn("Hedefli Proje Araması: 'fastapi'", targeted)

            # Footprint
            fp = engine.measure_footprint()
            self.assertGreater(fp["level1_overview_chars"], 50)
            self.assertGreater(fp["level1_overview_est_tokens"], 10)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_agent_integration_and_tool_executor_governance(self):
        """Verify agent integration and tool executor governance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# Demo Workspace", encoding="utf-8")
            (root / "package.json").write_text('{"name": "demo", "scripts": {"test": "jest"}}', encoding="utf-8")

            cfg = load_config(workspace_root=root)
            orchestrator = AgentOrchestrator(config=cfg, system_prompt="Test agent")

            out_overview = orchestrator.tool_executor.execute(
                tool_name="get_project_overview",
                arguments={},
                caller_role="main",
                caller_agent="unit_test",
                is_trust_mode=False,
            )
            self.assertIn("Proje Mimarisi ve Teknoloji Özeti", out_overview)
            self.assertIn("JavaScript", out_overview)

            # 2. get_project_modules
            out_mods = orchestrator.tool_executor.execute(
                tool_name="get_project_modules",
                arguments={},
                caller_role="researcher",
                caller_agent="unit_test",
                is_trust_mode=False,
            )
            self.assertIn("demo", out_mods)

            # 3. get_important_files
            out_files = orchestrator.tool_executor.execute(
                tool_name="get_important_files",
                arguments={"category": "readme"},
                caller_role="architect",
                caller_agent="unit_test",
                is_trust_mode=False,
            )
            self.assertIn("README.md", out_files)

            # 4. refresh_project_map
            out_refresh = orchestrator.tool_executor.execute(
                tool_name="refresh_project_map",
                arguments={"force": False},
                caller_role="main",
                caller_agent="unit_test",
                is_trust_mode=False,
            )
            self.assertIn("Artımlı proje taraması tamamlandı", out_refresh)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_performance_and_scan_benchmark(self):
        """Verify performance and scan benchmark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for i in range(50):
                d = root / f"pkg_{i % 5}"
                d.mkdir(exist_ok=True)
                (d / f"module_{i}.py").write_text(f"# module {i}\ndef f_{i}(): pass\n", encoding="utf-8")

            (root / "pyproject.toml").write_text('[project]\nname = "bench"\ndependencies = ["fastapi"]', encoding="utf-8")

            engine = RepoIntelligenceEngine(root)

            t0 = time.perf_counter()
            pmap1 = engine.rescan_and_analyze()
            cold_duration_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            pmap_cache = engine.get_project_map(force_refresh=False)
            cache_duration_ms = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            pmap_inc, diff = engine.refresh(force=False)
            inc_duration_ms = (time.perf_counter() - t2) * 1000

            # Metrikler
            fp = engine.measure_footprint()

            self.assertGreaterEqual(pmap1.metadata.total_files_scanned, 51)
            self.assertLess(cache_duration_ms, cold_duration_ms)
            self.assertEqual(diff.get("added_count"), 0)
            self.assertEqual(diff.get("modified_count"), 0)

            print(
                f"\n[BENCHMARK] Soğuk Tarama: {cold_duration_ms:.2f}ms | "
                f"Önbellek Okuma: {cache_duration_ms:.3f}ms | "
                f"Artımlı Kontrol: {inc_duration_ms:.2f}ms | "
                f"Taranan Dosya: {pmap1.metadata.total_files_scanned} | "
                f"Level 1 Boyutu: {fp['level1_overview_chars']} krk (~{fp['level1_overview_est_tokens']} token)"
            )


if __name__ == "__main__":
    unittest.main()
