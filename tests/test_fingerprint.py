"""Test fingerprint module."""

import json
import tempfile
import unittest
from pathlib import Path

from evren_code.core.fingerprint import WorkspaceFingerprinter
from evren_code.core.context import ContextManager
from evren_code.config import load_config


class TestWorkspaceFingerprinter(unittest.TestCase):
    def test_single_project_python(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text('[project]\nname = "demo"\ndependencies = ["fastapi", "pytest"]', encoding="utf-8")
            (root / "src").mkdir()

            fp = WorkspaceFingerprinter(root).detect()
            self.assertIn("Tekil Proje", fp)
            self.assertIn("Python (FastAPI)", fp)
            self.assertIn("pytest", fp)

    def test_monorepo_node_and_java(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fe_dir = root / "frontend"
            fe_dir.mkdir()
            (fe_dir / "package.json").write_text(json.dumps({
                "dependencies": {
                    "next": "16.4.3",
                    "react": "19.3.0",
                    "tailwindcss": "^4"
                },
                "devDependencies": {
                    "typescript": "^5"
                }
            }), encoding="utf-8")

            be_dir = root / "backend"
            be_dir.mkdir()
            (be_dir / "pom.xml").write_text("""
            <project>
                <properties><java.version>21</java.version></properties>
                <dependencies>
                    <dependency><artifactId>spring-boot-starter-web</artifactId></dependency>
                    <dependency><artifactId>postgresql</artifactId></dependency>
                </dependencies>
            </project>
            """, encoding="utf-8")

            fp = WorkspaceFingerprinter(root).detect()
            self.assertIn("Çoklu Modül / Monorepo", fp)
            self.assertIn("frontend", fp)
            self.assertIn("Next.js 16", fp)
            self.assertIn("TypeScript", fp)
            self.assertIn("backend", fp)
            self.assertIn("Java Spring Boot (Maven)", fp)
            self.assertIn("Java 21", fp)
            self.assertIn("PostgreSQL", fp)

    def test_context_manager_fingerprint_injection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pubspec.yaml").write_text("name: flutter_app\nenvironment:\n  sdk: '>=3.0.0 <4.0.0'", encoding="utf-8")

            cfg = load_config(workspace_root=root)
            ctx = ContextManager(config=cfg, base_system_prompt="Sen bir test ajanısın.")

            sys_msg = ctx.build_system_message()
            content = sys_msg["content"]
            self.assertIn("Flutter / Dart", content)
            self.assertIn("Workspace Fingerprint", content)


if __name__ == "__main__":
    unittest.main()
