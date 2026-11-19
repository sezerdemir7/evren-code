"""Test filesystem patching module."""

from pathlib import Path
import tempfile
import unittest

from evren_code.config import load_config
from evren_code.tools.filesystem import register_filesystem_tools
from evren_code.tools.registry import ToolRegistry


class TestFilesystemPatching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.config = load_config(workspace_root=self.workspace)
        self.config.workspace_root = self.workspace
        self.registry = ToolRegistry()
        register_filesystem_tools(self.registry, self.config)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fuzzy_micro_single_quote_double_quote(self):
        """Verify fuzzy micro single quote double quote."""
        file_p = self.workspace / "config.py"
        file_p.write_text("DEBUG = True\nPORT = 8080\nSECRET_KEY = 'insecure'\n", encoding="utf-8")

        res = self.registry.execute(
            "edit_file",
            {
                "path": "config.py",
                "target": 'SECRET_KEY="insecure"',
                "replacement": "SECRET_KEY = os.getenv('SECRET_KEY')",
            },
        )
        self.assertIn("Başarılı", res)
        updated = file_p.read_text(encoding="utf-8")
        self.assertIn("os.getenv('SECRET_KEY')", updated)
        self.assertNotIn("'insecure'", updated)

    def test_fuzzy_micro_single_line_high_similarity(self):
        """Verify fuzzy micro single line high similarity."""
        file_p = self.workspace / "math_utils.py"
        file_p.write_text(
            "def calculate_discount(price: float) -> float:\n    # Standart indirim\n    return price * 0.90\n",
            encoding="utf-8",
        )

        res = self.registry.execute(
            "edit_file",
            {
                "path": "math_utils.py",
                "target": "return price * 0.9",
                "replacement": "return price * 0.85",
            },
        )
        self.assertIn("Başarılı", res)
        updated = file_p.read_text(encoding="utf-8")
        self.assertIn("return price * 0.85", updated)

    def test_apply_patch_unified_diff(self):
        """Verify apply patch unified diff."""
        file_p = self.workspace / "service.py"
        file_p.write_text(
            "def run_service():\n    status = 'starting'\n    print(status)\n    return True\n",
            encoding="utf-8",
        )

        patch = (
            "@@ -1,4 +1,4 @@\n"
            " def run_service():\n"
            "-    status = 'starting'\n"
            "+    status = 'running'\n"
            "     print(status)\n"
            "     return True\n"
        )

        res = self.registry.execute(
            "apply_patch",
            {
                "path": "service.py",
                "patch": patch,
            },
        )
        self.assertIn("Başarılı", res)
        updated = file_p.read_text(encoding="utf-8")
        self.assertIn("status = 'running'", updated)
        self.assertNotIn("status = 'starting'", updated)

    def test_edit_file_delegates_to_apply_patch_when_diff_given(self):
        """Verify edit file delegates to apply patch when diff given."""
        file_p = self.workspace / "app.py"
        file_p.write_text("def index():\n    return 'v1'\n", encoding="utf-8")

        patch = (
            "@@ -1,2 +1,2 @@\n"
            " def index():\n"
            "-    return 'v1'\n"
            "+    return 'v2'\n"
        )

        res = self.registry.execute(
            "edit_file",
            {
                "path": "app.py",
                "target": patch,
                "replacement": "",
            },
        )
        self.assertIn("Başarılı", res)
        updated = file_p.read_text(encoding="utf-8")
        self.assertIn("return 'v2'", updated)


if __name__ == "__main__":
    unittest.main()
