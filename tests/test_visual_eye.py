"""Test visual eye module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.parser import ParsedAssistantResponse
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.vision import (
    capture_web_page_screenshot,
    find_system_browser,
    register_vision_tools,
)


class TestVisualEye(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.config = load_config()
        self.config.workspace_root = self.workspace
        self.registry = ToolRegistry()
        self.mock_llm = MagicMock()

        # Mock LLM vision response
        self.mock_llm.chat.return_value = ParsedAssistantResponse(
            content="Görsel Düzen: Başlık ortalanmış, butonlar kontrastlı ve taşma tespit edilmedi."
        )
        register_vision_tools(self.registry, self.config, self.mock_llm)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_find_system_browser(self) -> None:
        """Verify find system browser."""
        browser = find_system_browser()
        self.assertIsNotNone(browser)
        self.assertTrue(Path(browser).is_file())

    def test_capture_web_page_screenshot_desktop_and_mobile(self) -> None:
        """Verify capture web page screenshot desktop and mobile."""
        html_file = self.workspace / "index.html"
        html_file.write_text(
            "<!DOCTYPE html><html><body style='background:#111;color:#fff;'><h1>Test Landing Page</h1></body></html>",
            encoding="utf-8",
        )

        ok_d, msg_d, path_d = capture_web_page_screenshot(
            self.workspace,
            "index.html",
            viewport="desktop",
        )
        self.assertTrue(ok_d, f"Masaüstü screenshot hatası: {msg_d}")
        self.assertIsNotNone(path_d)
        self.assertTrue(path_d.is_file())
        self.assertGreater(path_d.stat().st_size, 0)

        ok_m, msg_m, path_m = capture_web_page_screenshot(
            self.workspace,
            "index.html",
            viewport="mobile",
        )
        self.assertTrue(ok_m, f"Mobil screenshot hatası: {msg_m}")
        self.assertIsNotNone(path_m)
        self.assertTrue(path_m.is_file())
        self.assertGreater(path_m.stat().st_size, 0)

    def test_capture_web_screenshot_tool(self) -> None:
        """Verify capture web screenshot tool."""
        html_file = self.workspace / "test.html"
        html_file.write_text("<!DOCTYPE html><html><body><h1>Tool Test</h1></body></html>", encoding="utf-8")

        res = self.registry.execute("capture_web_screenshot", {"path_or_url": "test.html"})
        self.assertIn("Başarılı", res)
        self.assertIn("ekran görüntüsü kaydedildi", res)

    def test_inspect_web_page_visually_tool(self) -> None:
        """Verify inspect web page visually tool."""
        html_file = self.workspace / "landing.html"
        html_file.write_text(
            "<!DOCTYPE html><html><body><div class='hero'>Hero Section</div></body></html>",
            encoding="utf-8",
        )

        res = self.registry.execute(
            "inspect_web_page_visually",
            {"path_or_url": "landing.html", "viewport": "desktop"},
        )
        self.assertIn("Visual Eye - Canlı Görsel UI İnceleme Raporu", res)
        self.assertIn("Görsel Düzen:", res)
        self.mock_llm.chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
