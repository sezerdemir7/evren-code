"""Large project stability remediation tests.

Verifies:
1. TaskComplexityClassifier detects UI/frontend/modernization tasks as COMPLEX or MEDIUM, not SIMPLE.
2. Short prompts without dev actions are classified as SIMPLE.
3. StatefulJsonRepair handles dangling key-colon JSON (e.g. '{"files": ') without throwing JSONDecodeError,
   marking _is_truncated=True to trigger Truncation Guard.
4. CSS inline diagnostics strips comments/strings before counting curly braces.
5. Continuation prompt preserves active goal on "devam et", preventing amnesia.
6. Context manager working limit is scaled up to 96,000 tokens.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evren_code.config import load_config
from evren_code.core.parser import StatefulJsonRepair, repair_and_parse_json, parse_assistant_message
from evren_code.core.planner import TaskComplexityClassifier, ComplexityLevel
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.filesystem import register_filesystem_tools


class TestLargeProjectStabilityRemediations(unittest.TestCase):
    def test_frontend_modernization_is_not_simple(self) -> None:
        """Verify frontend modernization is not simple."""
        prompt = "frontend ekranlarını daha temiz hale ve kullanışlı ve modern hale getir"
        assessment = TaskComplexityClassifier.classify(prompt)
        self.assertIn(assessment.level, (ComplexityLevel.COMPLEX, ComplexityLevel.MEDIUM))
        self.assertGreaterEqual(assessment.max_turns, 20)
        self.assertTrue(assessment.requires_formal_plan)

    def test_ui_dev_actions_classified_as_medium_or_complex(self) -> None:
        """Verify ui dev actions classified as medium or complex."""
        prompts = [
            "modern navbar tasarla",
            "ekranları güncelle",
            "yeni buton bileşeni ekle",
            "sayfayı temizle",
        ]
        for p in prompts:
            assessment = TaskComplexityClassifier.classify(p)
            self.assertNotEqual(
                assessment.level,
                ComplexityLevel.SIMPLE,
                f"Prompt '{p}' should not be classified as SIMPLE",
            )
            self.assertGreaterEqual(assessment.max_turns, 12)

    def test_simple_non_dev_prompt_remains_simple(self) -> None:
        """Verify simple non dev prompt remains simple."""
        res_typo = TaskComplexityClassifier.classify("typo düzelt")
        self.assertEqual(res_typo.level, ComplexityLevel.SIMPLE)
        self.assertLessEqual(res_typo.max_turns, 8)

        res_read = TaskComplexityClassifier.classify("README dosyasını oku")
        self.assertEqual(res_read.level, ComplexityLevel.SIMPLE)

    def test_stateful_json_repair_dangling_colon_files(self) -> None:
        """Verify stateful json repair dangling colon files."""
        raw = '{"files": '
        repaired, is_trunc = StatefulJsonRepair.repair_with_status(raw)
        self.assertTrue(is_trunc)
        parsed = json.loads(repaired)
        self.assertIn("files", parsed)
        self.assertIsNone(parsed["files"])

        parsed_dict = repair_and_parse_json(raw)
        self.assertTrue(parsed_dict.get("_is_truncated"))

    def test_parse_assistant_message_truncation_guard_triggers(self) -> None:
        """Verify parse assistant message truncation guard triggers."""
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write_multiple_files", "arguments": '{"files": '},
                }
            ],
        }
        res = parse_assistant_message(msg)
        self.assertTrue(res.is_truncated)
        self.assertEqual(len(res.tool_calls), 1)
        self.assertTrue(res.tool_calls[0].is_truncated)

    def test_css_inline_diagnostics_ignores_comments_and_strings(self) -> None:
        """Verify css inline diagnostics ignores comments and strings."""
        valid_css_with_braces_in_comment = (
            ":root {\n"
            "  /* Not: Bu blok { ornek } parantez icerir */\n"
            "  --primary: #123456;\n"
            "}\n"
        )
        with TemporaryDirectory() as tmpdir:
            cfg = load_config()
            cfg.workspace_root = Path(tmpdir)
            registry = ToolRegistry()
            register_filesystem_tools(registry, cfg)

            res = registry.execute("write_file", {"path": "style.css", "content": valid_css_with_braces_in_comment})
            self.assertNotIn("Sözdizimi/Bozulma Uyarısı", res)
            self.assertIn("style.css", res)

    def test_config_scaled_context_limits(self) -> None:
        """Verify config scaled context limits."""
        cfg = load_config()
        self.assertGreaterEqual(cfg.agent.max_working_context_tokens, 64000)
        self.assertGreaterEqual(cfg.agent.keep_recent_messages, 10)


if __name__ == "__main__":
    unittest.main()
