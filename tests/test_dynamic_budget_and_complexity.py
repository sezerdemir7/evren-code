"""Test dynamic budget and complexity module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.planner import (
    ComplexityLevel,
    TaskComplexityClassifier,
)


class TestDynamicBudgetAndComplexity(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="evren_test_budget_"))
        self.config = load_config()
        self.config.workspace_root = self.tmp_dir

    def test_short_prompts_with_development_verbs_are_not_simple(self) -> None:
        """Verify short prompts with development verbs are not simple."""
        res_paket = TaskComplexityClassifier.classify("paket b yi yap şimdi")
        self.assertEqual(res_paket.level, ComplexityLevel.COMPLEX)
        self.assertGreaterEqual(res_paket.max_turns, 40)

        res_login = TaskComplexityClassifier.classify("login ekranını yap")
        self.assertEqual(res_login.level, ComplexityLevel.MEDIUM)
        self.assertGreaterEqual(res_login.max_turns, 20)

        res_auth = TaskComplexityClassifier.classify("auth akışını bağla ve kur")
        self.assertEqual(res_auth.level, ComplexityLevel.MEDIUM)
        self.assertGreaterEqual(res_auth.max_turns, 20)

        res_tests = TaskComplexityClassifier.classify("bütün testleri düzelt")
        self.assertEqual(res_tests.level, ComplexityLevel.COMPLEX)
        self.assertGreaterEqual(res_tests.max_turns, 40)

    def test_truly_simple_inquiries_and_typos_remain_simple(self) -> None:
        """Verify truly simple inquiries and typos remain simple."""
        res_typo = TaskComplexityClassifier.classify("typo düzelt utils.py")
        self.assertEqual(res_typo.level, ComplexityLevel.SIMPLE)
        self.assertLessEqual(res_typo.max_turns, 8)

        res_q = TaskComplexityClassifier.classify("bu fonksiyon nedir açıkla")
        self.assertEqual(res_q.level, ComplexityLevel.SIMPLE)
        self.assertLessEqual(res_q.max_turns, 8)

    def test_dynamic_budget_expansion_on_active_progress(self) -> None:
        """Verify dynamic budget expansion on active progress."""
        orc = AgentOrchestrator(self.config)

        prompt = "paket b yi yap şimdi"
        assessment = TaskComplexityClassifier.classify(prompt)

        self.assertGreaterEqual(assessment.max_turns, 35)
        self.assertFalse(assessment.level == ComplexityLevel.SIMPLE)


if __name__ == "__main__":
    unittest.main()
