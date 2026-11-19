"""Test superpowers module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.memory import ProjectMemoryManager
from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem
from evren_code.core.verification.engine import VerificationEngine
from evren_code.core.verification.tools import register_verification_tools
from evren_code.tools.registry import ToolRegistry


class TestSuperpowers(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.config = load_config()
        self.config.workspace_root = self.workspace
        self.config.agent.dual_review_enabled = False

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


    def test_dual_model_pair_review_approved(self) -> None:
        """Verify dual model pair review approved."""
        orchestrator = AgentOrchestrator(config=self.config)
        orchestrator.trust_mode = True
        orchestrator.config.agent.dual_review_enabled = True

        tc = ToolCallItem(
            id="call_w1",
            name="write_file",
            arguments={"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}
        )

        mock_step1 = ParsedAssistantResponse(
            content="Toplama fonksiyonu yazıldı.",
            tool_calls=[tc],
        )
        mock_step2 = ParsedAssistantResponse(content="Tamamlandı.")

        call_idx = [0]
        def mock_chat(**kwargs):
            call_idx[0] += 1
            if kwargs.get("model_override") == "deepseek-v4.1-flash":
                return ParsedAssistantResponse(content="APPROVED")
            if call_idx[0] == 1:
                return mock_step1
            return mock_step2

        class MockLLM:
            active_model = orchestrator.config.models.get_model("coder")
            chat = staticmethod(mock_chat)

        orchestrator.llm_client = MockLLM()
        result = orchestrator.run_turn("Hesaplama modülü yaz", stream=False)

        self.assertIn("Mimari Hakem Onayı (Pair Review)", result)

    def test_dual_model_pair_review_rejected_triggers_correction(self) -> None:
        """Verify dual model pair review rejected triggers correction."""
        orchestrator = AgentOrchestrator(config=self.config)
        orchestrator.trust_mode = True
        orchestrator.config.agent.dual_review_enabled = True

        tc1 = ToolCallItem(
            id="call_bad",
            name="write_file",
            arguments={"path": "auth.py", "content": "API_KEY = 'hardcoded_secret_123'\n"}
        )

        mock_step1 = ParsedAssistantResponse(content="Yazıldı.", tool_calls=[tc1])
        mock_step2 = ParsedAssistantResponse(content="Bitti.")

        tc2 = ToolCallItem(
            id="call_fix",
            name="write_file",
            arguments={"path": "auth.py", "content": "import os\nAPI_KEY = os.environ.get('API_KEY')\n"}
        )
        mock_step3 = ParsedAssistantResponse(content="Düzeltildi.", tool_calls=[tc2])
        mock_step4 = ParsedAssistantResponse(content="Son teslim.")

        step_counter = [0]
        arch_reviews = [0]
        def mock_chat_dual(**kwargs):
            if kwargs.get("model_override") == "deepseek-v4.1-flash":
                arch_reviews[0] += 1
                if arch_reviews[0] == 1:
                    return ParsedAssistantResponse(content="REJECTED: Sert kodlanmış gizli anahtar (hardcoded secret) güvenlik riski taşıyor.")
                return ParsedAssistantResponse(content="APPROVED")

            step_counter[0] += 1
            if step_counter[0] == 1:
                return mock_step1
            elif step_counter[0] == 2:
                return mock_step2
            elif step_counter[0] == 3:
                return mock_step3
            return mock_step4

        class MockLLM:
            active_model = orchestrator.config.models.get_model("coder")
            chat = staticmethod(mock_chat_dual)

        orchestrator.llm_client = MockLLM()
        result = orchestrator.run_turn("Auth servisi yaz", stream=False)

        self.assertGreaterEqual(arch_reviews[0], 2)
        self.assertIn("Mimari Hakem Onayı (Pair Review)", result)


    def test_synthesize_unit_test(self) -> None:
        """Verify synthesize unit test."""
        src_file = self.workspace / "math_utils.py"
        src_file.write_text("def multiply(x, y):\n    return x * y\n", encoding="utf-8")

        reg = ToolRegistry()
        mock_executor = MagicMock()
        mock_executor.execute.return_value = "Ran 1 test in 0.001s - OK"

        mock_ve = MagicMock()
        mock_ve.workspace_root = self.workspace
        mock_ve.tool_executor = mock_executor

        mock_llm = MagicMock()
        mock_llm.config = self.config
        test_py_code = (
            "import unittest\n"
            "from math_utils import multiply\n\n"
            "class TestMath(unittest.TestCase):\n"
            "    def test_multiply(self):\n"
            "        self.assertEqual(multiply(2, 3), 6)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        mock_llm.chat.return_value = ParsedAssistantResponse(content=f"```python\n{test_py_code}\n```")

        register_verification_tools(reg, mock_ve, llm_client=mock_llm)

        res = reg.execute("synthesize_unit_test", {"target_file": "math_utils.py", "execute_immediately": True})
        self.assertIn("Otomatik Test Sentezi Tamamlandı", res)
        self.assertIn("test_math_utils.py", res)

        test_file = self.workspace / "tests" / "test_math_utils.py"
        self.assertTrue(test_file.is_file())
        self.assertIn("def test_multiply", test_file.read_text(encoding="utf-8"))


    def test_query_project_memory(self) -> None:
        """Verify query project memory."""
        mgr = ProjectMemoryManager(self.workspace)
        mgr.remember_fact("Backend testleri pytest -m unit ile çalıştırılır.", category="Test")
        mgr.remember_fact("JWT token gizli anahtarı .env dosyasından okunur.", category="Güvenlik")
        mgr.remember_fact("Veritabanı bağlantı havuzu boyutu 20 olarak yapılandırılmıştır.", category="Mimari")

        # 1. JWT sorgusu
        res_jwt = mgr.query_memory("JWT token")
        self.assertIn("JWT token gizli anahtarı", res_jwt)
        self.assertNotIn("Veritabanı bağlantı havuzu", res_jwt)

        # 2. Test komutu sorgusu
        res_test = mgr.query_memory("test çalıştırma komutu")
        self.assertIn("pytest -m unit", res_test)

    def test_self_healing_auto_learns_to_memory(self) -> None:
        """Verify self healing auto learns to memory."""
        orchestrator = AgentOrchestrator(config=self.config)
        orchestrator.memory_manager.remember_fact(
            "Proje doğrulama komutu `py -3.11 -m unittest` testleri başarıyla geçiyor.",
            category="Derleme/Test",
        )
        mem = orchestrator.memory_manager.load_memory()
        self.assertIn("py -3.11 -m unittest", mem)
        self.assertIn("Derleme/Test", mem)


if __name__ == "__main__":
    unittest.main()
