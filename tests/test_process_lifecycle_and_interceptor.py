"""Tests for process tree lifecycle, fast command dispatch, and extensible command interceptors."""

from pathlib import Path
import unittest

from evren_code.config import AppConfig
from evren_code.core.agent import is_continuation_prompt
from evren_code.core.symbol_graph.parsers.registry import ParserRegistry
from evren_code.core.symbol_graph.models import SymbolType
from evren_code.core.verification.test_reporter import TestOutputParser
from evren_code.tools.lsp import LSPClient
from evren_code.tools.skeletonizer import skeletonize_generic
from evren_code.tools.shell import (
    CommandInterceptorRegistry,
    calculate_effective_timeout,
    execute_command_with_tree_kill,
    _should_use_powershell,
    _build_shell_invocation,
)


class TestProcessLifecycleAndInterceptor(unittest.TestCase):
    def test_continuation_prompt_recognition(self):
        """Verifies semantic continuation phrases are recognized to prevent task abortion."""
        self.assertTrue(is_continuation_prompt("devam"))
        self.assertTrue(is_continuation_prompt("devam et"))
        self.assertTrue(is_continuation_prompt("yarım kalan son işlemi devam ettir"))
        self.assertTrue(is_continuation_prompt("kaldığın yerden devam et"))
        self.assertTrue(is_continuation_prompt("devam ettir lütfen"))

        self.assertFalse(is_continuation_prompt("yeni bir proje oluştur"))
        self.assertFalse(is_continuation_prompt("analiz raporu hazırla"))

    def test_command_interceptor_extensibility(self):
        """Verifies CommandInterceptorRegistry allows pluggable command transformations and timeouts."""
        registry = CommandInterceptorRegistry()

        # Register a plugin interceptor that appends a headless flag and caps timeout
        def custom_headless_interceptor(cmd: str, tout: int | None) -> tuple[str, int | None]:
            if "my-cli" in cmd and "--headless" not in cmd:
                return f"{cmd} --headless", tout or 120
            return cmd, tout

        registry.register(custom_headless_interceptor)

        # Transformation check
        processed_cmd, timeout = registry.process("my-cli run --profile fast", None)
        self.assertEqual(processed_cmd, "my-cli run --profile fast --headless")
        self.assertEqual(timeout, 120)

        # Non-matching command remains unmodified
        norm_cmd, norm_timeout = registry.process("git status", 30)
        self.assertEqual(norm_cmd, "git status")
        self.assertEqual(norm_timeout, 30)

    def test_calculate_effective_timeout_generic_build_tools(self):
        """Verifies standard build systems (docker, cargo, npm, cmake) receive build timeout."""
        from unittest.mock import MagicMock
        cfg = MagicMock()
        cfg.tools.command_timeout_seconds = 60
        cfg.tools.build_command_timeout_seconds = 300

        self.assertEqual(calculate_effective_timeout("cargo build --release", cfg), 300)
        self.assertEqual(calculate_effective_timeout("docker build -t my-app .", cfg), 300)
        self.assertEqual(calculate_effective_timeout("cmake --build build/", cfg), 300)
        self.assertEqual(calculate_effective_timeout("git status", cfg), 60)

    def test_csharp_skeletonizer_generic_class(self):
        """Verifies C# methods with Allman braces are folded and achieve high token savings."""
        csharp_code = '''using System;

namespace Enterprise.Services
{
    public class OrderProcessor
    {
        public int ProcessCount = 0;

        public void ProcessOrder(int orderId)
        {
            ProcessCount++;
            if (orderId <= 0)
            {
                Cancel();
            }
        }

        private void Cancel()
        {
            Console.WriteLine("Order cancelled");
        }
    }
}
'''
        skel = skeletonize_generic(csharp_code, ".cs")
        self.assertIn("class OrderProcessor", skel)
        self.assertIn("ProcessOrder", skel)
        self.assertIn("Cancel", skel)
        self.assertIn("gövde gizlendi", skel)
        self.assertNotIn('Console.WriteLine("Order cancelled")', skel)

        # Test focus_symbol keeps full body
        skel_focused = skeletonize_generic(csharp_code, ".cs", focus_symbol="Cancel")
        self.assertIn('Console.WriteLine("Order cancelled")', skel_focused)

    def test_csharp_symbol_parsing(self):
        """Verifies CSharpParser extracts classes and methods from .cs files."""
        csharp_code = '''using System;

namespace Enterprise.Services
{
    public class UserService
    {
        public void Authenticate()
        {
        }
    }
}
'''
        reg = ParserRegistry()
        res = reg.parse_file("UserService.cs", csharp_code)
        self.assertEqual(res.language, "csharp")
        sym_names = [s.name for s in res.symbols]
        self.assertIn("UserService", sym_names)
        self.assertIn("Authenticate", sym_names)

    def test_csharp_lsp_diagnostics(self):
        """Verifies C# syntax errors like unclosed braces are caught."""
        tmp = Path("tmp_syntax_test.cs")
        try:
            tmp.write_text("using System\npublic class Foo {\n", encoding="utf-8")
            client = LSPClient(Path("."))
            diags = client.get_diagnostics("tmp_syntax_test.cs")
            self.assertTrue(len(diags) >= 1)
            msgs = " ".join(d.message for d in diags)
            self.assertIn("süslü parantez", msgs.lower())
        finally:
            tmp.unlink(missing_ok=True)

    def test_test_output_parser_inspection_not_falsely_verified(self):
        """Verifies inspection command outputs do not claim tests passed."""
        ps_output = "Handles  NPM(K)    PM(K)      WS(K)     CPU(s)     Id ProcessName\n"
        metrics = TestOutputParser.parse_output(ps_output)
        self.assertEqual(metrics.tests_executed, 0)
        self.assertEqual(metrics.build_status, "UNKNOWN")

        # Genuine pytest output
        pytest_out = "===== 3 passed in 0.12s ====="
        metrics_pt = TestOutputParser.parse_output(pytest_out)
        self.assertEqual(metrics_pt.framework, "pytest")
        self.assertEqual(metrics_pt.tests_passed, 3)
        self.assertEqual(metrics_pt.build_status, "SUCCESS")

    def test_execute_command_with_tree_kill_timeout(self):
        """Verifies execute_command_with_tree_kill handles timeouts cleanly without pipe deadlocks."""
        import subprocess
        # Run a command that sleeps longer than timeout
        with self.assertRaises(subprocess.TimeoutExpired):
            execute_command_with_tree_kill(
                ["powershell.exe", "-NoProfile", "-Command", "Start-Sleep -Seconds 10"],
                cwd=".",
                timeout=1,
            )

    def test_should_use_powershell_and_cmd_dispatch(self):
        """Verifies CLI commands without PS syntax are routed through fast CMD."""
        self.assertFalse(_should_use_powershell("git status -sb"))
        self.assertFalse(_should_use_powershell("git commit -m 'feat: test'"))
        self.assertFalse(_should_use_powershell("pytest -v tests/"))
        self.assertFalse(_should_use_powershell("python script.py"))

        self.assertTrue(_should_use_powershell("& my-tool.exe --run"))
        self.assertTrue(_should_use_powershell("Get-Process node"))
        self.assertTrue(_should_use_powershell("Select-String -Path foo.log -Pattern 'err'"))
        self.assertTrue(_should_use_powershell("$x = 10"))

        args, is_shell = _build_shell_invocation("git status")
        self.assertEqual(args[0], "cmd.exe")


if __name__ == "__main__":
    unittest.main()
