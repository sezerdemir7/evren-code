"""Skeletonizer module."""

import tempfile
import unittest
from pathlib import Path

from evren_code.config import load_config
from evren_code.tools.profiler import register_profiler_tools, run_benchmark_subprocess
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.skeletonizer import (
    register_skeletonizer_tools,
    skeletonize_file_content,
    skeletonize_generic,
    skeletonize_python,
)


class TestSkeletonizer(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.config = load_config()
        self.config.workspace_root = self.workspace
        self.registry = ToolRegistry()
        register_skeletonizer_tools(self.registry, self.config)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_skeletonize_python_with_focus_symbol(self) -> None:
        code = '''"""Örnek modül başlığı."""
import os
import sys

GLOBAL_VAR = "test"

def helper_one(x: int) -> int:
    """Birinci yardımcı."""
    a = x * 2
    b = a + 5
    return b

def critical_algorithm(data: list[int]) -> int:
    """Kritik algoritma."""
    total = 0
    for item in data:
        total += item * 2
    return total

class DataProcessor:
    """Veri işleyici sınıfı."""
    threshold: int = 100

    def process_items(self, items: list[str]) -> list[str]:
        """Tüm öğeleri işle."""
        res = []
        for it in items:
            res.append(it.strip().upper())
        return res
'''
        skel_all = skeletonize_python(code, focus_symbol=None)
        self.assertIn("def helper_one(x: int) -> int:", skel_all)
        self.assertIn("def critical_algorithm(data: list[int]) -> int:", skel_all)
        self.assertIn("class DataProcessor:", skel_all)
        self.assertIn("gövde gizlendi", skel_all)
        self.assertNotIn("total += item * 2", skel_all)

        skel_focus = skeletonize_python(code, focus_symbol="critical_algorithm")
        self.assertIn("total += item * 2", skel_focus)
        self.assertIn("def helper_one(x: int) -> int:", skel_focus)
        self.assertIn("gövde gizlendi", skel_focus)

    def test_skeletonize_generic_ts(self) -> None:
        ts_code = """import { useState } from 'react';

export function calculateTotal(items: number[]): number {
    let sum = 0;
    for (const n of items) {
        sum += n;
    }
    return sum;
}

export function unusedHelper(): void {
    console.log("Unused");
}
"""
        skel_ts = skeletonize_generic(ts_code, ".ts", focus_symbol="calculateTotal")
        self.assertIn("calculateTotal", skel_ts)
        self.assertIn("sum += n;", skel_ts)
        self.assertIn("unusedHelper", skel_ts)
        self.assertIn("gövde gizlendi", skel_ts)

    def test_skeletonize_code_tool(self) -> None:
        py_file = self.workspace / "service.py"
        py_file.write_text(
            'def foo():\n    return 1\ndef bar():\n    return 2\n',
            encoding="utf-8",
        )
        res = self.registry.execute("skeletonize_code", {"path": "service.py", "focus_symbol": "foo"})
        self.assertIn("AST Code Skeleton", res)
        self.assertIn("Token Tasarrufu", res)
        self.assertIn("return 1", res)


class TestProfiler(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.config = load_config()
        self.config.workspace_root = self.workspace
        self.registry = ToolRegistry()
        register_profiler_tools(self.registry, self.config)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_run_benchmark_subprocess_snippet(self) -> None:
        snippet = "x = [i**2 for i in range(1000)]"
        stats = run_benchmark_subprocess(
            workspace_root=self.workspace,
            code_to_profile=snippet,
            iterations=20,
            warmup_runs=2,
            timeout_seconds=5.0,
        )
        self.assertTrue(stats.get("success"), f"Benchmark hatası: {stats.get('error')}")
        self.assertEqual(stats["iterations"], 20)
        self.assertGreater(stats["mean_ms"], 0.0)
        self.assertGreaterEqual(stats["peak_memory_kb"], 0.0)

    def test_run_benchmark_timeout_safety(self) -> None:
        infinite_loop = "while True: pass"
        stats = run_benchmark_subprocess(
            workspace_root=self.workspace,
            code_to_profile=infinite_loop,
            iterations=1,
            warmup_runs=0,
            timeout_seconds=1.0,
        )
        self.assertFalse(stats.get("success"))
        self.assertTrue("timed out" in stats.get("error", "").lower() or "zaman aşımına" in stats.get("error", "").lower())

    def test_profile_code_performance_tool(self) -> None:
        res = self.registry.execute(
            "profile_code_performance",
            {
                "code_snippet": "data = sum(range(5000))",
                "iterations": 15,
            },
        )
        self.assertIn("Otonom Performans & Bellek Profil Raporu", res)
        self.assertIn("Yürütme Süresi (Latency)", res)
        self.assertIn("Tepe Bellek Tüketimi", res)
        self.assertIn("Mühendislik Değerlendirmesi", res)


if __name__ == "__main__":
    unittest.main()
