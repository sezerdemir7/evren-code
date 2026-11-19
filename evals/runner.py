"""Evren-Code Evaluation (Benchmark) Runner:
Measures agent task pass rate, step count, token consumption, and loop resilience.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from evren_code.config import AppConfig, load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.events import EventBus, EventType
from evren_code.core.evren_auth import EvrenAuthManager


class EvalRunner:
    def __init__(
        self,
        tasks_file: Path,
        model_override: str | None = None,
        task_filter: str | None = None,
        output_path: Path | None = None,
        use_mock: bool = False,
    ) -> None:
        self.tasks_file = tasks_file
        self.model_override = model_override
        self.task_filter = task_filter
        self.output_path = output_path
        self.use_mock = use_mock

    def load_tasks(self) -> List[Dict[str, Any]]:
        with open(self.tasks_file, "r", encoding="utf-8") as f:
            tasks: List[Dict[str, Any]] = json.load(f)
        if self.task_filter:
            flt = self.task_filter.lower()
            tasks = [
                t
                for t in tasks
                if flt in str(t.get("id", "")).lower()
                or flt in str(t.get("category", "")).lower()
                or flt in str(t.get("name", "")).lower()
            ]
        return tasks

    def run_all(self) -> List[Dict[str, Any]]:
        tasks = self.load_tasks()
        print(f"\n=========================================================================")
        print(f"🚀 Evren-Code Evaluation Benchmark Engine Initializing...")
        print(f"📊 Total Task Count: {len(tasks)}")
        if self.model_override:
            print(f"🤖 Model: {self.model_override}")
        print(f"=========================================================================\n")

        results = []
        total_count = len(tasks)
        for idx, task in enumerate(tasks, start=1):
            task["total"] = total_count
            res = self.run_single_task(idx, task)
            results.append(res)

        self._print_summary(results)
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"💾 Benchmark report saved: {self.output_path}\n")
        return results

    def run_single_task(self, index: int, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task.get("id", f"task_{index}")
        name = task.get("name", task_id)
        max_allowed_lines = int(task.get("max_lines", 300))
        print(f"[{index}/{task.get('total', 12)}] Running: {name} ({task_id})...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Write Setup Files
            setup_files = task.get("setup_files", {})
            for rel_p, content in setup_files.items():
                target_p = tmp_path / rel_p
                target_p.parent.mkdir(parents=True, exist_ok=True)
                target_p.write_text(content, encoding="utf-8")

            # 2. Initialize Config and Agent in Isolated Workspace
            config = load_config(workspace_root=tmp_path)
            auth_mgr = EvrenAuthManager(config)
            stored_key = auth_mgr.load_stored_key()
            if stored_key:
                config.api.api_key = stored_key
            auth_mgr.load_models_cache()
            config.workspace_root = tmp_path
            config.agent.max_iterations = task.get("max_steps", 6)
            config.agent.auto_continue = False  # Fixed turn quota during evals
            config.tools.max_recommended_file_lines = max_allowed_lines
            config.telemetry.enabled = False
            config.multi_agent.enabled = False
            config.multi_agent.auto_decompose = False

            event_bus = EventBus()
            warnings_count = 0
            steps_taken = 0
            tool_trace: List[str] = []

            def on_warning(event):
                nonlocal warnings_count
                warnings_count += 1
                print(f"    ⚠ Warning: {event.data.get('message', '')}")

            def on_step(event):
                nonlocal steps_taken
                steps_taken += 1

            def on_tool(event):
                tname = event.data.get("tool_name", "")
                tool_trace.append(f"#{steps_taken}:{tname}")

            event_bus.subscribe(on_warning, EventType.WARNING)
            event_bus.subscribe(on_step, EventType.STEP_STARTED)
            event_bus.subscribe(on_tool, EventType.TOOL_CALL_DETECTED)

            orchestrator = AgentOrchestrator(
                config=config,
                event_bus=event_bus,
            )
            orchestrator.trust_mode = True  # Auto-approve actions in eval mode

            if self.use_mock:
                from evals.mock_provider import MockLLMClient
                orchestrator.llm_client = MockLLMClient(config, task)

            start_t = time.time()
            prompt = task.get("user_prompt", "")
            try:
                orchestrator.run_turn(
                    prompt,
                    model_override=self.model_override,
                    stream=True,
                )
                duration = round(time.time() - start_t, 2)
            except Exception as e:
                duration = round(time.time() - start_t, 2)
                print(f"  ❌ Execution Error: {e}")
                return {
                    "id": task_id,
                    "name": name,
                    "passed": False,
                    "reason": f"Exception: {e}",
                    "steps": steps_taken,
                    "duration": duration,
                    "tokens": 0,
                    "loc": 0,
                    "lean_ok": False,
                    "warnings": warnings_count,
                }

            tokens_used = (
                orchestrator.last_turn_usage.get("total_tokens")
                or orchestrator.context.estimate_tokens()
            )

            # 3. Verify Expected Files, Code Size (Anti-Bloat), and Execution
            expected_files = task.get("expected_files", {})
            passed = True
            lean_ok = True
            total_loc = 0
            failure_reasons = []

            for exp_path, exp_snippet in expected_files.items():
                actual_file = tmp_path / exp_path
                if not actual_file.exists():
                    passed = False
                    failure_reasons.append(f"File '{exp_path}' was not created/found.")
                    continue
                content = actual_file.read_text(encoding="utf-8", errors="replace")
                line_cnt = len(content.splitlines())
                total_loc += line_cnt
                if exp_snippet not in content:
                    passed = False
                    failure_reasons.append(f"Expected snippet not found in '{exp_path}': `{exp_snippet}`")
                if line_cnt > max_allowed_lines:
                    passed = False
                    lean_ok = False
                    failure_reasons.append(
                        f"Anti-bloat limit exceeded: '{exp_path}' has {line_cnt} lines (max: {max_allowed_lines})."
                    )

            verify_py = task.get("verify_python")
            if passed and verify_py:
                import subprocess

                proc = subprocess.run(
                    [sys.executable, "-c", verify_py],
                    cwd=str(tmp_path),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode != 0:
                    passed = False
                    failure_reasons.append(
                        f"Functional validation failure: {(proc.stderr or proc.stdout).strip()[:200]}"
                    )

            status_icon = "✔ PASS" if passed else "❌ FAIL"
            trace_str = " -> ".join(tool_trace) if tool_trace else "direct response"
            print(
                f"  {status_icon} ({duration}s, {steps_taken} steps [{trace_str}], ~{tokens_used} tokens, {total_loc} lines of code, {warnings_count} warnings)"
            )
            if not passed:
                for r in failure_reasons:
                    print(f"     • {r}")

            return {
                "id": task_id,
                "name": name,
                "passed": passed,
                "reason": "; ".join(failure_reasons) if failure_reasons else "OK",
                "steps": steps_taken,
                "duration": duration,
                "tokens": tokens_used,
                "loc": total_loc,
                "lean_ok": lean_ok,
                "warnings": warnings_count,
            }

    def _print_summary(self, results: List[Dict[str, Any]]) -> None:
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        lean_passed = sum(1 for r in results if r.get("lean_ok", True))
        rate = round((passed / total) * 100, 1) if total > 0 else 0.0
        avg_steps = round(sum(r["steps"] for r in results) / total, 2) if total > 0 else 0.0
        total_time = round(sum(r["duration"] for r in results), 2)
        total_tokens = sum(r.get("tokens", 0) for r in results)
        total_loc = sum(r.get("loc", 0) for r in results)

        print("\n" + "=" * 88)
        print("📋 EVREN-CODE EVAL BENCHMARK RESULTS (SPEED, TOKEN, AND LEAN CODE METRICS)")
        print("=" * 88)
        print(
            f"{'Task ID':<28} | {'Status':<8} | {'Steps':<5} | {'Duration':<7} | {'Tokens':<8} | {'LOC':<9} | {'Warnings'}"
        )
        print("-" * 88)
        for r in results:
            icon = "PASS ✔" if r["passed"] else "FAIL ❌"
            print(
                f"{r['id']:<28} | {icon:<8} | {r['steps']:<5} | {r['duration']:<5}s  | {r.get('tokens', 0):<8} | {r.get('loc', 0):<9} | {r['warnings']}"
            )
        print("=" * 88)
        print(
            f"🏆 Pass Rate: {rate}% ({passed}/{total}) | "
            f"⚡ Avg Steps: {avg_steps} | "
            f"⏱ Total Time: {total_time}s | "
            f"🪙 Total Tokens: {total_tokens} | "
            f"📏 Total LOC: {total_loc} (Lean Code: {lean_passed}/{total})"
        )
        print("=" * 88 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evren-Code Evaluation Runner")
    parser.add_argument(
        "--tasks",
        type=str,
        default=str(ROOT_DIR / "evals" / "tasks.json"),
        help="Path to evaluation tasks JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target model ID override (e.g. qwen3.8-flash-next, glm-5.3, auto)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter tasks by ID or category substring",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON evaluation report",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock provider instead of live API",
    )
    args = parser.parse_args()

    runner = EvalRunner(
        Path(args.tasks),
        model_override=args.model,
        task_filter=args.filter,
        output_path=Path(args.output) if args.output else None,
        use_mock=args.mock,
    )
    runner.run_all()


if __name__ == "__main__":
    main()
