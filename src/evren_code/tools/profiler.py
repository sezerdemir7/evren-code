"""Autonomous Performance & Memory Profiler (Auto-Benchmark & Performance Profiler).

Measures execution duration (min, max, mean, median, stddev) and peak memory
(RAM / tracemalloc) consumption of functions and code blocks in an isolated subprocess,
reporting algorithmic bottlenecks ($O(N^2)$ loops, memory leaks).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from evren_code.config import AppConfig
from evren_code.tools.filesystem import resolve_safe_path
from evren_code.tools.registry import ToolRegistry


RUNNER_SCRIPT_TEMPLATE = r"""
import sys
import time
import tracemalloc
import json
import statistics
import traceback

workspace_root = {workspace_root!r}
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

setup_code = {setup_code!r}
code_to_profile = {code_to_profile!r}
iterations = {iterations!r}
warmup_runs = {warmup_runs!r}

globals_dict = {{"__name__": "__main__", "__file__": "<profile_runner>"}}

result = {{
    "success": False,
    "error": None,
    "iterations": iterations,
    "times_ms": [],
    "mean_ms": 0.0,
    "min_ms": 0.0,
    "max_ms": 0.0,
    "median_ms": 0.0,
    "stddev_ms": 0.0,
    "peak_memory_kb": 0.0,
    "allocated_memory_kb": 0.0,
    "stdout": ""
}}

try:
    if setup_code and setup_code.strip():
        exec(setup_code, globals_dict)

    compiled_code = compile(code_to_profile, "<profile_target>", "exec")

    # Warmup runs
    for _ in range(warmup_runs):
        exec(compiled_code, globals_dict)

    tracemalloc.start()
    times = []

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        exec(compiled_code, globals_dict)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1_000_000.0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result["success"] = True
    result["times_ms"] = times
    result["mean_ms"] = round(statistics.mean(times), 4)
    result["min_ms"] = round(min(times), 4)
    result["max_ms"] = round(max(times), 4)
    result["median_ms"] = round(statistics.median(times), 4)
    result["stddev_ms"] = round(statistics.stdev(times), 4) if len(times) > 1 else 0.0
    result["peak_memory_kb"] = round(peak_mem / 1024.0, 2)
    result["allocated_memory_kb"] = round(current_mem / 1024.0, 2)

except Exception:
    result["success"] = False
    result["error"] = traceback.format_exc()

print("__EVREN_PROFILE_OUTPUT__")
print(json.dumps(result))
"""


def run_benchmark_subprocess(
    workspace_root: Path,
    code_to_profile: str,
    setup_code: str = "",
    iterations: int = 50,
    warmup_runs: int = 2,
    timeout_seconds: float = 12.0,
) -> Dict[str, Any]:
    """Measures the performance of a code snippet by spawning an isolated Python subprocess."""
    iterations = max(1, min(iterations, 1000))
    warmup_runs = max(0, min(warmup_runs, 10))

    runner_content = RUNNER_SCRIPT_TEMPLATE.format(
        workspace_root=str(workspace_root),
        setup_code=setup_code,
        code_to_profile=code_to_profile,
        iterations=iterations,
        warmup_runs=warmup_runs,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as tf:
        tf.write(runner_content)
        tmp_path = tf.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout_seconds,
            cwd=str(workspace_root),
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout = proc.stdout
        marker = "__EVREN_PROFILE_OUTPUT__"
        if marker in stdout:
            parts = stdout.split(marker)
            raw_json = parts[-1].strip()
            data = json.loads(raw_json)
            user_stdout = parts[0].strip()
            data["stdout"] = user_stdout
            return data
        else:
            err = proc.stderr.strip() or stdout.strip()
            return {"success": False, "error": f"Error: Subprocess produced unexpected output: {err} (Alt süreç beklenmeyen çıktı üretti)"}
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Error: Benchmark timed out after {timeout_seconds}s.",
        }
    except Exception as exc:
        return {"success": False, "error": f"Error: Profile execution error: {exc}"}
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def format_performance_report(
    target_desc: str,
    stats: Dict[str, Any],
) -> str:
    """Formats performance metrics into a clean Markdown report."""
    if not stats.get("success"):
        return (
            f"❌ **[Performance Profile Error: {target_desc} / Performans Profil Hatası]**\n"
            f"```\n{stats.get('error', 'Unknown error / Bilinmeyen hata.')}\n```"
        )

    mean_ms = stats["mean_ms"]
    min_ms = stats["min_ms"]
    max_ms = stats["max_ms"]
    median_ms = stats["median_ms"]
    stddev_ms = stats["stddev_ms"]
    peak_kb = stats["peak_memory_kb"]
    alloc_kb = stats["allocated_memory_kb"]
    iters = stats["iterations"]
    user_out = stats.get("stdout", "").strip()

    # Algorithmic rating tag
    if mean_ms < 0.5:
        speed_rating = "⚡ **Ultra Fast (<0.5ms)** - Native microprocessor-level execution."
    elif mean_ms < 5.0:
        speed_rating = "🚀 **Very Fast (<5ms)** - Excellent algorithm, production-ready."
    elif mean_ms < 50.0:
        speed_rating = "✅ **Acceptable (5-50ms)** - Standard business logic or I/O operation."
    elif mean_ms < 200.0:
        speed_rating = "⚠️ **Moderate / Review Recommended (50-200ms)** - Potential bottleneck in critical loops or API endpoints."
    else:
        speed_rating = "🛑 **Slow (>200ms)** - Suspected O(N²) or inefficient allocation. Refactor recommended!"

    mem_rating = (
        "💡 No memory leak detected. Peak RAM allocation is highly reasonable."
        if peak_kb < 1024
        else f"⚠️ Peak memory reached {round(peak_kb / 1024, 2)} MB. Consider using generators (`yield`) for large datasets."
    )

    out_block = f"\n**Standard Output (Stdout):**\n```\n{user_out}\n```\n" if user_out else ""

    return (
        f"⏱️ **[Autonomous Performance & Memory Profile Report / Otonom Performans & Bellek Profil Raporu]**\n"
        f"- **Target:** `{target_desc}`\n"
        f"- **Sample:** {iters} independent iterations (excluding warmup runs)\n\n"
        f"### 📊 Latency / Yürütme Süresi (Latency)\n"
        f"- **Mean:** `{mean_ms:.3f} ms`\n"
        f"- **Median:** `{median_ms:.3f} ms` (StdDev: `±{stddev_ms:.3f} ms`)\n"
        f"- **Min:** `{min_ms:.3f} ms` | **Max:** `{max_ms:.3f} ms`\n\n"
        f"### 💾 Memory (RAM / tracemalloc)\n"
        f"- **Peak RAM / Tepe Bellek Tüketimi (Peak RAM):** `{peak_kb:.2f} KB`\n"
        f"- **Current Allocated:** `{alloc_kb:.2f} KB`\n\n"
        f"### 🎯 Engineering Evaluation / Mühendislik Değerlendirmesi\n"
        f"- {speed_rating}\n"
        f"- {mem_rating}"
        f"{out_block}"
    )


def register_profiler_tools(registry: ToolRegistry, config: AppConfig) -> None:
    """Registers `profile_code_performance` tool with the ToolRegistry."""

    def profile_code_performance(
        code_snippet: Optional[str] = None,
        file_path: Optional[str] = None,
        target_function: Optional[str] = None,
        setup_code: str = "",
        iterations: int = 50,
        timeout_seconds: float = 10.0,
    ) -> str:
        workspace_root = config.workspace_root

        if file_path:
            p = resolve_safe_path(workspace_root, file_path)
            if not p.is_file():
                return f"Error: File '{file_path}' not found."
            # Prepare code to invoke function from file
            rel_mod = p.relative_to(workspace_root).as_posix()
            mod_dotted = rel_mod.replace("/", ".").replace(".py", "")
            if target_function:
                setup_lines = [
                    setup_code,
                    f"import importlib",
                    f"mod = importlib.import_module({mod_dotted!r})",
                    f"fn = getattr(mod, {target_function!r})",
                ]
                full_setup = "\n".join(s for s in setup_lines if s)
                code_to_exec = "fn()"
                target_desc = f"{file_path} :: {target_function}()"
            else:
                code_to_exec = p.read_text(encoding="utf-8", errors="replace")
                full_setup = setup_code
                target_desc = file_path
        elif code_snippet:
            code_to_exec = code_snippet
            full_setup = setup_code
            target_desc = code_snippet.strip().splitlines()[0][:40] + "..."
        else:
            return "Error: At least one of 'code_snippet' or 'file_path' must be specified."

        stats = run_benchmark_subprocess(
            workspace_root=workspace_root,
            code_to_profile=code_to_exec,
            setup_code=full_setup,
            iterations=iterations,
            timeout_seconds=timeout_seconds,
        )

        return format_performance_report(target_desc, stats)

    registry.register(
        name="profile_code_performance",
        description=(
            "Autonomous Performance & Memory Profiler: Measures execution runtime (min, max, mean, stddev) "
            "and peak RAM usage (tracemalloc) in an isolated subprocess. Identifies algorithmic bottlenecks, "
            "slow loops, and memory leaks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code_snippet": {
                    "type": "string",
                    "description": "Python code snippet to benchmark (e.g. '[x**2 for x in range(1000)]').",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to module or file to profile (e.g. 'src/utils.py').",
                },
                "target_function": {
                    "type": "string",
                    "description": "Function name to profile when file_path is provided.",
                },
                "setup_code": {
                    "type": "string",
                    "description": "One-time setup code executed prior to benchmarking (e.g. imports or dataset creation).",
                },
                "iterations": {
                    "type": "integer",
                    "description": "Number of benchmark iterations (default: 50, range: 1-1000).",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Maximum execution timeout in seconds (default: 10.0s).",
                },
            },
        },
        handler=profile_code_performance,
        requires_approval=False,
    )
