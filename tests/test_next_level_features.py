"""Unit tests for next-level platform capabilities:
1. External MCP Client (`MCPClientManager`, `.evren/mcp.json`, stdio JSON-RPC 2.0)
2. Live Web & Documentation Search (`web_search`, `parse_duckduckgo_html`)
3. Bracketed Multiline Paste & `@clipboard` Visual/Text Expansion (`create_prompt_key_bindings`, `expand_prompt_mentions`)
4. End-of-Turn Compact Engineering Summary Badge & Git Commit Assistant (`build_compact_turn_summary`, `git_commit_assistant`)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evren_code.cli import create_prompt_key_bindings, expand_prompt_mentions
from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.mcp_client import MCPClientManager
from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem
from evren_code.tools.git_tools import build_conventional_commit_proposal, register_git_tools
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.web import parse_duckduckgo_html, register_web_tools


class TestNextLevelFeatures(unittest.TestCase):
    """Verifies MCP Client, web_search, @clipboard/bracketed paste, and compact turn summary."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.config = load_config(workspace_root=self.workspace)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_01_external_mcp_client_stdio_discovery_and_execution(self) -> None:
        """Verifies that MCPClientManager discovers `.evren/mcp.json`, connects over stdio JSON-RPC, and executes tools."""
        dummy_server = self.workspace / "dummy_mcp_server.py"
        dummy_server.write_text(
            "import json, sys\n"
            "for line in sys.stdin:\n"
            "    line = line.strip()\n"
            "    if not line: continue\n"
            "    msg = json.loads(line)\n"
            "    mid = msg.get('id')\n"
            "    method = msg.get('method')\n"
            "    if method == 'initialize':\n"
            "        resp = {'jsonrpc': '2.0', 'id': mid, 'result': {'protocolVersion': '2024-11-05', 'capabilities': {}}}\n"
            "        sys.stdout.write(json.dumps(resp) + '\\n'); sys.stdout.flush()\n"
            "    elif method == 'tools/list':\n"
            "        tools = [{'name': 'query_db', 'description': 'Run SQL query', 'inputSchema': {'type': 'object', 'properties': {'sql': {'type': 'string'}}, 'required': ['sql']}}]\n"
            "        resp = {'jsonrpc': '2.0', 'id': mid, 'result': {'tools': tools}}\n"
            "        sys.stdout.write(json.dumps(resp) + '\\n'); sys.stdout.flush()\n"
            "    elif method == 'tools/call':\n"
            "        args = msg.get('params', {}).get('arguments', {})\n"
            "        sql = args.get('sql', '')\n"
            "        resp = {'jsonrpc': '2.0', 'id': mid, 'result': {'content': [{'type': 'text', 'text': f'Rows for: {sql}'}], 'isError': False}}\n"
            "        sys.stdout.write(json.dumps(resp) + '\\n'); sys.stdout.flush()\n",
            encoding="utf-8",
        )

        evren_dir = self.workspace / ".evren"
        evren_dir.mkdir(parents=True, exist_ok=True)
        mcp_json = evren_dir / "mcp.json"
        mcp_json.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "postgres": {
                            "command": sys.executable,
                            "args": ["-u", str(dummy_server)],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        reg = ToolRegistry()
        mgr = MCPClientManager(workspace_root=self.workspace, global_dir=self.workspace / "no_global")
        try:
            registered = mgr.connect_and_register_tools(reg)
            self.assertIn("mcp_postgres_query_db", registered)
            out = reg.execute("mcp_postgres_query_db", {"sql": "SELECT * FROM users"})
            self.assertEqual(out, "Rows for: SELECT * FROM users")
        finally:
            mgr.close_all()

    def test_02_web_search_duckduckgo_html_parser_and_tool(self) -> None:
        """Verifies DuckDuckGo HTML parsing and web_search tool execution."""
        sample_html = (
            '<div class="result">'
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ffastapi.tiangolo.com%2Fadvanced%2Flifespan%2F">FastAPI Lifespan Events</a>'
            '<a class="result__snippet">Use lifespan context manager in FastAPI applications.</a>'
            "</div>"
        )
        items = parse_duckduckgo_html(sample_html, max_results=5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "FastAPI Lifespan Events")
        self.assertEqual(items[0]["url"], "https://fastapi.tiangolo.com/advanced/lifespan/")
        self.assertIn("lifespan context manager", items[0]["snippet"])

        reg = ToolRegistry()
        register_web_tools(reg, self.config)
        self.assertIsNotNone(reg.get("web_search"))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = sample_html
        with patch("httpx.Client.post", return_value=mock_resp):
            res = reg.execute("web_search", {"query": "FastAPI lifespan", "domain": "fastapi.tiangolo.com"})
            self.assertIn("FastAPI Lifespan Events", res)
            self.assertIn("https://fastapi.tiangolo.com/advanced/lifespan/", res)

    def test_03_bracketed_paste_and_clipboard_mentions(self) -> None:
        """Verifies bracketed multiline paste keybindings and @clipboard image/text expansion."""
        kb = create_prompt_key_bindings()
        self.assertGreaterEqual(len(kb.bindings), 2)

        sample_file = self.workspace / "hello.py"
        sample_file.write_text("print('hi')\n", encoding="utf-8")
        expanded_file = expand_prompt_mentions("Check @hello.py", self.workspace)
        self.assertIn("[@hello.py]:", expanded_file)
        self.assertIn("print('hi')", expanded_file)

        fake_shot = self.workspace / ".evren" / "screenshots" / "clipboard_123.png"
        fake_shot.parent.mkdir(parents=True, exist_ok=True)
        fake_shot.write_bytes(b"PNGDATA")

        with patch("evren_code.tools.vision.save_clipboard_image", return_value=fake_shot):
            expanded_img = expand_prompt_mentions("Fix this UI bug @clipboard", self.workspace)
            self.assertIn("analyze_image", expanded_img)
            self.assertIn("clipboard_123.png", expanded_img)

        with patch("evren_code.tools.vision.save_clipboard_image", return_value=None), patch(
            "evren_code.tools.vision.get_clipboard_text", return_value="Traceback: ValueError on line 42"
        ):
            expanded_txt = expand_prompt_mentions("Explain @clipboard", self.workspace)
            self.assertIn("[@clipboard]:", expanded_txt)
            self.assertIn("Traceback: ValueError on line 42", expanded_txt)

    def test_04_compact_turn_summary_and_git_commit_assistant(self) -> None:
        """Verifies end-of-turn compact engineering summary badge and Conventional Commit assistant."""
        orch = AgentOrchestrator(config=self.config)
        orch.trust_mode = True
        try:
            steps = [0]

            def _mock_chat(**kwargs):
                steps[0] += 1
                if steps[0] == 1:
                    return ParsedAssistantResponse(
                        content="",
                        tool_calls=[
                            ToolCallItem(
                                id="tc_w1",
                                name="write_file",
                                arguments={"path": "app.py", "content": "def run():\n    return 42\n"},
                            )
                        ],
                        usage={"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
                    )
                return ParsedAssistantResponse(
                    content="Implemented app.py.",
                    tool_calls=[],
                    usage={"prompt_tokens": 1400, "completion_tokens": 100, "total_tokens": 1500},
                )

            orch.llm_client.chat = _mock_chat  # type: ignore
            orch.run_turn("Create app.py with run()", stream=False)

            badge = getattr(orch, "last_turn_summary_badge", "")
            self.assertIn("1 dosya düzenlendi", badge)
            self.assertIn("+2 / -0 satır", badge)
            self.assertIn("token", badge)

            subprocess.run(["git", "init"], cwd=str(self.workspace), capture_output=True, check=False)
            reg = ToolRegistry()
            register_git_tools(reg, self.config)
            draft = reg.execute("git_commit_assistant", {"scope": "app", "commit_type": "feat"})
            self.assertIn("feat(app):", draft)
            self.assertIn("app.py", draft)
        finally:
            orch.close()


if __name__ == "__main__":
    unittest.main()
