"""External Model Context Protocol (MCP) Stdio Client Manager (`mcp_client.py`).

Discovers external MCP server definitions from `.evren/mcp.json` and `~/.evren-code/mcp.json`,
handshakes over JSON-RPC 2.0 stdio (`initialize`, `tools/list`, `tools/call`), and dynamically
registers discovered tools into Evren-Code's `ToolRegistry`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evren_code.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single external stdio MCP server."""

    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    timeout_seconds: float = 30.0
    disabled: bool = False


@dataclass
class MCPToolSpec:
    """Metadata for a tool discovered from an external MCP server."""

    server_name: str
    name: str
    registered_name: str
    description: str
    input_schema: Dict[str, Any]


class MCPStdioConnection:
    """Thread-safe synchronous JSON-RPC 2.0 stdio connection to an MCP server process."""

    def __init__(self, config: MCPServerConfig, default_cwd: Path) -> None:
        self.config = config
        self.default_cwd = default_cwd
        self._proc: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._req_id = 0
        self.initialized = False

    def start_and_initialize(self) -> bool:
        """Spawns the MCP server subprocess and performs the JSON-RPC `initialize` handshake."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self.initialized:
                return True

            merged_env = os.environ.copy()
            merged_env.update(self.config.env or {})

            work_dir = self.config.cwd or str(self.default_cwd)
            cmd_list = [self.config.command] + list(self.config.args)

            try:
                self._proc = subprocess.Popen(
                    cmd_list,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=merged_env,
                    cwd=work_dir,
                    bufsize=1,
                )
            except Exception as exc:
                logger.warning("Failed to start MCP server '%s': %s", self.config.name, exc)
                self._proc = None
                return False

            init_resp = self._send_request_unlocked(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "evren-code", "version": "0.2.0"},
                },
            )
            if init_resp is None or "error" in init_resp:
                logger.warning("MCP server '%s' initialize handshake failed: %s", self.config.name, init_resp)
                self.close_unlocked()
                return False

            self._send_notification_unlocked("notifications/initialized", {})
            self.initialized = True
            return True

    def list_tools(self) -> List[MCPToolSpec]:
        """Queries `tools/list` on the connected MCP server."""
        with self._lock:
            if not self.initialized and not self.start_and_initialize():
                return []
            resp = self._send_request_unlocked("tools/list", {})
            if not resp or "result" not in resp:
                return []

            raw_tools = resp["result"].get("tools", [])
            specs: List[MCPToolSpec] = []
            for t in raw_tools:
                if not isinstance(t, dict) or not t.get("name"):
                    continue
                t_name = str(t["name"]).strip()
                reg_name = f"mcp_{self.config.name}_{t_name}".replace("-", "_")
                schema = t.get("inputSchema") or {"type": "object", "properties": {}}
                specs.append(
                    MCPToolSpec(
                        server_name=self.config.name,
                        name=t_name,
                        registered_name=reg_name,
                        description=f"[MCP:{self.config.name}] {t.get('description', t_name)}",
                        input_schema=schema,
                    )
                )
            return specs

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Invokes `tools/call` on the MCP server and formats content blocks as text."""
        with self._lock:
            if not self.initialized and not self.start_and_initialize():
                return f"Error: Could not connect to MCP server '{self.config.name}'."
            resp = self._send_request_unlocked(
                "tools/call",
                {"name": tool_name, "arguments": arguments or {}},
            )
            if resp is None:
                return f"Error: No response from MCP server '{self.config.name}' for tool '{tool_name}'."
            if "error" in resp:
                err_obj = resp["error"]
                msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
                return f"MCP Tool Error ({self.config.name}/{tool_name}): {msg}"

            result = resp.get("result", {})
            if isinstance(result, dict):
                is_err = bool(result.get("isError", False))
                content_blocks = result.get("content", [])
                if isinstance(content_blocks, list):
                    parts: List[str] = []
                    for block in content_blocks:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(str(block.get("text", "")))
                            else:
                                parts.append(json.dumps(block, ensure_ascii=False))
                        else:
                            parts.append(str(block))
                    joined = "\n".join(parts).strip()
                    return f"Error: {joined}" if is_err else (joined or "(Empty MCP response)")
            return json.dumps(result, ensure_ascii=False)

    def _send_notification_unlocked(self, method: str, params: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except Exception:
            pass

    def _send_request_unlocked(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            return None
        self._req_id += 1
        req_id = self._req_id
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()

            while True:
                line = self._proc.stdout.readline()
                if not line:
                    return None
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict) and msg.get("id") == req_id:
                    return msg
        except Exception as exc:
            logger.debug("MCP stdio transport error on '%s': %s", self.config.name, exc)
            return None

    def close_unlocked(self) -> None:
        self.initialized = False
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            finally:
                self._proc = None

    def close(self) -> None:
        with self._lock:
            self.close_unlocked()


class MCPClientManager:
    """Manages external MCP server configurations, connections, and tool registrations."""

    def __init__(self, workspace_root: Path, global_dir: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.global_dir = (global_dir or (Path.home() / ".evren-code")).expanduser().resolve()
        self.servers: Dict[str, MCPServerConfig] = {}
        self.connections: Dict[str, MCPStdioConnection] = {}
        self.discovered_tools: Dict[str, MCPToolSpec] = {}

    def load_configs(self) -> Dict[str, MCPServerConfig]:
        """Loads MCP server configurations from global (`~/.evren-code/mcp.json`) and workspace (`.evren/mcp.json`)."""
        candidates = [
            self.global_dir / "mcp.json",
            self.workspace_root / ".evren" / "mcp.json",
        ]
        loaded: Dict[str, MCPServerConfig] = {}
        for cfg_path in candidates:
            if not cfg_path.is_file():
                continue
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                logger.warning("Failed to parse MCP config '%s': %s", cfg_path, exc)
                continue

            servers_dict = raw.get("mcpServers", raw) if isinstance(raw, dict) else {}
            if not isinstance(servers_dict, dict):
                continue

            for sname, sdata in servers_dict.items():
                if not isinstance(sdata, dict):
                    continue
                cmd = str(sdata.get("command", "")).strip()
                if not cmd:
                    continue
                args = [str(a) for a in (sdata.get("args") or []) if a is not None]
                env = {str(k): str(v) for k, v in (sdata.get("env") or {}).items()}
                loaded[str(sname)] = MCPServerConfig(
                    name=str(sname),
                    command=cmd,
                    args=args,
                    env=env,
                    cwd=sdata.get("cwd"),
                    timeout_seconds=float(sdata.get("timeout", 30.0)),
                    disabled=bool(sdata.get("disabled", False)),
                )

        self.servers = loaded
        return loaded

    def connect_and_register_tools(self, registry: ToolRegistry) -> List[str]:
        """Connects to all enabled MCP servers and registers their tools into `ToolRegistry`."""
        if not self.servers:
            self.load_configs()

        registered_names: List[str] = []
        for sname, scfg in self.servers.items():
            if scfg.disabled:
                continue
            conn = MCPStdioConnection(scfg, default_cwd=self.workspace_root)
            if not conn.start_and_initialize():
                continue
            self.connections[sname] = conn
            tools = conn.list_tools()
            for tspec in tools:
                self.discovered_tools[tspec.registered_name] = tspec
                self._register_single_tool(registry, conn, tspec)
                registered_names.append(tspec.registered_name)

        return registered_names

    @staticmethod
    def _register_single_tool(
        registry: ToolRegistry,
        conn: MCPStdioConnection,
        tspec: MCPToolSpec,
    ) -> None:
        def _handler(**kwargs: Any) -> str:
            return conn.call_tool(tspec.name, kwargs)

        registry.register(
            name=tspec.registered_name,
            description=tspec.description,
            parameters=tspec.input_schema,
            handler=_handler,
            requires_approval=False,
        )

    def close_all(self) -> None:
        """Closes all active MCP server subprocess connections."""
        for conn in list(self.connections.values()):
            try:
                conn.close()
            except Exception:
                pass
        self.connections.clear()
