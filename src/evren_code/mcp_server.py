"""EVREN-Code Model Context Protocol (MCP) Server Module.

Allows host agents such as Claude Code, Claude Desktop, Antigravity, and Cursor
to invoke EVREN AI models and autonomous software engineering capabilities
directly over stdio JSON-RPC as a specialized subagent/tool provider.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from evren_code.config import AppConfig, load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.events import EventBus
from evren_code.core.evren_auth import EvrenAuthManager
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.ui.console import ConsoleUI

logger = logging.getLogger(__name__)


def _create_orchestrator_for_workspace(
    workspace_path: Path,
    role_or_model: Optional[str] = None,
) -> AgentOrchestrator:
    """Creates a secure, headless ReAct orchestrator for the target workspace.
    
    Note: To prevent breaking the MCP stdio protocol, all UI and logging output is strictly directed to stderr.
    """
    workspace = workspace_path.resolve()
    config: AppConfig = load_config(workspace_root=workspace)
    config.telemetry.enabled = False

    event_bus = EventBus()
    err_console = Console(stderr=True, quiet=True)
    ui = ConsoleUI(event_bus=event_bus, console=err_console)
    ui._trust_all_active = True

    auth_mgr = EvrenAuthManager(config=config)
    auth_mgr.load_models_cache()
    auth_mgr.load_stored_key()

    llm_client = EvrenLLMClient(config=config, event_bus=event_bus)
    if role_or_model:
        llm_client.set_model(role_or_model)

    orchestrator = AgentOrchestrator(
        config=config,
        llm_client=llm_client,
        event_bus=event_bus,
        approval_delegate=ui,
    )
    orchestrator.trust_mode = True
    return orchestrator


def _create_llm_client_for_mcp(model_id: Optional[str] = None) -> EvrenLLMClient:
    """Creates lightweight client for direct LLM queries."""
    config: AppConfig = load_config()
    event_bus = EventBus()
    auth_mgr = EvrenAuthManager(config=config)
    auth_mgr.load_models_cache()
    auth_mgr.load_stored_key()
    client = EvrenLLMClient(config=config, event_bus=event_bus)
    if model_id:
        client.set_model(model_id)
    return client


def create_mcp_server():
    """Initializes MCP server instance and registers tool handlers."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise RuntimeError(
            "MCP library not installed. Install via `py -m pip install mcp>=2.0.0`."
        ) from exc

    server = MCPServer("evren-code")

    @server.tool()
    def evren_task(
        task: str,
        workspace: str = ".",
        role: str = "orchestrator",
    ) -> str:
        """Evren task."""
        target_ws = Path(workspace).resolve()
        orchestrator = _create_orchestrator_for_workspace(target_ws, role_or_model=role)
        try:
            output = orchestrator.run_turn(task, stream=False)
            
            modified = []
            if hasattr(orchestrator, "undo_manager") and orchestrator.undo_manager._stack:
                for item in orchestrator.undo_manager._stack:
                    p = item.get("path")
                    if p:
                        try:
                            rel = Path(p).relative_to(target_ws)
                            modified.append(str(rel).replace("\\", "/"))
                        except Exception:
                            modified.append(str(p).replace("\\", "/"))

            res = {
                "status": "success",
                "model_used": orchestrator.llm_client.active_model.id,
                "role": role,
                "workspace": str(target_ws),
                "summary": output,
                "modified_files": list(set(modified)),
            }
            return json.dumps(res, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps({
                "status": "error",
                "role": role,
                "workspace": str(target_ws),
                "error": str(exc),
            }, ensure_ascii=False, indent=2)
        finally:
            orchestrator.close()

    @server.tool()
    def evren_architect(
        task: str,
        workspace: str = ".",
    ) -> str:
        """Evren architect (reuse)."""
        target_ws = Path(workspace).resolve()
        orchestrator = _create_orchestrator_for_workspace(target_ws, role_or_model="architect")
        try:
            architect_prompt = (
                f"Lütfen aşağıdaki görev için kod yazmadan sadece mimari tasarım, "
                f"mevcut sembol/kod yeniden kullanımı ve risk analizi planı çıkar:\n\n{task}"
            )
            output = orchestrator.run_turn(architect_prompt, stream=False)
            return output
        except Exception as exc:
            return f"Mimari analiz hatası: {exc}"
        finally:
            orchestrator.close()

    @server.tool()
    def evren_code_review(
        path: str = ".",
        workspace: str = ".",
    ) -> str:
        """Evren code review (code review)."""
        target_ws = Path(workspace).resolve()
        orchestrator = _create_orchestrator_for_workspace(target_ws, role_or_model="reviewer")
        try:
            review_prompt = (
                f"Lütfen '{path}' konumundaki kodları ve son değişiklikleri; temiz kod, "
                f"güvenlik açıkları, hata yönetimi ve performans açısından kapsamlı bir şekilde incele."
            )
            output = orchestrator.run_turn(review_prompt, stream=False)
            return output
        except Exception as exc:
            return f"Kod inceleme hatası: {exc}"
        finally:
            orchestrator.close()

    @server.tool()
    def evren_llm_call(
        prompt: str,
        model: str = "deepseek-v4.1-flash",
        system_prompt: Optional[str] = None,
        include_thinking: bool = True,
        temperature: float = 0.2,
    ) -> str:
        """Evren llm call."""
        client = _create_llm_client_for_mcp(model_id=model)
        try:
            messages: List[Dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            resp = client.chat(
                messages=messages,
                tools_schema=None,
                stream=False,
                temperature=temperature,
                silent=True,
            )

            res = {
                "status": "success",
                "model": client.active_model.id,
                "content": resp.content,
                "thinking": resp.thinking if include_thinking else None,
                "usage": resp.usage,
                "finish_reason": resp.finish_reason,
            }
            return json.dumps(res, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps({
                "status": "error",
                "model": model,
                "error": str(exc),
            }, ensure_ascii=False, indent=2)
        finally:
            client.close()

    @server.tool()
    def evren_models() -> str:
        """Evren models."""
        config = load_config()
        catalog = {}
        for mid, spec in config.models.catalog.items():
            catalog[mid] = {
                "name": spec.name,
                "context_length": spec.context_length,
                "supports_tools": spec.tools,
                "supports_vision": spec.supports_vision,
            }
        return json.dumps({
            "default_model": config.models.default,
            "roles": config.models.roles,
            "catalog": catalog,
        }, ensure_ascii=False, indent=2)

    return server


def run_mcp_server() -> int:
    """Run mcp server."""
    import asyncio

    if sys.platform == "win32":
        try:
            import msvcrt
            import os
            msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
            msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        except Exception:
            pass

    server = create_mcp_server()
    try:
        asyncio.run(server.run_stdio_async())
        return 0
    except (KeyboardInterrupt, SystemExit):
        return 0
    except Exception as exc:
        sys.stderr.write(f"MCP Server error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_mcp_server())
