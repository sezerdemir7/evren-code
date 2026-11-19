"""Test performance and parity module."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evren_code.config import load_config
from evren_code.core.context import ContextManager
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.tools.filesystem import register_filesystem_tools
from evren_code.tools.registry import ToolRegistry
from evren_code.tools.git_tools import ShadowGitManager
from evren_code.tools.shell import PersistentShellSession


def test_persistent_shell_cd_tracking(tmp_path: Path):
    """Verify persistent shell cd tracking."""
    sub_dir = tmp_path / "submodule"
    sub_dir.mkdir()

    session = PersistentShellSession(tmp_path)
    try:
        rc, out, err = session.execute("cd submodule")
        assert rc == 0
        assert session.current_cwd.resolve() == sub_dir.resolve()

        rc, out, err = session.execute("echo HELLO_SUB")
        assert rc == 0
        assert "HELLO_SUB" in out
    finally:
        session.close()


def test_fuzzy_indent_replacement(tmp_path: Path):
    """Verify fuzzy indent replacement."""
    sample_file = tmp_path / "calc.py"
    sample_file.write_text(
        "def compute():\n"
        "    # Step 1\n"
        "    x = 10\n"
        "    y = 20\n"
        "    return x + y\n",
        encoding="utf-8",
    )

    config = load_config()
    setattr(config, "workspace_root", tmp_path)

    registry = ToolRegistry()
    register_filesystem_tools(registry, config)

    target_from_model = (
        "x = 10\n"
        "y = 20"
    )
    replacement_from_model = (
        "x = 15\n"
        "y = 25"
    )

    edit_tool = registry.get("edit_file")
    res = edit_tool.handler(
        path="calc.py",
        target=target_from_model,
        replacement=replacement_from_model,
    )
    assert "Başarılı" in res
    updated_code = sample_file.read_text(encoding="utf-8")
    assert "    x = 15" in updated_code
    assert "    y = 25" in updated_code
    assert "    # Step 1" in updated_code


def test_hierarchical_project_rules_and_auto_learn(tmp_path: Path):
    """Verify hierarchical project rules and auto learn."""
    config = load_config()
    setattr(config, "workspace_root", tmp_path)

    ctx = ContextManager(config, "Sen EVREN ajanısın.")

    root_rule = tmp_path / "EVREN.md"
    root_rule.write_text("Projede TypeScript zorunludur.", encoding="utf-8")

    ui_dir = tmp_path / "src" / "ui"
    ui_dir.mkdir(parents=True)
    ui_rule = ui_dir / "EVREN.md"
    ui_rule.write_text("UI bileşenlerinde TailwindCSS kullanılmalıdır.", encoding="utf-8")

    rules_text = ctx.load_project_rules(active_filepath="src/ui/Button.tsx")
    assert "TypeScript zorunludur" in rules_text
    assert "TailwindCSS kullanılmalıdır" in rules_text

    user_instruction = "Bundan sonra her zaman async fonksiyonlarda try/catch kullan"
    note = ctx.auto_learn_user_preference(user_instruction)
    assert note is not None
    assert (tmp_path / "EVREN.md").is_file()
    saved_content = (tmp_path / "EVREN.md").read_text(encoding="utf-8")
    assert "try/catch kullan" in saved_content


def test_auto_model_direct_provider_requests():
    """Verify auto model direct provider requests."""
    config = load_config()
    client = EvrenLLMClient(config)

    spec = client.set_model("auto")
    assert spec.id == "auto"
    assert spec.name == "EVREN Auto"
    assert client.active_model.id == "auto"
    assert not hasattr(client, "resolve_smart_model")
    assert not hasattr(client, "auto_routing_enabled")


def test_tool_call_preparing_event_and_console_spinner():
    """Verify tool call preparing event and console spinner."""
    from evren_code.core.events import EventBus, EventType, AgentEvent
    from evren_code.ui.console import ConsoleUI
    from evren_code.core.stream_filter import StreamTagFilter

    events = []
    bus = EventBus()
    bus.subscribe(None, lambda ev: events.append(ev))

    console = ConsoleUI(event_bus=bus)
    console.on_event(AgentEvent(type=EventType.TEXT_DELTA, data={"delta": "Dosya yeniden yazılıyor..."}))
    assert console._in_text is True

    console.on_event(AgentEvent(type=EventType.TOOL_CALL_PREPARING, data={"tool_name": "write_file", "bytes_received": 1536}))
    assert console._in_text is False
    assert console._status is not None

    console.on_event(AgentEvent(type=EventType.TOOL_CALL_DETECTED, data={"tool_name": "write_file", "arguments": {"path": "test.cs"}}))
    assert console._status is None

    prep_calls = []
    tag_filter = StreamTagFilter(
        on_text=lambda t: None,
        on_thinking=lambda th: None,
        on_tool_preparing=lambda name, bcount: prep_calls.append((name, bcount)),
    )
    tag_filter.feed("<tool_call>{\"name\": \"write_file\", \"content\": \"hello\"}")
    assert len(prep_calls) > 0


def test_console_does_not_interrupt_text_stream_with_idle_spinner():
    """Verify console does not interrupt text stream with idle spinner."""
    import time
    from evren_code.core.events import EventBus, EventType, AgentEvent
    from evren_code.ui.console import ConsoleUI

    bus = EventBus()
    console = ConsoleUI(event_bus=bus)
    try:
        console.on_event(AgentEvent(type=EventType.TEXT_DELTA, data={"delta": "Output stream chunk: DemoApp"}))
        assert console._in_text is True
        assert console._status is None

        time.sleep(0.45)
        assert console._in_text is True
        assert console._status is None

        console.on_event(AgentEvent(type=EventType.TOOL_CALL_PREPARING, data={"tool_name": "ask_user", "bytes_received": 128}))
        assert console._in_text is False
        assert console._status is not None

        console.on_event(AgentEvent(type=EventType.TURN_COMPLETED, data={}))
        assert console._status is None
    finally:
        console._cancel_idle_timer()
        console._stop_status()


def test_tool_stream_chunk_without_arguments_does_not_raise_unbound_local():
    """Verify tool stream chunk without arguments does not raise unbound local."""
    from unittest.mock import MagicMock, patch
    from evren_code.config import load_config
    from evren_code.core.llm_client import EvrenLLMClient

    config = load_config()
    config.api.api_key = "test_key"
    client = EvrenLLMClient(config)

    sse_lines = [
        "data: {\"choices\": [{\"delta\": {\"tool_calls\": [{\"index\": 0, \"id\": \"call_1\", \"function\": {\"name\": \"run_command\"}}]}}]}\n\n",
        "data: {\"choices\": [{\"delta\": {\"tool_calls\": [{\"index\": 0, \"function\": {\"arguments\": \"{\\\"command\\\": \\\"git status\\\"}\"}}]}}]}\n\n",
        "data: [DONE]\n\n",
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = sse_lines
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("httpx.Client.stream", return_value=mock_resp):
        res = client.chat(
            messages=[{"role": "user", "content": "git durumunu kontrol et"}],
            stream=True,
        )
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0].name == "run_command"


def test_console_does_not_duplicate_agent_header_or_split_words_on_jitter():
    """Verify console does not duplicate agent header or split words on jitter."""
    import time
    from unittest.mock import MagicMock
    from evren_code.core.events import EventBus, EventType, AgentEvent
    from evren_code.ui.console import ConsoleUI

    bus = EventBus()
    mock_console = MagicMock()
    ui = ConsoleUI(event_bus=bus, console=mock_console)

    ui.on_event(AgentEvent(type=EventType.STEP_STARTED, data={"step": 1, "model_name": "EVREN"}))

    ui.on_event(AgentEvent(type=EventType.TEXT_DELTA, data={"delta": "commit at"}))
    assert ui._has_printed_agent_header is True

    time.sleep(0.40)
    assert ui._in_text is True
    assert ui._status is None

    ui.on_event(AgentEvent(type=EventType.TEXT_DELTA, data={"delta": "ıyorum."}))

    header_calls = [
        args for args, _ in mock_console.print.call_args_list
        if args and any("● Evren-Code:" in str(a) for a in args)
    ]
    assert len(header_calls) == 1
    ui._close_stream_line()




