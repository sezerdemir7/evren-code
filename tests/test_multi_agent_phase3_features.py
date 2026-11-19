"""Test multi agent phase3 features module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evren_code.config import AppConfig, load_config
from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem
from evren_code.core.multi_agent.agent_factory import SpecialistAgent
from evren_code.core.multi_agent.context_sharing import ContextSharingEngine
from evren_code.core.multi_agent.models import (
    AgentRole,
    AgentTaskResult,
    EngineeringTask,
    TaskStatus,
)
from evren_code.core.multi_agent.task_graph import TaskGraph


def test_zero_step_file_preload(tmp_path: Path):
    """Verify zero step file preload."""
    sample_file = tmp_path / "sample.py"
    sample_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    config = load_config()
    setattr(config, "workspace_root", tmp_path)

    llm_client_mock = MagicMock()
    tool_executor_mock = MagicMock()
    registry_mock = MagicMock()
    file_ownership_mock = MagicMock()

    agent = SpecialistAgent(
        agent_id="dev_preload_test",
        role=AgentRole.DEVELOPER,
        config=config,
        llm_client=llm_client_mock,
        tool_executor=tool_executor_mock,
        registry=registry_mock,
        file_ownership=file_ownership_mock,
    )

    task = EngineeringTask(
        task_id="task_preload",
        title="Test Preload",
        description="Düzenle",
        role=AgentRole.DEVELOPER,
        assigned_files=["sample.py"],
    )

    with patch.object(llm_client_mock, "chat") as mock_chat:
        mock_chat.return_value = ParsedAssistantResponse(
            content="Tamamlandı",
            tool_calls=[],
        )
        res = agent.execute_task(task)
        assert res.success is True

        args, kwargs = mock_chat.call_args
        messages = kwargs.get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "### [Ön-Yüklenen Dosya]: `sample.py`" in user_msg
        assert "def hello():" in user_msg


def test_parallel_readonly_tool_execution(tmp_path: Path):
    """Verify parallel readonly tool execution."""
    config = load_config()
    setattr(config, "workspace_root", tmp_path)

    llm_client_mock = MagicMock()
    tool_executor_mock = MagicMock()
    tool_executor_mock.execute.side_effect = lambda tool_name, **kw: f"Result of {tool_name}"
    registry_mock = MagicMock()
    file_ownership_mock = MagicMock()

    agent = SpecialistAgent(
        agent_id="dev_parallel_test",
        role=AgentRole.DEVELOPER,
        config=config,
        llm_client=llm_client_mock,
        tool_executor=tool_executor_mock,
        registry=registry_mock,
        file_ownership=file_ownership_mock,
    )

    task = EngineeringTask(
        task_id="task_parallel",
        title="Parallel Tools",
        description="Çoklu okuma",
        role=AgentRole.DEVELOPER,
    )

    tool_calls = [
        ToolCallItem(id="call_1", name="read_file", arguments={"path": "a.txt"}),
        ToolCallItem(id="call_2", name="search_code", arguments={"query": "foo"}),
    ]

    with patch.object(llm_client_mock, "chat") as mock_chat:
        mock_chat.side_effect = [
            ParsedAssistantResponse(content="Okuyorum", tool_calls=tool_calls),
            ParsedAssistantResponse(content="Bitti", tool_calls=[]),
        ]
        res = agent.execute_task(task)
        assert res.success is True
        assert res.tool_calls_count == 2
        assert tool_executor_mock.execute.call_count == 2


def test_ast_symbol_contract_sharing(tmp_path: Path):
    """Verify ast symbol contract sharing."""
    symbol_engine_mock = MagicMock()
    mock_sym = MagicMock()
    mock_sym.kind = "class"
    mock_sym.name = "UserModel"
    symbol_engine_mock.get_file_symbols.return_value = [mock_sym]

    tg = TaskGraph()
    t1 = EngineeringTask(
        task_id="t1",
        title="Modelleri Yaz",
        description="UserModel yaz",
        role=AgentRole.DEVELOPER,
        assigned_files=["models.py"],
        provides_contracts=["UserContract"],
    )
    t1.status = TaskStatus.COMPLETED
    t1.result = AgentTaskResult(
        task_id="t1",
        agent_id="dev1",
        role=AgentRole.DEVELOPER,
        success=True,
        output="UserModel sınıfı eklendi.",
        modified_files=["models.py"],
    )

    t2 = EngineeringTask(
        task_id="t2",
        title="Servisi Yaz",
        description="UserModel kullanan servis",
        role=AgentRole.DEVELOPER,
        dependencies=["t1"],
        consumes_contracts=["UserContract"],
    )

    tg.add_task(t1)
    tg.add_task(t2)

    sharing_engine = ContextSharingEngine(symbol_engine=symbol_engine_mock)
    ctx = sharing_engine.prepare_task_context(t2, tg)

    assert "contracts" in ctx
    assert "UserContract" in ctx["contracts"]
    contract_info = ctx["contracts"]["UserContract"]
    assert contract_info["provider_task_id"] == "t1"
    assert "class UserModel" in contract_info["contract_data"]
