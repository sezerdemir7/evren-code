"""Evren-Code Evaluation Mock LLM Provider:
Deterministic simulation engine to test TurnPipeline, Gatekeeper, ToolRunner,
and filesystem tools without live API keys or internet connection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from evren_code.config import AppConfig, ModelSpec
from evren_code.core.parser import ParsedAssistantResponse, ToolCallItem


class MockLLMClient:
    """Mock client simulating `chat` calls with deterministic steps for evaluation suites."""

    def __init__(self, config: AppConfig, task: Dict[str, Any]) -> None:
        self.config = config
        self.task = task
        self.step_index = 0
        self._active_model_id = "mock-glm-5.3"
        self.active_model = ModelSpec(
            id="mock-glm-5.3",
            name="Mock Model (Eval Mode)",
            task="chat",
            context_length=262144,
            max_output_tokens=16384,
            tools=True,
            thinking=True,
        )
        self._custom_steps: List[Dict[str, Any]] = list(task.get("mock_steps") or [])

    def close(self) -> None:
        pass

    def cancel(self) -> None:
        pass

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        model_override: Optional[str] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
        on_tool_speculate: Optional[Any] = None,
    ) -> ParsedAssistantResponse:
        # Execute custom mock steps sequentially if provided
        if self.step_index < len(self._custom_steps):
            step_cfg = self._custom_steps[self.step_index]
            self.step_index += 1

            t_calls: List[ToolCallItem] = []
            for raw_tc in step_cfg.get("tool_calls", []):
                t_calls.append(
                    ToolCallItem(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=raw_tc.get("name", ""),
                        arguments=dict(raw_tc.get("arguments", {})),
                        raw_arguments=str(raw_tc.get("arguments", {})),
                    )
                )

            return ParsedAssistantResponse(
                content=step_cfg.get("content", ""),
                thinking=step_cfg.get("thinking", "Task analyzed, executing step."),
                tool_calls=t_calls,
                finish_reason="stop",
                usage={"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
            )

        # Default fallback behavior: return completion message when mock steps are exhausted
        self.step_index += 1
        return ParsedAssistantResponse(
            content="All operations completed and verified successfully.",
            thinking="Required modifications completed.",
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
        )
