"""Agent Registry module."""

from __future__ import annotations

import threading
import uuid
from typing import Dict, List, Optional

from evren_code.config import AppConfig
from evren_code.core.events import EventBus
from evren_code.core.llm_client import EvrenLLMClient
from evren_code.core.multi_agent.agent_factory import SpecialistAgent
from evren_code.core.multi_agent.file_ownership import FileOwnershipManager
from evren_code.core.multi_agent.models import AgentRole, AgentStatus
from evren_code.core.security import ToolExecutor
from evren_code.tools.registry import ToolRegistry


class AgentRegistry:
    """Agent registry implementation."""

    def __init__(
        self,
        config: AppConfig,
        llm_client: EvrenLLMClient,
        tool_executor: ToolExecutor,
        registry: ToolRegistry,
        file_ownership: FileOwnershipManager,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.registry = registry
        self.file_ownership = file_ownership
        self.event_bus = event_bus or EventBus()
        self._lock = threading.RLock()
        self._agents: Dict[str, SpecialistAgent] = {}

    def get_or_create_agent(self, role: AgentRole, agent_id: Optional[str] = None) -> SpecialistAgent:
        """Get or create agent."""
        with self._lock:
            if agent_id and agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.status = AgentStatus.RUNNING
                return agent

            for agent in self._agents.values():
                if agent.role == role:
                    if agent.status == AgentStatus.IDLE:
                        agent.status = AgentStatus.RUNNING
                        return agent
                    if role == AgentRole.PLANNER:
                        agent.status = AgentStatus.RUNNING
                        return agent

            new_id = agent_id or f"{role.value}_{uuid.uuid4().hex[:6]}"
            agent = SpecialistAgent(
                agent_id=new_id,
                role=role,
                config=self.config,
                llm_client=self.llm_client,
                tool_executor=self.tool_executor,
                registry=self.registry,
                file_ownership=self.file_ownership,
                event_bus=self.event_bus,
            )
            agent.status = AgentStatus.RUNNING
            self._agents[new_id] = agent
            return agent

    def list_agents(self) -> List[Dict[str, Any]]:
        """List agents."""
        with self._lock:
            return [
                {
                    "agent_id": a.agent_id,
                    "role": a.role.value,
                    "status": a.status.value,
                    "model": a.model_spec.id,
                }
                for a in self._agents.values()
            ]

    def reset_agents(self) -> None:
        """Reset agents."""
        with self._lock:
            for a in self._agents.values():
                a.status = AgentStatus.IDLE
