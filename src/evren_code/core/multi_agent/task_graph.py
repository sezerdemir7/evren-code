"""Task graph module."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from evren_code.core.multi_agent.models import EngineeringTask, TaskStatus


class TaskGraphError(Exception):
    """Task graph error implementation."""
    pass


class TaskGraph:
    """Task graph implementation."""

    def __init__(self) -> None:
        self._tasks: Dict[str, EngineeringTask] = {}
        # edges: parent -> children (dependency -> dependent)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)

    @property
    def tasks(self) -> Dict[str, EngineeringTask]:
        return self._tasks

    def add_task(self, task: EngineeringTask) -> None:
        """Add task."""
        if not task.task_id:
            raise TaskGraphError("Görev ID boş olamaz.")
        self._tasks[task.task_id] = task
        for dep in task.dependencies:
            self._dependents[dep].add(task.task_id)

    def get_task(self, task_id: str) -> Optional[EngineeringTask]:
        return self._tasks.get(task_id)

    def add_dependency(self, task_id: str, depends_on_id: str) -> None:
        """Add dependency."""
        task = self._tasks.get(task_id)
        if task and depends_on_id not in task.dependencies:
            task.dependencies.append(depends_on_id)
            self._dependents[depends_on_id].add(task_id)

    def remove_dependency(self, task_id: str, depends_on_id: str) -> None:
        """Remove dependency."""
        task = self._tasks.get(task_id)
        if task and depends_on_id in task.dependencies:
            task.dependencies.remove(depends_on_id)
            self._dependents[depends_on_id].discard(task_id)

    def _rebuild_dependents(self) -> None:
        """Rebuild dependents."""
        self._dependents.clear()
        for tid, task in self._tasks.items():
            for dep in task.dependencies:
                self._dependents[dep].add(tid)

    def remove_task(self, task_id: str) -> None:
        """Remove task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
        if task_id in self._dependents:
            del self._dependents[task_id]
        for deps in self._dependents.values():
            deps.discard(task_id)

    def validate_dag(self) -> Tuple[bool, List[str]]:
        """Validate dag."""
        self._rebuild_dependents()
        errors: List[str] = []

        for tid, task in self._tasks.items():
            for dep in task.dependencies:
                if dep == tid:
                    errors.append(f"Öz-bağımlılık hatası: '{tid}' görevi kendisine bağımlı olamaz.")
                elif dep not in self._tasks:
                    errors.append(f"Eksik bağımlılık: '{tid}' görevinin bağımlı olduğu '{dep}' mevcut değil.")

        if errors:
            return False, errors

        in_degree: Dict[str, int] = {tid: len(task.dependencies) for tid, task in self._tasks.items()}
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for child in self._dependents.get(curr, set()):
                if child in in_degree:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if visited_count < len(self._tasks):
            cycle_nodes = [tid for tid, deg in in_degree.items() if deg > 0]
            errors.append(f"Çizgede döngüsel bağımlılık (çevrim) tespit edildi: {cycle_nodes}")
            return False, errors

        return True, []

    def topological_sort(self) -> List[str]:
        """Topological sort."""
        is_valid, errors = self.validate_dag()
        if not is_valid:
            raise TaskGraphError(f"Geçersiz görev çizgesi: {'; '.join(errors)}")

        in_degree: Dict[str, int] = {tid: len(task.dependencies) for tid, task in self._tasks.items()}
        queue = deque(sorted([tid for tid, deg in in_degree.items() if deg == 0]))
        result: List[str] = []

        while queue:
            curr = queue.popleft()
            result.append(curr)
            for child in sorted(self._dependents.get(curr, set())):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def get_ready_tasks(self) -> List[EngineeringTask]:
        """Get ready tasks."""
        ready: List[EngineeringTask] = []
        for task in self._tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.READY):
                deps_met = True
                for dep_id in task.dependencies:
                    dep_task = self._tasks.get(dep_id)
                    if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                        deps_met = False
                        break
                if deps_met:
                    task.status = TaskStatus.READY
                    ready.append(task)
        return ready

    def mark_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Mark task status."""
        task = self._tasks.get(task_id)
        if not task:
            return
        task.status = status

        if status in (TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED):
            self.cascade_failure(task_id)

    def cascade_failure(self, failed_task_id: str) -> None:
        """Cascade failure."""
        visited: Set[str] = set()
        queue = deque([failed_task_id])

        while queue:
            curr = queue.popleft()
            for child_id in self._dependents.get(curr, set()):
                if child_id not in visited:
                    visited.add(child_id)
                    child_task = self._tasks.get(child_id)
                    if child_task and not child_task.is_terminal:
                        child_task.status = TaskStatus.BLOCKED
                        queue.append(child_id)

    def all_completed(self) -> bool:
        """All completed."""
        if not self._tasks:
            return True
        return all(t.status == TaskStatus.COMPLETED for t in self._tasks.values())

    def is_finished(self) -> bool:
        """Is finished (COMPLETED, FAILED, BLOCKED, CANCELLED)."""
        if not self._tasks:
            return True
        return all(t.is_terminal for t in self._tasks.values())

    def to_ascii_tree(self) -> str:
        """To ascii tree."""
        lines: List[str] = ["Görev Çizgesi (DAG):"]
        status_icons = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.READY: "[*]",
            TaskStatus.ASSIGNED: "[~]",
            TaskStatus.RUNNING: "[/]",
            TaskStatus.COMPLETED: "[✔]",
            TaskStatus.FAILED: "[✖]",
            TaskStatus.BLOCKED: "[⊘]",
            TaskStatus.CANCELLED: "[-]",
        }
        for tid in self.topological_sort() if self.validate_dag()[0] else self._tasks.keys():
            t = self._tasks[tid]
            icon = status_icons.get(t.status, "[?]")
            deps_str = f" <- [{', '.join(t.dependencies)}]" if t.dependencies else ""
            files_str = f" [files: {', '.join(t.assigned_files)}]" if t.assigned_files else ""
            lines.append(f"  {icon} {t.task_id} ({t.role.value}): {t.title}{deps_str}{files_str}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": {tid: t.to_dict() for tid, t in self._tasks.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskGraph:
        graph = cls()
        tasks_data = data.get("tasks", {})
        for tid, tdict in tasks_data.items():
            graph.add_task(EngineeringTask.from_dict(tdict))
        return graph
