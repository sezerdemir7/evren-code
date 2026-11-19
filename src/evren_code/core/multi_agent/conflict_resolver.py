"""Conflict Resolver module."""

from __future__ import annotations

import fnmatch
import uuid
from typing import Dict, List, Set, Tuple

from evren_code.core.multi_agent.models import ConflictReport, EngineeringTask
from evren_code.core.multi_agent.task_graph import TaskGraph


def _patterns_conflict(p1: str, p2: str) -> bool:
    """Patterns conflict."""
    if p1 == p2:
        return True
    if p1 in ("*", "**", "/*", "./*") or p2 in ("*", "**", "/*", "./*"):
        return True
    if fnmatch.fnmatch(p1, p2) or fnmatch.fnmatch(p2, p1):
        return True
    norm1 = p1.rstrip("/*").rstrip("/") + "/"
    norm2 = p2.rstrip("/*").rstrip("/") + "/"
    if norm1.startswith(norm2) or norm2.startswith(norm1):
        if "/" in p1 or "/" in p2:
            return True
    return False


class ConflictResolver:
    """Conflict resolver implementation."""

    def __init__(self) -> None:
        pass

    def detect_conflicts(self, task_graph: TaskGraph) -> List[ConflictReport]:
        """Detect conflicts."""
        conflicts: List[ConflictReport] = []
        tasks = list(task_graph.tasks.values())
        n = len(tasks)

        reachable: Dict[str, Set[str]] = {}
        for t in tasks:
            visited = set()
            stack = list(t.dependencies)
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    if curr in task_graph.tasks:
                        stack.extend(task_graph.tasks[curr].dependencies)
            reachable[t.task_id] = visited

        for i in range(n):
            for j in range(i + 1, n):
                t1 = tasks[i]
                t2 = tasks[j]

                if t1.task_id in reachable.get(t2.task_id, set()) or t2.task_id in reachable.get(t1.task_id, set()):
                    continue

                set1 = [f.strip().lower().replace("\\", "/") for f in t1.assigned_files if f.strip()]
                set2 = [f.strip().lower().replace("\\", "/") for f in t2.assigned_files if f.strip()]

                conflicting_items: List[str] = []
                for f1 in set1:
                    for f2 in set2:
                        if _patterns_conflict(f1, f2):
                            if f1 == f2:
                                conflicting_items.append(f1)
                            else:
                                conflicting_items.append(f"{f1} ~ {f2}")

                if conflicting_items:
                    conflicting_items = sorted(list(dict.fromkeys(conflicting_items)))
                    report = ConflictReport(
                        conflict_id=f"cnf_{uuid.uuid4().hex[:6]}",
                        task_a_id=t1.task_id,
                        task_b_id=t2.task_id,
                        conflicting_files=conflicting_items,
                        resolution_strategy="sequentialize",
                        resolved=False,
                        details=(
                            f"Görev '{t1.task_id}' ve '{t2.task_id}' bağımsız ancak ortak veya çakışan dosyaları "
                            f"({', '.join(conflicting_items)}) değiştirmeyi hedefliyor."
                        ),
                    )
                    conflicts.append(report)

        return conflicts

    def resolve_conflicts(
        self,
        task_graph: TaskGraph,
        strategy: str = "sequentialize",
    ) -> List[ConflictReport]:
        """Resolve conflicts."""
        conflicts = self.detect_conflicts(task_graph)
        for report in conflicts:
            if strategy == "sequentialize":
                t2 = task_graph.get_task(report.task_b_id)
                if t2 and report.task_a_id not in t2.dependencies:
                    task_graph.add_dependency(report.task_b_id, report.task_a_id)
                    valid, errs = task_graph.validate_dag()
                    if valid:
                        report.resolved = True
                        report.details += f" -> Çözüldü: '{report.task_b_id}' görevi '{report.task_a_id}' sonrasına sıralandı."
                    else:
                        task_graph.remove_dependency(report.task_b_id, report.task_a_id)
                        t1 = task_graph.get_task(report.task_a_id)
                        if t1 and report.task_b_id not in t1.dependencies:
                            task_graph.add_dependency(report.task_a_id, report.task_b_id)
                            valid2, _ = task_graph.validate_dag()
                            if valid2:
                                report.resolved = True
                                report.details += f" -> Çözüldü: '{report.task_a_id}' görevi '{report.task_b_id}' sonrasına sıralandı."
                            else:
                                task_graph.remove_dependency(report.task_a_id, report.task_b_id)
                                report.resolved = False
                                report.details += " -> Sıralama çevrim yarattığı için otomatik çözülemedi."
            else:
                report.resolved = False
                report.details += f" -> Desteklenmeyen veya uygulanamayan strateji: {strategy}"

        return conflicts
