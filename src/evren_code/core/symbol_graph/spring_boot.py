"""Spring boot module."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from evren_code.core.symbol_graph.index import SymbolIndexDB
from evren_code.core.symbol_graph.models import CodeSymbol, RelationshipType


class SpringBootAnalyzer:
    """Spring boot analyzer implementation."""

    def __init__(self, db: SymbolIndexDB) -> None:
        self.db = db

    def analyze_architecture(self) -> Dict[str, Any]:
        """Analyze architecture."""
        controllers: List[Dict[str, Any]] = []
        services: List[Dict[str, Any]] = []
        repositories: List[Dict[str, Any]] = []
        entities: List[Dict[str, Any]] = []
        components: List[Dict[str, Any]] = []

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, name, fqn, file_path, signature, metadata_json
                FROM symbols
                WHERE language = 'java' AND symbol_type IN ('class', 'interface', 'record')
                """
            )
            for r in cursor.fetchall():
                import json
                meta = json.loads(r["metadata_json"] or "{}")
                annotations = [a.lower() for a in meta.get("annotations", [])]
                entry = {
                    "id": r["id"],
                    "name": r["name"],
                    "fqn": r["fqn"],
                    "file_path": r["file_path"],
                    "signature": r["signature"],
                    "annotations": meta.get("annotations", []),
                }

                if any("restcontroller" in a or "controller" in a for a in annotations):
                    controllers.append(entry)
                elif any("service" in a for a in annotations):
                    services.append(entry)
                elif any("repository" in a for a in annotations):
                    repositories.append(entry)
                elif any("entity" in a or "table" in a for a in annotations):
                    entities.append(entry)
                elif any("component" in a or "configuration" in a for a in annotations):
                    components.append(entry)

        return {
            "controllers": controllers,
            "services": services,
            "repositories": repositories,
            "entities": entities,
            "components": components,
            "total_spring_components": (
                len(controllers) + len(services) + len(repositories) + len(entities) + len(components)
            ),
        }

    def trace_component_chain(self, class_name: str) -> Dict[str, Any]:
        """Trace component chain."""
        arch = self.analyze_architecture()
        matching = None
        role = "unknown"

        for group_name, items in arch.items():
            if isinstance(items, list):
                for item in items:
                    if item["name"].lower() == class_name.lower() or item["fqn"].lower().endswith(f".{class_name.lower()}"):
                        matching = item
                        role = group_name[:-1]  # 'controller', 'service', vb.
                        break

        if not matching:
            return {"error": f"Spring Boot bileşeni '{class_name}' bulunamadı."}

        injected_dependencies = []
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT target_name, target_symbol_id, relationship_type, evidence
                FROM symbol_references
                WHERE source_symbol_id LIKE ? AND relationship_type IN ('REFERENCES', 'CALLS', 'IMPLEMENTS')
                """,
                (f"%::{matching['fqn']}%",),
            )
            for r in cursor.fetchall():
                injected_dependencies.append(dict(r))

        return {
            "component": matching,
            "spring_role": role,
            "dependencies": injected_dependencies,
        }
