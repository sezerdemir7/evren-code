"""Test symbol graph module."""

import os
import tempfile
import time
import unittest
from pathlib import Path

from evren_code.config import load_config
from evren_code.core.agent import AgentOrchestrator
from evren_code.core.symbol_graph.engine import CodebaseSymbolEngine
from evren_code.core.symbol_graph.graph import SymbolGraphEngine
from evren_code.core.symbol_graph.impact_analyzer import ImpactAnalyzer
from evren_code.core.symbol_graph.index import SymbolIndexDB
from evren_code.core.symbol_graph.models import (
    CodeSymbol,
    FileImport,
    ParserConfidence,
    RelationshipType,
    ResolutionStatus,
    SymbolReference,
    SymbolType,
)
from evren_code.core.symbol_graph.parsers.go_parser import GoParser
from evren_code.core.symbol_graph.parsers.java_parser import JavaParser
from evren_code.core.symbol_graph.parsers.python_parser import PythonParser
from evren_code.core.symbol_graph.parsers.registry import ParserRegistry
from evren_code.core.symbol_graph.parsers.rust_parser import RustParser
from evren_code.core.symbol_graph.parsers.ts_js_parser import TypeScriptJavaScriptParser
from evren_code.core.symbol_graph.reference_resolver import ReferenceResolver
from evren_code.core.symbol_graph.spring_boot import SpringBootAnalyzer
from evren_code.core.symbol_graph.test_linker import TestLinker


class TestSymbolGraph(unittest.TestCase):
    """Test symbol graph implementation."""

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_python_nested_classes_and_methods(self):
        """Verify python nested classes and methods."""
        code = '''
class Outer:
    """Outer doc."""
    def outer_method(self, a: int) -> str:
        return str(a)

    class Inner:
        @staticmethod
        def inner_func():
            pass
'''
        parser = PythonParser()
        res = parser.parse("src/nested.py", code)
        self.assertFalse(res.is_partial)

        names = {s.name: s for s in res.symbols}
        self.assertIn("Outer", names)
        self.assertEqual(names["Outer"].symbol_type, SymbolType.CLASS)
        self.assertEqual(names["Outer"].docstring, "Outer doc.")

        self.assertIn("outer_method", names)
        self.assertEqual(names["outer_method"].symbol_type, SymbolType.METHOD)
        self.assertEqual(names["outer_method"].parent_symbol_id, "src/nested.py::Outer::class")

        self.assertIn("Inner", names)
        self.assertEqual(names["Inner"].parent_symbol_id, "src/nested.py::Outer::class")

        self.assertIn("inner_func", names)
        self.assertEqual(names["inner_func"].parent_symbol_id, "src/nested.py::Outer.Inner::class")

    def test_java_interfaces_implementations_annotations(self):
        """Verify java interfaces implementations annotations."""
        code = '''
package com.example.demo;

import org.springframework.stereotype.Service;

@Service
public class UserService implements IUserService {
    private final UserRepository repository;

    public UserService(UserRepository repository) {
        this.repository = repository;
    }

    @Override
    public User findById(Long id) {
        return repository.findUser(id);
    }
}
'''
        parser = JavaParser()
        res = parser.parse("src/UserService.java", code)

        names = {s.name for s in res.symbols}
        classes = [s for s in res.symbols if s.symbol_type == SymbolType.CLASS]
        self.assertEqual(len(classes), 1)
        u_class = classes[0]
        self.assertEqual(u_class.name, "UserService")
        self.assertEqual(u_class.symbol_type, SymbolType.CLASS)
        self.assertIn("@Service", u_class.metadata.get("annotations", []))
        self.assertIn("IUserService", u_class.metadata.get("interfaces", []))

        # Constructor
        ctors = [s for s in res.symbols if s.symbol_type == SymbolType.CONSTRUCTOR]
        self.assertEqual(len(ctors), 1)
        self.assertEqual(ctors[0].name, "UserService")

        self.assertIn("findById", names)
        self.assertTrue(any(r.target_name == "findUser" and r.relationship_type == RelationshipType.CALLS for r in res.references))

    def test_typescript_interfaces_types_and_arrow_functions(self):
        """Verify typescript interfaces types and arrow functions."""
        code = '''
import { Item } from './models';

export interface IItemService {
    getItem(id: string): Item;
}

export type ItemId = string | number;

export class ItemService implements IItemService {
    getItem(id: string): Item {
        return fetchItem(id);
    }
}

export const calculateTax = (amount: number): number => {
    return amount * 0.18;
};
'''
        parser = TypeScriptJavaScriptParser()
        res = parser.parse("src/items.ts", code)

        names = {s.name: s for s in res.symbols}
        self.assertIn("IItemService", names)
        self.assertEqual(names["IItemService"].symbol_type, SymbolType.INTERFACE)

        self.assertIn("ItemId", names)
        self.assertEqual(names["ItemId"].symbol_type, SymbolType.TYPE_ALIAS)

        self.assertIn("ItemService", names)
        self.assertEqual(names["ItemService"].symbol_type, SymbolType.CLASS)

        self.assertIn("calculateTax", names)
        self.assertEqual(names["calculateTax"].symbol_type, SymbolType.FUNCTION)

        self.assertTrue(any(r.target_name == "IItemService" and r.relationship_type == RelationshipType.IMPLEMENTS for r in res.references))

    def test_go_methods_and_structs(self):
        """Verify go methods and structs."""
        code = '''
package service

type Order struct {
    ID int
}

func (o *Order) Total() float64 {
    return 100.0
}

func CreateOrder() *Order {
    return &Order{ID: 1}
}
'''
        parser = GoParser()
        res = parser.parse("order.go", code)

        names = {s.name: s for s in res.symbols}
        self.assertIn("Order", names)
        self.assertEqual(names["Order"].symbol_type, SymbolType.STRUCT)

        self.assertIn("Total", names)
        self.assertEqual(names["Total"].symbol_type, SymbolType.METHOD)
        self.assertEqual(names["Total"].parent_symbol_id, "order.go::service.Order::struct")

        self.assertIn("CreateOrder", names)
        self.assertEqual(names["CreateOrder"].symbol_type, SymbolType.FUNCTION)

    def test_rust_impl_blocks_and_traits(self):
        """Verify rust impl blocks and traits."""
        code = '''
pub struct Config {
    pub port: u16,
}

pub trait Runner {
    fn run(&self);
}

impl Runner for Config {
    fn run(&self) {
        start_server();
    }
}
'''
        parser = RustParser()
        res = parser.parse("src/lib.rs", code)

        names = {s.name: s for s in res.symbols}
        self.assertIn("Config", names)
        self.assertEqual(names["Config"].symbol_type, SymbolType.STRUCT)

        self.assertIn("Runner", names)
        self.assertEqual(names["Runner"].symbol_type, SymbolType.TRAIT)

        self.assertIn("run", names)
        self.assertEqual(names["run"].symbol_type, SymbolType.METHOD)

        self.assertTrue(any(r.target_name == "Runner" and r.relationship_type == RelationshipType.IMPLEMENTS for r in res.references))

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_reference_resolution_same_file_and_imports(self):
        """Verify reference resolution same file and imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db = SymbolIndexDB(root)
            resolver = ReferenceResolver(db)

            s1 = CodeSymbol(
                id="utils.py::format_text::function",
                name="format_text",
                fully_qualified_name="utils.format_text",
                symbol_type=SymbolType.FUNCTION,
                language="python",
                file_path="utils.py",
                start_line=1,
                end_line=5,
                signature="def format_text(s: str) -> str",
            )
            db.save_file_data("utils.py", 100.0, "python", [s1], [], [])

            s2 = CodeSymbol(
                id="app.py::main::function",
                name="main",
                fully_qualified_name="app.main",
                symbol_type=SymbolType.FUNCTION,
                language="python",
                file_path="app.py",
                start_line=5,
                end_line=10,
                signature="def main()",
            )
            imp = FileImport(file_path="app.py", imported_module="utils", imported_symbols=["format_text"], line_number=1)
            ref = SymbolReference(
                source_file="app.py",
                source_symbol_id="app.py::main::function",
                target_name="format_text",
                target_symbol_id=None,
                relationship_type=RelationshipType.CALLS,
                line_number=7,
                resolution_status=ResolutionStatus.UNRESOLVED,
                evidence="call format_text()",
            )
            db.save_file_data("app.py", 100.0, "python", [s2], [imp], [ref])

            resolved, _, _ = resolver.resolve_all_unresolved()
            self.assertEqual(resolved, 1)

            with db.get_connection() as conn:
                cur = conn.execute("SELECT target_symbol_id, resolution_status FROM symbol_references WHERE source_file = 'app.py'")
                r = cur.fetchone()
                self.assertEqual(r["target_symbol_id"], "utils.py::format_text::function")
                self.assertEqual(r["resolution_status"], "RESOLVED")

    def test_comments_and_strings_not_treated_as_references(self):
        """Verify comments and strings not treated as references."""
        code = '''
# Bu fonksiyonda UserService veya DatabaseService kelimeleri geçiyor.
def process():
    msg = "OrderRepository silindi"
    return msg
'''
        parser = PythonParser()
        res = parser.parse("src/dummy.py", code)
        call_targets = [r.target_name for r in res.references if r.relationship_type == RelationshipType.CALLS]
        self.assertNotIn("UserService", call_targets)
        self.assertNotIn("DatabaseService", call_targets)
        self.assertNotIn("OrderRepository", call_targets)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_graph_traversal_and_cycle_safety(self):
        """Verify graph traversal and cycle safety."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db = SymbolIndexDB(root)
            graph = SymbolGraphEngine(db)

            sym_a = CodeSymbol(id="sym_a", name="fn_a", fully_qualified_name="fn_a", symbol_type=SymbolType.FUNCTION, language="python", file_path="a.py", start_line=1, end_line=5)
            sym_b = CodeSymbol(id="sym_b", name="fn_b", fully_qualified_name="fn_b", symbol_type=SymbolType.FUNCTION, language="python", file_path="b.py", start_line=1, end_line=5)

            ref_a_to_b = SymbolReference(source_file="a.py", source_symbol_id="sym_a", target_name="fn_b", target_symbol_id="sym_b", relationship_type=RelationshipType.CALLS, line_number=2, resolution_status=ResolutionStatus.RESOLVED)
            ref_b_to_a = SymbolReference(source_file="b.py", source_symbol_id="sym_b", target_name="fn_a", target_symbol_id="sym_a", relationship_type=RelationshipType.CALLS, line_number=2, resolution_status=ResolutionStatus.RESOLVED)

            db.save_file_data("a.py", 1.0, "python", [sym_a], [], [ref_a_to_b])
            db.save_file_data("b.py", 1.0, "python", [sym_b], [], [ref_b_to_a])

            steps = graph.traverse_graph("sym_a", direction="outgoing", max_depth=5, limit=10)
            self.assertEqual(len(steps), 2)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_incremental_indexing_workflow(self):
        """Verify incremental indexing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            f1 = root / "service.py"
            f1.write_text("def run_task(): pass\n", encoding="utf-8")
            f2 = root / "helper.py"
            f2.write_text("def compute(): pass\n", encoding="utf-8")

            engine = CodebaseSymbolEngine(root)

            stats1 = engine.index_codebase(force_reindex=False)
            self.assertEqual(stats1.files_parsed, 2)
            self.assertEqual(stats1.files_skipped, 0)
            self.assertEqual(stats1.symbols_extracted, 2)

            stats2 = engine.index_codebase(force_reindex=False)
            self.assertEqual(stats2.files_parsed, 0)
            self.assertEqual(stats2.files_skipped, 2)

            time.sleep(0.05)
            f1.write_text("def run_task(): pass\ndef extra(): pass\n", encoding="utf-8")
            f3 = root / "new_module.py"
            f3.write_text("class Config: pass\n", encoding="utf-8")

            stats3 = engine.index_codebase(force_reindex=False)
            self.assertEqual(stats3.files_parsed, 2)
            self.assertEqual(stats3.files_skipped, 1)

            f2.unlink()
            engine.index_codebase(force_reindex=False)
            self.assertIsNone(engine.db.get_symbol_by_id("helper.py::compute::function"))

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_spring_boot_controller_service_repository_impact_analysis(self):
        """Spring Boot Controller -> Service -> Repository zincirinde etki analizini test eder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # 1. Entity
            (root / "User.java").write_text("package com.demo;\npublic class User { private Long id; }", encoding="utf-8")

            # 2. Repository Interface
            (root / "UserRepository.java").write_text(
                """
                package com.demo;
                public interface UserRepository {
                    User findUser(Long id);
                }
                """,
                encoding="utf-8",
            )

            # 3. Service
            (root / "UserService.java").write_text(
                """
                package com.demo;
                import org.springframework.stereotype.Service;

                @Service
                public class UserService {
                    private final UserRepository userRepository;
                    public UserService(UserRepository repo) { this.userRepository = repo; }
                    public User getUser(Long id) {
                        return userRepository.findUser(id);
                    }
                }
                """,
                encoding="utf-8",
            )

            # 4. Controller
            (root / "UserController.java").write_text(
                """
                package com.demo;
                import org.springframework.web.bind.annotation.RestController;

                @RestController
                public class UserController {
                    private final UserService userService;
                    public UserController(UserService service) { this.userService = service; }
                    public User handleGet(Long id) {
                        return userService.getUser(id);
                    }
                }
                """,
                encoding="utf-8",
            )

            (root / "UserServiceTest.java").write_text(
                """
                package com.demo;
                import org.junit.jupiter.api.Test;

                public class UserServiceTest {
                    @Test
                    public void testGetUser() {
                        UserService s = new UserService(null);
                        s.getUser(1L);
                    }
                }
                """,
                encoding="utf-8",
            )

            engine = CodebaseSymbolEngine(root)
            engine.index_codebase(force_reindex=True)

            # UserService.getUser metodunun etki analizini yap
            user_service_sym = [s for s in engine.db.get_symbols_by_name("getUser") if "UserService" in s.id][0]
            report = engine.impact_analyzer.analyze_symbol(user_service_sym.id)

            self.assertTrue(any("UserController" in c["source_file"] for c in report.direct_callers))

            self.assertTrue(any("UserServiceTest" in t["test_file"] for t in report.related_test_files))

            report_text = engine.analyze_change_impact("getUser")
            self.assertIn("Değişiklik Etki Analizi", report_text)
            self.assertIn("UserController", report_text)
            self.assertIn("UserServiceTest", report_text)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_agent_tool_executor_governance_and_read_only(self):
        """Verify agent tool executor governance and read only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("def calculate_total(x: int): return x * 10\n", encoding="utf-8")

            cfg = load_config(workspace_root=root)
            orchestrator = AgentOrchestrator(config=cfg, system_prompt="Test agent")

            # 1. index_codebase
            out_index = orchestrator.tool_executor.execute(
                tool_name="index_codebase",
                arguments={"force": True},
                caller_role="main",
                caller_agent="unit_test",
                is_trust_mode=False,
            )
            self.assertIn("sembol indeksleme tamamlandı", out_index)

            # 2. search_symbols
            out_search = orchestrator.tool_executor.execute(
                tool_name="search_symbols",
                arguments={"query": "calculate_total"},
                caller_role="researcher",
                caller_agent="unit_test",
                is_trust_mode=False,
            )
            self.assertIn("calculate_total", out_search)

            # 3. find_symbol_definition
            out_def = orchestrator.tool_executor.execute(
                tool_name="find_symbol_definition",
                arguments={"name_or_id": "calculate_total"},
                caller_role="architect",
                caller_agent="unit_test",
                is_trust_mode=False,
            )
            self.assertIn("Sembol Tanımı: `calculate_total`", out_def)

            # 4. analyze_change_impact
            out_impact = orchestrator.tool_executor.execute(
                tool_name="analyze_change_impact",
                arguments={"symbol_name_or_id": "calculate_total"},
                caller_role="reviewer",
                caller_agent="unit_test",
                is_trust_mode=False,
            )
            self.assertIn("Değişiklik Etki Analizi", out_impact)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def test_performance_and_benchmark(self):
        """Verify performance and benchmark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            for i in range(15):
                (root / f"Service_{i}.java").write_text(
                    f"package com.bench;\npublic class Service_{i} {{\n    public void process_{i}() {{}}\n}}\n",
                    encoding="utf-8",
                )
                (root / f"module_{i}.py").write_text(
                    f"def worker_{i}(x):\n    return x + {i}\n",
                    encoding="utf-8",
                )

            engine = CodebaseSymbolEngine(root)

            t0 = time.perf_counter()
            stats_cold = engine.index_codebase(force_reindex=True)
            cold_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            stats_warm = engine.index_codebase(force_reindex=False)
            warm_sec = time.perf_counter() - t1

            t2 = time.perf_counter()
            impact_out = engine.analyze_change_impact("process_0")
            impact_sec = time.perf_counter() - t2

            db_size_bytes = engine.db.db_path.stat().st_size if engine.db.db_path.exists() else 0
            symbols_per_sec = stats_cold.symbols_extracted / cold_sec if cold_sec > 0 else 0

            print(
                f"\n[PHASE 2.2 BENCHMARK] Soğuk İndeksleme: {cold_sec * 1000:.2f}ms | "
                f"Sıcak İndeksleme: {warm_sec * 1000:.2f}ms | "
                f"Sembol/Saniye: {symbols_per_sec:.1f} | "
                f"Etki Analizi: {impact_sec * 1000:.2f}ms | "
                f"DB Boyutu: {db_size_bytes / 1024:.1f} KB | "
                f"Çıkarılan Sembol: {stats_cold.symbols_extracted}"
            )

            self.assertGreaterEqual(stats_cold.symbols_extracted, 30)
            self.assertLess(warm_sec, cold_sec)


if __name__ == "__main__":
    unittest.main()
